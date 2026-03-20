from mltabpipe.common import (
    pd, np, plt, sns, 
    StratifiedKFold, train_test_split, 
    StandardScaler, roc_auc_score, get_eval_score,
    TargetEncoder as skTargetEncoder,
    LogisticRegression as skLogReg
)

# RAPIDS GPU Imports (Ensure your Kaggle environment has GPU enabled!)
try:
    import cudf
    import cupy as cp
    from cuml.preprocessing import TargetEncoder
    from cuml.linear_model import LogisticRegression as cuLogReg
except ImportError:
    cudf = cp = TargetEncoder = cuLogReg = None
    print("Warning: RAPIDS (cudf, cuml, cupy) not found. GPU LogReg will not work.")

# ============================================================
# HELPER FUNCTIONS FOR GPU LOGISTIC REGRESSION
# ============================================================
def label_encode_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list):
    """Encodes categorical features consistently across train and test sets."""
    train_out = train_df.copy()
    test_out  = test_df.copy()

    for c in cols:
        tr, te = train_out[c], test_out[c]
        tr_s = tr.astype("string").fillna("__MISSING__")
        te_s = te.astype("string").fillna("__MISSING__")

        all_vals = pd.concat([tr_s, te_s], axis=0)
        uniq = pd.Index(all_vals.unique())

        mapping = pd.Series(np.arange(len(uniq), dtype=np.int32), index=uniq)
        train_out[c] = tr_s.map(mapping).astype(np.int32)
        test_out[c]  = te_s.map(mapping).astype(np.int32)

    return train_out, test_out

def _clip01(x, eps=1e-5):
    return np.clip(x, eps, 1.0 - eps)

def _logit(x, eps=1e-5):
    x = _clip01(x, eps)
    return np.log(x / (1.0 - x)).astype(np.float32)

def make_logit3_features(tr_m, va_m, te_m, eps=1e-5):
    """Generates z, z^2, z^3 polynomial features from Target Encoded probabilities."""
    z_tr, z_va, z_te = _logit(tr_m, eps=eps), _logit(va_m, eps=eps), _logit(te_m, eps=eps)
    X_tr = np.hstack([z_tr, z_tr**2, z_tr**3]).astype(np.float32)
    X_va = np.hstack([z_va, z_va**2, z_va**3]).astype(np.float32)
    X_te = np.hstack([z_te, z_te**2, z_te**3]).astype(np.float32)
    return X_tr, X_va, X_te

# ============================================================
# LEVEL 1 MODELS: GPU / CPU PIPELINE
# ============================================================
def train_te_logit_model(train_df, test_df, features, target_col, n_folds=5, te_n_folds=5, random_states=[42], use_gpu=True):
    """
    Trains a Logistic Regression model using pairwise Target Encoding with Seed Ensembling.
    Automatically uses GPU if available, otherwise falls back to CPU.
    """
    if isinstance(random_states, int):
        random_states = [random_states]

    # Check if GPU libraries were successfully imported earlier
    has_gpu = use_gpu and 'cuLogReg' in globals()
    mode = "GPU" if has_gpu else "CPU"
    print(f"--- Training Pairwise TE Logistic Regression ({mode}) with {len(random_states)} seeds ---")
    
    # 1. Label Encode
    train_enc, test_enc = label_encode_train_test(train_df, test_df, features)
    
    y_all = train_df[target_col].values.astype(np.int32)
    N = len(train_df)
    n_feat = len(features)
    
    # 3. Create Pair combinations
    pair_cols = [(features[i], features[j]) for i in range(n_feat) for j in range(i + 1, n_feat)]
    n_pair = len(pair_cols)
    
    print(f"Num raw features: {n_feat} | Num pairs: {n_pair}")
    if not has_gpu:
        print("WARNING: Running pairwise iterations on CPU. This may take a while depending on feature count!")
        
    # 2. Move to GPU (Only if GPU is enabled)
    if has_gpu:
        X_all_g  = cudf.DataFrame({f: cudf.Series(train_enc[f].values).astype("int32") for f in features})
        X_test_g = cudf.DataFrame({f: cudf.Series(test_enc[f].values).astype("int32") for f in features})
    
    oof_preds = np.zeros(N, dtype=np.float32)
    test_preds = np.zeros(len(test_df), dtype=np.float32)
    all_metrics = []
    
    for seed in random_states:
        print(f"Seed {seed}")
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        # 4. Leak-Free Outer CV Loop
        for fold, (tr_idx, va_idx) in enumerate(kf.split(np.zeros(N), y_all)):
            print(f"Fold {fold + 1}/{n_folds}")
            
            y_tr, y_va = y_all[tr_idx], y_all[va_idx]
            
            tr_pair = np.zeros((len(tr_idx), n_pair), dtype=np.float32)
            va_pair = np.zeros((len(va_idx), n_pair), dtype=np.float32)
            te_pair = np.zeros((len(test_df), n_pair), dtype=np.float32)
            
            if has_gpu:
                # ---------------- GPU PATH (RAPIDS) ----------------
                tr_idx_g, va_idx_g = cudf.Series(tr_idx), cudf.Series(va_idx)
                X_tr_g, X_va_g = X_all_g.take(tr_idx_g), X_all_g.take(va_idx_g)
                y_tr_g = cudf.Series(y_tr)
                
                te = TargetEncoder(n_folds=te_n_folds, smooth=0, seed=seed + fold, split_method="random", stat="mean", output_type="cupy")
                
                for t, (f1, f2) in enumerate(pair_cols):
                    # We combine strings to create true interaction features
                    inter_tr = X_tr_g[f1].astype(str) + "_" + X_tr_g[f2].astype(str)
                    inter_va = X_va_g[f1].astype(str) + "_" + X_va_g[f2].astype(str)
                    inter_te = X_test_g[f1].astype(str) + "_" + X_test_g[f2].astype(str)
                    
                    tr_oof_cp = te.fit_transform(cudf.DataFrame({'inter': inter_tr}), y_tr_g)
                    tr_pair[:, t] = cp.asnumpy(tr_oof_cp).ravel().astype(np.float32)
                    
                    te.fit(cudf.DataFrame({'inter': inter_tr}), y_tr_g)
                    va_pair[:, t] = cp.asnumpy(te.transform(cudf.DataFrame({'inter': inter_va}))).ravel().astype(np.float32)
                    te_pair[:, t] = cp.asnumpy(te.transform(cudf.DataFrame({'inter': inter_te}))).ravel().astype(np.float32)
                    
            else:
                # ---------------- CPU PATH (Scikit-Learn) ----------------
                # sklearn's TargetEncoder natively handles smoothing and cross-fitting to prevent leaks
                te = skTargetEncoder(cv=te_n_folds, smooth="auto", random_state=seed + fold)
                
                for t, (f1, f2) in enumerate(pair_cols):
                    # Combine Pandas series for interaction
                    inter_tr = (train_enc.iloc[tr_idx][f1].astype(str) + "_" + train_enc.iloc[tr_idx][f2].astype(str)).values.reshape(-1, 1)
                    inter_va = (train_enc.iloc[va_idx][f1].astype(str) + "_" + train_enc.iloc[va_idx][f2].astype(str)).values.reshape(-1, 1)
                    inter_te = (test_enc[f1].astype(str) + "_" + test_enc[f2].astype(str)).values.reshape(-1, 1)
                    
                    tr_pair[:, t] = te.fit_transform(inter_tr, y_tr).ravel().astype(np.float32)
                    va_pair[:, t] = te.transform(inter_va).ravel().astype(np.float32)
                    te_pair[:, t] = te.transform(inter_te).ravel().astype(np.float32)
                
            # Feature Engineering & Scaling
            X_tr_raw, X_va_raw, X_te_raw = make_logit3_features(tr_pair, va_pair, te_pair)
            
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_raw).astype(np.float32)
            X_va = scaler.transform(X_va_raw).astype(np.float32)
            X_te = scaler.transform(X_te_raw).astype(np.float32)
            
            if has_gpu:
                meta = cuLogReg(penalty="l2", C=0.5, max_iter=4000, tol=1e-4, fit_intercept=True, verbose=0)
                meta.fit(cp.asarray(X_tr), cp.asarray(y_tr))
                oof_va = cp.asnumpy(meta.predict_proba(cp.asarray(X_va))[:, 1]).astype(np.float32)
                test_fold_preds = cp.asnumpy(meta.predict_proba(cp.asarray(X_te))[:, 1]).astype(np.float32)
            else:
                meta = skLogReg(penalty="l2", C=0.5, max_iter=4000, tol=1e-4, fit_intercept=True)
                meta.fit(X_tr, y_tr)
                oof_va = meta.predict_proba(X_va)[:, 1].astype(np.float32)
                test_fold_preds = meta.predict_proba(X_te)[:, 1].astype(np.float32)
            
            oof_preds[va_idx] += oof_va / len(random_states)
            test_preds += (test_fold_preds / n_folds) / len(random_states)
            
            fold_auc = roc_auc_score(y_va, oof_va)
            all_metrics.append(fold_auc)
            print(f"Fold {fold + 1} ROC AUC: {fold_auc:.5f}")

    print(f"Overall OOF ROC AUC (Ensembled): {roc_auc_score(y_all, oof_preds):.5f}")
    print("-" * 30)
    
    return oof_preds, test_preds, all_metrics