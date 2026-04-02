import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

try:
    import cuml
    from cuml.preprocessing import TargetEncoder as cuTargetEncoder
    from cuml.linear_model import LogisticRegression as cuLogReg
    CUML_AVAILABLE = True
except ImportError:
    # We fallback to sklearn for LogisticRegression but TargetEncoder is tricky in older sklearn
    from sklearn.linear_model import LogisticRegression as skLogReg
    try:
        from sklearn.preprocessing import TargetEncoder as skTargetEncoder
    except ImportError:
        skTargetEncoder = None
    CUML_AVAILABLE = False

from mltabpipe.core.common import get_eval_score

def train_te_logit_model(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    features: list, 
    target_col: str, 
    params: dict = None, 
    task: str = 'classification',
    n_folds: int = 5, 
    random_states: list = [42]
):
    """
    Target Encoding + Logistic Regression model.
    Inspired by Chris Deotte's te_logit implementation.
    """
    if params is None:
        params = {'C': 0.1, 'max_iter': 1000}

    if isinstance(random_states, int):
        random_states = [random_states]

    print(f"--- Training TE-Logit Model ({task}) ---")
    
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    
    # We only apply Target Encoding to categorical features (strings/objects)
    cat_features = train_df[features].select_dtypes(include=['object', 'category']).columns.tolist()
    num_features = [f for f in features if f not in cat_features]
    
    for seed in random_states:
        print(f"Seed {seed}")
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_df, train_df[target_col])):
            print(f"Fold {fold + 1}/{n_folds}")
            
            X_train, y_train = train_df.iloc[train_idx], train_df.iloc[train_idx][target_col]
            X_val, y_val = train_df.iloc[val_idx], train_df.iloc[val_idx][target_col]
            X_test = test_df.copy()
            
            # Target Encoding
            if cat_features:
                if CUML_AVAILABLE:
                    te = cuTargetEncoder(n_folds=5, smooth='auto')
                    X_train_cat_te = te.fit_transform(X_train[cat_features], y_train)
                    X_val_cat_te = te.transform(X_val[cat_features])
                    X_test_cat_te = te.transform(X_test[cat_features])
                else:
                    if skTargetEncoder is None:
                        print("Warning: sklearn.preprocessing.TargetEncoder not available. Skipping categorical features.")
                        X_train_cat_te = pd.DataFrame(index=X_train.index)
                        X_val_cat_te = pd.DataFrame(index=X_val.index)
                        X_test_cat_te = pd.DataFrame(index=X_test.index)
                    else:
                        te = skTargetEncoder()
                        X_train_cat_te = te.fit_transform(X_train[cat_features], y_train)
                        X_val_cat_te = te.transform(X_val[cat_features])
                        X_test_cat_te = te.transform(X_test[cat_features])
            else:
                X_train_cat_te, X_val_cat_te, X_test_cat_te = None, None, None

            # Combine numeric and TE features
            def combine_features(df_num, df_cat_te):
                if df_cat_te is not None and not df_cat_te.empty:
                    if isinstance(df_cat_te, np.ndarray):
                        return np.column_stack([df_num.values, df_cat_te])
                    return np.column_stack([df_num.values, df_cat_te.values])
                return df_num.values

            X_tr_final = combine_features(X_train[num_features], X_train_cat_te)
            X_va_final = combine_features(X_val[num_features], X_val_cat_te)
            X_te_final = combine_features(X_test[num_features], X_test_cat_te)
            
            # Logistic Regression
            if CUML_AVAILABLE:
                model = cuLogReg(**params)
            else:
                model = skLogReg(**params, solver='lbfgs')
            
            model.fit(X_tr_final.astype(np.float32), y_train)
            
            val_preds = model.predict_proba(X_va_final.astype(np.float32))[:, 1]
            test_fold_preds = model.predict_proba(X_te_final.astype(np.float32))[:, 1]
            
            oof_preds[val_idx] += val_preds / len(random_states)
            test_preds += (test_fold_preds / n_folds) / len(random_states)
            
    overall_score = get_eval_score(train_df[target_col], oof_preds, task)
    print(f"Overall OOF Score: {overall_score:.5f}")
    return oof_preds, test_preds