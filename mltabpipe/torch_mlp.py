import time
from mltabpipe.common import np, pd, StratifiedKFold, StandardScaler, roc_auc_score

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    pass

def make_vocab_maps(train_df: pd.DataFrame, cols):
    maps, sizes = {}, {}
    for c in cols:
        uniq = pd.Series(train_df[c].values).astype(str).unique().tolist()
        v2i = {v: i + 1 for i, v in enumerate(uniq)}
        maps[c], sizes[c] = v2i, len(v2i) + 1
    return maps, sizes

def encode_with_maps(df: pd.DataFrame, cols, maps):
    X = np.zeros((len(df), len(cols)), dtype=np.int64)
    for j, c in enumerate(cols):
        X[:, j] = pd.Series(df[c].values).astype(str).map(maps[c]).fillna(0).astype(np.int64).values
    return X

def emb_dim_from_card(card: int) -> int:
    return int(np.clip(int(round(1.8 * (card ** 0.25))), 4, 64))

def build_numeric_snapper(train_series: pd.Series, rare_min_count: int):
    s = pd.to_numeric(train_series, errors="coerce").astype(np.float32)
    vc = pd.Series(s).value_counts(dropna=False)
    frequent_vals = vc[vc >= rare_min_count].index.values
    frequent_vals = np.array([v for v in frequent_vals if pd.notna(v)], dtype=np.float32)
    if frequent_vals.size == 0: frequent_vals = np.array(pd.Series(s.dropna()).unique(), dtype=np.float32)
    frequent_vals = np.sort(frequent_vals)
    frequent_set = set(frequent_vals.tolist())

    def transform(series_any: pd.Series):
        x = pd.to_numeric(series_any, errors="coerce").astype(np.float32).values
        is_nan = np.isnan(x)
        is_rare = np.ones_like(x, dtype=np.int32)
        for i, v in enumerate(x):
            is_rare[i] = 1 if np.isnan(v) else (0 if float(v) in frequent_set else 1)
        x_snapped = x.copy()
        if frequent_vals.size > 0:
            idx_snap = np.where((~is_nan) & (is_rare == 1))[0]
            if idx_snap.size > 0:
                v = x[idx_snap]
                pos = np.clip(np.searchsorted(frequent_vals, v), 0, len(frequent_vals) - 1)
                left = np.clip(pos - 1, 0, len(frequent_vals) - 1)
                choose_right = (np.abs(v - frequent_vals[pos]) <= np.abs(v - frequent_vals[left]))
                x_snapped[idx_snap] = np.where(choose_right, frequent_vals[pos], frequent_vals[left]).astype(np.float32)
        return x_snapped.astype(np.float32), is_rare.astype(np.int32)
    return transform

class TabMixDataset(Dataset):
    def __init__(self, X_cat, X_num, y=None):
        self.Xc = torch.as_tensor(X_cat, dtype=torch.long)
        self.Xn = torch.as_tensor(X_num, dtype=torch.float32)
        self.y  = None if y is None else torch.as_tensor(y, dtype=torch.float32)
    def __len__(self): return self.Xc.shape[0]
    def __getitem__(self, idx): return (self.Xc[idx], self.Xn[idx]) if self.y is None else (self.Xc[idx], self.Xn[idx], self.y[idx])

class EmbMLP_Mixed(nn.Module):
    def __init__(self, cat_cardinals, n_num, hidden=(256, 128), emb_dropout=0.1, mlp_dropout=0.2):
        super().__init__()
        self.emb_layers = nn.ModuleList([nn.Embedding(c, emb_dim_from_card(c)) for c in cat_cardinals])
        self.emb_drop = nn.Dropout(emb_dropout)
        in_dim = sum(emb_dim_from_card(c) for c in cat_cardinals) + n_num
        layers = []
        for h in hidden:
            layers.extend([nn.Linear(in_dim, h), nn.LayerNorm(h), nn.ReLU(inplace=True), nn.Dropout(mlp_dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)
        for emb in self.emb_layers: nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(self, x_cat, x_num):
        embs = [emb(x_cat[:, j]) for j, emb in enumerate(self.emb_layers)]
        z = torch.cat(embs, dim=1) if embs else x_num
        if embs: z = torch.cat([self.emb_drop(z), x_num], dim=1)
        return self.mlp(z).squeeze(1)

@torch.no_grad()
def pytorch_predict_proba(model, loader):
    model.eval()
    return np.concatenate([torch.sigmoid(model(b[0].to(DEVICE), b[1].to(DEVICE))).cpu().numpy() for b in loader])

class SmoothBCE(nn.Module):
    def __init__(self, eps=0.02):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets):
        return nn.functional.binary_cross_entropy_with_logits(logits, targets * (1 - self.eps) + 0.5 * self.eps)

def train_pytorch_mlp_model(train_df, test_df, num_features, cat_features, target_col, params=None, n_folds=5, random_states=[42]):
    if params is None:
        params = {'epochs': 10, 'batch_size': 256, 'lr': 2.5e-5, 'patience': 10, 'weight_decay': 3e-4, 
                  'emb_dropout': 0.10, 'mlp_dropout': 0.30, 'hidden': (512, 256), 'warmup_epochs': 1, 'rare_min_count': 25}
                  
    if isinstance(random_states, int):
        random_states = [random_states]

    print(f"--- Training PyTorch MLP Model with {len(random_states)} seeds ---")
    y = train_df[target_col].values.astype(np.float32)
    
    oof_preds = np.zeros(len(train_df), dtype=np.float32)
    test_preds = np.zeros(len(test_df), dtype=np.float32)
    all_metrics = []

    for seed in random_states:
        print(f"Seed {seed}")
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(train_df)), y)):
            print(f"Fold {fold + 1}/{n_folds}")
            fold_t0 = time.time()
            
            tr_df, va_df = train_df.iloc[tr_idx].reset_index(drop=True), train_df.iloc[va_idx].reset_index(drop=True)
            tr_cat_df, va_cat_df, te_cat_df = tr_df[cat_features].copy(), va_df[cat_features].copy(), test_df[cat_features].copy()
            
            for col in num_features:
                snapper = build_numeric_snapper(tr_df[col], params['rare_min_count'])
                tr_cat_df[f"{col}__cat"], tr_cat_df[f"{col}__is_rare"] = [v.astype(str) for v in snapper(tr_df[col])]
                va_cat_df[f"{col}__cat"], va_cat_df[f"{col}__is_rare"] = [v.astype(str) for v in snapper(va_df[col])]
                te_cat_df[f"{col}__cat"], te_cat_df[f"{col}__is_rare"] = [v.astype(str) for v in snapper(test_df[col])]

            CAT_ALL = list(tr_cat_df.columns)
            maps, sizes = make_vocab_maps(tr_cat_df, CAT_ALL)
            Xc_tr, Xc_va, Xc_te = encode_with_maps(tr_cat_df, CAT_ALL, maps), encode_with_maps(va_cat_df, CAT_ALL, maps), encode_with_maps(te_cat_df, CAT_ALL, maps)

            scaler = StandardScaler()
            Xn_tr = scaler.fit_transform(tr_df[num_features].astype(np.float32)).astype(np.float32)
            Xn_va, Xn_te = scaler.transform(va_df[num_features].astype(np.float32)).astype(np.float32), scaler.transform(test_df[num_features].astype(np.float32)).astype(np.float32)

            y_tr, y_va = y[tr_idx], y[va_idx]
            model = EmbMLP_Mixed([sizes[c] for c in CAT_ALL], Xn_tr.shape[1], params['hidden'], params['emb_dropout'], params['mlp_dropout']).to(DEVICE)
            opt, loss_fn = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay']), SmoothBCE(0.02)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, params['epochs'] - params['warmup_epochs']), eta_min=params['lr'] * 0.05)

            dl_tr = DataLoader(TabMixDataset(Xc_tr, Xn_tr, y_tr), batch_size=params['batch_size'], shuffle=True)
            dl_va = DataLoader(TabMixDataset(Xc_va, Xn_va, y_va), batch_size=params['batch_size'], shuffle=False)
            dl_te = DataLoader(TabMixDataset(Xc_te, Xn_te), batch_size=params['batch_size'], shuffle=False)

            best_auc, best_state, bad = -1.0, None, 0

            for epoch in range(1, params['epochs'] + 1):
                model.train()
                if epoch <= params['warmup_epochs']:
                    for pg in opt.param_groups: pg["lr"] = params['lr'] * (0.1 + 0.9 * (epoch / params['warmup_epochs']))
                for xc, xn, yb in dl_tr:
                    opt.zero_grad(set_to_none=True)
                    loss_fn(model(xc.to(DEVICE), xn.to(DEVICE)), yb.to(DEVICE)).backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                if epoch > params['warmup_epochs']: sched.step()
                
                auc = roc_auc_score(y_va, pytorch_predict_proba(model, dl_va))
                if auc > best_auc + 1e-6:
                    best_auc, bad, best_state = auc, 0, {k: v.cpu().clone() for k, v in model.state_dict().items()}
                elif (bad := bad + 1) >= params['patience']: break

            if best_state: model.load_state_dict(best_state)
            
            val_preds = pytorch_predict_proba(model, dl_va)
            test_fold_preds = pytorch_predict_proba(model, dl_te)
            
            oof_preds[va_idx] += val_preds / len(random_states)
            test_preds += (test_fold_preds / n_folds) / len(random_states)
            
            all_metrics.append(best_auc)
            print(f"Fold {fold + 1} Best AUC: {best_auc:.6f} | Time: {time.time()-fold_t0:.1f}s")

    print(f"Overall OOF ROC AUC (Ensembled): {roc_auc_score(y, oof_preds):.5f}\n" + "-" * 30)
    return oof_preds, test_preds, all_metrics