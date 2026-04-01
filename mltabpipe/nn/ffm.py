import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from mltabpipe.core.common import StratifiedKFold, KFold, roc_auc_score, get_eval_score, StandardScaler

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

class FFMModel(nn.Module):
    def __init__(self, cat_cardinalities, n_num, emb_dim=4):
        super().__init__()
        self.n_cat = len(cat_cardinalities)
        self.n_num = n_num
        self.n_fields = self.n_cat + self.n_num
        self.emb_dim = emb_dim
        
        # Linear part (first-order)
        self.cat_linear = nn.ModuleList([nn.Embedding(c, 1) for c in cat_cardinalities])
        self.num_linear = nn.Linear(n_num, 1)
        
        # FFM part (second-order)
        # For each feature, we need (n_fields) embeddings of size (emb_dim)
        # This can be very memory intensive if n_fields or card is high
        self.cat_ffm_embeddings = nn.ModuleList([
            nn.Embedding(c, self.n_fields * emb_dim) for c in cat_cardinalities
        ])
        
        # Numerical features can also be treated as fields with 1 "value" 
        # but here we'll simplify and only do cat-cat and cat-num interactions 
        # or just categorical FFM as in the writeup
        
    def forward(self, x_cat, x_num):
        # Linear part
        cat_lin = torch.sum(torch.stack([self.cat_linear[i](x_cat[:, i]) for i in range(self.n_cat)], dim=1), dim=1)
        num_lin = self.num_linear(x_num)
        linear_out = cat_lin + num_lin
        
        # FFM part (Categorical interactions only for simplicity/efficiency)
        # (Batch, n_cat, n_fields * k)
        cat_embs = torch.stack([self.cat_ffm_embeddings[i](x_cat[:, i]) for i in range(self.n_cat)], dim=1)
        cat_embs = cat_embs.view(-1, self.n_cat, self.n_fields, self.emb_dim)
        
        ffm_out = torch.zeros(x_cat.size(0), 1).to(DEVICE)
        
        for i in range(self.n_cat):
            for j in range(i + 1, self.n_cat):
                # Interaction between field i and field j
                # v_{i, f_j} * v_{j, f_i}
                v_ifj = cat_embs[:, i, j, :]
                v_jfi = cat_embs[:, j, i, :]
                dot = torch.sum(v_ifj * v_jfi, dim=1, keepdim=True)
                ffm_out += dot
                
        return (linear_out + ffm_out).squeeze()

class FFMDataset(Dataset):
    def __init__(self, X_cat, X_num, y=None):
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self): return self.X_cat.shape[0]
    def __getitem__(self, idx):
        if self.y is not None: return self.X_cat[idx], self.X_num[idx], self.y[idx]
        return self.X_cat[idx], self.X_num[idx]

def train_ffm_model(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    num_features: list, 
    cat_features: list, 
    target_col: str, 
    params: dict = None, 
    task: str = 'classification',
    n_folds: int = 5, 
    random_states: list = [42]
):
    """
    Trains a Field-aware Factorization Machine (FFM) using Cross Validation.
    """
    if params is None:
        params = {
            'emb_dim': 4, # Small for memory efficiency
            'lr': 0.005,
            'epochs': 20,
            'batch_size': 1024,
            'patience': 5
        }

    if isinstance(random_states, int):
        random_states = [random_states]

    print(f"--- Training FFM Model ({task}) with {len(random_states)} seeds ---")
    
    # Preprocessing
    X_num_train = train_df[num_features].fillna(0).values.astype(np.float32)
    X_num_test = test_df[num_features].fillna(0).values.astype(np.float32)
    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(X_num_train)
    X_num_test = scaler.transform(X_num_test)
    
    cat_cardinalities = []
    X_cat_train = np.zeros((len(train_df), len(cat_features)), dtype=np.int64)
    X_cat_test = np.zeros((len(test_df), len(cat_features)), dtype=np.int64)
    
    for i, col in enumerate(cat_features):
        codes, uniques = pd.factorize(pd.concat([train_df[col], test_df[col]]))
        X_cat_train[:, i] = codes[:len(train_df)]
        X_cat_test[:, i] = codes[len(train_df):]
        cat_cardinalities.append(len(uniques))

    y = train_df[target_col].values.astype(np.float32)
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))

    for seed in random_states:
        print(f"Seed {seed}")
        if task == 'classification':
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_df, y)):
            print(f"Fold {fold + 1}/{n_folds}")
            ds_tr = FFMDataset(X_cat_train[train_idx], X_num_train[train_idx], y[train_idx])
            ds_va = FFMDataset(X_cat_train[val_idx], X_num_train[val_idx], y[val_idx])
            dl_tr = DataLoader(ds_tr, batch_size=params['batch_size'], shuffle=True)
            dl_va = DataLoader(ds_va, batch_size=params['batch_size'], shuffle=False)
            
            model = FFMModel(cat_cardinalities, len(num_features), params['emb_dim']).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
            criterion = nn.BCEWithLogitsLoss() if task == 'classification' else nn.MSELoss()
            
            best_val_score = -np.inf if task == 'classification' else np.inf
            best_state = None
            patience_cnt = 0
            
            for epoch in range(params['epochs']):
                model.train()
                for xc, xn, yb in dl_tr:
                    xc, xn, yb = xc.to(DEVICE), xn.to(DEVICE), yb.to(DEVICE)
                    optimizer.zero_grad()
                    out = model(xc, xn)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()
                
                model.eval()
                v_preds_fold = []
                with torch.no_grad():
                    for xc, xn, yb in dl_va:
                        xc, xn = xc.to(DEVICE), xn.to(DEVICE)
                        out = model(xc, xn)
                        v_preds_fold.append(torch.sigmoid(out).cpu().numpy() if task == 'classification' else out.cpu().numpy())
                
                v_preds_fold = np.concatenate(v_preds_fold)
                score = get_eval_score(y[val_idx], v_preds_fold, task)
                
                improved = (task == 'classification' and score > best_val_score) or \
                           (task == 'regression' and score < best_val_score)
                
                if improved:
                    best_val_score = score
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                
                if patience_cnt >= params['patience']: break

            model.load_state_dict(best_state)
            model.eval()
            
            v_preds = []
            with torch.no_grad():
                for xc, xn, yb in dl_va:
                    xc, xn = xc.to(DEVICE), xn.to(DEVICE)
                    out = model(xc, xn)
                    v_preds.append(torch.sigmoid(out).cpu().numpy() if task == 'classification' else out.cpu().numpy())
            oof_preds[val_idx] += np.concatenate(v_preds) / len(random_states)
            
            ds_te = FFMDataset(X_cat_test, X_num_test)
            dl_te = DataLoader(ds_te, batch_size=params['batch_size'], shuffle=False)
            t_preds = []
            with torch.no_grad():
                for xc, xn in dl_te:
                    xc, xn = xc.to(DEVICE), xn.to(DEVICE)
                    out = model(xc, xn)
                    t_preds.append(torch.sigmoid(out).cpu().numpy() if task == 'classification' else out.cpu().numpy())
            test_preds += (np.concatenate(t_preds) / n_folds) / len(random_states)
            
            print(f"Fold {fold+1} Best Score: {best_val_score:.5f}")

    overall_score = get_eval_score(y, oof_preds, task)
    print(f"Overall OOF Score: {overall_score:.5f}")
    return oof_preds, test_preds
