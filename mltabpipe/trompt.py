import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mltabpipe.common import StratifiedKFold, KFold, roc_auc_score, get_eval_score

try:
    import torch_frame
    from torch_frame import stype
    from torch_frame.data import Dataset, DataLoader as TFDataLoader
    from torch_frame.nn import Trompt
    TORCH_FRAME_AVAILABLE = True
except ImportError:
    TORCH_FRAME_AVAILABLE = False

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

def train_trompt_model(
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
    Trains a Trompt model using Cross Validation via torch-frame.
    Trompt uses prompt-based learning and layer-wise supervision.
    """
    if not TORCH_FRAME_AVAILABLE:
        raise ImportError("torch-frame is not installed. Please run 'pip install pytorch-frame'.")

    # Defaults from 1st place solution / torch-frame docs
    if params is None:
        params = {
            'channels': 128,
            'num_prompts': 128,
            'num_layers': 6,
            'batch_size': 256,
            'lr': 1e-4,
            'epochs': 30,
            'patience': 10
        }

    if isinstance(random_states, int):
        random_states = [random_states]

    print(f"--- Training Trompt Model ({task}) with {len(random_states)} seeds ---")
    
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    
    col_to_stype = {col: stype.numerical for col in num_features}
    for col in cat_features:
        col_to_stype[col] = stype.categorical

    y_true = train_df[target_col].values

    for seed in random_states:
        print(f"Seed {seed}")
        if task == 'classification':
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_df, y_true)):
            print(f"Fold {fold + 1}/{n_folds}")
            
            df_train = train_df.iloc[train_idx].reset_index(drop=True)
            df_val = train_df.iloc[val_idx].reset_index(drop=True)
            
            train_dataset = Dataset(df_train, col_to_stype=col_to_stype, target_col=target_col)
            val_dataset = Dataset(df_val, col_to_stype=col_to_stype, target_col=target_col)
            test_dataset = Dataset(test_df, col_to_stype=col_to_stype, target_col=target_col)
            
            train_dataset.materialize()
            val_dataset.materialize(train_dataset.col_stats)
            test_dataset.materialize(train_dataset.col_stats)
            
            train_loader = TFDataLoader(train_dataset.tensor_frame, batch_size=params['batch_size'], shuffle=True)
            val_loader = TFDataLoader(val_dataset.tensor_frame, batch_size=params['batch_size'], shuffle=False)
            test_loader = TFDataLoader(test_dataset.tensor_frame, batch_size=params['batch_size'], shuffle=False)
            
            # Trompt typically uses 1 out_channel for binary or regression
            model = Trompt(
                channels=params['channels'],
                out_channels=1,
                num_prompts=params['num_prompts'],
                num_layers=params['num_layers'],
                col_stats=train_dataset.col_stats,
                col_names_dict=train_dataset.tensor_frame.col_names_dict,
            ).to(DEVICE)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'])
            criterion = nn.BCEWithLogitsLoss() if task == 'classification' else nn.MSELoss()
            
            best_val_score = -np.inf if task == 'classification' else np.inf
            best_state = None
            patience_cnt = 0
            
            for epoch in range(params['epochs']):
                model.train()
                for tf in train_loader:
                    tf = tf.to(DEVICE)
                    optimizer.zero_grad()
                    # Trompt returns (batch_size, num_layers, out_channels)
                    out = model(tf) 
                    # Layer-wise supervision: loss is aggregated over all layers
                    loss = 0
                    for layer in range(params['num_layers']):
                        loss += criterion(out[:, layer, 0], tf.y.to(torch.float32))
                    loss /= params['num_layers']
                    loss.backward()
                    optimizer.step()
                
                model.eval()
                val_fold_preds_list = []
                with torch.no_grad():
                    for tf in val_loader:
                        tf = tf.to(DEVICE)
                        out = model(tf)
                        # Average predictions across all layers for final prediction
                        layer_avg_out = out.mean(dim=1).squeeze()
                        preds = torch.sigmoid(layer_avg_out).cpu().numpy() if task == 'classification' else layer_avg_out.cpu().numpy()
                        val_fold_preds_list.append(preds)
                
                val_fold_preds = np.concatenate(val_fold_preds_list)
                score = get_eval_score(y_true[val_idx], val_fold_preds, task)
                
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
                for tf in val_loader:
                    tf = tf.to(DEVICE)
                    out = model(tf).mean(dim=1).squeeze()
                    v_preds.append(torch.sigmoid(out).cpu().numpy() if task == 'classification' else out.cpu().numpy())
            
            t_preds = []
            with torch.no_grad():
                for tf in test_loader:
                    tf = tf.to(DEVICE)
                    out = model(tf).mean(dim=1).squeeze()
                    t_preds.append(torch.sigmoid(out).cpu().numpy() if task == 'classification' else out.cpu().numpy())

            oof_preds[val_idx] += np.concatenate(v_preds) / len(random_states)
            test_preds += (np.concatenate(t_preds) / n_folds) / len(random_states)
            print(f"Fold {fold+1} Best Score: {best_val_score:.5f}")

    overall_score = get_eval_score(y_true, oof_preds, task)
    print(f"Overall OOF Score: {overall_score:.5f}")
    return oof_preds, test_preds
