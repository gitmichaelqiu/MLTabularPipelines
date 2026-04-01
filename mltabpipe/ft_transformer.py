import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mltabpipe.common import StratifiedKFold, KFold, roc_auc_score, get_eval_score

try:
    import torch_frame
    from torch_frame import stype
    from torch_frame.data import Dataset, DataLoader as TFDataLoader
    from torch_frame.nn import FTTransformer
    from torch_frame.nn.encoder import EmbeddingEncoder, LinearEncoder
    TORCH_FRAME_AVAILABLE = True
except ImportError:
    TORCH_FRAME_AVAILABLE = False

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

def train_ft_transformer(
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
    Trains an FT-Transformer model using Cross Validation via torch-frame.
    """
    if not TORCH_FRAME_AVAILABLE:
        raise ImportError("torch-frame is not installed. Please run 'pip install pytorch-frame'.")

    if params is None:
        params = {
            'channels': 128,
            'num_layers': 3,
            'batch_size': 256,
            'lr': 1e-4,
            'epochs': 50,
            'patience': 10
        }

    if isinstance(random_states, int):
        random_states = [random_states]

    print(f"--- Training FT-Transformer Model ({task}) with {len(random_states)} seeds ---")
    
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    all_metrics = []
    
    # Define semantic types
    col_to_stype = {col: stype.numerical for col in num_features}
    for col in cat_features:
        col_to_stype[col] = stype.categorical
    
    target_stype = stype.categorical if task == 'classification' else stype.numerical

    for seed in random_states:
        print(f"Seed {seed}")
        if task == 'classification':
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_df, train_df[target_col])):
            print(f"Fold {fold + 1}/{n_folds}")
            
            df_train = train_df.iloc[train_idx].reset_index(drop=True)
            df_val = train_df.iloc[val_idx].reset_index(drop=True)
            
            # Create torch-frame Datasets
            train_dataset = Dataset(df_train, col_to_stype=col_to_stype, target=target_col)
            val_dataset = Dataset(df_val, col_to_stype=col_to_stype, target=target_col)
            test_dataset = Dataset(test_df, col_to_stype=col_to_stype, target=target_col)
            
            # Materialize train_dataset FIRST to get stats
            train_dataset.materialize()
            # Materialize val/test using train stats
            val_dataset.materialize(train_dataset.col_stats)
            test_dataset.materialize(train_dataset.col_stats)
            
            train_loader = TFDataLoader(train_dataset.tensor_frame, batch_size=params['batch_size'], shuffle=True)
            val_loader = TFDataLoader(val_dataset.tensor_frame, batch_size=params['batch_size'], shuffle=False)
            test_loader = TFDataLoader(test_dataset.tensor_frame, batch_size=params['batch_size'], shuffle=False)
            
            stype_encoder_dict = {
                stype.numerical: LinearEncoder(),
                stype.categorical: EmbeddingEncoder(),
            }
            
            model = FTTransformer(
                channels=params['channels'],
                out_channels=1, # binary classification or regression
                num_layers=params['num_layers'],
                col_stats=train_dataset.col_stats,
                col_names_dict=train_dataset.col_names_dict,
                stype_encoder_dict=stype_encoder_dict,
            ).to(DEVICE)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'])
            criterion = nn.BCEWithLogitsLoss() if task == 'classification' else nn.MSELoss()
            
            best_val_score = -np.inf if task == 'classification' else np.inf
            best_model_state = None
            patience_counter = 0
            
            for epoch in range(params['epochs']):
                model.train()
                for tf in train_loader:
                    tf = tf.to(DEVICE)
                    optimizer.zero_grad()
                    out = model(tf).squeeze()
                    loss = criterion(out, tf.y.to(torch.float32))
                    loss.backward()
                    optimizer.step()
                
                model.eval()
                val_fold_preds = []
                with torch.no_grad():
                    for tf in val_loader:
                        tf = tf.to(DEVICE)
                        out = model(tf).squeeze()
                        if task == 'classification':
                            val_fold_preds.append(torch.sigmoid(out).cpu().numpy())
                        else:
                            val_fold_preds.append(out.cpu().numpy())
                
                val_fold_preds = np.concatenate(val_fold_preds)
                current_score = get_eval_score(df_val[target_col], val_fold_preds, task)
                
                improved = (task == 'classification' and current_score > best_val_score) or \
                           (task == 'regression' and current_score < best_val_score)
                
                if improved:
                    best_val_score = current_score
                    best_model_state = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= params['patience']:
                    break
            
            # Load best model
            model.load_state_dict(best_model_state)
            model.eval()
            
            # Final Validation and Test Preds
            val_fold_preds = []
            test_fold_preds = []
            with torch.no_grad():
                for tf in val_loader:
                    tf = tf.to(DEVICE)
                    out = model(tf).squeeze()
                    val_fold_preds.append(torch.sigmoid(out).cpu().numpy() if task == 'classification' else out.cpu().numpy())
                for tf in test_loader:
                    tf = tf.to(DEVICE)
                    out = model(tf).squeeze()
                    test_fold_preds.append(torch.sigmoid(out).cpu().numpy() if task == 'classification' else out.cpu().numpy())
            
            val_fold_preds = np.concatenate(val_fold_preds)
            test_fold_preds = np.concatenate(test_fold_preds)
            
            oof_preds[val_idx] += val_fold_preds / len(random_states)
            test_preds += (test_fold_preds / n_folds) / len(random_states)
            
            all_metrics.append(best_val_score)
            print(f"Fold {fold + 1} Score: {best_val_score:.5f}")
            
    overall_score = get_eval_score(train_df[target_col], oof_preds, task)
    print(f"Overall OOF Score (Ensembled): {overall_score:.5f}")
    return oof_preds, test_preds, all_metrics
