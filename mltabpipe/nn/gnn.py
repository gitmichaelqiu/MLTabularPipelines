import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from mltabpipe.core.common import StratifiedKFold, KFold, roc_auc_score, get_eval_score, StandardScaler

try:
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv
    from torch_geometric.loader import NeighborLoader
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

try:
    import cuml
    from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

from sklearn.neighbors import NearestNeighbors

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        return x.squeeze()

def build_knn_graph(X, k=5):
    """ Builds a KNN graph and returns edge_index. """
    if CUML_AVAILABLE:
        knn = cuNearestNeighbors(n_neighbors=k+1) # +1 because the point itself is included
    else:
        knn = NearestNeighbors(n_neighbors=k+1)
    
    knn.fit(X)
    distances, indices = knn.kneighbors(X)
    
    # Construct edge_index (exclude self-loops)
    source_nodes = np.repeat(np.arange(X.shape[0]), k)
    target_nodes = indices[:, 1:].flatten() # index 0 is the node itself
    
    edge_index = torch.tensor(np.array([source_nodes, target_nodes]), dtype=torch.long)
    return edge_index

def train_gnn_sage_model(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    features: list, 
    target_col: str, 
    params: dict = None, 
    task: str = 'classification',
    n_folds: int = 5, 
    random_states: list = [42],
    k_neighbors: int = 5
):
    """
    Trains a GraphSAGE GNN model using Cross Validation.
    Constructs a KNN graph on the combined train+test data.
    """
    if not PYG_AVAILABLE:
        raise ImportError("torch_geometric is not installed. Please run 'pip install torch-geometric'.")

    if params is None:
        params = {
            'hidden_channels': 64,
            'num_layers': 2,
            'lr': 1e-3,
            'epochs': 100,
            'patience': 15,
            'batch_size': 1024
        }

    if isinstance(random_states, int):
        random_states = [random_states]

    print(f"--- Training GraphSAGE Model ({task}) with {len(random_states)} seeds ---")
    
    # Prepare data
    X_train_full = train_df[features].values.astype(np.float32)
    X_test_full = test_df[features].values.astype(np.float32)
    y_full = train_df[target_col].values
    
    # Scaling is crucial for KNN and GNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test_full)
    
    # Combine for graph construction (transductive approach)
    X_all = np.vstack([X_train_scaled, X_test_scaled])
    num_train = len(train_df)
    num_total = len(X_all)
    
    print("Building KNN Graph...")
    edge_index = build_knn_graph(X_all, k=k_neighbors)
    x_tensor = torch.tensor(X_all, dtype=torch.float32)
    
    oof_preds = np.zeros(num_train)
    test_preds = np.zeros(len(test_df))
    
    for seed in random_states:
        print(f"Seed {seed}")
        if task == 'classification':
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full, y_full)):
            print(f"Fold {fold + 1}/{n_folds}")
            
            # Masks for this fold (relative to the full concat x_tensor)
            # Train nodes are 0:num_train, test nodes are num_train:num_total
            train_mask = torch.zeros(num_total, dtype=torch.bool)
            train_mask[train_idx] = True
            
            val_mask = torch.zeros(num_total, dtype=torch.bool)
            val_mask[val_idx] = True
            
            y_tensor = torch.zeros(num_total)
            y_tensor[:num_train] = torch.tensor(y_full, dtype=torch.float32)
            
            model = GraphSAGEModel(
                in_channels=len(features),
                hidden_channels=params['hidden_channels'],
                out_channels=1,
                num_layers=params['num_layers']
            ).to(DEVICE)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
            criterion = nn.BCEWithLogitsLoss() if task == 'classification' else nn.MSELoss()
            
            # Using simple full-batch training for demonstration. 
            # For very large graphs, NeighborLoader should be used.
            x_tensor_dev = x_tensor.to(DEVICE)
            edge_index_dev = edge_index.to(DEVICE)
            y_tensor_dev = y_tensor.to(DEVICE)
            train_mask_dev = train_mask.to(DEVICE)
            val_mask_dev = val_mask.to(DEVICE)
            
            best_val_score = -np.inf if task == 'classification' else np.inf
            best_state = None
            patience_cnt = 0
            
            for epoch in range(params['epochs']):
                model.train()
                optimizer.zero_grad()
                out = model(x_tensor_dev, edge_index_dev)
                loss = criterion(out[train_mask_dev], y_tensor_dev[train_mask_dev])
                loss.backward()
                optimizer.step()
                
                model.eval()
                with torch.no_grad():
                    out = model(x_tensor_dev, edge_index_dev)
                    v_preds = torch.sigmoid(out[val_mask_dev]).cpu().numpy() if task == 'classification' else out[val_mask_dev].cpu().numpy()
                    score = get_eval_score(y_full[val_idx], v_preds, task)
                    
                    improved = (task == 'classification' and score > best_val_score) or \
                               (task == 'regression' and score < best_val_score)
                    
                    if improved:
                        best_val_score = score
                        best_state = model.state_dict()
                        patience_cnt = 0
                    else:
                        patience_cnt += 1
                
                if patience_cnt >= params['patience']:
                    break
            
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                out = model(x_tensor_dev, edge_index_dev)
                v_preds = torch.sigmoid(out[val_mask_dev]).cpu().numpy() if task == 'classification' else out[val_mask_dev].cpu().numpy()
                t_preds = torch.sigmoid(out[num_train:]).cpu().numpy() if task == 'classification' else out[num_train:].cpu().numpy()
                
                oof_preds[val_idx] += v_preds / len(random_states)
                test_preds += (t_preds / n_folds) / len(random_states)
                print(f"Fold {fold+1} Best Score: {best_val_score:.5f}")

    overall_score = get_eval_score(y_full, oof_preds, task)
    print(f"Overall OOF Score: {overall_score:.5f}")
    return oof_preds, test_preds
