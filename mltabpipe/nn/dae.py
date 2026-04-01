import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from mltabpipe.core.common import StandardScaler

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

class DAEModel(nn.Module):
    def __init__(self, in_dim, architecture=(512, 256, 128)):
        super().__init__()
        # Encoder
        encoder_layers = []
        curr_dim = in_dim
        for h in architecture:
            encoder_layers.append(nn.Linear(curr_dim, h))
            encoder_layers.append(nn.ReLU())
            curr_dim = h
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        for h in reversed(architecture[:-1]):
            decoder_layers.append(nn.Linear(curr_dim, h))
            decoder_layers.append(nn.ReLU())
            curr_dim = h
        decoder_layers.append(nn.Linear(curr_dim, in_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

def apply_swap_noise(x, noise_level=0.15):
    """
    Applies swap noise to tabular data. 
    Randomly replaces values in each column with values from other rows.
    """
    if noise_level <= 0:
        return x
    
    x_noisy = x.copy()
    n_rows, n_cols = x.shape
    
    for col in range(n_cols):
        n_swap = int(n_rows * noise_level)
        if n_swap == 0:
            continue
        
        idx_to_swap = np.random.choice(n_rows, n_swap, replace=False)
        idx_to_use = np.random.choice(n_rows, n_swap, replace=False)
        
        x_noisy[idx_to_swap, col] = x[idx_to_use, col]
        
    return x_noisy

class DAEDataset(Dataset):
    def __init__(self, x_original, noise_level=0.15):
        self.x_original = torch.tensor(x_original, dtype=torch.float32)
        self.noise_level = noise_level

    def __len__(self):
        return self.x_original.shape[0]

    def __getitem__(self, idx):
        x = self.x_original[idx]
        # On-the-fly swap noise for this row is hard with Dataset, 
        # usually easier to pre-calculate or do it per-batch.
        return x

def train_dae_and_extract_features(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    features: list, 
    params: dict = None
):
    """
    Trains a Denoising Autoencoder on combined data and extracts bottleneck features.
    """
    if params is None:
        params = {
            'architecture': (512, 256, 128),
            'noise_level': 0.15,
            'lr': 1e-3,
            'epochs': 50,
            'batch_size': 512,
        }

    print("--- Training Denoising Autoencoder for Feature Extraction ---")
    
    # Preprocessing
    X_train = train_df[features].fillna(0).values.astype(np.float32)
    X_test = test_df[features].fillna(0).values.astype(np.float32)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Combine for DAE training (Self-supervised)
    X_all = np.vstack([X_train_scaled, X_test_scaled])
    
    ds = DAEDataset(X_all, noise_level=params['noise_level'])
    dl = DataLoader(ds, batch_size=params['batch_size'], shuffle=True)
    
    model = DAEModel(len(features), params['architecture']).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(params['epochs']):
        epoch_loss = 0
        for x_orig in dl:
            x_orig = x_orig.to(DEVICE)
            
            # Apply swap noise to the batch
            x_noisy = x_orig.clone()
            n_batch, n_feat = x_orig.shape
            mask = torch.rand(x_orig.shape) < params['noise_level']
            mask = mask.to(DEVICE)
            
            # Simple noise for demonstration: Gaussian or masking
            # Real swap noise is better done on CPU pre-batching
            x_noisy = x_orig + torch.randn_like(x_orig) * 0.1 * mask
            
            optimizer.zero_grad()
            x_recon, _ = model(x_noisy)
            loss = criterion(x_recon, x_orig)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{params['epochs']} Loss: {epoch_loss/len(dl):.6f}")

    # Extract features
    model.eval()
    print("Extracting DAE Features...")
    with torch.no_grad():
        x_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
        x_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)
        
        _, train_features = model(x_train_tensor)
        _, test_features = model(x_test_tensor)
        
    train_dae_feats = train_features.cpu().numpy()
    test_dae_feats = test_features.cpu().numpy()
    
    # Return as DataFrames or arrays
    feat_names = [f"dae_feat_{i}" for i in range(train_dae_feats.shape[1])]
    df_train_dae = pd.DataFrame(train_dae_feats, columns=feat_names)
    df_test_dae = pd.DataFrame(test_dae_feats, columns=feat_names)
    
    return df_train_dae, df_test_dae
