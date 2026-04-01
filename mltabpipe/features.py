import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def add_snap_features(df, original_df, cols=['MonthlyCharges', 'TotalCharges']):
    """
    Map each synthetic float to its nearest value in the original IBM dataset.
    This recovers the 'true' original feature archetype.
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns or col not in original_df.columns:
            continue
            
        # Get unique values from original data
        orig_values = np.sort(original_df[col].dropna().unique())
        
        # Find nearest original value for each synthetic value
        # Using searchsorted for 1D nearest neighbor
        idx = np.searchsorted(orig_values, df[col].values)
        idx = np.clip(idx, 0, len(orig_values) - 1)
        
        # Check if previous index is closer
        idx_prev = np.clip(idx - 1, 0, len(orig_values) - 1)
        dist_curr = np.abs(df[col].values - orig_values[idx])
        dist_prev = np.abs(df[col].values - orig_values[idx_prev])
        
        best_idx = np.where(dist_curr <= dist_prev, idx, idx_prev)
        snap_values = orig_values[best_idx]
        
        df[f"{col}_snap"] = snap_values
        df[f"{col}_snap_diff"] = df[col] - snap_values
        
    return df

def add_digit_features(df, cols=['MonthlyCharges', 'TotalCharges']):
    """
    Extract the digit at every decimal position.
    Exploits artifacts left by the generator's rounding behavior.
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
            
        x = df[col].values
        frac = x - np.floor(x)
        
        df[f"{col}_d1"] = np.floor(frac * 10).astype(int)              # 1st decimal digit
        df[f"{col}_d2"] = np.floor(frac * 100).astype(int) % 10         # 2nd decimal digit
        df[f"{col}_frac100"] = np.round(frac * 100).astype(int)         # 2-digit integer representation
        df[f"{col}_mod10"] = np.floor(x).astype(int) % 10               # Last digit of integer part
        
        # Indicators for "round" numbers
        df[f"{col}_is_round_05"] = (np.abs(frac - 0.5) < 0.005).astype(int)
        df[f"{col}_is_int"] = (frac < 0.005).astype(int)
        
    return df

def add_arithmetic_interactions(df):
    """
    Powerful churn-specific interactions identified in the 1st place solution.
    """
    df = df.copy()
    
    # Deviation captures how much actual total charges differ from predicted monthly billing
    if 'TotalCharges' in df.columns and 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
        df['TC_deviation'] = df['TotalCharges'] - df['tenure'] * df['MonthlyCharges']
        df['TC_per_month'] = df['TotalCharges'] / (df['tenure'] + 1e-9)
        df['MC_to_TC_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1e-9)
        
    # Snap-based deviations (if snap features were already added)
    if 'TotalCharges_snap' in df.columns and 'MonthlyCharges_snap' in df.columns:
        df['TC_snap_exp_dev'] = df['TotalCharges_snap'] - df['tenure'] * df['MonthlyCharges_snap']
        
    return df

def add_binning_features(df, cols=['MonthlyCharges', 'TotalCharges'], n_bins=5000):
    """
    Multi-scale quantile binning to recover original distributions.
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
            
        # Fine-grained quantile bins essentially recover unique values in noisy data
        df[f"{col}_bin_{n_bins}"] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
        
    return df

def add_service_counts(df, svc_cols=None):
    """
    Aggregated service flags.
    """
    if svc_cols is None:
        svc_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", 
                    "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines"]
    
    df = df.copy()
    available_cols = [c for c in svc_cols if c in df.columns]
    
    if available_cols:
        df['svc_yes_count'] = (df[available_cols] == "Yes").sum(axis=1)
        df['has_internet'] = (df['InternetService'] != "No").astype(int) if 'InternetService' in df.columns else 0
        
    return df

def apply_full_kag_pipeline(df, original_df=None):
    """
    Helper to apply all feature engineering steps at once.
    """
    df = add_digit_features(df)
    if original_df is not None:
        df = add_snap_features(df, original_df)
    df = add_arithmetic_interactions(df)
    df = add_binning_features(df)
    df = add_service_counts(df)
    return df
