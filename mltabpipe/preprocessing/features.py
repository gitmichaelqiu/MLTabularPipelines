import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def add_snap_features(df, original_df, cols):
    """
    Map each synthetic float to its nearest value in an original/reference dataset.
    Recovers archetypes from noisy or synthetic data.
    """
    df = df.copy()
    if not cols:
        return df
        
    print(f"--- Snapping features for {len(cols)} columns ---")
    for col in cols:
        if col not in df.columns or col not in original_df.columns:
            continue
            
        # Get unique values from original/reference data
        orig_values = np.sort(original_df[col].dropna().unique())
        if len(orig_values) == 0:
            continue
            
        # Find nearest original value for each synthetic value
        idx = np.searchsorted(orig_values, df[col].values)
        idx = np.clip(idx, 0, len(orig_values) - 1)
        
        # Check if previous index is closer
        idx_prev = np.clip(idx - 1, 0, len(orig_values) - 1)
        dist_curr = np.abs(df[col].values - orig_values[idx])
        dist_prev = np.abs(df[col].values - orig_values[idx_prev])
        
        best_idx = np.where(dist_curr <= dist_prev, idx, idx_prev)
        snap_values = orig_values[best_idx]
        
        # Count snapped values (rows where the value has been moved)
        # Using a small epsilon for floating point comparison
        snapped_mask = np.abs(df[col].values - snap_values) > 1e-9
        snapped_count = np.sum(snapped_mask)
        
        df[f"{col}_snap"] = snap_values
        df[f"{col}_snap_diff"] = df[col] - snap_values
        
        print(f"  Column '{col}': Snapped {snapped_count} / {len(df)} rows using {len(orig_values)} archetypes.")
        
    return df

def add_digit_features(df, cols):
    """
    Extract the digit at every decimal position.
    Helpful for identifying artifacts in synthetic data rounding/sampling.
    """
    df = df.copy()
    if not cols:
        return df
        
    print(f"--- Extracting digit features for {len(cols)} columns ---")
    for col in cols:
        if col not in df.columns:
            continue
            
        x = df[col].values
        frac = x - np.floor(x)
        
        df[f"{col}_d1"] = np.floor(frac * 10).astype(int)              
        df[f"{col}_d2"] = np.floor(frac * 100).astype(int) % 10         
        df[f"{col}_frac100"] = np.round(frac * 100).astype(int)         
        df[f"{col}_mod10"] = np.floor(x).astype(int) % 10               
        
        # Generic indicators
        df[f"{col}_is_int"] = (frac < 0.005).astype(int)
        
    return df

def add_arithmetic_interactions(df, interactions):
    """
    Apply generic arithmetic interactions.
    'interactions' should be a list of tuples: (col_a, op, col_b, name)
    """
    df = df.copy()
    if not interactions:
        return df
        
    print(f"--- Creating {len(interactions)} arithmetic interactions ---")
    for col_a, op, col_b, name in interactions:
        if col_a not in df.columns or col_b not in df.columns:
            continue
            
        if op == '-':
            df[name] = df[col_a] - df[col_b]
        elif op == '+':
            df[name] = df[col_a] + df[col_b]
        elif op == '*':
            df[name] = df[col_a] * df[col_b]
        elif op == '/':
            df[name] = df[col_a] / (df[col_b] + 1e-9)
            
    return df

def add_binning_features(df, cols, n_bins=1000):
    """
    Quantile binning for a list of features.
    """
    df = df.copy()
    if not cols:
        return df
        
    print(f"--- Applying quantile binning ({n_bins}) for {len(cols)} columns ---")
    for col in cols:
        if col not in df.columns:
            continue
            
        df[f"{col}_bin_{n_bins}"] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
        
    return df

def add_flag_counts(df, cols, flag_value=1, name="flag_count"):
    """
    Counts specific value occurrences ('flag_value') across 'cols'.
    Generic version of service counts.
    """
    df = df.copy()
    available_cols = [c for c in cols if c in df.columns]
    if not available_cols:
        return df
        
    print(f"--- Creating flag count '{name}' across {len(available_cols)} columns ---")
    df[name] = (df[available_cols] == flag_value).sum(axis=1)
    return df

def add_frequency_encoding(df, cols):
    """
    Add frequency encoding (normalized counts).
    """
    df = df.copy()
    if not cols:
        return df
        
    print(f"--- Adding frequency encoding for {len(cols)} columns ---")
    for col in cols:
        if col not in df.columns:
            continue
        freq = df[col].value_counts(normalize=True)
        df[f"{col}_freq"] = df[col].map(freq)
    return df

def apply_modular_pipeline(df, config, original_df=None):
    """
    Applies the modular pipeline based on a configuration dictionary.
    config keys: 'digit_cols', 'snap_cols', 'interactions', 'binning_configs', 'flag_configs'
    """
    initial_cols = len(df.columns)
    print("--- Applying Modular Feature Pipeline ---")

    if 'digit_cols' in config:
        df = add_digit_features(df, config['digit_cols'])
        
    if 'snap_cols' in config and original_df is not None:
        df = add_snap_features(df, original_df, config['snap_cols'])
        
    if 'interactions' in config:
        df = add_arithmetic_interactions(df, config['interactions'])
        
    if 'binning_configs' in config:
        for bin_cfg in config['binning_configs']:
            df = add_binning_features(df, bin_cfg['cols'], bin_cfg.get('n_bins', 1000))
            
    if 'flag_configs' in config:
        for fl_cfg in config['flag_configs']:
            df = add_flag_counts(df, fl_cfg['cols'], fl_cfg.get('value', 1), fl_cfg.get('name', 'count'))
            
    if 'freq_cols' in config:
        df = add_frequency_encoding(df, config['freq_cols'])
        
    final_cols = len(df.columns)
    print(f"--- Modular Pipeline Completed. Features: {initial_cols} -> {final_cols} (+{final_cols - initial_cols}) ---")

    return df
