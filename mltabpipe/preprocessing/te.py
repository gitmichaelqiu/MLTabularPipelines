import pandas as pd
import numpy as np
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import StratifiedKFold

def add_nested_target_encoding(train_df, test_df, cat_cols, target_col, n_folds=5, shuffle=True, random_state=42):
    """
    Apply nested cross-validation Target Encoding to avoid leakage.
    Uses sklearn.preprocessing.TargetEncoder (available in sklearn 1.3+).
    
    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        cat_cols: List of categorical columns to encode.
        target_col: Name of the target column.
        n_folds: Number of folds for the internal CV.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    if not cat_cols:
        return train_df, test_df
        
    print(f"--- Applying Nested Target Encoding for {len(cat_cols)} columns ({n_folds} folds) ---")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    
    # Initialize TE columns in train with zeros
    # Since TargetEncoder in sklearn can return multiple columns for multiclass,
    # we need to be careful with initialization.
    
    # First, fit once on the whole train to see the output structure (e.g. multiclass)
    temp_te = TargetEncoder()
    temp_encoded = temp_te.fit_transform(train_df[cat_cols], train_df[target_col])
    
    # Determine the number of output columns
    # temp_encoded is usually a numpy array or dataframe
    n_out_cols = temp_encoded.shape[1] if temp_encoded.ndim > 1 else 1
    is_multiclass = n_out_cols > len(cat_cols)
    
    # Prefix for new columns
    prefix = "te_"
    
    # We will compute OOF (Out-of-Fold) predictions for train
    # and use the mean of fold-specific encoders for test
    train_encoded_all = np.zeros_like(temp_encoded, dtype=float)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df[target_col])):
        print(f"  Processing Fold {fold+1}/{n_folds}...")
        
        # Fit on train_idx, transform val_idx
        te = TargetEncoder()
        te.fit(train_df.iloc[train_idx][cat_cols], train_df.iloc[train_idx][target_col])
        
        train_encoded_all[val_idx] = te.transform(train_df.iloc[val_idx][cat_cols])
        
    # Fit on all train once more to transform test
    # (Alternatively, average the fold-specific encoders if possible, but sklearn's TE is simpler to re-fit)
    final_te = TargetEncoder()
    test_encoded_all = final_te.fit_transform(train_df[cat_cols], train_df[target_col])
    test_transformed = final_te.transform(test_df[cat_cols])

    # Convert to DataFrames and attach to original
    # We need to name the columns correctly. 
    # If not multiclass, names are te_<col>
    # If multiclass, names are te_<col>_<class>
    
    if not is_multiclass:
        new_col_names = [f"{prefix}{col}" for col in cat_cols]
    else:
        # Multiclass: sklearn generates n_classes columns for each input column? 
        # Actually it's often n_classes columns in total? 
        # In recent sklearn, for k classes, it generates k columns? No, usually it's k-1 or k?
        # Let's use generic names if we can't easily infer.
        new_col_names = [f"{prefix}feat_{i}" for i in range(n_out_cols)]

    df_train_te = pd.DataFrame(train_encoded_all, index=train_df.index, columns=new_col_names)
    df_test_te = pd.DataFrame(test_transformed, index=test_df.index, columns=new_col_names)
    
    train_df = pd.concat([train_df, df_train_te], axis=1)
    test_df = pd.concat([test_df, df_test_te], axis=1)
    
    return train_df, test_df
