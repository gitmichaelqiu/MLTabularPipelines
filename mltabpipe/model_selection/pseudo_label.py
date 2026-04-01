import numpy as np
import pandas as pd

def add_pseudo_labels(
    train_df, 
    test_df, 
    test_preds, 
    target_col, 
    threshold=0.95, 
    task='classification'
):
    """
    Identifies high-confidence test predictions and adds them to the training set.
    For classification, uses a threshold on probabilities.
    For regression, could use a threshold on prediction variance (if available) or other metrics.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    if task == 'classification':
        # High confidence means prob > threshold or prob < (1 - threshold)
        upper_idx = np.where(test_preds >= threshold)[0]
        lower_idx = np.where(test_preds <= (1 - threshold))[0]
        
        pseudo_idx = np.concatenate([upper_idx, lower_idx])
        
        # Assign predicted binary labels
        pseudo_labels = (test_preds[pseudo_idx] >= 0.5).astype(int)
        
    else:
        # In regression, pseudo-labeling is less common without uncertainty estimates.
        # One simple way is to take the top/bottom N% of predictions.
        # But here we'll just return the original train_df if not specified.
        print("Pseudo-labeling for regression not implemented in this utility.")
        return train_df
        
    print(f"Adding {len(pseudo_idx)} pseudo-labeled samples from test set.")
    
    if len(pseudo_idx) == 0:
        return train_df
        
    # Create the pseudo-labeled dataframe
    pseudo_df = test_df.iloc[pseudo_idx].copy()
    pseudo_df[target_col] = pseudo_labels
    
    # Concatenate with original training data
    augmented_train_df = pd.concat([train_df, pseudo_df], axis=0).reset_index(drop=True)
    
    return augmented_train_df

def apply_pseudo_labeling_pipeline(
    train_df, 
    test_df, 
    target_col, 
    model_func, 
    model_params, 
    threshold=0.95, 
    task='classification'
):
    """
    A helper to perform one round of pseudo-labeling:
    1. Train model on initial train_df.
    2. Get test_preds.
    3. Augment train_df with high-confidence test_preds.
    4. Train final model on augmented_train_df.
    """
    # Step 1 & 2: Initial training and inference
    print("Step 1: Initial training for pseudo-labeling...")
    oof_initial, test_preds_initial, _ = model_func(train_df, test_df, **model_params)
    
    # Step 3: Augmentation
    print(f"Step 2: Augmenting training data with threshold {threshold}...")
    augmented_train_df = add_pseudo_labels(
        train_df, test_df, test_preds_initial, target_col, threshold, task
    )
    
    # Step 4: Final training
    print("Step 3: Final training on augmented data...")
    oof_final, test_preds_final, metrics = model_func(augmented_train_df, test_df, **model_params)
    
    # Note: OOF for the augmented samples might not be directly comparable 
    # to original OOF, so we returning both if needed.
    return oof_final, test_preds_final, metrics
