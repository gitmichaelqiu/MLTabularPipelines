import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from mltabpipe.core.common import get_eval_score

def train_ridge_model(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    features: list, 
    target_col: str, 
    params: dict = None, 
    task: str = 'classification',
    n_folds: int = 5, 
    random_states: list = [42]
):
    """
    Trains a simple Ridge Regression (or Classifier) using Cross Validation.
    """
    if params is None:
        params = {'alpha': 1.0}

    if isinstance(random_states, int):
        random_states = [random_states]

    print(f"--- Training Ridge Model ({task}) with {len(random_states)} seeds ---")
    
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    
    X = train_df[features].fillna(0)
    y = train_df[target_col]
    X_test = test_df[features].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    for seed in random_states:
        print(f"Seed {seed}")
        if task == 'classification':
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y)):
            X_train, y_train = X_scaled[train_idx], y.iloc[train_idx]
            X_val, y_val = X_scaled[val_idx], y.iloc[val_idx]
            
            # Ridge can be used for classification by predicting 0/1 (or probabilities via Decision Function)
            # but usually for stacking we use pure Ridge regression on probabilities.
            model = Ridge(**params, random_state=seed)
            model.fit(X_train, y_train)
            
            val_preds = model.predict(X_val)
            test_fold_preds = model.predict(X_test_scaled)
                
            oof_preds[val_idx] += val_preds / len(random_states)
            test_preds += (test_fold_preds / n_folds) / len(random_states)
            
    overall_score = get_eval_score(y, oof_preds, task)
    print(f"Overall OOF Score: {overall_score:.5f}")
    return oof_preds, test_preds