import time
import numpy as np
import pandas as pd
from mltabpipe.common import StratifiedKFold, KFold, roc_auc_score, get_eval_score

try:
    from pytabkit.models.sklearn.sklearn_interfaces import TabM_TD_Classifier, TabM_TD_Regressor
    PYTABKIT_AVAILABLE = True
except ImportError:
    PYTABKIT_AVAILABLE = False

def train_tabm_model(
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
    Trains a TabM model using Cross Validation with Seed Ensembling via pytabkit.
    """
    if not PYTABKIT_AVAILABLE:
        raise ImportError("pytabkit is not installed or missing dependencies (e.g., lightning, httpcore).")

    if params is None:
        params = {} # Using Tuned Defaults (TD) by default

    if isinstance(random_states, int):
        random_states = [random_states]

    print(f"--- Training TabM Model ({task}) with {len(random_states)} seeds ---")
    
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    all_metrics = []
    
    X = train_df[features]
    y = train_df[target_col]
    X_test = test_df[features]

    for seed in random_states:
        print(f"Seed {seed}")
        if task == 'classification':
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f"Fold {fold + 1}/{n_folds}")
            
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            
            if task == 'classification':
                model = TabM_TD_Classifier(random_state=seed, **params)
            else:
                model = TabM_TD_Regressor(random_state=seed, **params)
            
            model.fit(X_train, y_train)
            
            if task == 'classification':
                val_preds = model.predict_proba(X_val)[:, 1]
                test_fold_preds = model.predict_proba(X_test)[:, 1]
            else:
                val_preds = model.predict(X_val)
                test_fold_preds = model.predict(X_test)
                
            oof_preds[val_idx] += val_preds / len(random_states)
            test_preds += (test_fold_preds / n_folds) / len(random_states)
            
            fold_score = get_eval_score(y_val, val_preds, task)
            all_metrics.append(fold_score)
            print(f"Fold {fold + 1} Score: {fold_score:.5f}")
            
    overall_score = get_eval_score(y, oof_preds, task)
    print(f"Overall OOF Score (Ensembled): {overall_score:.5f}")
    return oof_preds, test_preds, all_metrics
