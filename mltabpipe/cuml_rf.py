import time
import numpy as np
import pandas as pd
from mltabpipe.common import StratifiedKFold, KFold, roc_auc_score, get_eval_score

try:
    import cuml
    from cuml.ensemble import RandomForestClassifier as cuRFClassifier, RandomForestRegressor as cuRFRegressor
    CUML_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier as skRFClassifier, RandomForestRegressor as skRFRegressor
    CUML_AVAILABLE = False

def train_rf_model(
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
    Trains a Random Forest model using Cross Validation with Seed Ensembling.
    Uses cuml (GPU) if available, otherwise falls back to sklearn (CPU).
    """
    if params is None:
        params = {
            'n_estimators': 500,
            'max_depth': 12,
            'max_features': 'sqrt',
            'n_jobs': -1 if not CUML_AVAILABLE else None
        }

    if isinstance(random_states, int):
        random_states = [random_states]

    engine = "RAPIDS cuML (GPU)" if CUML_AVAILABLE else "scikit-learn (CPU)"
    print(f"--- Training Random Forest Model ({task}) using {engine} ---")
    
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
                if CUML_AVAILABLE:
                    model = cuRFClassifier(random_state=seed, **params)
                else:
                    model = skRFClassifier(random_state=seed, **params)
            else:
                if CUML_AVAILABLE:
                    model = cuRFRegressor(random_state=seed, **params)
                else:
                    model = skRFRegressor(random_state=seed, **params)
            
            model.fit(X_train.astype(np.float32), y_train)
            
            if task == 'classification':
                val_preds = model.predict_proba(X_val.astype(np.float32))[:, 1]
                test_fold_preds = model.predict_proba(X_test.astype(np.float32))[:, 1]
            else:
                val_preds = model.predict(X_val.astype(np.float32))
                test_fold_preds = model.predict(X_test.astype(np.float32))
                
            oof_preds[val_idx] += val_preds / len(random_states)
            test_preds += (test_fold_preds / n_folds) / len(random_states)
            
            fold_score = get_eval_score(y_val, val_preds, task)
            all_metrics.append(fold_score)
            print(f"Fold {fold + 1} Score: {fold_score:.5f}")
            
    overall_score = get_eval_score(y, oof_preds, task)
    print(f"Overall OOF Score: {overall_score:.5f}")
    return oof_preds, test_preds
