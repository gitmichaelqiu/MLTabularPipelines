import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression as skLogReg, Ridge as skRidge

try:
    import cuml
    from cuml.linear_model import LogisticRegression as cuLogReg, Ridge as cuRidge
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

from mltabpipe.core.common import get_eval_score

def train_stacker(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    oof_list: list, 
    test_preds_list: list, 
    target_col: str, 
    task: str = 'classification',
    params: dict = None,
    n_folds: int = 5,
    seed: int = 42
):
    """
    Trains a meta-learner (stacker) to combine multiple model predictions.
    Handles multiclass tasks by concatenating class probabilities as meta-features.
    """
    if params is None:
        # High regularization (low C) is often good for stacking
        params = {'C': 0.1} if task == 'classification' else {'alpha': 1.0}

    print(f"--- Training Meta-Learner (Stacker) for {task} ---")
    
    # Construct meta-features
    # We must handle both 1D (binary/reg) and 2D (multiclass) predictions
    def process_meta_features(pred_list):
        processed = []
        for p in pred_list:
            if len(p.shape) == 1:
                processed.append(p.reshape(-1, 1))
            else:
                processed.append(p)
        return np.column_stack(processed)

    X_meta_train = process_meta_features(oof_list)
    X_meta_test = process_meta_features(test_preds_list)
    y = train_df[target_col].values
    
    # Detect n_classes from y
    n_unique = len(np.unique(y))
    is_multiclass = (task == 'classification' and n_unique > 2)

    # Initialize OOF and test predictions
    if is_multiclass:
        oof_stacker = np.zeros((len(train_df), n_unique))
        test_preds_stacker = np.zeros((len(test_df), n_unique))
    else:
        oof_stacker = np.zeros(len(train_df))
        test_preds_stacker = np.zeros(len(test_df))
    
    if task == 'classification':
        # Even for multi-class, SKF is suitable
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_meta_train, y)):
        X_tr, y_tr = X_meta_train[tr_idx], y[tr_idx]
        X_va, y_va = X_meta_train[va_idx], y[va_idx]
        
        if task == 'classification':
            if CUML_AVAILABLE:
                model = cuLogReg(penalty='l2', **params)
            else:
                # Use sklearn solver as fallback. 
                # Note: 'multi_class' is deprecated/removed in recent sklearn; 
                # the model now handles it automatically based on 'y'.
                model = skLogReg(penalty='l2', **params, solver='lbfgs')
            model.fit(X_tr, y_tr)
            
            if is_multiclass:
                val_preds_fold = model.predict_proba(X_va)
                test_fold_preds = model.predict_proba(X_meta_test)
            else:
                val_preds_fold = model.predict_proba(X_va)[:, 1]
                test_fold_preds = model.predict_proba(X_meta_test)[:, 1]
        else:
            if CUML_AVAILABLE:
                model = cuRidge(**params)
            else:
                model = skRidge(**params)
            model.fit(X_tr, y_tr)
            val_preds_fold = model.predict(X_va)
            test_fold_preds = model.predict(X_meta_test)
            
        oof_stacker[va_idx] = val_preds_fold
        test_preds_stacker += test_fold_preds / n_folds
        
        fold_score = get_eval_score(y_va, val_preds_fold, task)
        print(f"Stacker Fold {fold+1} Score: {fold_score:.5f}")

    overall_score = get_eval_score(y, oof_stacker, task)
    print(f"Overall Stacker OOF Score: {overall_score:.5f}")
    
    return oof_stacker, test_preds_stacker
