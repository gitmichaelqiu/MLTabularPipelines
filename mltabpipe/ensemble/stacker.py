import numpy as np
import pandas as pd
from mltabpipe.core.common import StratifiedKFold, KFold, roc_auc_score, get_eval_score

try:
    import cuml
    from cuml.linear_model import LogisticRegression as cuLogReg, Ridge as cuRidge
    CUML_AVAILABLE = True
except ImportError:
    from sklearn.linear_model import LogisticRegression as skLogReg, Ridge as skRidge
    CUML_AVAILABLE = False

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
    Uses L2-penalized Logistic Regression (Classification) or Ridge (Regression).
    """
    if params is None:
        # High regularization (low C) is often good for stacking
        params = {'C': 0.1} if task == 'classification' else {'alpha': 1.0}

    print(f"--- Training Meta-Learner (Stacker) for {task} ---")
    
    # Construct meta-features
    X_meta_train = np.column_stack(oof_list)
    X_meta_test = np.column_stack(test_preds_list)
    y = train_df[target_col].values
    
    # We do a standard CV for the stacker as well to get its OOF
    oof_stacker = np.zeros(len(train_df))
    test_preds_stacker = np.zeros(len(test_df))
    
    if task == 'classification':
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
                model = skLogReg(penalty='l2', **params, solver='lbfgs')
            model.fit(X_tr, y_tr)
            val_preds = model.predict_proba(X_va)[:, 1]
            test_fold_preds = model.predict_proba(X_meta_test)[:, 1]
        else:
            if CUML_AVAILABLE:
                model = cuRidge(**params)
            else:
                model = skRidge(**params)
            model.fit(X_tr, y_tr)
            val_preds = model.predict(X_va)
            test_fold_preds = model.predict(X_meta_test)
            
        oof_stacker[va_idx] = val_preds
        test_preds_stacker += test_fold_preds / n_folds
        
        fold_score = get_eval_score(y_va, val_preds, task)
        print(f"Stacker Fold {fold+1} Score: {fold_score:.5f}")

    overall_score = get_eval_score(y, oof_stacker, task)
    print(f"Overall Stacker OOF Score: {overall_score:.5f}")
    
    return oof_stacker, test_preds_stacker
