import numpy as np
import pandas as pd
import os
import shutil

try:
    from autogluon.tabular import TabularPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False

from mltabpipe.core.common import get_eval_score

def train_autogluon_model(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    target_col: str, 
    params: dict = None, 
    task: str = 'classification',
    path: str = 'ag_models',
    presets: str = 'best_quality',
    time_limit: int = 3600
):
    """
    Trains an AutoGluon TabularPredictor.
    Uses true Out-of-Fold (OOF) predictions for standard scoring.
    """
    if not AUTOGLUON_AVAILABLE:
        raise ImportError("AutoGluon is not installed. Please run 'pip install autogluon'.")

    print(f"--- Training AutoGluon Model ({task}) with presets='{presets}' ---")
    
    # AutoGluon handles task detection, but we can specify the problem type
    # binary, multiclass, regression
    if task == 'classification':
        # Check if binary or multi
        n_unique = train_df[target_col].nunique()
        problem_type = 'binary' if n_unique == 2 else 'multiclass'
        eval_metric = 'roc_auc' if problem_type == 'binary' else 'log_loss'
    else:
        problem_type = 'regression'
        eval_metric = 'root_mean_squared_error'

    # Remove existing model path to avoid conflicts
    if os.path.exists(path):
        shutil.rmtree(path)

    predictor = TabularPredictor(
        label=target_col, 
        problem_type=problem_type,
        eval_metric=eval_metric,
        path=path,
        verbosity=2
    ).fit(
        train_df, 
        presets=presets,
        time_limit=time_limit
    )

    # Get True Out-Of-Fold (OOF) predictions
    # This requires bagging to be enabled (e.g. presets='best_quality')
    try:
        if problem_type == 'binary':
            # Returns probabilities for the positive class only
            oof_preds = predictor.predict_proba_oof(as_multiclass=False).values
        elif problem_type == 'multiclass':
            # Returns probabilities for all classes
            oof_preds = predictor.predict_proba_oof().values
        else:
            # For regression
            oof_preds = predictor.predict_oof().values
            
    except Exception as e:
        # If OOF is not available (e.g. bagging disabled), we raise a clear error to avoid biased scoring.
        error_msg = (
            f"Failed to extract True OOF predictions from AutoGluon. Error: {e}\n"
            "This usually happens when bagging (cross-validation) is disabled. "
            "Ensure you are using presets like 'best_quality' or manually enabled bagging."
        )
        raise RuntimeError(error_msg)

    # Test predictions
    if problem_type == 'binary':
        test_preds = predictor.predict_proba(test_df, as_multiclass=False).values
    elif problem_type == 'multiclass':
        test_preds = predictor.predict_proba(test_df).values
    else:
        test_preds = predictor.predict(test_df).values

    # Scoring
    y_true = train_df[target_col].values
    score = get_eval_score(y_true, oof_preds, task)
    print(f"AutoGluon Overall OOF Score: {score:.5f}")

    return oof_preds, test_preds
