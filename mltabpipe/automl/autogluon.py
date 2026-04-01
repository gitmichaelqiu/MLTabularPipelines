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
    time_limit: int = 3600,
    overwrite: bool = True
):
    """
    Trains an AutoGluon TabularPredictor with persistence control and dynamic type detection.
    Uses true Out-of-Fold (OOF) predictions for standard scoring.
    """
    if not AUTOGLUON_AVAILABLE:
        raise ImportError("AutoGluon is not installed. Please run 'pip install autogluon'.")

    print(f"--- Training AutoGluon Model ({task}) with presets='{presets}' ---")
    
    # Handle path persistence
    if overwrite and os.path.exists(path):
        print(f"Overwriting existing model path: {path}")
        shutil.rmtree(path)

    # Initialize predictor - let AutoGluon infer problem_type and eval_metric by default
    # but we can still pass hints from the 'task' argument if needed.
    predictor = TabularPredictor(
        label=target_col, 
        path=path,
        verbosity=2
    ).fit(
        train_df, 
        presets=presets,
        time_limit=time_limit
    )

    # Reliable problem type detection after training
    problem_type = predictor.problem_type
    print(f"AutoGluon inferred problem_type: {problem_type}")

    # Extract True Out-Of-Fold (OOF) predictions
    # This requires bagging to be enabled (e.g. presets='best_quality')
    try:
        if problem_type == 'binary':
            # Returns probabilities for the positive class only
            oof_preds = predictor.predict_proba_oof(as_multiclass=False).values
        elif problem_type == 'multiclass':
            # Returns probabilities for all classes (as NumPy array)
            oof_preds = predictor.predict_proba_oof().values
        else:
            # For regression (returns scalar values)
            oof_preds = predictor.predict_oof().values
            
    except Exception as e:
        error_msg = (
            f"Failed to extract True OOF predictions from AutoGluon (problem_type={problem_type}). Error: {e}\n"
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

    # Convert to NumPy if they are still Pandas objects (though .values should have handled it)
    if isinstance(oof_preds, (pd.Series, pd.DataFrame)):
        oof_preds = oof_preds.values
    if isinstance(test_preds, (pd.Series, pd.DataFrame)):
        test_preds = test_preds.values

    # Scoring
    y_true = train_df[target_col].values
    score = get_eval_score(y_true, oof_preds, task)
    print(f"AutoGluon Overall OOF Score: {score:.5f}")

    return oof_preds, test_preds
