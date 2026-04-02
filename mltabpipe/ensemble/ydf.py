import numpy as np
import pandas as pd

try:
    import ydf
    YDF_AVAILABLE = True
except ImportError:
    YDF_AVAILABLE = False

from mltabpipe.core.common import get_eval_score

def train_ydf_model(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    features: list, 
    target_col: str, 
    params: dict = None, 
    task: str = 'classification'
):
    """
    Trains a Yggdrasil Decision Forests (YDF) model.
    YDF handles its own CV and validation internally if configured, 
    but we can also use a simple train/test or OOF extraction.
    """
    if not YDF_AVAILABLE:
        raise ImportError("YDF is not installed. Please run 'pip install ydf'.")

    if params is None:
        params = {
            'num_trees': 500,
            'max_depth': 12,
        }

    print(f"--- Training YDF Model ({task}) ---")
    
    # YDF likes its own task definition
    ydf_task = ydf.Task.CLASSIFICATION if task == 'classification' else ydf.Task.REGRESSION
    
    # Create the learner
    # Random Forest is a good default for YDF
    learner = ydf.RandomForestLearner(label=target_col, task=ydf_task, **params)
    
    # Training
    model = learner.train(train_df[features + [target_col]])
    
    # OOF Predictions
    # YDF can provide OOF predictions for Random Forest learners
    try:
        oof_preds = model.out_of_fold_predictions()
        if task == 'classification':
            # For binary, it returns probabilities for all classes
            # We assume binary for now and take the positive class (usually index 1)
            oof_preds = oof_preds[:, 1]
    except Exception:
        print("Warning: Could not extract OOF predictions from YDF model. Using in-sample (biased) predictions.")
        # Fallback to in-sample (not ideal, but YDF RF usually has OOF)
        if task == 'classification':
            oof_preds = model.predict(train_df[features])[:, 1]
        else:
            oof_preds = model.predict(train_df[features])

    # Test Predictions
    test_preds = model.predict(test_df[features])
    if task == 'classification':
        test_preds = test_preds[:, 1]

    # Scoring
    y_true = train_df[target_col].values
    score = get_eval_score(y_true, oof_preds, task)
    print(f"YDF Overall Score: {score:.5f}")

    return oof_preds, test_preds