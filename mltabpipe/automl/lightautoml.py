import numpy as np
import pandas as pd

try:
    from lightautoml.automl.presets.tabular_presets import TabularAutoML
    from lightautoml.tasks import Task
    LIGHTAUTOML_AVAILABLE = True
except ImportError:
    LIGHTAUTOML_AVAILABLE = False

from mltabpipe.core.common import get_eval_score

def train_lightautoml_model(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    target_col: str, 
    task: str = 'classification',
    timeout: int = 3600,
    cpu_limit: int = 4
):
    """
    Trains a LightAutoML TabularAutoML model.
    Standardizes OOF and test predictions.
    """
    if not LIGHTAUTOML_AVAILABLE:
        raise ImportError("LightAutoML is not installed. Please run 'pip install lightautoml'.")

    print(f"--- Training LightAutoML Model ({task}) with timeout={timeout}s ---")
    
    # Define task for LAMA
    if task == 'classification':
        # Default is binary, check for multi
        n_unique = train_df[target_col].nunique()
        task_name = 'binary' if n_unique == 2 else 'multiclass'
        metric = 'auc' if task_name == 'binary' else 'logloss'
    else:
        task_name = 'reg'
        metric = 'rmse'

    lama_task = Task(name=task_name, metric=metric)
    
    # Initialize LAMA
    automl = TabularAutoML(
        task=lama_task, 
        timeout=timeout,
        cpu_limit=cpu_limit,
        general_params={'use_algos': [['lgb', 'cb', 'linear_l2']]},
        reader_params={'n_jobs': cpu_limit, 'cv': 5, 'random_state': 42}
    )

    # Fit and get OOF
    # LightAutoML roles specify columns. 'target' is required.
    roles = {'target': target_col}
    
    # fit_predict returns OOF predictions (in LAMA's format)
    # The output is a LAMA predictions object
    oof_preds_lama = automl.fit_predict(train_df, roles=roles, verbose=1)
    
    # Convert LAMA predictions to numpy
    # For binary, it's typically (N, 1) probabilities
    # For multi, it's (N, K)
    # For reg, it's (N, 1)
    oof_preds = oof_preds_lama.data
    if task_name == 'binary' or task_name == 'reg':
        oof_preds = oof_preds.squeeze()
    
    # Test predictions
    test_preds_lama = automl.predict(test_df)
    test_preds = test_preds_lama.data
    if task_name == 'binary' or task_name == 'reg':
        test_preds = test_preds.squeeze()

    # Scoring
    y_true = train_df[target_col].values
    score = get_eval_score(y_true, oof_preds, task)
    print(f"LightAutoML Overall OOF Score: {score:.5f}")

    return oof_preds, test_preds
