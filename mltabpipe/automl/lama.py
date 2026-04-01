import numpy as np
import pandas as pd

try:
    from lightautoml.automl.presets.tabular_presets import TabularAutoML
    from lightautoml.tasks import Task
    LIGHTAUTOML_AVAILABLE = True
except ImportError:
    LIGHTAUTOML_AVAILABLE = False

from mltabpipe.core.common import get_eval_score

def train_lama_model(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    target_col: str, 
    task: str = 'classification',
    timeout: int = 3600,
    cpu_limit: int = -1
):
    """
    Trains a LightAutoML TabularAutoML model.
    Standardizes OOF and test predictions using robust shaping and label alignment.
    """
    if not LIGHTAUTOML_AVAILABLE:
        raise ImportError("LightAutoML is not installed. Please run 'pip install lightautoml'.")

    print(f"--- Training LightAutoML Model ({task}) with timeout={timeout}s ---")
    
    # Avoid side effects on the original dataframe
    train_data = train_df.copy()
    test_data = test_df.copy()

    # Define task for LAMA
    if task == 'classification':
        n_unique = train_data[target_col].nunique()
        task_name = 'binary' if n_unique == 2 else 'multiclass'
        metric = 'auc' if task_name == 'binary' else 'logloss'

        # Explicitly encode categorical targets to integers to avoid "The Multi Trap" (label mismatch)
        if not np.issubdtype(train_data[target_col].dtype, np.integer):
            print(f"Label encoding categorical target: {target_col}")
            train_data[target_col], labels = pd.factorize(train_data[target_col])
            # Note: We don't necessarily need to encode test_df's target as it's usually empty/placeholder
    else:
        task_name = 'reg'
        metric = 'rmse'

    lama_task = Task(name=task_name, metric=metric)
    
    # Initialize LAMA
    # redundant n_jobs removed from reader_params as it's handled by cpu_limit
    automl = TabularAutoML(
        task=lama_task, 
        timeout=timeout,
        cpu_limit=cpu_limit,
        general_params={'use_algos': [['lgb', 'cb', 'linear_l2']]},
        reader_params={'cv': 5, 'random_state': 42} 
    )

    # Fit and get OOF
    roles = {'target': target_col}
    
    # fit_predict returns OOF predictions (in LAMA's format)
    oof_preds_lama = automl.fit_predict(train_data, roles=roles, verbose=1)
    
    # Convert LAMA predictions to numpy
    # Use .ravel() ensure the array is 1D for binary/regression (prevents batch-size-1 scalar errors)
    oof_preds = oof_preds_lama.data
    if task_name == 'binary' or task_name == 'reg':
        oof_preds = oof_preds.ravel()
    
    # Test predictions
    test_preds_lama = automl.predict(test_data)
    test_preds = test_preds_lama.data
    if task_name == 'binary' or task_name == 'reg':
        test_preds = test_preds.ravel()

    # Scoring - ensure we score against the (potentially encoded) target
    y_true = train_data[target_col].values
    score = get_eval_score(y_true, oof_preds, task)
    print(f"LightAutoML Overall OOF Score: {score:.5f}")

    return oof_preds, test_preds
