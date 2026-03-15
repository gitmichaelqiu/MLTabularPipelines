import time
import warnings
from mltabpipe.common import np, pd, StratifiedKFold, KFold, get_eval_score

try:
    import ydf
except ImportError:
    pass

def train_ydf_model(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    features: list, 
    target_col: str, 
    params: dict = None, 
    task: str = 'classification',
    n_folds: int = 5, 
    random_state: int = 42
):
    """
    Trains a Yggdrasil Decision Forests (YDF) Gradient Boosted Trees model.
    YDF handles categorical variables and missing values natively.
    """
    if params is None:
        params = {
            'shrinkage': 0.1,
            'early_stopping_num_trees_look_ahead': 300,
            'max_depth': 2,
            'growing_strategy': 'BEST_FIRST_GLOBAL',
            'categorical_algorithm': 'RANDOM',
            'num_trees': 10000,
        }
        
    print(f"--- Training YDF Model ({task}) ---")
    
    # Suppress verbose YDF C++ engine outputs for a clean notebook
    try:
        ydf.verbose(0)
    except:
        pass
    warnings.filterwarnings('ignore')
    
    # Map our task string to the YDF Task Enum
    ydf_task = ydf.Task.CLASSIFICATION if task == 'classification' else ydf.Task.REGRESSION
    
    # YDF requires the target column to be passed inside the model parameters
    learner_params = params.copy()
    learner_params['label'] = target_col
    learner_params['task'] = ydf_task
    
    oof_preds = np.zeros(len(train_df), dtype=np.float32)
    test_preds = np.zeros(len(test_df), dtype=np.float32)
    metrics = []
    
    if task == 'classification':
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
    # YDF expects the label column to be inside the DataFrame during training
    train_cols = features + [target_col]
    test_df_feats = test_df[features].copy()
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df[features], train_df[target_col])):
        fold_t0 = time.time()
        print(f"Fold {fold + 1}/{n_folds}")
        
        # Slicing the dataframes for this fold
        tr_df = train_df.iloc[tr_idx][train_cols]
        va_df = train_df.iloc[va_idx][train_cols]
        y_va = train_df.iloc[va_idx][target_col]
        
        # Initialize and Train
        learner = ydf.GradientBoostedTreesLearner(**learner_params)
        model = learner.train(tr_df, valid=va_df) 
        
        # For classification, YDF predict() returns 1D array of positive class probabilities natively
        val_preds = model.predict(va_df[features])
        test_fold_preds = model.predict(test_df_feats)
        
        oof_preds[va_idx] = val_preds
        test_preds += test_fold_preds / n_folds
        
        # Calculate metric
        fold_score = get_eval_score(y_va, val_preds, task)
        metrics.append(fold_score)
        
        metric_name = "ROC AUC" if task == 'classification' else "RMSE"
        print(f"Fold {fold + 1} {metric_name}: {fold_score:.5f} | Time: {time.time()-fold_t0:.1f}s")
        
    overall_score = get_eval_score(train_df[target_col], oof_preds, task)
    metric_name = "ROC AUC" if task == 'classification' else "RMSE"
    print(f"Overall OOF {metric_name}: {overall_score:.5f}")
    print("-" * 30)
    
    return oof_preds, test_preds, metrics