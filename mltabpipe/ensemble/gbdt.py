import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
except ImportError:
    CatBoostClassifier = None

from mltabpipe.core.common import roc_auc_score, get_eval_score

def _get_gpu_params(model_type, user_params=None):
    """
    Utility to detect GPU/CUDA availability and provide relevant model parameters.
    Returns a dictionary of parameters to merge.
    """
    if user_params is None:
        user_params = {}
    
    gpu_params = {}
    
    if model_type == 'xgb':
        if xgb is not None:
            try:
                from xgboost import device_is_available
                if device_is_available("cuda"):
                    if "device" not in user_params and "tree_method" not in user_params:
                        gpu_params = {"device": "cuda"}
            except ImportError:
                pass 
                
    elif model_type == 'lgbm':
        if lgb is not None:
            if "device" not in user_params:
                # Force 'gpu' by default, if it fails, fallback to 'cpu' in fit
                gpu_params = {"device": "gpu"}

    elif model_type == 'cb':
        if CatBoostClassifier is not None:
            if "task_type" not in user_params:
                try:
                    from catboost.utils import get_gpu_device_count
                    if get_gpu_device_count() > 0:
                        gpu_params = {"task_type": "GPU"}
                except:
                    pass
                    
    return gpu_params


def train_xgb_model(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    features: list, 
    target_col: str, 
    params: dict, 
    task: str = 'classification',
    n_folds: int = 5, 
    random_states: list = [42]
):
    """
    Trains an XGBoost model using Cross Validation with Seed Ensembling.
    Handles binary and multiclass tasks automatically.
    """
    if xgb is None:
        raise ImportError("XGBoost is not installed. Please run 'pip install xgboost'.")

    # Detect & apply GPU acceleration if possible
    gpu_params = _get_gpu_params('xgb', params)
    final_params = {**params, **gpu_params}
    if "device" in gpu_params and gpu_params["device"] == "cuda":
        print(f"  XGBoost: Using GPU acceleration (device='cuda').")

    if isinstance(random_states, int):
        random_states = [random_states]

    print(f"--- Training XGB Model ({task}) with {len(random_states)} seeds ---")
    
    # Detect n_classes
    vals = train_df[target_col].dropna().unique()
    n_classes = len(vals)
    is_multiclass = (task == 'classification' and n_classes > 2)

    if is_multiclass:
        print(f"  Detected multiclass task ({n_classes} classes). Predictions will be shape (N, {n_classes}).")
        oof_preds = np.zeros((len(train_df), n_classes))
        test_preds = np.zeros((len(test_df), n_classes))
    else:
        if task == 'classification':
             print(f"  Detected binary task. Predictions will be shape (N,).")
        oof_preds = np.zeros(len(train_df))
        test_preds = np.zeros(len(test_df))
    
    all_metrics = []
    
    for seed in random_states:
        print(f"Seed {seed}")
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed) if task == 'classification' else KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_df[features], train_df[target_col])):
            print(f"Fold {fold + 1}/{n_folds}")
            
            X_train, y_train = train_df.iloc[train_idx][features], train_df.iloc[train_idx][target_col]
            X_val, y_val = train_df.iloc[val_idx][features], train_df.iloc[val_idx][target_col]
            
            if task == 'classification':
                model = xgb.XGBClassifier(**final_params, random_state=seed, early_stopping_rounds=100, enable_categorical=True)
            else:
                model = xgb.XGBRegressor(**final_params, random_state=seed, early_stopping_rounds=100, enable_categorical=True)
            
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            # Prediction logic
            if is_multiclass:
                val_preds_fold = model.predict_proba(X_val)
                test_fold_preds = model.predict_proba(test_df[features])
            elif task == 'classification':
                val_preds_fold = model.predict_proba(X_val)[:, 1]
                test_fold_preds = model.predict_proba(test_df[features])[:, 1]
            else:
                val_preds_fold = model.predict(X_val)
                test_fold_preds = model.predict(test_df[features])
                
            oof_preds[val_idx] += val_preds_fold / len(random_states)
            test_preds += (test_fold_preds / n_folds) / len(random_states)
            fold_score = get_eval_score(y_val, val_preds_fold, task)
            all_metrics.append(fold_score)
            print(f"Fold {fold + 1} Score: {fold_score:.5f}")
            
    overall_score = get_eval_score(train_df[target_col], oof_preds, task)
    print(f"Overall OOF Score (Ensembled): {overall_score:.5f}")
    return oof_preds, test_preds, all_metrics


def train_lgbm_model(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    features: list, 
    target_col: str, 
    params: dict, 
    boosting_type: str = 'gbdt',
    task: str = 'classification',
    n_folds: int = 5, 
    random_states: list = [42]
):
    """
    Trains a LightGBM model using Cross Validation with Seed Ensembling.
    Handles binary and multiclass tasks automatically.
    Uses GPU by default with automatic CPU fallback if needed.
    """
    if lgb is None:
        raise ImportError("LightGBM is not installed. Please run 'pip install lightgbm'.")

    # Detect & apply GPU acceleration if possible
    gpu_params = _get_gpu_params('lgbm', params)
    lgbm_params = {**params, **gpu_params}
    lgbm_params['boosting_type'] = boosting_type
    
    if "device" in gpu_params and gpu_params["device"] == "gpu":
        print(f"  LightGBM: Attempting GPU acceleration (device='gpu').")

    if isinstance(random_states, int):
        random_states = [random_states]

    print(f"--- Training LGBM Model ({task}) with {boosting_type} and {len(random_states)} seeds ---")
    
    # Detect n_classes
    vals = train_df[target_col].dropna().unique()
    n_classes = len(vals)
    is_multiclass = (task == 'classification' and n_classes > 2)

    if is_multiclass:
        print(f"  Detected multiclass task ({n_classes} classes). Predictions will be shape (N, {n_classes}).")
        oof_preds = np.zeros((len(train_df), n_classes))
        test_preds = np.zeros((len(test_df), n_classes))
    else:
        if task == 'classification':
             print(f"  Detected binary task. Predictions will be shape (N,).")
        oof_preds = np.zeros(len(train_df))
        test_preds = np.zeros(len(test_df))
    
    all_metrics = []
    
    for seed in random_states:
        print(f"Seed {seed}")
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed) if task == 'classification' else KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_df[features], train_df[target_col])):
            print(f"Fold {fold + 1}/{n_folds}")
            
            X_train, y_train = train_df.iloc[train_idx][features], train_df.iloc[train_idx][target_col]
            X_val, y_val = train_df.iloc[val_idx][features], train_df.iloc[val_idx][target_col]
            
            if task == 'classification':
                model = lgb.LGBMClassifier(**lgbm_params, random_state=seed, verbosity=-1)
            else:
                model = lgb.LGBMRegressor(**lgbm_params, random_state=seed, verbosity=-1)
            
            try:
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                          callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
            except Exception as e:
                # If GPU failed, fallback to CPU
                if "device" in lgbm_params and lgbm_params["device"] == "gpu":
                    print(f"  LightGBM: GPU failed ({str(e)[:50]}...). Falling back to CPU for all remaining folds.")
                    lgbm_params["device"] = "cpu"
                    if task == 'classification':
                        model = lgb.LGBMClassifier(**lgbm_params, random_state=seed, verbosity=-1)
                    else:
                        model = lgb.LGBMRegressor(**lgbm_params, random_state=seed, verbosity=-1)
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                              callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
                else:
                    raise e
            
            if is_multiclass:
                val_preds_fold = model.predict_proba(X_val)
                test_fold_preds = model.predict_proba(test_df[features])
            elif task == 'classification':
                val_preds_fold = model.predict_proba(X_val)[:, 1]
                test_fold_preds = model.predict_proba(test_df[features])[:, 1]
            else:
                val_preds_fold = model.predict(X_val)
                test_fold_preds = model.predict(test_df[features])
                
            oof_preds[val_idx] += val_preds_fold / len(random_states)
            test_preds += (test_fold_preds / n_folds) / len(random_states)
            fold_score = get_eval_score(y_val, val_preds_fold, task)
            all_metrics.append(fold_score)
            print(f"Fold {fold + 1} Score: {fold_score:.5f}")
            
    overall_score = get_eval_score(train_df[target_col], oof_preds, task)
    print(f"Overall OOF Score (Ensembled): {overall_score:.5f}")
    return oof_preds, test_preds, all_metrics


def train_cb_model(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    features: list, 
    target_col: str, 
    params: dict, 
    task: str = 'classification',
    n_folds: int = 5, 
    random_states: list = [42]
):
    """
    Trains a CatBoost model using Cross Validation with Seed Ensembling.
    Handles binary and multiclass tasks automatically.
    """
    if CatBoostClassifier is None:
        raise ImportError("CatBoost is not installed. Please run 'pip install catboost'.")

    # Detect & apply GPU acceleration if possible
    gpu_params = _get_gpu_params('cb', params)
    final_params = {**params, **gpu_params}
    if "task_type" in gpu_params and gpu_params["task_type"] == "GPU":
        print(f"  CatBoost: Using GPU acceleration (task_type='GPU').")

    if isinstance(random_states, int):
        random_states = [random_states]

    print(f"--- Training CB Model ({task}) with {len(random_states)} seeds ---")
    
    # Detect n_classes
    vals = train_df[target_col].dropna().unique()
    n_classes = len(vals)
    is_multiclass = (task == 'classification' and n_classes > 2)

    if is_multiclass:
        print(f"  Detected multiclass task ({n_classes} classes). Predictions will be shape (N, {n_classes}).")
        oof_preds = np.zeros((len(train_df), n_classes))
        test_preds = np.zeros((len(test_df), n_classes))
    else:
        if task == 'classification':
             print(f"  Detected binary task. Predictions will be shape (N,).")
        oof_preds = np.zeros(len(train_df))
        test_preds = np.zeros(len(test_df))
    
    all_metrics = []
    
    for seed in random_states:
        print(f"Seed {seed}")
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed) if task == 'classification' else KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_df[features], train_df[target_col])):
            print(f"Fold {fold + 1}/{n_folds}")
            
            X_train, y_train = train_df.iloc[train_idx][features], train_df.iloc[train_idx][target_col]
            X_val, y_val = train_df.iloc[val_idx][features], train_df.iloc[val_idx][target_col]
            
            cat_features_idx = X_train.select_dtypes(include=['category']).columns.tolist()
            train_pool = Pool(X_train, y_train, cat_features=cat_features_idx)
            val_pool = Pool(X_val, y_val, cat_features=cat_features_idx)
            
            if task == 'classification':
                model = CatBoostClassifier(**final_params, random_state=seed, early_stopping_rounds=100, verbose=False)
            else:
                model = CatBoostRegressor(**final_params, random_state=seed, early_stopping_rounds=100, verbose=False)
            
            model.fit(train_pool, eval_set=val_pool)
            
            if is_multiclass:
                val_preds_fold = model.predict_proba(val_pool)
                test_fold_preds = model.predict_proba(test_df[features])
            elif task == 'classification':
                val_preds_fold = model.predict_proba(val_pool)[:, 1]
                test_fold_preds = model.predict_proba(test_df[features])[:, 1]
            else:
                val_preds_fold = model.predict(val_pool)
                test_fold_preds = model.predict(test_df[features])
                
            oof_preds[val_idx] += val_preds_fold / len(random_states)
            test_preds += (test_fold_preds / n_folds) / len(random_states)
            fold_score = get_eval_score(y_val, val_preds_fold, task)
            all_metrics.append(fold_score)
            print(f"Fold {fold + 1} Score: {fold_score:.5f}")
            
    overall_score = get_eval_score(train_df[target_col], oof_preds, task)
    print(f"Overall OOF Score (Ensembled): {overall_score:.5f}")
    return oof_preds, test_preds, all_metrics


def tune_xgb_hyperparameters(train_df, features, target_col, task='classification', n_trials=20):
    """
    Optuna Hyperparameter tuning for XGBoost.
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for hyperparameter tuning. Please install it via 'pip install optuna'.")
    if xgb is None:
        raise ImportError("XGBoost is not installed.")

    gpu_params = _get_gpu_params('xgb')

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            **gpu_params
        }
        
        X_train, X_val, y_train, y_val = train_test_split(
            train_df[features], train_df[target_col], test_size=0.2, random_state=42
        )
        
        if task == 'classification':
            model = xgb.XGBClassifier(**params, random_state=42, early_stopping_rounds=50, enable_categorical=True)
        else:
            model = xgb.XGBRegressor(**params, random_state=42, early_stopping_rounds=50, enable_categorical=True)
            
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        n_classes = train_df[target_col].nunique()
        if task == 'classification' and n_classes == 2:
            preds = model.predict_proba(X_val)[:, 1]
        else:
            preds = model.predict_proba(X_val) if task == 'classification' else model.predict(X_val)
            
        score = get_eval_score(y_val, preds, task)
        print(f"  Trial {trial.number} | Score: {score:.5f}")
        return score

    direction = 'maximize' if task == 'classification' else 'minimize'
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)
    
    best_params = {**study.best_trial.params, **gpu_params}
    print("Best XGBoost Trial:", best_params)
    return best_params

def tune_lgbm_hyperparameters(train_df, features, target_col, task='classification', n_trials=20, boosting_type='gbdt'):
    """
    Optuna Hyperparameter tuning for LightGBM.
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for hyperparameter tuning. Please install it via 'pip install optuna'.")
    if lgb is None:
        raise ImportError("LightGBM is not installed.")

    gpu_params = _get_gpu_params('lgbm')

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'boosting_type': boosting_type,
            **gpu_params
        }
        
        X_train, X_val, y_train, y_val = train_test_split(
            train_df[features], train_df[target_col], test_size=0.2, random_state=42
        )
        
        if task == 'classification':
            model = lgb.LGBMClassifier(**params, random_state=42, verbosity=-1)
        else:
            model = lgb.LGBMRegressor(**params, random_state=42, verbosity=-1)
            
        try:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                      callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        except Exception as e:
            if "device" in params and params["device"] == "gpu":
                print(f"  LightGBM: Tuning GPU failed. Swapping to CPU for remaining trials.")
                params["device"] = "cpu"
                if task == 'classification':
                    model = lgb.LGBMClassifier(**params, random_state=42, verbosity=-1)
                else:
                    model = lgb.LGBMRegressor(**params, random_state=42, verbosity=-1)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                          callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
            else:
                raise e
        
        n_classes = train_df[target_col].nunique()
        if task == 'classification' and n_classes == 2:
            preds = model.predict_proba(X_val)[:, 1]
        else:
            preds = model.predict_proba(X_val) if task == 'classification' else model.predict(X_val)
            
        score = get_eval_score(y_val, preds, task)
        print(f"  Trial {trial.number} | Score: {score:.5f}")
        return score

    direction = 'maximize' if task == 'classification' else 'minimize'
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)
    
    print("Best LightGBM Trial:", study.best_trial.params)
    return study.best_trial.params

def tune_cb_hyperparameters(train_df, features, target_col, task='classification', n_trials=20):
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for hyperparameter tuning. Please install it via 'pip install optuna'.")
    if CatBoostClassifier is None:
        raise ImportError("CatBoost is not installed.")

    gpu_params = _get_gpu_params('cb')

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            **gpu_params
        }
        X_tr, X_vl, y_tr, y_vl = train_test_split(train_df[features], train_df[target_col], test_size=0.2, random_state=42)
        cat_feat = X_tr.select_dtypes(include=['category']).columns.tolist()
        train_pool = Pool(X_tr, y_tr, cat_features=cat_feat)
        val_pool = Pool(X_vl, y_vl, cat_features=cat_feat)
        if task == 'classification':
            model = CatBoostClassifier(**params, random_state=42, early_stopping_rounds=50, verbose=False)
        else:
            model = CatBoostRegressor(**params, random_state=42, early_stopping_rounds=50, verbose=False)
        model.fit(train_pool, eval_set=val_pool)
        
        n_classes = train_df[target_col].nunique()
        if task == 'classification' and n_classes == 2:
            preds = model.predict_proba(val_pool)[:, 1]
        else:
            preds = model.predict_proba(val_pool) if task == 'classification' else model.predict(val_pool)
            
        score = get_eval_score(y_vl, preds, task)
        print(f"  Trial {trial.number} | Score: {score:.5f}")
        return score

    direction = 'maximize' if task == 'classification' else 'minimize'
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)
    
    best_params = {**study.best_trial.params, **gpu_params}
    print("Best CatBoost Trial:", best_params)
    return best_params
