import optuna
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from mltabpipe.common import (
    pd, np, plt, sns, 
    StratifiedKFold, train_test_split, 
    roc_auc_score, get_eval_score
)

def train_cv_model(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    features: list, 
    target_col: str, 
    model_name: str, 
    params: dict, 
    task: str = 'classification',
    n_folds: int = 5, 
    random_state: int = 42
):
    """
    Trains a machine learning model using Cross Validation.
    Adapts to either classification or regression tasks.
    """
    print(f"--- Training {model_name.upper()} Model ({task}) ---")
    
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    metrics = []
    
    if task == 'classification':
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df[features], train_df[target_col])):
        print(f"Fold {fold + 1}/{n_folds}")
        
        X_train, y_train = train_df.iloc[train_idx][features], train_df.iloc[train_idx][target_col]
        X_val, y_val = train_df.iloc[val_idx][features], train_df.iloc[val_idx][target_col]
        
        if model_name == 'lgb':
            if task == 'classification':
                model = lgb.LGBMClassifier(**params, random_state=random_state, verbosity=-1)
            else:
                model = lgb.LGBMRegressor(**params, random_state=random_state, verbosity=-1)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                      callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
            val_preds = model.predict_proba(X_val)[:, 1] if task == 'classification' else model.predict(X_val)
            test_fold_preds = model.predict_proba(test_df[features])[:, 1] if task == 'classification' else model.predict(test_df[features])
            
        elif model_name == 'xgb':
            if task == 'classification':
                model = xgb.XGBClassifier(**params, random_state=random_state, early_stopping_rounds=100, enable_categorical=True)
            else:
                model = xgb.XGBRegressor(**params, random_state=random_state, early_stopping_rounds=100, enable_categorical=True)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            val_preds = model.predict_proba(X_val)[:, 1] if task == 'classification' else model.predict(X_val)
            test_fold_preds = model.predict_proba(test_df[features])[:, 1] if task == 'classification' else model.predict(test_df[features])
            
        elif model_name == 'cat':
            cat_features = X_train.select_dtypes(include=['category']).columns.tolist()
            train_pool = Pool(X_train, y_train, cat_features=cat_features)
            val_pool = Pool(X_val, y_val, cat_features=cat_features)
            if task == 'classification':
                model = CatBoostClassifier(**params, random_state=random_state, early_stopping_rounds=100, verbose=False)
            else:
                model = CatBoostRegressor(**params, random_state=random_state, early_stopping_rounds=100, verbose=False)
            model.fit(train_pool, eval_set=val_pool)
            val_preds = model.predict_proba(val_pool)[:, 1] if task == 'classification' else model.predict(val_pool)
            test_fold_preds = model.predict_proba(test_df[features])[:, 1] if task == 'classification' else model.predict(test_df[features])
            
        else:
            raise ValueError("Unsupported model_name!")
            
        oof_preds[val_idx] = val_preds
        test_preds += test_fold_preds / n_folds 
        fold_score = get_eval_score(y_val, val_preds, task)
        metrics.append(fold_score)
        print(f"Fold {fold + 1} Score: {fold_score:.5f}")
        
    overall_score = get_eval_score(train_df[target_col], oof_preds, task)
    print(f"Overall OOF Score: {overall_score:.5f}")
    return oof_preds, test_preds, metrics

def tune_xgb_hyperparameters(train_df, features, target_col, task='classification', n_trials=20):
    """
    Optuna Hyperparameter tuning for XGBoost.
    """
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
        }
        
        X_train, X_val, y_train, y_val = train_test_split(
            train_df[features], train_df[target_col], test_size=0.2, random_state=42
        )
        
        if task == 'classification':
            model = xgb.XGBClassifier(**params, random_state=42, early_stopping_rounds=50, enable_categorical=True)
        else:
            model = xgb.XGBRegressor(**params, random_state=42, early_stopping_rounds=50, enable_categorical=True)
            
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        preds = model.predict_proba(X_val)[:, 1] if task == 'classification' else model.predict(X_val)
        return get_eval_score(y_val, preds, task)

    direction = 'maximize' if task == 'classification' else 'minimize'
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)
    
    print("Best XGBoost Trial:", study.best_trial.params)
    return study.best_trial.params

def tune_lgb_hyperparameters(train_df, features, target_col, task='classification', n_trials=20):
    """
    Optuna Hyperparameter tuning for LightGBM.
    """
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
        }
        
        # Quick Evaluation using a simple train_test_split (or 3-fold CV for speed)
        X_train, X_val, y_train, y_val = train_test_split(
            train_df[features], train_df[target_col], test_size=0.2, random_state=42
        )
        
        if task == 'classification':
            model = lgb.LGBMClassifier(**params, random_state=42, verbosity=-1)
        else:
            model = lgb.LGBMRegressor(**params, random_state=42, verbosity=-1)
            
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                  callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        
        preds = model.predict_proba(X_val)[:, 1] if task == 'classification' else model.predict(X_val)
        return get_eval_score(y_val, preds, task)

    # Optuna needs to know whether to maximize (AUC) or minimize (RMSE)
    direction = 'maximize' if task == 'classification' else 'minimize'
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)
    
    print("Best LightGBM Trial:", study.best_trial.params)
    return study.best_trial.params

def tune_cat_hyperparameters(train_df, features, target_col, task='classification', n_trials=20):
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255)
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
        preds = model.predict_proba(val_pool)[:, 1] if task == 'classification' else model.predict(val_pool)
        return get_eval_score(y_vl, preds, task)
    direction = 'maximize' if task == 'classification' else 'minimize'
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)
    print("Best CatBoost Trial:", study.best_trial.params)
    return study.best_trial.params
