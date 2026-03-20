import time
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy.special import expit

from mltabpipe.common import (
    np, pd, StratifiedKFold, KFold, get_eval_score, 
    StandardScaler, OneHotEncoder
)

def train_ridge_model(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    features: list, 
    target_col: str, 
    cat_features: list = None,
    params: dict = None, 
    task: str = 'classification',
    n_folds: int = 5, 
    random_states: list = [42]
):
    """
    Trains a Ridge Regression / Classifier model with Seed Ensembling.
    Automatically handles missing value imputation, scaling, and One-Hot Encoding 
    within the cross-validation loop to prevent data leakage.
    """
    if params is None:
        params = {'alpha': 1.0}  # L2 Regularization strength
        
    if isinstance(random_states, int):
        random_states = [random_states]

    print(f"--- Training Ridge Model ({task}) with {len(random_states)} seeds ---")
    
    cat_features = cat_features or []
    num_features = [c for c in features if c not in cat_features]
    
    # 1. Define Preprocessing Pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ])
    
    oof_preds = np.zeros(len(train_df), dtype=np.float32)
    test_preds = np.zeros(len(test_df), dtype=np.float32)
    all_metrics = []
    
    for seed in random_states:
        print(f"Seed {seed}")
        if task == 'classification':
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            
        for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df[features], train_df[target_col])):
            fold_t0 = time.time()
            print(f"Fold {fold + 1}/{n_folds}")
            
            X_train, y_train = train_df.iloc[tr_idx][features], train_df.iloc[tr_idx][target_col]
            X_val, y_val = train_df.iloc[va_idx][features], train_df.iloc[va_idx][target_col]
            
            # 2. Fit and Transform Preprocessor to avoid leakage
            X_tr_proc = preprocessor.fit_transform(X_train)
            X_va_proc = preprocessor.transform(X_val)
            X_te_proc = preprocessor.transform(test_df[features])
            
            # 3. Train Model
            if task == 'classification':
                model = RidgeClassifier(**params, random_state=seed)
                model.fit(X_tr_proc, y_train)
                
                # RidgeClassifier outputs distances to the hyperplane via `decision_function`
                # We use `expit` (the sigmoid function) to squash these into [0, 1] probabilities
                val_preds = expit(model.decision_function(X_va_proc))
                test_fold_preds = expit(model.decision_function(X_te_proc))
                
            else:
                model = Ridge(**params, random_state=seed)
                model.fit(X_tr_proc, y_train)
                
                val_preds = model.predict(X_va_proc)
                test_fold_preds = model.predict(X_te_proc)
                
            oof_preds[va_idx] += val_preds / len(random_states)
            test_preds += (test_fold_preds / n_folds) / len(random_states)
            
            # 4. Evaluate
            fold_score = get_eval_score(y_val, val_preds, task)
            all_metrics.append(fold_score)
            
            metric_name = "ROC AUC" if task == 'classification' else "RMSE"
            print(f"Fold {fold + 1} {metric_name}: {fold_score:.5f} | Time: {time.time()-fold_t0:.1f}s")
        
    overall_score = get_eval_score(train_df[target_col], oof_preds, task)
    metric_name = "ROC AUC" if task == 'classification' else "RMSE"
    print(f"Overall OOF {metric_name} (Ensembled): {overall_score:.5f}")
    print("-" * 30)
    
    return oof_preds, test_preds, all_metrics