import sys
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.metrics import roc_auc_score, mean_squared_error, log_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
try:
    from sklearn.preprocessing import TargetEncoder
except ImportError:
    # Older sklearn versions don't have TargetEncoder
    TargetEncoder = None
from sklearn.linear_model import LogisticRegression

def get_eval_score(y_true, y_pred, task):
    """Helper function to calculate the evaluation metric based on the task."""
    if task == 'classification':
        # Multiclass or Binary?
        # Check shape of y_pred: (N, n_classes) for multiclass, (N,) for binary
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            # For multi-class, log_loss is a more standard default evaluation metric 
            return log_loss(y_true, y_pred)
        else:
            # AUC is default for binary classification
            return roc_auc_score(y_true, y_pred)
    else:
        try:
            return np.sqrt(mean_squared_error(y_true, y_pred))
        except Exception:
            return np.sqrt(np.mean((y_true - y_pred)**2))
