import sys
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.metrics import roc_auc_score, mean_squared_error
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
        return roc_auc_score(y_true, y_pred)
    else:
        try:
            return np.sqrt(mean_squared_error(y_true, y_pred))
        except NameError:
            # Fallback if mean_squared_error is not imported correctly or available
            return np.sqrt(np.mean((y_true - y_pred)**2))
