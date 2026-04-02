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

def get_torch_device():
    """Returns 'cuda' if available, otherwise 'cpu'."""
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'

def is_cuml_available():
    """Checks if RAPIDS cuML is installed and compatible."""
    try:
        import cuml
        return True
    except ImportError:
        return False

def get_eval_score(y_true, y_pred, task):
    """Helper function to calculate the evaluation metric based on the task."""
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if task == 'classification':
            # Is it multiclass? 
            # 1. Check y_pred shape: (N, n_classes)
            # 2. Check y_true unique count
            n_classes_pred = y_pred.shape[1] if len(y_pred.shape) > 1 else 1
            n_classes_true = len(np.unique(y_true[~np.isnan(y_true)])) if y_true.dtype.kind in 'if' else len(np.unique(y_true))
            
            if n_classes_pred > 1:
                # Probabilities for all classes provided -> LogLoss or Multiclass AUC
                return log_loss(y_true, y_pred)
            elif n_classes_true > 2:
                # Multiclass true values but only 1D predictions. 
                # This usually happens if the model was trained as binary despite >2 classes.
                # In this case, standard AUC fails. We call with 'ovr' but it will likely still fail 
                # unless y_pred is labels (which it shouldn't be).
                try:
                    return roc_auc_score(y_true, y_pred, multi_class='ovr')
                except Exception:
                    return 0.5 # Safety fallback
            else:
                # Binary classification
                return roc_auc_score(y_true, y_pred)
        else:
            return np.sqrt(mean_squared_error(y_true, y_pred))
    except Exception as e:
        # Fallback to simple MSE or generic handling
        try:
            return np.sqrt(np.mean((y_true - y_pred)**2))
        except:
            return 0.0
