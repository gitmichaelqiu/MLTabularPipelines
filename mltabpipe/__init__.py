from .ensemble import (
    train_xgb_model, 
    train_lgbm_model, 
    train_cb_model, 
    train_rf_model, 
    train_stacker,
    train_ridge_model,
    train_ydf_model,
    train_te_logit_model
)

from .core.update_checker import check_for_updates

__version__ = "0.1.1"
check_for_updates(__version__)

from .nn import (
    train_mlp_model,
    train_tabm_model,
    train_realmlp_model,
    train_ft_transformer,
    train_gnn_model,
    train_deepfm_model,
    train_ffm_model,
    train_trompt_model,
    train_dae_model
)

from .preprocessing import (
    add_snap_features,
    add_digit_features,
    add_arithmetic_interactions,
    add_binning_features,
    add_flag_counts,
    add_frequency_encoding,
    apply_modular_pipeline
)

from .model_selection import (
    add_pseudo_labels,
    apply_pseudo_labeling_pipeline
)

from .automl import (
    train_autogluon_model,
    train_lama_model
)
