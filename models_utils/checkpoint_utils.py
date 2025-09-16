# models_utils/checkpoint_utils.py
import os
import pickle
import torch
import json
from pathlib import Path

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def get_checkpoint_path(dataset_type, causal_config, scenario, model_family, 
                       model_name, repeat_idx):
    """
    Generate standardized checkpoint path.
    
    Example: model_checkpoints/synthetic/RCT_0_5/scenario_A/t_learner/
             t_learner_lasso_Pseudo_obs_repeat0.pkl
    """
    base_path = os.path.join(
        "model_checkpoints",
        dataset_type,
        causal_config,
        scenario,
        model_family
    )
    ensure_dir(base_path)
    filename = f"{model_name}_repeat{repeat_idx}.pkl"
    return os.path.join(base_path, filename)