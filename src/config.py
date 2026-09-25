"""
Configuration module for Amazon ML Challenge 2026.
Contains global paths, constants, model parameters, and blocking hyperparameters.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Config:
    # Project paths
    project_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir: str = os.path.join(project_root, "data")
    train_dir: str = os.path.join(data_dir, "train")
    test_dir: str = os.path.join(data_dir, "test")
    models_dir: str = os.path.join(project_root, "models")
    output_dir: str = os.path.join(project_root, "output")
    submissions_dir: str = os.path.join(project_root, "submissions")
    experiments_file: str = os.path.join(project_root, "experiments", "experiments.csv")
    leaderboard_file: str = os.path.join(project_root, "experiments", "leaderboard.csv")
    reports_dir: str = os.path.join(project_root, "reports")
    
    # Random seed
    random_seed: int = 42
    
    # Train/Val split ratio (by Source 1 entity)
    val_split_ratio: float = 0.20
    
    # Blocking parameters
    tfidf_top_k: int = 10
    tfidf_min_similarity: float = 0.25
    ngram_range: tuple = (2, 4)
    rare_token_min_freq: int = 2
    rare_token_max_freq: int = 25
    
    # Decision threshold optimization range
    threshold_grid: List[float] = field(default_factory=lambda: [
        0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
    ])
    default_threshold: float = 0.65
    
    # Model parameters
    model_type: str = "lightgbm"  # "lightgbm", "xgboost", or "sklearn"
    lgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1
    })
    
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "random_state": 42,
        "n_jobs": -1
    })

DEFAULT_CONFIG = Config()
