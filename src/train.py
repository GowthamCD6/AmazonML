"""
Model training module for Amazon ML Challenge 2026.
Trains pairwise binary classifiers (LightGBM / XGBoost / GradientBoosting) with class weighting and model persistence.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from src.features import FEATURE_COLUMNS

def train_pairwise_model(
    df_train_features: pd.DataFrame,
    feature_cols: List[str] = None,
    model_type: str = "lightgbm",
    scale_pos_weight: float = None,
    random_state: int = 42
) -> Tuple[Any, List[str], Dict[str, float]]:
    """
    Train a gradient boosting model on pairwise similarity features.
    """
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLUMNS if c in df_train_features.columns]
        
    X_train = df_train_features[feature_cols].values
    y_train = df_train_features["label"].values.astype(int)
    
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    print(f"Training Pairwise Model: {len(X_train)} samples ({pos_count} positive, {neg_count} negative, {len(feature_cols)} features)")
    
    if scale_pos_weight is None:
        # Precision-weighted ratio: slight conservative weight for negative class to protect precision
        scale_pos_weight = max(1.0, float(neg_count) / max(1, pos_count) * 0.8)

    model = None
    
    if model_type == "lightgbm":
        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                objective="binary",
                boosting_type="gbdt",
                n_estimators=350,
                learning_rate=0.04,
                num_leaves=31,
                max_depth=6,
                min_child_samples=15,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"LightGBM failed ({e}), falling back to XGBoost/HistGradientBoosting")
            model_type = "xgboost"

    if model_type == "xgboost" and model is None:
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=300,
                learning_rate=0.04,
                max_depth=6,
                scale_pos_weight=scale_pos_weight,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"XGBoost failed ({e}), falling back to HistGradientBoostingClassifier")
            model_type = "sklearn"

    if model is None or model_type == "sklearn":
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            max_depth=6,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
    # Extract feature importances if available
    feature_importances = {}
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        for f, imp in zip(feature_cols, importances):
            feature_importances[f] = float(imp)
            
    return model, feature_cols, feature_importances

def save_model_artifact(
    model: Any,
    feature_cols: List[str],
    best_threshold: float,
    metadata: Dict[str, Any],
    save_dir: str
) -> str:
    """Save trained model, feature configuration, threshold and metadata to directory."""
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, "model.joblib")
    meta_path = os.path.join(save_dir, "metadata.json")
    
    payload = {
        "model": model,
        "feature_cols": feature_cols,
        "best_threshold": best_threshold,
        "metadata": metadata
    }
    joblib.dump(payload, model_path)
    
    import json
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Saved model artifact to {model_path}")
    return model_path

def load_model_artifact(load_dir: str) -> Dict[str, Any]:
    """Load model artifact dictionary from directory."""
    model_path = os.path.join(load_dir, "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}")
    return joblib.load(model_path)
