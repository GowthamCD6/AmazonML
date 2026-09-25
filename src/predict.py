"""
Inference and prediction module for Amazon ML Challenge 2026.
Applies blocking, feature extraction, and trained classifier to generate test predictions.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd

from src.normalization import normalize_dataframe
from src.blocking import BlockingEngine
from src.features import extract_pairwise_features

def generate_test_predictions(
    df_test_s1: pd.DataFrame,
    df_test_s2: pd.DataFrame,
    df_test_s3: pd.DataFrame,
    model: Any,
    feature_cols: List[str],
    threshold: float,
    blocking_engine: BlockingEngine = None
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Run full test inference pipeline:
    1. Normalize test records
    2. Block to generate candidate pairs
    3. Extract pairwise features
    4. Predict match probabilities
    5. Filter by validated threshold
    6. Aggregate matches per Source 1 entity
    """
    if blocking_engine is None:
        blocking_engine = BlockingEngine()
        
    print("Normalizing test datasets...")
    df_s1_norm = normalize_dataframe(df_test_s1)
    df_s2_norm = normalize_dataframe(df_test_s2)
    df_s3_norm = normalize_dataframe(df_test_s3)
    
    df_targets = pd.concat([df_s2_norm, df_s3_norm], ignore_index=True)
    
    print("Generating candidate pairs on test data...")
    df_candidates = blocking_engine.generate_candidates(df_s1_norm, df_targets)
    print(f"Generated {len(df_candidates)} test candidate pairs.")
    
    if len(df_candidates) == 0:
        all_s1_ids = df_test_s1["entity_id"].tolist()
        return df_candidates, {s1_id: [] for s1_id in all_s1_ids}
        
    print("Extracting pairwise features for test candidates...")
    df_features = extract_pairwise_features(df_candidates, df_s1_norm, df_targets)
    
    X_test = df_features[feature_cols].values
    
    print(f"Predicting match probabilities with model (threshold={threshold:.2f})...")
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        # Map decision function to [0, 1] via sigmoid
        scores = model.decision_function(X_test)
        probs = 1.0 / (1.0 + np.exp(-scores))
    else:
        probs = model.predict(X_test).astype(float)
        
    df_candidates["match_probability"] = probs
    
    # Aggregate predictions by S1
    all_s1_ids = set(df_test_s1["entity_id"].tolist())
    predictions_by_s1 = {s1_id: [] for s1_id in all_s1_ids}
    
    for _, row in df_candidates.iterrows():
        s1_id = row["source1_entity_id"]
        c_id = row["candidate_entity_id"]
        prob = row["match_probability"]
        
        if prob >= threshold:
            predictions_by_s1[s1_id].append(c_id)
            
    # Deduplicate and sort IDs
    for s1_id in predictions_by_s1:
        predictions_by_s1[s1_id] = list(dict.fromkeys(predictions_by_s1[s1_id]))
        
    num_matches = sum(len(v) for v in predictions_by_s1.values())
    singletons = sum(1 for v in predictions_by_s1.values() if len(v) == 0)
    print(f"Prediction Complete: {len(all_s1_ids)} S1 entities -> {num_matches} total matches, {singletons} singletons.")
    
    return df_candidates, predictions_by_s1
