"""
Validation split and evaluation module for Amazon ML Challenge 2026.
Provides strict Source 1 entity-level splitting to prevent data leakage and measures performance.
"""

import random
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import pandas as pd

from src.scoring import evaluate_entity_macro_f05

def split_s1_entities(
    df_s1: pd.DataFrame,
    val_ratio: float = 0.20,
    seed: int = 42
) -> Tuple[Set[str], Set[str]]:
    """
    Split Source 1 entities into train and validation sets with zero overlap.
    """
    all_s1_ids = sorted(df_s1["entity_id"].unique())
    rng = random.Random(seed)
    shuffled = all_s1_ids.copy()
    rng.shuffle(shuffled)
    
    val_size = int(len(shuffled) * val_ratio)
    val_ids = set(shuffled[:val_size])
    train_ids = set(shuffled[val_size:])
    
    return train_ids, val_ids

def evaluate_validation_predictions(
    df_val_candidates_with_probs: pd.DataFrame,
    val_gt_map: Dict[str, list],
    threshold: float,
    val_s1_ids: Set[str] = None
) -> Dict[str, Any]:
    """
    Evaluate validation set predictions given candidate probabilities and a decision threshold.
    """
    if val_s1_ids is None:
        val_s1_ids = set(val_gt_map.keys())
        
    val_gt_subset = {k: val_gt_map.get(k, []) for k in val_s1_ids}
    predictions_by_s1 = {s1_id: [] for s1_id in val_s1_ids}
    
    for _, row in df_val_candidates_with_probs.iterrows():
        s1_id = row["source1_entity_id"]
        if s1_id not in val_s1_ids:
            continue
        prob = row["match_probability"]
        cand_id = row["candidate_entity_id"]
        
        if prob >= threshold:
            predictions_by_s1[s1_id].append(cand_id)
            
    # Remove duplicates
    for s1_id in predictions_by_s1:
        predictions_by_s1[s1_id] = list(dict.fromkeys(predictions_by_s1[s1_id]))
        
    metrics = evaluate_entity_macro_f05(predictions_by_s1, val_gt_subset)
    metrics["threshold"] = threshold
    metrics["predicted_matches_count"] = sum(len(v) for v in predictions_by_s1.values())
    metrics["predicted_singletons_count"] = sum(1 for v in predictions_by_s1.values() if len(v) == 0)
    
    return metrics
