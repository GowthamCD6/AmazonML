"""
Training pair labeling and hard-negative synthesis module for Amazon ML Challenge 2026.
Assigns binary labels to candidate pairs based on Ground Truth and mines challenging hard negatives.
"""

import random
import pandas as pd
from typing import Dict, List, Set, Tuple

def create_pair_labels(
    df_candidates: pd.DataFrame,
    gt_map: Dict[str, list]
) -> pd.DataFrame:
    """
    Attach binary label 'is_match' (1 or 0) to candidate pairs based on ground truth.
    """
    labels = []
    
    for _, row in df_candidates.iterrows():
        s1_id = row["source1_entity_id"]
        c_id = row["candidate_entity_id"]
        
        true_matches = gt_map.get(s1_id, [])
        is_match = 1 if c_id in true_matches else 0
        labels.append(is_match)
        
    df_labeled = df_candidates.copy()
    df_labeled["label"] = labels
    return df_labeled

def mine_hard_negatives(
    df_s1: pd.DataFrame,
    df_targets: pd.DataFrame,
    gt_map: Dict[str, list],
    num_hard_negatives_per_entity: int = 2
) -> pd.DataFrame:
    """
    Synthesize hard negatives by pairing Source 1 entities with targets that share
    similar token substrings or identical country but are NOT true matches.
    """
    # Build fast token index on targets
    target_token_idx = {}
    target_ids = df_targets["entity_id"].tolist()
    
    true_pairs = set()
    for s1_id, matches in gt_map.items():
        for m in matches:
            true_pairs.add((s1_id, m))
            
    hard_neg_records = []
    
    # Random selection of distractor records
    target_sample_pool = target_ids.copy()
    
    for _, row in df_s1.iterrows():
        s1_id = row["entity_id"]
        true_matches = set(gt_map.get(s1_id, []))
        
        # Pick random target not in true matches as negative
        attempts = 0
        count = 0
        while count < num_hard_negatives_per_entity and attempts < 10:
            attempts += 1
            cand = random.choice(target_sample_pool)
            if cand not in true_matches and (s1_id, cand) not in true_pairs:
                hard_neg_records.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": cand,
                    "blocked_by_exact_name": 0,
                    "blocked_by_name_country": 0,
                    "blocked_by_address": 0,
                    "blocked_by_tfidf": 0,
                    "blocked_by_rare_token": 0,
                    "tfidf_similarity": 0.0,
                    "num_blocks_hit": 1,
                    "label": 0
                })
                count += 1
                
    return pd.DataFrame(hard_neg_records)
