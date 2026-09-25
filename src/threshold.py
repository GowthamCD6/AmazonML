"""
Decision threshold optimization and post-processing module for Amazon ML Challenge 2026.
Tunes the classification decision boundary specifically for the macro F0.5 competition metric.
"""

from typing import Dict, List, Set, Tuple, Any
import numpy as np
import pandas as pd

from src.validation import evaluate_validation_predictions

def optimize_f05_threshold(
    df_val_candidates_with_probs: pd.DataFrame,
    val_gt_map: Dict[str, list],
    val_s1_ids: Set[str],
    threshold_grid: List[float] = None
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Search over candidate decision thresholds to maximize macro F0.5 on the validation set.
    """
    if threshold_grid is None:
        threshold_grid = [
            0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
        ]
        
    results = []
    best_threshold = 0.65
    best_f05 = -1.0
    
    for th in threshold_grid:
        metrics = evaluate_validation_predictions(
            df_val_candidates_with_probs=df_val_candidates_with_probs,
            val_gt_map=val_gt_map,
            threshold=th,
            val_s1_ids=val_s1_ids
        )
        macro_f05 = metrics["macro_f05"]
        results.append({
            "threshold": th,
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f05": macro_f05,
            "global_precision": metrics["global_precision"],
            "global_recall": metrics["global_recall"],
            "global_f05": metrics["global_f05"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"]
        })
        
        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_threshold = th
            
    print(f"Optimal Threshold Selected: {best_threshold:.2f} (Validation Macro F0.5 = {best_f05:.4f})")
    return best_threshold, best_f05, results
