"""
Official scoring metric implementation for Amazon ML Challenge 2026.
Calculates Precision, Recall, and F0.5 at global pair-level and entity macro-level.
"""

from typing import Dict, List, Set, Tuple, Any

def compute_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """
    Compute F-beta score: ((1 + beta^2) * P * R) / (beta^2 * P + R).
    For beta=0.5: (1.25 * P * R) / (0.25 * P + R).
    """
    if precision + recall == 0:
        return 0.0
    beta_sq = beta ** 2
    numerator = (1.0 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall
    if denominator == 0:
        return 0.0
    return numerator / denominator

def evaluate_pair_predictions(
    predicted_pairs: Set[Tuple[str, str]],
    true_pairs: Set[Tuple[str, str]]
) -> Dict[str, float]:
    """
    Evaluate global pair-level Precision, Recall, and F0.5.
    """
    tp = len(predicted_pairs.intersection(true_pairs))
    fp = len(predicted_pairs - true_pairs)
    fn = len(true_pairs - predicted_pairs)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if len(true_pairs) == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f05 = compute_f_beta(precision, recall, beta=0.5)
    f1 = compute_f_beta(precision, recall, beta=1.0)
    
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f05": f05,
        "f1": f1
    }

def evaluate_entity_macro_f05(
    predictions_by_s1: Dict[str, List[str]],
    ground_truth_by_s1: Dict[str, List[str]]
) -> Dict[str, float]:
    """
    Compute entity-level macro-averaged F0.5 across all Source 1 entities,
    giving full credit (1.0) to correctly identified singletons (empty match lists).
    """
    s1_entities = set(ground_truth_by_s1.keys())
    entity_f05_scores = []
    entity_precision_scores = []
    entity_recall_scores = []
    
    tp_total = 0
    fp_total = 0
    fn_total = 0
    
    for s1_id in s1_entities:
        true_targets = set(ground_truth_by_s1.get(s1_id, []))
        pred_targets = set(predictions_by_s1.get(s1_id, []))
        
        tp = len(pred_targets.intersection(true_targets))
        fp = len(pred_targets - true_targets)
        fn = len(true_targets - pred_targets)
        
        tp_total += tp
        fp_total += fp
        fn_total += fn
        
        if len(true_targets) == 0:
            # Singleton case
            if len(pred_targets) == 0:
                # Correctly predicted empty
                entity_precision_scores.append(1.0)
                entity_recall_scores.append(1.0)
                entity_f05_scores.append(1.0)
            else:
                # False positive on singleton
                entity_precision_scores.append(0.0)
                entity_recall_scores.append(1.0)
                entity_f05_scores.append(0.0)
        else:
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f05 = compute_f_beta(p, r, beta=0.5)
            
            entity_precision_scores.append(p)
            entity_recall_scores.append(r)
            entity_f05_scores.append(f05)
            
    macro_p = sum(entity_precision_scores) / max(1, len(entity_precision_scores))
    macro_r = sum(entity_recall_scores) / max(1, len(entity_recall_scores))
    macro_f05 = sum(entity_f05_scores) / max(1, len(entity_f05_scores))
    
    # Also calculate global micro scores
    global_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    global_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    global_f05 = compute_f_beta(global_p, global_r, beta=0.5)
    
    return {
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f05": macro_f05,
        "global_precision": global_p,
        "global_recall": global_r,
        "global_f05": global_f05,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total
    }
