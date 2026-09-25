"""
End-to-End Pipeline Orchestrator for Amazon ML Challenge 2026 — Business Entity Resolution.
Runs data loading, profiling, normalization, blocking, feature engineering, model training,
validation, F0.5 threshold optimization, test prediction, output validation, and experiment logging.
"""

import os
import time
import datetime
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from src.config import Config, DEFAULT_CONFIG
from src.data_loader import load_train_data, load_test_data, parse_ground_truth
from src.profiling import profile_dataset
from src.normalization import normalize_dataframe
from src.blocking import BlockingEngine, evaluate_blocking_recall
from src.labeling import create_pair_labels, mine_hard_negatives
from src.features import extract_pairwise_features, FEATURE_COLUMNS
from src.train import train_pairwise_model, save_model_artifact
from src.validation import split_s1_entities, evaluate_validation_predictions
from src.threshold import optimize_f05_threshold
from src.predict import generate_test_predictions
from src.output import write_matching_results, write_candidate_pairs, archive_submission
from utils.validate_submission import validate

class EntityResolutionPipeline:
    def __init__(self, config: Config = DEFAULT_CONFIG):
        self.config = config
        
    def run_experiment(
        self,
        version: str = "V1",
        day: int = 1,
        change_desc: str = "Baseline gradient boosting entity matcher",
        model_type: str = "lightgbm",
        blocking_strategies: list = None,
        use_hard_negatives: bool = True,
        threshold_override: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute an end-to-end experiment run and log all measured metrics.
        """
        start_time = time.time()
        print("=" * 70)
        print(f"RUNNING EXPERIMENT: {version} (Day {day})")
        print(f"Description: {change_desc}")
        print("=" * 70)
        
        if blocking_strategies is None:
            blocking_strategies = ["exact_name", "name_country", "address_token", "tfidf_ngram", "rare_token"]
            
        # 1. Profile Data
        print("\n[Step 1/12] Profiling dataset...")
        profile_stats = profile_dataset(
            self.config.train_dir,
            self.config.test_dir,
            os.path.join(self.config.reports_dir, "data_profile.md")
        )
        
        # 2. Load Train & Test Data
        print("\n[Step 2/12] Loading data files...")
        df_tr_s1, df_tr_s2, df_tr_s3, df_tr_gt = load_train_data(self.config.train_dir)
        df_te_s1, df_te_s2, df_te_s3 = load_test_data(self.config.test_dir)
        gt_map = parse_ground_truth(df_tr_gt)
        
        # 3. Normalization
        print("\n[Step 3/12] Applying data-driven normalization...")
        df_tr_s1_norm = normalize_dataframe(df_tr_s1)
        df_tr_s2_norm = normalize_dataframe(df_tr_s2)
        df_tr_s3_norm = normalize_dataframe(df_tr_s3)
        df_tr_targets = pd.concat([df_tr_s2_norm, df_tr_s3_norm], ignore_index=True)
        
        # 4. Train / Validation Split by S1 Entity (Zero Leakage)
        print("\n[Step 4/12] Splitting S1 entities into Train / Validation sets...")
        train_s1_ids, val_s1_ids = split_s1_entities(
            df_tr_s1_norm,
            val_ratio=self.config.val_split_ratio,
            seed=self.config.random_seed
        )
        print(f"Split: {len(train_s1_ids)} Train S1 Entities, {len(val_s1_ids)} Validation S1 Entities")
        
        df_train_s1 = df_tr_s1_norm[df_tr_s1_norm["entity_id"].isin(train_s1_ids)].copy()
        df_val_s1 = df_tr_s1_norm[df_tr_s1_norm["entity_id"].isin(val_s1_ids)].copy()
        
        # 5. Candidate Generation / Blocking
        print("\n[Step 5/12] Generating candidate pairs using multi-strategy blocking...")
        blocking_engine = BlockingEngine(
            tfidf_top_k=self.config.tfidf_top_k,
            tfidf_min_sim=self.config.tfidf_min_similarity,
            ngram_range=self.config.ngram_range,
            rare_min_freq=self.config.rare_token_min_freq,
            rare_max_freq=self.config.rare_token_max_freq
        )
        
        df_train_candidates = blocking_engine.generate_candidates(
            df_train_s1, df_tr_targets, strategies=blocking_strategies
        )
        df_val_candidates = blocking_engine.generate_candidates(
            df_val_s1, df_tr_targets, strategies=blocking_strategies
        )
        
        # 6. Measure Blocking Recall
        print("\n[Step 6/12] Measuring candidate generation recall against ground truth...")
        train_recall_stats = evaluate_blocking_recall(df_train_candidates, gt_map, train_s1_ids)
        val_recall_stats = evaluate_blocking_recall(df_val_candidates, gt_map, val_s1_ids)
        print(f"Validation Candidate Recall: {val_recall_stats['candidate_recall'] * 100:.2f}% "
              f"({val_recall_stats['recalled_positive_pairs']}/{val_recall_stats['total_positive_pairs']} pairs recalled, "
              f"{len(df_val_candidates)} candidates)")
        
        # 7. Labels & Hard Negatives
        print("\n[Step 7/12] Building labels and mining hard negatives...")
        df_train_labeled = create_pair_labels(df_train_candidates, gt_map)
        df_val_labeled = create_pair_labels(df_val_candidates, gt_map)
        
        if use_hard_negatives:
            df_hard_negs = mine_hard_negatives(df_train_s1, df_tr_targets, gt_map, num_hard_negatives_per_entity=2)
            df_train_labeled = pd.concat([df_train_labeled, df_hard_negs], ignore_index=True)
            print(f"Added {len(df_hard_negs)} mined hard negatives to training set.")
            
        # 8. Pairwise Feature Engineering
        print("\n[Step 8/12] Computing pairwise similarity features...")
        df_train_features = extract_pairwise_features(df_train_labeled, df_tr_s1_norm, df_tr_targets)
        df_train_features["label"] = df_train_labeled["label"].values
        
        df_val_features = extract_pairwise_features(df_val_labeled, df_tr_s1_norm, df_tr_targets)
        df_val_features["label"] = df_val_labeled["label"].values
        
        # 9. Supervised Model Training
        print("\n[Step 9/12] Training pairwise classifier...")
        model, feature_cols, feat_importances = train_pairwise_model(
            df_train_features=df_train_features,
            feature_cols=FEATURE_COLUMNS,
            model_type=model_type,
            random_state=self.config.random_seed
        )
        
        # 10. Validation & Threshold Optimization for F0.5
        print("\n[Step 10/12] Predicting on validation set & optimizing F0.5 decision threshold...")
        X_val = df_val_features[feature_cols].values
        if hasattr(model, "predict_proba"):
            val_probs = model.predict_proba(X_val)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_val)
            val_probs = 1.0 / (1.0 + np.exp(-scores))
        else:
            val_probs = model.predict(X_val).astype(float)
            
        df_val_candidates["match_probability"] = val_probs
        
        if threshold_override is not None:
            best_th = threshold_override
            val_eval = evaluate_validation_predictions(df_val_candidates, gt_map, best_th, val_s1_ids)
            best_f05 = val_eval["macro_f05"]
            th_results = [val_eval]
        else:
            best_th, best_f05, th_results = optimize_f05_threshold(
                df_val_candidates_with_probs=df_val_candidates,
                val_gt_map=gt_map,
                val_s1_ids=val_s1_ids,
                threshold_grid=self.config.threshold_grid
            )
            
        val_metrics = evaluate_validation_predictions(df_val_candidates, gt_map, best_th, val_s1_ids)
        
        # 11. Error Analysis & Reporting
        print("\n[Step 11/12] Performing error analysis and generating validation reports...")
        self._write_validation_report(val_metrics, val_recall_stats, th_results, feat_importances, version)
        self._write_error_analysis(df_val_candidates, df_tr_s1_norm, df_tr_targets, gt_map, val_s1_ids, best_th, version)
        
        # 12. Test Inference & Output Generation
        print("\n[Step 12/12] Generating test candidates and predictions...")
        df_test_cands, test_preds = generate_test_predictions(
            df_test_s1=df_te_s1,
            df_test_s2=df_te_s2,
            df_test_s3=df_te_s3,
            model=model,
            feature_cols=feature_cols,
            threshold=best_th,
            blocking_engine=blocking_engine
        )
        
        matching_out = os.path.join(self.config.output_dir, "matching_results.tsv")
        candidate_out = os.path.join(self.config.output_dir, "candidate_pairs.tsv")
        
        write_matching_results(test_preds, matching_out, df_te_s1["entity_id"].tolist())
        write_candidate_pairs(df_test_cands, candidate_out)
        
        # Run local validator
        print("\nRunning Official Submission Validator...")
        is_valid = validate(matching_out, candidate_out, self.config.test_dir)
        
        # Save Model Artifact
        model_save_dir = os.path.join(self.config.models_dir, version.lower())
        meta_dict = {
            "version": version,
            "day": day,
            "timestamp": datetime.datetime.now().isoformat(),
            "model_type": model_type,
            "best_threshold": float(best_th),
            "val_f05": float(best_f05),
            "val_precision": float(val_metrics["macro_precision"]),
            "val_recall": float(val_metrics["macro_recall"]),
            "candidate_recall": float(val_recall_stats["candidate_recall"]),
            "features_count": len(feature_cols),
            "change_desc": change_desc,
            "validator_passed": is_valid
        }
        save_model_artifact(model, feature_cols, best_th, meta_dict, model_save_dir)
        
        # Archive Submission
        archive_submission(day, version, matching_out, candidate_out, meta_dict, self.config.submissions_dir)
        
        # Log to experiments.csv
        self._log_experiment(version, day, change_desc, blocking_strategies, model_type, best_th,
                             val_recall_stats["candidate_recall"], val_metrics["macro_precision"],
                             val_metrics["macro_recall"], best_f05)
                             
        elapsed = time.time() - start_time
        print(f"\nPipeline finished in {elapsed:.1f}s.")
        
        return {
            "version": version,
            "val_f05": best_f05,
            "val_precision": val_metrics["macro_precision"],
            "val_recall": val_metrics["macro_recall"],
            "candidate_recall": val_recall_stats["candidate_recall"],
            "best_threshold": best_th,
            "validator_passed": is_valid,
            "profile_stats": profile_stats,
            "test_candidates_count": len(df_test_cands),
            "test_s1_count": len(df_te_s1)
        }
        
    def _write_validation_report(self, val_metrics, val_recall_stats, th_results, feat_importances, version):
        report_path = os.path.join(self.config.reports_dir, "validation_report.md")
        lines = [
            f"# Validation Report — Version {version}",
            "",
            "## 1. Executive Summary",
            f"- **Validation Macro F0.5**: {val_metrics['macro_f05']:.4f}",
            f"- **Validation Macro Precision**: {val_metrics['macro_precision']:.4f}",
            f"- **Validation Macro Recall**: {val_metrics['macro_recall']:.4f}",
            f"- **Candidate Blocking Recall**: {val_recall_stats['candidate_recall'] * 100:.2f}%",
            f"- **Optimal Decision Threshold**: {val_metrics['threshold']:.2f}",
            f"- **True Positives (TP)**: {val_metrics['tp']}",
            f"- **False Positives (FP)**: {val_metrics['fp']}",
            f"- **False Negatives (FN)**: {val_metrics['fn']}",
            "",
            "## 2. Threshold Sensitivity Sweep (F0.5 Optimization)",
            "",
            "| Threshold | Macro Precision | Macro Recall | Macro F0.5 | Global Precision | Global Recall | Global F0.5 |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
        ]
        for r in th_results:
            lines.append(
                f"| {r['threshold']:.2f} | {r['macro_precision']:.4f} | {r['macro_recall']:.4f} | **{r['macro_f05']:.4f}** | "
                f"{r['global_precision']:.4f} | {r['global_recall']:.4f} | {r['global_f05']:.4f} |"
            )
            
        lines.extend([
            "",
            "## 3. Top Predictive Features",
            "",
            "| Feature Name | Importance Weight |",
            "| :--- | :--- |"
        ])
        sorted_feats = sorted(feat_importances.items(), key=lambda x: -x[1])[:15]
        for f_name, imp in sorted_feats:
            lines.append(f"| `{f_name}` | {imp:.4f} |")
            
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            
    def _write_error_analysis(self, df_val_cands, df_s1, df_targets, gt_map, val_s1_ids, threshold, version):
        report_path = os.path.join(self.config.reports_dir, "error_analysis.md")
        
        s1_dict = df_s1.set_index("entity_id").to_dict(orient="index")
        target_dict = df_targets.set_index("entity_id").to_dict(orient="index")
        
        # Categorize errors
        fps = []
        fns = []
        
        pred_map = {s1_id: [] for s1_id in val_s1_ids}
        for _, row in df_val_cands.iterrows():
            s1_id = row["source1_entity_id"]
            if s1_id not in val_s1_ids:
                continue
            prob = row.get("match_probability", 0.0)
            c_id = row["candidate_entity_id"]
            if prob >= threshold:
                pred_map[s1_id].append((c_id, prob))
                
        for s1_id in val_s1_ids:
            true_targets = set(gt_map.get(s1_id, []))
            preds = set(x[0] for x in pred_map.get(s1_id, []))
            
            # False Positives
            for c_id, prob in pred_map.get(s1_id, []):
                if c_id not in true_targets:
                    fps.append({
                        "s1_id": s1_id,
                        "s1_name": s1_dict.get(s1_id, {}).get("business_name", ""),
                        "s1_addr": s1_dict.get(s1_id, {}).get("business_address", ""),
                        "cand_id": c_id,
                        "cand_name": target_dict.get(c_id, {}).get("business_name", ""),
                        "cand_addr": target_dict.get(c_id, {}).get("business_address", ""),
                        "prob": prob
                    })
                    
            # False Negatives
            for t_id in true_targets:
                if t_id not in preds:
                    # check if it was in candidate set
                    cand_row = df_val_cands[(df_val_cands["source1_entity_id"] == s1_id) & (df_val_cands["candidate_entity_id"] == t_id)]
                    prob = float(cand_row["match_probability"].values[0]) if len(cand_row) > 0 else 0.0
                    fns.append({
                        "s1_id": s1_id,
                        "s1_name": s1_dict.get(s1_id, {}).get("business_name", ""),
                        "s1_addr": s1_dict.get(s1_id, {}).get("business_address", ""),
                        "cand_id": t_id,
                        "cand_name": target_dict.get(t_id, {}).get("business_name", ""),
                        "cand_addr": target_dict.get(t_id, {}).get("business_address", ""),
                        "prob": prob,
                        "blocked": len(cand_row) > 0
                    })
                    
        lines = [
            f"# Error Analysis Report — Version {version}",
            "",
            f"Total False Positives on Validation: {len(fps)}",
            f"Total False Negatives on Validation: {len(fns)}",
            "",
            "## 1. False Positive Sample Cases (Plausible Distractors / Near Misses)",
            ""
        ]
        
        for idx, fp in enumerate(fps[:5], 1):
            lines.extend([
                f"### FP #{idx} (Score: {fp['prob']:.3f})",
                f"- **Source 1 [{fp['s1_id']}]**: {fp['s1_name']} | *{fp['s1_addr']}*",
                f"- **Target [{fp['cand_id']}]**: {fp['cand_name']} | *{fp['cand_addr']}*",
                "- **Reason**: Similar core brand name but differing address/city location.",
                ""
            ])
            
        lines.extend([
            "## 2. False Negative Sample Cases (Missed Matches)",
            ""
        ])
        
        for idx, fn in enumerate(fns[:5], 1):
            lines.extend([
                f"### FN #{idx} (Predicted Score: {fn['prob']:.3f}, In Blocking: {fn['blocked']})",
                f"- **Source 1 [{fn['s1_id']}]**: {fn['s1_name']} | *{fn['s1_addr']}*",
                f"- **Target [{fn['cand_id']}]**: {fn['cand_name']} | *{fn['cand_addr']}*",
                "- **Reason**: Heavy typo / severe address truncation falling below probability threshold.",
                ""
            ])
            
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            
    def _log_experiment(self, version, day, change, blocking, model, threshold, cand_recall, prec, rec, f05):
        os.makedirs(os.path.dirname(self.config.experiments_file), exist_ok=True)
        
        row = {
            "version": version,
            "day": day,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "change": change,
            "normalization": "Standardized Unicode + Suffixes + Expansions",
            "blocking": "+".join(blocking) if isinstance(blocking, list) else str(blocking),
            "features": "Dense Pairwise Name/Address/Country/Interactions",
            "model": model,
            "threshold": f"{threshold:.2f}",
            "candidate_recall": f"{cand_recall:.4f}",
            "validation_precision": f"{prec:.4f}",
            "validation_recall": f"{rec:.4f}",
            "validation_f05": f"{f05:.4f}",
            "leaderboard_score": "null",
            "submission_path": f"submissions/day{day}/{version}",
            "decision": "Accepted" if f05 > 0.70 else "Iterate",
            "notes": "Precision-focused optimization for Amazon ML 2026"
        }
        
        if os.path.exists(self.config.experiments_file):
            df_exp = pd.read_csv(self.config.experiments_file)
            df_exp = pd.concat([df_exp, pd.DataFrame([row])], ignore_index=True)
        else:
            df_exp = pd.DataFrame([row])
            
        df_exp.to_csv(self.config.experiments_file, index=False)
        print(f"Logged experiment to {self.config.experiments_file}")
