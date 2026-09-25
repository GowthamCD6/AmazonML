"""
Main execution entry point for Amazon ML Challenge 2026 — Business Entity Resolution.
Supports CLI arguments for running full end-to-end pipeline, training, validation, profiling, and prediction.
"""

import argparse
import sys
import os

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import DEFAULT_CONFIG
from src.pipeline import EntityResolutionPipeline
from src.profiling import profile_dataset
from utils.validate_submission import validate

def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 — Business Entity Resolution Pipeline")
    parser.add_argument("--mode", default="full", choices=["full", "train", "validate", "profile", "predict"],
                        help="Execution mode (default: full)")
    parser.add_argument("--version", default="V1", help="Submission/Experiment version identifier (e.g. V1, V2)")
    parser.add_argument("--day", type=int, default=1, help="Competition Day (1, 2, or 3)")
    parser.add_argument("--model-type", default="lightgbm", choices=["lightgbm", "xgboost", "sklearn"],
                        help="Model architecture for pairwise classifier")
    parser.add_argument("--threshold", type=float, default=None, help="Fixed decision threshold override")
    parser.add_argument("--desc", default="Baseline LightGBM entity resolution with multi-strategy blocking",
                        help="Brief description of the experiment")
    
    args = parser.parse_args()
    
    pipeline = EntityResolutionPipeline(DEFAULT_CONFIG)
    
    if args.mode == "profile":
        print("Running dataset profiling...")
        profile_dataset(DEFAULT_CONFIG.train_dir, DEFAULT_CONFIG.test_dir)
        return
        
    elif args.mode == "validate":
        print("Validating current output files...")
        matching_file = os.path.join(DEFAULT_CONFIG.output_dir, "matching_results.tsv")
        candidate_file = os.path.join(DEFAULT_CONFIG.output_dir, "candidate_pairs.tsv")
        success = validate(matching_file, candidate_file, DEFAULT_CONFIG.test_dir)
        sys.exit(0 if success else 1)
        
    elif args.mode in ["full", "train", "predict"]:
        res = pipeline.run_experiment(
            version=args.version,
            day=args.day,
            change_desc=args.desc,
            model_type=args.model_type,
            threshold_override=args.threshold
        )
        print("\n" + "=" * 70)
        print(f"EXPERIMENT {args.version} SUMMARY:")
        print(f"  Validation Macro F0.5: {res['val_f05']:.4f}")
        print(f"  Validation Precision:  {res['val_precision']:.4f}")
        print(f"  Validation Recall:     {res['val_recall']:.4f}")
        print(f"  Candidate Recall:      {res['candidate_recall'] * 100:.2f}%")
        print(f"  Optimal Threshold:     {res['best_threshold']:.2f}")
        print(f"  Validator Passed:      {res['validator_passed']}")
        print("=" * 70)

if __name__ == "__main__":
    main()
