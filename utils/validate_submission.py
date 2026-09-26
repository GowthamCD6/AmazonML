"""
Official Submission Validator for Amazon ML Challenge 2026 — Business Entity Resolution
Validates submission compliance against all challenge formatting rules.
"""

import argparse
import os
import sys
import pandas as pd

def validate(matching_file, candidate_file, test_dir):
    print("=" * 60)
    print("AMAZON ML CHALLENGE 2026 — SUBMISSION VALIDATOR")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    # 1. Check file existence
    if not os.path.exists(matching_file):
        errors.append(f"Matching file not found: {matching_file}")
    if not os.path.exists(candidate_file):
        errors.append(f"Candidate file not found: {candidate_file}")
    
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")
    
    for p in [s1_path, s2_path, s3_path]:
        if not os.path.exists(p):
            errors.append(f"Test reference file not found: {p}")
            
    if errors:
        print("CRITICAL ERRORS FOUND:")
        for e in errors:
            print(f"  [ERROR] {e}")
        return False
        
    # 2. Load test source IDs
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str)
    df_s2 = pd.read_csv(s2_path, sep="\t", dtype=str)
    df_s3 = pd.read_csv(s3_path, sep="\t", dtype=str)
    
    valid_s1_ids = set(df_s1["entity_id"].dropna().tolist())
    valid_s2_ids = set(df_s2["entity_id"].dropna().tolist())
    valid_s3_ids = set(df_s3["entity_id"].dropna().tolist())
    valid_target_ids = valid_s2_ids.union(valid_s3_ids)
    
    print(f"Loaded Test Reference Data:")
    print(f"  Source 1 IDs: {len(valid_s1_ids)}")
    print(f"  Source 2 IDs: {len(valid_s2_ids)}")
    print(f"  Source 3 IDs: {len(valid_s3_ids)}")
    print(f"  Total Valid Target IDs: {len(valid_target_ids)}")
    
    # 3. Validate matching_results.tsv
    try:
        df_match = pd.read_csv(matching_file, sep="\t", dtype=str).fillna("")
    except Exception as e:
        print(f"  [ERROR] Failed to read matching file as TSV: {e}")
        return False
        
    if "source1_entity_id" not in df_match.columns or "matched_entity_ids" not in df_match.columns:
        errors.append("matching_results.tsv must have columns: ['source1_entity_id', 'matched_entity_ids']")
    
    matching_s1_ids = list(df_match["source1_entity_id"])
    if len(matching_s1_ids) != len(valid_s1_ids):
        errors.append(f"Row count mismatch: matching_results.tsv has {len(matching_s1_ids)} rows, expected {len(valid_s1_ids)}")
        
    if len(matching_s1_ids) != len(set(matching_s1_ids)):
        errors.append("matching_results.tsv contains duplicate source1_entity_id rows")
        
    missing_s1 = valid_s1_ids - set(matching_s1_ids)
    if missing_s1:
        errors.append(f"matching_results.tsv is missing {len(missing_s1)} test Source 1 entities")
        
    extra_s1 = set(matching_s1_ids) - valid_s1_ids
    if extra_s1:
        errors.append(f"matching_results.tsv contains {len(extra_s1)} unknown Source 1 IDs")
        
    # Check match content
    predicted_pairs = set()
    total_matches = 0
    singletons = 0
    
    s1_vals = df_match["source1_entity_id"].astype(str).values
    match_vals = df_match["matched_entity_ids"].astype(str).values
    
    for s1_id, matches_str in zip(s1_vals, match_vals):
        matches_str = matches_str.strip()
        if not matches_str or matches_str.lower() in ["nan", "none"]:
            singletons += 1
            continue
            
        m_ids = [m.strip() for m in matches_str.split(",") if m.strip()]
        if len(m_ids) != len(set(m_ids)):
            errors.append(f"Entity {s1_id} contains duplicate matched IDs: {matches_str}")
            
        for m in m_ids:
            if m not in valid_target_ids:
                errors.append(f"Entity {s1_id} has invalid match ID not in test S2/S3: '{m}'")
            predicted_pairs.add((s1_id, m))
            total_matches += 1
            
    print(f"Matching Results Statistics:")
    print(f"  Total Source 1 Entities: {len(df_match)}")
    print(f"  Entities with Matches: {len(df_match) - singletons}")
    print(f"  Entities with No Matches (Singletons): {singletons}")
    print(f"  Total Predicted Pairs: {total_matches}")
    
    # 4. Validate candidate_pairs.tsv
    try:
        df_cand = pd.read_csv(candidate_file, sep="\t", dtype=str).fillna("")
    except Exception as e:
        print(f"  [ERROR] Failed to read candidate file as TSV: {e}")
        return False
        
    if "source1_entity_id" not in df_cand.columns or "candidate_entity_id" not in df_cand.columns:
        errors.append("candidate_pairs.tsv must have columns: ['source1_entity_id', 'candidate_entity_id']")
        
    candidate_pairs = set(zip(df_cand["source1_entity_id"], df_cand["candidate_entity_id"]))
    print(f"Candidate Pairs Statistics:")
    print(f"  Total Candidate Pairs: {len(candidate_pairs)}")
    
    # Check rule: every predicted match must be in candidate_pairs.tsv
    missed_cands = predicted_pairs - candidate_pairs
    if missed_cands:
        errors.append(f"CRITICAL: {len(missed_cands)} predicted matches are NOT present in candidate_pairs.tsv! (e.g. {list(missed_cands)[:3]})")
        
    print("-" * 60)
    if errors:
        print(f"FAILED: Found {len(errors)} validation error(s):")
        for e in errors[:10]:
            print(f"  [ERROR] {e}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors.")
        return False
    else:
        print("SUCCESS: All validation checks passed perfectly!")
        print("  [x] File formats: Valid TSV")
        print("  [x] Complete test entity coverage")
        print("  [x] No duplicate IDs")
        print("  [x] Only valid test target IDs")
        print("  [x] Every predicted match exists in candidate_pairs.tsv")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", default="output/matching_results.tsv", help="Path to matching_results.tsv")
    parser.add_argument("--candidate", default="output/candidate_pairs.tsv", help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", default="data/test", help="Path to test data directory")
    args = parser.parse_args()
    
    success = validate(args.matching, args.candidate, args.test_dir)
    sys.exit(0 if success else 1)
