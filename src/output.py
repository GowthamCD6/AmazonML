"""
Output generation and submission packaging module for Amazon ML Challenge 2026.
Produces compliant matching_results.tsv and candidate_pairs.tsv and archives submissions.
"""

import os
import shutil
import json
import pandas as pd
from typing import Dict, List, Set, Any

def write_matching_results(
    predictions_by_s1: Dict[str, List[str]],
    output_path: str,
    all_test_s1_ids: List[str] = None
) -> str:
    """
    Write matching_results.tsv in the exact required challenge format.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if all_test_s1_ids is None:
        all_test_s1_ids = sorted(predictions_by_s1.keys())
        
    records = []
    for s1_id in all_test_s1_ids:
        matches = predictions_by_s1.get(s1_id, [])
        # Deduplicate while preserving order
        unique_matches = list(dict.fromkeys(matches))
        matches_str = ",".join(unique_matches)
        records.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": matches_str
        })
        
    df_out = pd.DataFrame(records)
    df_out.to_csv(output_path, sep="\t", index=False)
    print(f"Saved matching results to: {output_path} ({len(df_out)} rows)")
    return output_path

def write_candidate_pairs(
    df_candidates: pd.DataFrame,
    output_path: str
) -> str:
    """
    Write candidate_pairs.tsv containing the exact candidates passed into the matching model.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df_out = df_candidates[["source1_entity_id", "candidate_entity_id"]].drop_duplicates()
    df_out.to_csv(output_path, sep="\t", index=False)
    print(f"Saved candidate pairs to: {output_path} ({len(df_out)} pairs)")
    return output_path

def archive_submission(
    day: int,
    version: str,
    matching_path: str,
    candidate_path: str,
    metadata: Dict[str, Any],
    submissions_dir: str = "submissions"
) -> str:
    """
    Archive submission files and metadata under submissions/day{day}/{version}/.
    """
    dest_dir = os.path.join(submissions_dir, f"day{day}", version)
    os.makedirs(dest_dir, exist_ok=True)
    
    dest_matching = os.path.join(dest_dir, "matching_results.tsv")
    dest_candidate = os.path.join(dest_dir, "candidate_pairs.tsv")
    dest_meta = os.path.join(dest_dir, "metadata.json")
    
    shutil.copyfile(matching_path, dest_matching)
    shutil.copyfile(candidate_path, dest_candidate)
    
    metadata["version"] = version
    metadata["day"] = day
    
    with open(dest_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Archived submission to: {dest_dir}")
    return dest_dir
