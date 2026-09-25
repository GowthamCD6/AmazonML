"""
Data loader module for Amazon ML Challenge 2026.
Provides robust TSV parsing, type preservation, schema validation, and safe null handling.
"""

import os
import pandas as pd
from typing import Tuple, Dict, Optional

REQUIRED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
REQUIRED_GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

def load_tsv(filepath: str, expected_columns: Optional[list] = None) -> pd.DataFrame:
    """
    Safely load a TSV file with tab separation and string dtype preservation.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
        
    df = pd.read_csv(
        filepath,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=["", "NA", "null", "None", "NaN"]
    )
    
    # Fill NaN values with empty string
    df = df.fillna("")
    
    # Strip whitespace from string columns
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        
    if expected_columns:
        missing = set(expected_columns) - set(df.columns)
        if missing:
            raise ValueError(f"File {filepath} is missing required columns: {missing}")
            
    return df

def load_train_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all training data: train_source1, train_source2, train_source3, train_ground_truth.
    """
    s1_path = os.path.join(data_dir, "train_source1.tsv")
    s2_path = os.path.join(data_dir, "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train_source3.tsv")
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")
    
    df_s1 = load_tsv(s1_path, REQUIRED_SOURCE_COLUMNS)
    df_s2 = load_tsv(s2_path, REQUIRED_SOURCE_COLUMNS)
    df_s3 = load_tsv(s3_path, REQUIRED_SOURCE_COLUMNS)
    df_gt = load_tsv(gt_path, REQUIRED_GT_COLUMNS)
    
    # Check ID uniqueness
    assert len(df_s1) == df_s1["entity_id"].nunique(), "Duplicate entity_ids detected in Source 1"
    assert len(df_s2) == df_s2["entity_id"].nunique(), "Duplicate entity_ids detected in Source 2"
    assert len(df_s3) == df_s3["entity_id"].nunique(), "Duplicate entity_ids detected in Source 3"
    
    return df_s1, df_s2, df_s3, df_gt

def load_test_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all test data: test_source1, test_source2, test_source3.
    """
    s1_path = os.path.join(data_dir, "test_source1.tsv")
    s2_path = os.path.join(data_dir, "test_source2.tsv")
    s3_path = os.path.join(data_dir, "test_source3.tsv")
    
    df_s1 = load_tsv(s1_path, REQUIRED_SOURCE_COLUMNS)
    df_s2 = load_tsv(s2_path, REQUIRED_SOURCE_COLUMNS)
    df_s3 = load_tsv(s3_path, REQUIRED_SOURCE_COLUMNS)
    
    assert len(df_s1) == df_s1["entity_id"].nunique(), "Duplicate entity_ids detected in Test Source 1"
    
    return df_s1, df_s2, df_s3

def parse_ground_truth(df_gt: pd.DataFrame) -> Dict[str, list]:
    """
    Parses ground truth into a dictionary mapping source1_entity_id -> list of matched target entity_ids.
    """
    gt_dict = {}
    for _, row in df_gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = str(row["matched_entity_ids"]).strip()
        if not matched_str or matched_str.lower() in ["nan", "none"]:
            gt_dict[s1_id] = []
        else:
            ids = [x.strip() for x in matched_str.split(",") if x.strip()]
            gt_dict[s1_id] = list(dict.fromkeys(ids))  # preserve order & unique
    return gt_dict
