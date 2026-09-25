"""
Pairwise feature engineering module for Amazon ML Challenge 2026.
Extracts name, address, country, cross-field, and blocking indicator features for candidate pairs.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple

try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import Levenshtein
except ImportError:
    # Basic pure python fallback if rapidfuzz is not yet loaded
    class fuzz:
        @staticmethod
        def ratio(s1, s2):
            return 100.0 if s1 == s2 else 0.0
        @staticmethod
        def partial_ratio(s1, s2):
            return 100.0 if s1 in s2 or s2 in s1 else 0.0
        @staticmethod
        def token_sort_ratio(s1, s2):
            return 100.0 if sorted(s1.split()) == sorted(s2.split()) else 0.0
        @staticmethod
        def token_set_ratio(s1, s2):
            return 100.0 if set(s1.split()) == set(s2.split()) else 0.0

RE_NUMBERS = re.compile(r"\b\d+\b")

def get_char_ngrams(text: str, n: int = 3) -> set:
    """Extract character n-grams."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def overlap_coefficient(set1: set, set2: set) -> float:
    """Compute Overlap / Szymkiewicz-Simpson coefficient: |A ∩ B| / min(|A|, |B|)."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    min_len = min(len(set1), len(set2))
    return intersection / min_len if min_len > 0 else 0.0

def extract_pairwise_features(
    df_candidates: pd.DataFrame,
    df_s1: pd.DataFrame,
    df_targets: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute dense numerical feature matrix for all candidate pairs in df_candidates.
    """
    # Create fast entity lookups
    s1_dict = df_s1.set_index("entity_id").to_dict(orient="index")
    target_dict = df_targets.set_index("entity_id").to_dict(orient="index")
    
    rows = []
    
    for _, row in df_candidates.iterrows():
        s1_id = row["source1_entity_id"]
        t_id = row["candidate_entity_id"]
        
        s1_rec = s1_dict.get(s1_id, {})
        t_rec = target_dict.get(t_id, {})
        
        # Raw & normalized strings
        s1_name = s1_rec.get("business_name", "")
        t_name = t_rec.get("business_name", "")
        s1_name_norm = s1_rec.get("business_name_norm", "")
        t_name_norm = t_rec.get("business_name_norm", "")
        s1_name_core = s1_rec.get("business_name_core", "")
        t_name_core = t_rec.get("business_name_core", "")
        
        s1_name_toks = set(s1_rec.get("business_name_tokens", []))
        t_name_toks = set(t_rec.get("business_name_tokens", []))
        
        s1_addr = s1_rec.get("business_address", "")
        t_addr = t_rec.get("business_address", "")
        s1_addr_norm = s1_rec.get("business_address_norm", "")
        t_addr_norm = t_rec.get("business_address_norm", "")
        s1_addr_toks = set(s1_rec.get("business_address_tokens", []))
        t_addr_toks = set(t_rec.get("business_address_tokens", []))
        
        s1_c = s1_rec.get("country_norm", "")
        t_c = t_rec.get("country_norm", "")
        
        # -------------------------------------------------------------
        # 1. Business Name Features
        # -------------------------------------------------------------
        f_name_exact = 1.0 if (s1_name_norm and s1_name_norm == t_name_norm) else 0.0
        f_name_core_exact = 1.0 if (s1_name_core and s1_name_core == t_name_core) else 0.0
        
        f_name_fuzz_ratio = fuzz.ratio(s1_name_norm, t_name_norm) / 100.0
        f_name_partial_ratio = fuzz.partial_ratio(s1_name_norm, t_name_norm) / 100.0
        f_name_token_sort_ratio = fuzz.token_sort_ratio(s1_name_norm, t_name_norm) / 100.0
        f_name_token_set_ratio = fuzz.token_set_ratio(s1_name_norm, t_name_norm) / 100.0
        
        f_name_jaccard = jaccard_similarity(s1_name_toks, t_name_toks)
        f_name_overlap = overlap_coefficient(s1_name_toks, t_name_toks)
        
        len_s1_n = len(s1_name_norm)
        len_t_n = len(t_name_norm)
        f_name_len_diff = abs(len_s1_n - len_t_n)
        f_name_len_ratio = (min(len_s1_n, len_t_n) / max(1, max(len_s1_n, len_t_n)))
        f_name_token_diff = abs(len(s1_name_toks) - len(t_name_toks))
        
        # 3-gram char similarity
        s1_ngrams = get_char_ngrams(s1_name_norm, 3)
        t_ngrams = get_char_ngrams(t_name_norm, 3)
        f_name_ngram_jaccard = jaccard_similarity(s1_ngrams, t_ngrams)
        
        f_name_prefix_3 = 1.0 if (len_s1_n >= 3 and len_t_n >= 3 and s1_name_norm[:3] == t_name_norm[:3]) else 0.0
        f_name_prefix_5 = 1.0 if (len_s1_n >= 5 and len_t_n >= 5 and s1_name_norm[:5] == t_name_norm[:5]) else 0.0

        # -------------------------------------------------------------
        # 2. Address Features
        # -------------------------------------------------------------
        f_addr_exact = 1.0 if (s1_addr_norm and s1_addr_norm == t_addr_norm) else 0.0
        f_addr_fuzz_ratio = fuzz.ratio(s1_addr_norm, t_addr_norm) / 100.0
        f_addr_partial_ratio = fuzz.partial_ratio(s1_addr_norm, t_addr_norm) / 100.0
        f_addr_token_sort_ratio = fuzz.token_sort_ratio(s1_addr_norm, t_addr_norm) / 100.0
        f_addr_token_set_ratio = fuzz.token_set_ratio(s1_addr_norm, t_addr_norm) / 100.0
        
        f_addr_jaccard = jaccard_similarity(s1_addr_toks, t_addr_toks)
        f_addr_overlap = overlap_coefficient(s1_addr_toks, t_addr_toks)
        
        len_s1_a = len(s1_addr_norm)
        len_t_a = len(t_addr_norm)
        f_addr_len_diff = abs(len_s1_a - len_t_a)
        f_addr_len_ratio = (min(len_s1_a, len_t_a) / max(1, max(len_s1_a, len_t_a)))
        
        # Numbers / postal codes match in address
        s1_nums = set(RE_NUMBERS.findall(s1_addr))
        t_nums = set(RE_NUMBERS.findall(t_addr))
        f_addr_num_overlap = len(s1_nums.intersection(t_nums))
        f_addr_num_jaccard = jaccard_similarity(s1_nums, t_nums)
        
        # Leading street number match
        s1_lead_num = s1_nums.pop() if s1_nums else ""
        t_lead_num = t_nums.pop() if t_nums else ""
        f_addr_lead_num_match = 1.0 if (s1_lead_num and s1_lead_num == t_lead_num) else 0.0

        # -------------------------------------------------------------
        # 3. Country Features
        # -------------------------------------------------------------
        f_country_match = 1.0 if (s1_c and t_c and s1_c == t_c) else 0.0
        f_country_missing = 1.0 if (not s1_c or not t_c) else 0.0
        f_country_conflict = 1.0 if (s1_c and t_c and s1_c != t_c) else 0.0

        # -------------------------------------------------------------
        # 4. Cross-Field & Interaction Features
        # -------------------------------------------------------------
        f_name_addr_mult = f_name_token_set_ratio * f_addr_token_set_ratio
        f_name_addr_mean = (f_name_token_set_ratio + f_addr_token_set_ratio) / 2.0
        
        # Harmonic mean (penalizes low address similarity when name is high)
        if (f_name_token_set_ratio + f_addr_token_set_ratio) > 0:
            f_name_addr_harmonic = 2 * (f_name_token_set_ratio * f_addr_token_set_ratio) / (f_name_token_set_ratio + f_addr_token_set_ratio)
        else:
            f_name_addr_harmonic = 0.0
            
        f_name_high_addr_low = 1.0 if (f_name_token_set_ratio > 0.85 and f_addr_token_set_ratio < 0.35) else 0.0
        f_addr_high_name_low = 1.0 if (f_addr_token_set_ratio > 0.85 and f_name_token_set_ratio < 0.35) else 0.0

        # -------------------------------------------------------------
        # 5. Blocking Signals
        # -------------------------------------------------------------
        f_block_exact_name = float(row.get("blocked_by_exact_name", 0))
        f_block_nc = float(row.get("blocked_by_name_country", 0))
        f_block_addr = float(row.get("blocked_by_address", 0))
        f_block_tfidf = float(row.get("blocked_by_tfidf", 0))
        f_block_rare = float(row.get("blocked_by_rare_token", 0))
        f_tfidf_sim = float(row.get("tfidf_similarity", 0.0))
        f_num_blocks = float(row.get("num_blocks_hit", 1))

        # Compile feature dict
        feats = {
            "source1_entity_id": s1_id,
            "candidate_entity_id": t_id,
            "f_name_exact": f_name_exact,
            "f_name_core_exact": f_name_core_exact,
            "f_name_fuzz_ratio": f_name_fuzz_ratio,
            "f_name_partial_ratio": f_name_partial_ratio,
            "f_name_token_sort_ratio": f_name_token_sort_ratio,
            "f_name_token_set_ratio": f_name_token_set_ratio,
            "f_name_jaccard": f_name_jaccard,
            "f_name_overlap": f_name_overlap,
            "f_name_len_diff": f_name_len_diff,
            "f_name_len_ratio": f_name_len_ratio,
            "f_name_token_diff": f_name_token_diff,
            "f_name_ngram_jaccard": f_name_ngram_jaccard,
            "f_name_prefix_3": f_name_prefix_3,
            "f_name_prefix_5": f_name_prefix_5,
            "f_addr_exact": f_addr_exact,
            "f_addr_fuzz_ratio": f_addr_fuzz_ratio,
            "f_addr_partial_ratio": f_addr_partial_ratio,
            "f_addr_token_sort_ratio": f_addr_token_sort_ratio,
            "f_addr_token_set_ratio": f_addr_token_set_ratio,
            "f_addr_jaccard": f_addr_jaccard,
            "f_addr_overlap": f_addr_overlap,
            "f_addr_len_diff": f_addr_len_diff,
            "f_addr_len_ratio": f_addr_len_ratio,
            "f_addr_num_overlap": f_addr_num_overlap,
            "f_addr_num_jaccard": f_addr_num_jaccard,
            "f_addr_lead_num_match": f_addr_lead_num_match,
            "f_country_match": f_country_match,
            "f_country_missing": f_country_missing,
            "f_country_conflict": f_country_conflict,
            "f_name_addr_mult": f_name_addr_mult,
            "f_name_addr_mean": f_name_addr_mean,
            "f_name_addr_harmonic": f_name_addr_harmonic,
            "f_name_high_addr_low": f_name_high_addr_low,
            "f_addr_high_name_low": f_addr_high_name_low,
            "f_block_exact_name": f_block_exact_name,
            "f_block_nc": f_block_nc,
            "f_block_addr": f_block_addr,
            "f_block_tfidf": f_block_tfidf,
            "f_block_rare": f_block_rare,
            "f_tfidf_sim": f_tfidf_sim,
            "f_num_blocks": f_num_blocks
        }
        rows.append(feats)
        
    df_features = pd.DataFrame(rows)
    return df_features

FEATURE_COLUMNS = [
    "f_name_exact", "f_name_core_exact", "f_name_fuzz_ratio", "f_name_partial_ratio",
    "f_name_token_sort_ratio", "f_name_token_set_ratio", "f_name_jaccard", "f_name_overlap",
    "f_name_len_diff", "f_name_len_ratio", "f_name_token_diff", "f_name_ngram_jaccard",
    "f_name_prefix_3", "f_name_prefix_5", "f_addr_exact", "f_addr_fuzz_ratio",
    "f_addr_partial_ratio", "f_addr_token_sort_ratio", "f_addr_token_set_ratio",
    "f_addr_jaccard", "f_addr_overlap", "f_addr_len_diff", "f_addr_len_ratio",
    "f_addr_num_overlap", "f_addr_num_jaccard", "f_addr_lead_num_match",
    "f_country_match", "f_country_missing", "f_country_conflict", "f_name_addr_mult",
    "f_name_addr_mean", "f_name_addr_harmonic", "f_name_high_addr_low",
    "f_addr_high_name_low", "f_block_exact_name", "f_block_nc", "f_block_addr",
    "f_block_tfidf", "f_block_rare", "f_tfidf_sim", "f_num_blocks"
]
