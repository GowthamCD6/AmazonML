"""
Multi-strategy candidate generation and blocking module for Amazon ML Challenge 2026.
Combines exact name, name+country, address tokens, character TF-IDF cosine similarity,
and rare-token inverted indexing using UNION blocking.
"""

import math
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def build_inverted_index(df: pd.DataFrame, key_func, id_col: str = "entity_id") -> Dict[Any, List[str]]:
    """Build an inverted index mapping from generated keys to list of entity IDs."""
    index = defaultdict(list)
    for _, row in df.iterrows():
        eid = row[id_col]
        keys = key_func(row)
        if isinstance(keys, (list, set, tuple)):
            for k in keys:
                if k:
                    index[k].append(eid)
        elif keys:
            index[keys].append(eid)
    return index

class BlockingEngine:
    def __init__(
        self,
        tfidf_top_k: int = 10,
        tfidf_min_sim: float = 0.30,
        ngram_range: Tuple[int, int] = (2, 4),
        rare_min_freq: int = 2,
        rare_max_freq: int = 30
    ):
        self.tfidf_top_k = tfidf_top_k
        self.tfidf_min_sim = tfidf_min_sim
        self.ngram_range = ngram_range
        self.rare_min_freq = rare_min_freq
        self.rare_max_freq = rare_max_freq
        
    def generate_candidates(
        self,
        df_s1: pd.DataFrame,
        df_targets: pd.DataFrame, # combined S2 and S3
        strategies: List[str] = None
    ) -> pd.DataFrame:
        """
        Generate candidate pairs between df_s1 and df_targets using the union of strategies.
        Returns a DataFrame of candidate pairs with blocking flags.
        """
        if strategies is None:
            strategies = ["exact_name", "name_country", "address_token", "tfidf_ngram", "rare_token"]
            
        candidate_map = defaultdict(lambda: {
            "blocked_by_exact_name": 0,
            "blocked_by_name_country": 0,
            "blocked_by_address": 0,
            "blocked_by_tfidf": 0,
            "blocked_by_rare_token": 0,
            "tfidf_similarity": 0.0
        })
        
        # -------------------------------------------------------------
        # Strategy 1: Exact Normalized Name Blocking
        # -------------------------------------------------------------
        if "exact_name" in strategies:
            target_name_idx = build_inverted_index(df_targets, lambda r: r["business_name_norm"])
            for _, row in df_s1.iterrows():
                s1_id = row["entity_id"]
                name_key = row["business_name_norm"]
                if name_key and name_key in target_name_idx:
                    for t_id in target_name_idx[name_key]:
                        candidate_map[(s1_id, t_id)]["blocked_by_exact_name"] = 1

        # -------------------------------------------------------------
        # Strategy 2: Name Core + Country Blocking
        # -------------------------------------------------------------
        if "name_country" in strategies:
            def name_country_key(r):
                c = r["country_norm"]
                n = r["business_name_core"]
                return f"{n}___{c}" if n and c else None
                
            target_nc_idx = build_inverted_index(df_targets, name_country_key)
            for _, row in df_s1.iterrows():
                s1_id = row["entity_id"]
                k = name_country_key(row)
                if k and k in target_nc_idx:
                    for t_id in target_nc_idx[k]:
                        candidate_map[(s1_id, t_id)]["blocked_by_name_country"] = 1

        # -------------------------------------------------------------
        # Strategy 3: Address Distinctive Token Blocking (Numbers + distinctive street)
        # -------------------------------------------------------------
        if "address_token" in strategies:
            common_addr_stops = {"street", "road", "avenue", "boulevard", "suite", "floor", "apartment", "lane", "drive", "main", "city"}
            def addr_tokens_key(r):
                tokens = r.get("business_address_tokens", [])
                informative = [t for t in tokens if t not in common_addr_stops and (len(t) > 3 or t.isdigit())]
                return informative[:4] # top 4 informative
                
            target_addr_idx = build_inverted_index(df_targets, addr_tokens_key)
            for _, row in df_s1.iterrows():
                s1_id = row["entity_id"]
                s1_tokens = addr_tokens_key(row)
                for tok in s1_tokens:
                    if tok in target_addr_idx:
                        # Avoid huge blocks
                        matches = target_addr_idx[tok]
                        if len(matches) <= 50:
                            for t_id in matches:
                                candidate_map[(s1_id, t_id)]["blocked_by_address"] = 1

        # -------------------------------------------------------------
        # Strategy 4: Character N-gram TF-IDF Cosine Similarity
        # -------------------------------------------------------------
        if "tfidf_ngram" in strategies:
            s1_names = df_s1["business_name_norm"].fillna("").tolist()
            target_names = df_targets["business_name_norm"].fillna("").tolist()
            
            s1_ids = df_s1["entity_id"].tolist()
            target_ids = df_targets["entity_id"].tolist()
            
            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=self.ngram_range,
                min_df=1,
                sublinear_tf=True
            )
            
            # Fit on all names
            all_names = s1_names + target_names
            vectorizer.fit(all_names)
            
            s1_matrix = vectorizer.transform(s1_names)
            target_matrix = vectorizer.transform(target_names)
            
            # Compute batch-wise cosine similarity (sparse dot product)
            # Process in chunks of 500 S1 queries to conserve memory
            batch_size = 500
            for i in range(0, s1_matrix.shape[0], batch_size):
                end_i = min(i + batch_size, s1_matrix.shape[0])
                batch_sim = s1_matrix[i:end_i].dot(target_matrix.T).toarray()
                
                for local_idx in range(batch_sim.shape[0]):
                    global_s1_idx = i + local_idx
                    s1_id = s1_ids[global_s1_idx]
                    sim_scores = batch_sim[local_idx]
                    
                    # Top-K indices
                    if len(sim_scores) > self.tfidf_top_k:
                        top_indices = np.argpartition(sim_scores, -self.tfidf_top_k)[-self.tfidf_top_k:]
                        top_indices = top_indices[np.argsort(-sim_scores[top_indices])]
                    else:
                        top_indices = np.argsort(-sim_scores)
                        
                    for t_idx in top_indices:
                        sim = float(sim_scores[t_idx])
                        if sim >= self.tfidf_min_sim:
                            t_id = target_ids[t_idx]
                            candidate_map[(s1_id, t_id)]["blocked_by_tfidf"] = 1
                            candidate_map[(s1_id, t_id)]["tfidf_similarity"] = max(
                                candidate_map[(s1_id, t_id)]["tfidf_similarity"], sim
                            )

        # -------------------------------------------------------------
        # Strategy 5: Rare-Token Inverted Index Blocking
        # -------------------------------------------------------------
        if "rare_token" in strategies:
            # Count token frequencies across both sets
            token_counts = Counter()
            for tokens in df_s1["business_name_tokens"]:
                token_counts.update(tokens)
            for tokens in df_targets["business_name_tokens"]:
                token_counts.update(tokens)
                
            rare_tokens = {
                t for t, count in token_counts.items()
                if self.rare_min_freq <= count <= self.rare_max_freq and len(t) >= 3
            }
            
            def rare_token_key(r):
                return [t for t in r.get("business_name_tokens", []) if t in rare_tokens]
                
            target_rare_idx = build_inverted_index(df_targets, rare_token_key)
            for _, row in df_s1.iterrows():
                s1_id = row["entity_id"]
                s1_rare = rare_token_key(row)
                for tok in s1_rare:
                    if tok in target_rare_idx:
                        matches = target_rare_idx[tok]
                        if len(matches) <= 30:
                            for t_id in matches:
                                candidate_map[(s1_id, t_id)]["blocked_by_rare_token"] = 1

        # Format output dataframe
        records = []
        for (s1_id, t_id), meta in candidate_map.items():
            records.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": t_id,
                "blocked_by_exact_name": meta["blocked_by_exact_name"],
                "blocked_by_name_country": meta["blocked_by_name_country"],
                "blocked_by_address": meta["blocked_by_address"],
                "blocked_by_tfidf": meta["blocked_by_tfidf"],
                "blocked_by_rare_token": meta["blocked_by_rare_token"],
                "tfidf_similarity": meta["tfidf_similarity"],
                "num_blocks_hit": (
                    meta["blocked_by_exact_name"] +
                    meta["blocked_by_name_country"] +
                    meta["blocked_by_address"] +
                    meta["blocked_by_tfidf"] +
                    meta["blocked_by_rare_token"]
                )
            })
            
        df_candidates = pd.DataFrame(records)
        if len(df_candidates) == 0:
            df_candidates = pd.DataFrame(columns=[
                "source1_entity_id", "candidate_entity_id", "blocked_by_exact_name",
                "blocked_by_name_country", "blocked_by_address", "blocked_by_tfidf",
                "blocked_by_rare_token", "tfidf_similarity", "num_blocks_hit"
            ])
            
        return df_candidates

def evaluate_blocking_recall(
    df_candidates: pd.DataFrame,
    gt_map: Dict[str, list],
    s1_subset_ids: Set[str] = None
) -> Dict[str, Any]:
    """
    Evaluate candidate generation recall against ground truth matches.
    """
    cand_pairs = set(zip(df_candidates["source1_entity_id"], df_candidates["candidate_entity_id"]))
    
    total_positives = 0
    recalled_positives = 0
    missed_pairs = []
    
    for s1_id, targets in gt_map.items():
        if s1_subset_ids is not None and s1_id not in s1_subset_ids:
            continue
        for t_id in targets:
            total_positives += 1
            if (s1_id, t_id) in cand_pairs:
                recalled_positives += 1
            else:
                missed_pairs.append((s1_id, t_id))
                
    recall = recalled_positives / max(1, total_positives)
    
    return {
        "candidate_pairs_count": len(df_candidates),
        "total_positive_pairs": total_positives,
        "recalled_positive_pairs": recalled_positives,
        "candidate_recall": recall,
        "missed_pairs_count": len(missed_pairs),
        "missed_pairs_sample": missed_pairs[:10]
    }
