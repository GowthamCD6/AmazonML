"""
Entity Resolver Service
Loads trained ML models and runs real-time candidate blocking, feature extraction, and classification.
"""

import os
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from backend.app.core.config import settings
from backend.app.api.schemas import BusinessEntity, EntityMatchResult, SingleResolveResponse, PairComparisonResponse
from src.normalization import normalize_dataframe
from src.blocking import BlockingEngine
from src.features import extract_pairwise_features, FEATURE_COLUMNS
from src.train import load_model_artifact
from src.data_loader import load_tsv

class EntityResolverService:
    def __init__(self):
        self.model = None
        self.feature_cols: List[str] = FEATURE_COLUMNS
        self.best_threshold: float = settings.DEFAULT_THRESHOLD
        self.metadata: Dict[str, Any] = {}
        self.df_targets: pd.DataFrame = pd.DataFrame()
        self.blocking_engine = BlockingEngine()
        self.is_loaded = False

    def initialize(self, model_version: str = "final"):
        """Load model artifact and target knowledge base into memory."""
        print(f"Initializing EntityResolverService (Model Version: {model_version})...")
        model_dir = os.path.join(settings.MODELS_DIR, model_version)
        if not os.path.exists(model_dir):
            model_dir = os.path.join(settings.MODELS_DIR, "v5")
            
        if os.path.exists(model_dir):
            artifact = load_model_artifact(model_dir)
            self.model = artifact["model"]
            self.feature_cols = artifact["feature_cols"]
            self.best_threshold = float(artifact.get("best_threshold", settings.DEFAULT_THRESHOLD))
            self.metadata = artifact.get("metadata", {})
            print(f"Loaded model artifact from {model_dir} (Threshold: {self.best_threshold:.2f})")
        else:
            print(f"Warning: Model directory {model_dir} not found. Run training pipeline first.")

        # Load target database (test S2 + S3 + train S2 + S3)
        targets = []
        for path, src_name in [
            (os.path.join(settings.DATA_DIR, "test", "test_source2.tsv"), "Source 2"),
            (os.path.join(settings.DATA_DIR, "test", "test_source3.tsv"), "Source 3"),
            (os.path.join(settings.DATA_DIR, "train", "train_source2.tsv"), "Source 2"),
            (os.path.join(settings.DATA_DIR, "train", "train_source3.tsv"), "Source 3"),
        ]:
            if os.path.exists(path):
                df = load_tsv(path)
                df["source"] = src_name
                targets.append(df)
                
        if targets:
            df_combined = pd.concat(targets, ignore_index=True).drop_duplicates(subset=["entity_id"])
            self.df_targets = normalize_dataframe(df_combined)
            print(f"Target business database loaded: {len(self.df_targets)} records.")
        else:
            print("Warning: No target database files found.")

        self.is_loaded = True

    def resolve_single(
        self,
        query: BusinessEntity,
        threshold: Optional[float] = None
    ) -> SingleResolveResponse:
        """Resolve a single business entity against target database."""
        start_time = time.time()
        th = threshold if threshold is not None else self.best_threshold
        query_id = query.entity_id or "QUERY_001"
        
        # Prepare query DataFrame
        df_query = pd.DataFrame([{
            "entity_id": query_id,
            "business_name": query.business_name,
            "business_address": query.business_address,
            "country": query.country or ""
        }])
        df_query_norm = normalize_dataframe(df_query)
        
        # Blocking
        df_cands = self.blocking_engine.generate_candidates(df_query_norm, self.df_targets)
        
        if len(df_cands) == 0:
            elapsed = (time.time() - start_time) * 1000
            return SingleResolveResponse(
                query_entity_id=query_id,
                query_name=query.business_name,
                query_address=query.business_address,
                query_country=query.country or "",
                threshold_applied=th,
                is_singleton=True,
                matches_count=0,
                candidates_count=0,
                matched_ids=[],
                matches=[],
                latency_ms=round(elapsed, 2)
            )

        # Feature Extraction
        df_features = extract_pairwise_features(df_cands, df_query_norm, self.df_targets)
        X = df_features[self.feature_cols].values
        
        # Model Scoring
        if self.model is not None and hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[:, 1]
        elif self.model is not None and hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)
            probs = 1.0 / (1.0 + np.exp(-scores))
        else:
            probs = np.full(len(X), 0.5)

        df_cands["match_probability"] = probs
        
        # Fast lookup on targets
        target_dict = self.df_targets.set_index("entity_id").to_dict(orient="index")
        
        match_results = []
        matched_ids = []
        
        # Sort candidates by probability descending
        df_cands_sorted = df_cands.sort_values(by="match_probability", ascending=False)
        
        for _, row in df_cands_sorted.iterrows():
            c_id = row["candidate_entity_id"]
            prob = float(row["match_probability"])
            is_m = prob >= th
            
            t_rec = target_dict.get(c_id, {})
            feat_row = df_features[df_features["candidate_entity_id"] == c_id].iloc[0] if len(df_features[df_features["candidate_entity_id"] == c_id]) > 0 else {}
            
            sim_summary = {
                "name_fuzzy_ratio": float(feat_row.get("f_name_fuzz_ratio", 0.0)),
                "name_token_sort_ratio": float(feat_row.get("f_name_token_sort_ratio", 0.0)),
                "name_token_set_ratio": float(feat_row.get("f_name_token_set_ratio", 0.0)),
                "addr_fuzzy_ratio": float(feat_row.get("f_addr_fuzz_ratio", 0.0)),
                "addr_token_set_ratio": float(feat_row.get("f_addr_token_set_ratio", 0.0)),
                "addr_jaccard": float(feat_row.get("f_addr_jaccard", 0.0)),
                "country_match": float(feat_row.get("f_country_match", 0.0)),
                "harmonic_similarity": float(feat_row.get("f_name_addr_harmonic", 0.0))
            }
            
            blocking_sigs = {
                "exact_name": int(row.get("blocked_by_exact_name", 0)),
                "name_country": int(row.get("blocked_by_name_country", 0)),
                "address_token": int(row.get("blocked_by_address", 0)),
                "tfidf_ngram": int(row.get("blocked_by_tfidf", 0)),
                "rare_token": int(row.get("blocked_by_rare_token", 0))
            }
            
            if is_m:
                matched_ids.append(c_id)
                
            match_results.append(EntityMatchResult(
                candidate_entity_id=c_id,
                business_name=t_rec.get("business_name", ""),
                business_address=t_rec.get("business_address", ""),
                country=t_rec.get("country", ""),
                source=t_rec.get("source", "Source 2/3"),
                match_probability=round(prob, 4),
                is_match=is_m,
                num_blocks_hit=int(row.get("num_blocks_hit", 1)),
                blocking_signals=blocking_sigs,
                similarity_summary=sim_summary
            ))

        elapsed = (time.time() - start_time) * 1000
        
        return SingleResolveResponse(
            query_entity_id=query_id,
            query_name=query.business_name,
            query_address=query.business_address,
            query_country=query.country or "",
            threshold_applied=th,
            is_singleton=len(matched_ids) == 0,
            matches_count=len(matched_ids),
            candidates_count=len(df_cands),
            matched_ids=matched_ids,
            matches=match_results,
            latency_ms=round(elapsed, 2)
        )

    def compare_pair(
        self,
        e1: BusinessEntity,
        e2: BusinessEntity,
        threshold: float = 0.30
    ) -> PairComparisonResponse:
        """Deep comparison of any two business entities."""
        df_1 = pd.DataFrame([{
            "entity_id": e1.entity_id or "E1",
            "business_name": e1.business_name,
            "business_address": e1.business_address,
            "country": e1.country or ""
        }])
        df_2 = pd.DataFrame([{
            "entity_id": e2.entity_id or "E2",
            "business_name": e2.business_name,
            "business_address": e2.business_address,
            "country": e2.country or ""
        }])
        
        df_1_norm = normalize_dataframe(df_1)
        df_2_norm = normalize_dataframe(df_2)
        
        df_cand = pd.DataFrame([{
            "source1_entity_id": "E1",
            "candidate_entity_id": "E2",
            "blocked_by_exact_name": 1,
            "blocked_by_name_country": 1,
            "blocked_by_address": 1,
            "blocked_by_tfidf": 1,
            "blocked_by_rare_token": 1,
            "tfidf_similarity": 1.0,
            "num_blocks_hit": 5
        }])
        
        df_feats = extract_pairwise_features(df_cand, df_1_norm, df_2_norm)
        X = df_feats[self.feature_cols].values
        
        if self.model is not None and hasattr(self.model, "predict_proba"):
            prob = float(self.model.predict_proba(X)[0, 1])
        elif self.model is not None and hasattr(self.model, "decision_function"):
            s = float(self.model.decision_function(X)[0])
            prob = 1.0 / (1.0 + np.exp(-s))
        else:
            prob = 0.5
            
        features_dict = {col: round(float(df_feats[col].values[0]), 4) for col in self.feature_cols if col in df_feats.columns}
        
        return PairComparisonResponse(
            entity_1_name=e1.business_name,
            entity_2_name=e2.business_name,
            entity_1_address=e1.business_address,
            entity_2_address=e2.business_address,
            entity_1_country=e1.country or "",
            entity_2_country=e2.country or "",
            match_probability=round(prob, 4),
            predicted_same_business=prob >= threshold,
            decision_threshold=threshold,
            features=features_dict
        )

resolver_service = EntityResolverService()
