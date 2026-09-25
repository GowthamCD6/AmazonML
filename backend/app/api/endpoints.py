"""
API Endpoints Router for Business Entity Resolution
"""

import os
import time
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, status

from backend.app.core.config import settings
from backend.app.api.schemas import (
    BusinessEntity, SingleResolveResponse, BatchResolveRequest, BatchResolveResponse,
    PairComparisonRequest, PairComparisonResponse, DatasetStatsResponse, ValidationCheckResponse
)
from backend.app.services.resolver import resolver_service
from backend.app.services.report_service import report_service
from src.profiling import profile_dataset
from utils.validate_submission import validate

router = APIRouter()

@router.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint indicating model readiness and knowledge base size."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "model_loaded": resolver_service.is_loaded,
        "target_records_count": len(resolver_service.df_targets),
        "default_threshold": resolver_service.best_threshold,
        "features_count": len(resolver_service.feature_cols)
    }

@router.post("/resolve", response_model=SingleResolveResponse, tags=["Entity Resolution"])
def resolve_entity(query: BusinessEntity, threshold: Optional[float] = Query(None, ge=0.0, le=1.0)):
    """
    Resolve a single business entity against the loaded Source 2 / Source 3 database.
    Performs real-time candidate blocking, 41 pairwise feature extractions, model scoring,
    and thresholding.
    """
    if not resolver_service.is_loaded:
        resolver_service.initialize()
    return resolver_service.resolve_single(query, threshold=threshold)

@router.post("/resolve/batch", response_model=BatchResolveResponse, tags=["Entity Resolution"])
def resolve_batch_entities(request: BatchResolveRequest):
    """Resolve a batch of Source 1 entities in real-time."""
    start_time = time.time()
    if not resolver_service.is_loaded:
        resolver_service.initialize()
        
    th = request.threshold if request.threshold is not None else resolver_service.best_threshold
    results = []
    total_matches = 0
    singletons = 0
    
    for entity in request.entities:
        res = resolver_service.resolve_single(entity, threshold=th)
        results.append(res)
        total_matches += res.matches_count
        if res.is_singleton:
            singletons += 1
            
    elapsed = (time.time() - start_time) * 1000
    return BatchResolveResponse(
        total_entities=len(request.entities),
        total_matches=total_matches,
        total_singletons=singletons,
        threshold_applied=th,
        results=results,
        latency_ms=round(elapsed, 2)
    )

@router.post("/match/pair", response_model=PairComparisonResponse, tags=["Inspection"])
def compare_business_pair(request: PairComparisonRequest):
    """
    Deep pairwise comparison between two business entity records.
    Computes all 41 similarity metrics, cross-field features, and model probability.
    """
    if not resolver_service.is_loaded:
        resolver_service.initialize()
    return resolver_service.compare_pair(
        request.entity_1, request.entity_2, threshold=request.threshold or 0.30
    )

@router.get("/experiments", tags=["Experiments"])
def get_experiments_history():
    """Retrieve complete experiment log history."""
    return report_service.get_experiments()

@router.get("/leaderboard", tags=["Experiments"])
def get_leaderboard_logs():
    """Retrieve submission tracking and leaderboard entries."""
    return report_service.get_leaderboard()

@router.get("/reports/{report_name}", tags=["Reports"])
def get_markdown_report(report_name: str):
    """
    Fetch markdown report content (data_profile, validation_report, error_analysis, methodology, readme).
    """
    content = report_service.get_markdown_report(report_name)
    if content is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report '{report_name}' not found. Available: [data_profile, validation_report, error_analysis, methodology, readme]"
        )
    return {"report_name": report_name, "content": content}

@router.get("/stats/dataset", response_model=DatasetStatsResponse, tags=["Dataset"])
def get_dataset_statistics():
    """Compute and retrieve measured dataset dimensions and distributions."""
    train_dir = os.path.join(settings.DATA_DIR, "train")
    test_dir = os.path.join(settings.DATA_DIR, "test")
    stats = profile_dataset(train_dir, test_dir)
    return DatasetStatsResponse(**stats)

@router.post("/pipeline/validate", response_model=ValidationCheckResponse, tags=["Validation"])
def run_submission_validation():
    """Run the official submission validator on output files."""
    matching_file = os.path.join(settings.OUTPUT_DIR, "matching_results.tsv")
    candidate_file = os.path.join(settings.OUTPUT_DIR, "candidate_pairs.tsv")
    test_dir = os.path.join(settings.DATA_DIR, "test")
    
    is_valid = validate(matching_file, candidate_file, test_dir)
    
    import pandas as pd
    df_match = pd.read_csv(matching_file, sep="\t", dtype=str).fillna("")
    df_cand = pd.read_csv(candidate_file, sep="\t", dtype=str).fillna("")
    
    singletons = sum(1 for m in df_match["matched_entity_ids"] if not str(m).strip())
    total_matches = sum(len(str(m).split(",")) for m in df_match["matched_entity_ids"] if str(m).strip())
    
    return ValidationCheckResponse(
        is_valid=is_valid,
        matching_file=matching_file,
        candidate_file=candidate_file,
        total_s1_entities=len(df_match),
        total_matches=total_matches,
        singletons=singletons,
        total_candidate_pairs=len(df_cand),
        errors=[]
    )
