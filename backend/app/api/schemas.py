"""
Pydantic schemas for API request and response data models.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class BusinessEntity(BaseModel):
    entity_id: Optional[str] = Field(None, example="S1_TEST_000001")
    business_name: str = Field(..., example="Acme Global Technologies Inc.")
    business_address: str = Field(..., example="123 Main St, Suite 400, New York, 10001, US")
    country: Optional[str] = Field(default="", example="US")

class EntityMatchResult(BaseModel):
    candidate_entity_id: str
    business_name: str
    business_address: str
    country: str
    source: str
    match_probability: float
    is_match: bool
    num_blocks_hit: int
    blocking_signals: Dict[str, int]
    similarity_summary: Dict[str, float]

class SingleResolveResponse(BaseModel):
    query_entity_id: str
    query_name: str
    query_address: str
    query_country: str
    threshold_applied: float
    is_singleton: bool
    matches_count: int
    candidates_count: int
    matched_ids: List[str]
    matches: List[EntityMatchResult]
    latency_ms: float

class BatchResolveRequest(BaseModel):
    entities: List[BusinessEntity]
    threshold: Optional[float] = None

class BatchResolveResponse(BaseModel):
    total_entities: int
    total_matches: int
    total_singletons: int
    threshold_applied: float
    results: List[SingleResolveResponse]
    latency_ms: float

class PairComparisonRequest(BaseModel):
    entity_1: BusinessEntity
    entity_2: BusinessEntity
    threshold: Optional[float] = 0.30

class PairComparisonResponse(BaseModel):
    entity_1_name: str
    entity_2_name: str
    entity_1_address: str
    entity_2_address: str
    entity_1_country: str
    entity_2_country: str
    match_probability: float
    predicted_same_business: bool
    decision_threshold: float
    features: Dict[str, float]

class DatasetStatsResponse(BaseModel):
    num_train_s1: int
    num_train_s2: int
    num_train_s3: int
    num_train_gt: int
    num_test_s1: int
    num_test_s2: int
    num_test_s3: int
    positive_pairs: int
    singletons: int

class ValidationCheckResponse(BaseModel):
    is_valid: bool
    matching_file: str
    candidate_file: str
    total_s1_entities: int
    total_matches: int
    singletons: int
    total_candidate_pairs: int
    errors: List[str]
