"""
Backend Core Configuration Settings
"""

import os
from pydantic import BaseModel

class Settings(BaseModel):
    PROJECT_NAME: str = "Amazon ML Challenge 2026 — Entity Resolution API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Base paths relative to backend root
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "output")
    REPORTS_DIR: str = os.path.join(BASE_DIR, "reports")
    EXPERIMENTS_DIR: str = os.path.join(BASE_DIR, "experiments")
    
    # Active model
    DEFAULT_MODEL_VERSION: str = "final"
    DEFAULT_THRESHOLD: float = 0.30
    
    # CORS
    CORS_ORIGINS: list = ["*"]

settings = Settings()
