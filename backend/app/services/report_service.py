"""
Report and Experiment Service for Backend
"""

import os
import pandas as pd
from typing import List, Dict, Any, Optional

from backend.app.core.config import settings

class ReportService:
    @staticmethod
    def get_experiments() -> List[Dict[str, Any]]:
        path = os.path.join(settings.EXPERIMENTS_DIR, "experiments.csv")
        if not os.path.exists(path):
            return []
        df = pd.read_csv(path).fillna("")
        return df.to_dict(orient="records")

    @staticmethod
    def get_leaderboard() -> List[Dict[str, Any]]:
        path = os.path.join(settings.EXPERIMENTS_DIR, "leaderboard.csv")
        if not os.path.exists(path):
            return []
        df = pd.read_csv(path).fillna("")
        return df.to_dict(orient="records")

    @staticmethod
    def get_markdown_report(report_name: str) -> Optional[str]:
        allowed_reports = {
            "data_profile": os.path.join(settings.REPORTS_DIR, "data_profile.md"),
            "validation_report": os.path.join(settings.REPORTS_DIR, "validation_report.md"),
            "error_analysis": os.path.join(settings.REPORTS_DIR, "error_analysis.md"),
            "methodology": os.path.join(settings.BASE_DIR, "methodology.md"),
            "readme": os.path.join(settings.BASE_DIR, "README.md")
        }
        target_path = allowed_reports.get(report_name.lower())
        if not target_path or not os.path.exists(target_path):
            return None
        with open(target_path, "r", encoding="utf-8") as f:
            return f.read()

report_service = ReportService()
