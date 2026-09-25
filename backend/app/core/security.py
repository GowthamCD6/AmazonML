"""
Backend Core Security & Utilities
"""

import time
from typing import Dict, Any

class RateLimiter:
    """Lightweight in-memory rate limiter."""
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = {}

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        timestamps = self.requests.get(client_ip, [])
        timestamps = [t for t in timestamps if now - t < self.window_seconds]
        if len(timestamps) >= self.max_requests:
            return False
        timestamps.append(now)
        self.requests[client_ip] = timestamps
        return True

rate_limiter = RateLimiter()
