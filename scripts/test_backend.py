"""
Test script for verifying all backend API endpoints
"""

import sys
import os
import json
from fastapi.testclient import TestClient

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.main import app

def run_tests():
    client = TestClient(app)
    
    print("\n--- 1. Testing GET /api/v1/health ---")
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    print("Health response:", resp.json())
    
    print("\n--- 2. Testing POST /api/v1/resolve (Single Query) ---")
    query = {
        "business_name": "Apex Logistics 128 Pty Ltd",
        "business_address": "4820 George Street, Sydney, 2000, AU",
        "country": "AU"
    }
    resp = client.post("/api/v1/resolve?threshold=0.30", json=query)
    assert resp.status_code == 200, f"Resolve failed: {resp.text}"
    resolve_data = resp.json()
    print(f"Resolve Result: {resolve_data['matches_count']} matches found in {resolve_data['latency_ms']}ms")
    for m in resolve_data["matches"][:3]:
        print(f"  -> Match [{m['candidate_entity_id']}]: {m['business_name']} (Prob: {m['match_probability'] * 100:.1f}%)")

    print("\n--- 3. Testing POST /api/v1/match/pair (Deep Pairwise Comparison) ---")
    pair_req = {
        "entity_1": {
            "business_name": "Clearwater Analytics Inc.",
            "business_address": "123 Main St, New York, 10001, US",
            "country": "US"
        },
        "entity_2": {
            "business_name": "Clearwater Analytics",
            "business_address": "123 Main Street #400, New York, 10001",
            "country": "US"
        },
        "threshold": 0.30
    }
    resp = client.post("/api/v1/match/pair", json=pair_req)
    assert resp.status_code == 200, f"Pair match failed: {resp.text}"
    pair_data = resp.json()
    print(f"Pair Result: Match={pair_data['predicted_same_business']} (Probability: {pair_data['match_probability'] * 100:.2f}%)")

    print("\n--- 4. Testing GET /api/v1/experiments ---")
    resp = client.get("/api/v1/experiments")
    assert resp.status_code == 200
    print(f"Experiments logged: {len(resp.json())} entries.")

    print("\n--- 5. Testing GET /api/v1/reports/validation_report ---")
    resp = client.get("/api/v1/reports/validation_report")
    assert resp.status_code == 200
    print(f"Report fetched successfully ({len(resp.json()['content'])} characters).")

    print("\n--- 6. Testing GET / (Static Dashboard HTML) ---")
    resp = client.get("/")
    assert resp.status_code == 200
    assert "<title>" in resp.text
    print("Dashboard static HTML verified successfully!")

    print("\n" + "=" * 60)
    print("ALL BACKEND ENDPOINTS AND SERVICES TESTED & VERIFIED!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
