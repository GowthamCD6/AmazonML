"""
Server runner script for FastAPI Backend
"""

import sys
import os
import argparse
import uvicorn

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    parser = argparse.ArgumentParser(description="Start Amazon ML Entity Resolution Backend Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload")
    args = parser.parse_args()

    print(f"Starting server on http://{args.host}:{args.port} ...")
    uvicorn.run("backend.app.main:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()
