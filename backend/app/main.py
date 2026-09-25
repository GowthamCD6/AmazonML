"""
FastAPI Application Entry Point
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.app.core.config import settings
from backend.app.api.endpoints import router as api_router
from backend.app.services.resolver import resolver_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize model and target database
    print("Starting Amazon ML Challenge Entity Resolution Service...")
    resolver_service.initialize()
    yield
    print("Shutting down Entity Resolution Service...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production Business Entity Resolution Backend & Live Inspection Engine for Amazon ML Challenge 2026",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Static and Client directory setup
client_dist = os.path.abspath(os.path.join(settings.BASE_DIR, "client", "dist"))
static_dir = os.path.join(os.path.dirname(__file__), "static")

if os.path.exists(client_dist) and os.path.exists(os.path.join(client_dist, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(client_dist, "assets")), name="assets")

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", tags=["Dashboard"])
def get_dashboard():
    """Serve the interactive Web Dashboard."""
    if os.path.exists(client_dist) and os.path.exists(os.path.join(client_dist, "index.html")):
        return FileResponse(os.path.join(client_dist, "index.html"))
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Welcome to Amazon ML Challenge 2026 Entity Resolution API. Visit /docs for OpenAPI documentation."}

