"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import emails, search
from app.config import get_settings
from app.models.schemas import APIResponse, HealthResponse

settings = get_settings()

app = FastAPI(
    title="Memory Layer API",
    description="Persistent Memory Layer for AI Agents - Email ingestion and semantic search",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(emails.router, prefix=settings.api_v1_prefix)
app.include_router(search.router, prefix=settings.api_v1_prefix)


@app.get("/", response_model=APIResponse, tags=["root"])
async def root() -> APIResponse:
    """Root endpoint with API information."""
    return APIResponse(
        success=True,
        data={
            "name": "Memory Layer API",
            "version": "0.1.0",
            "docs": "/docs",
        },
    )


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
    )
