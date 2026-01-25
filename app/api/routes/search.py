"""Semantic search endpoint."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.embeddings import embed_text
from app.db.chromadb import ChromaDBClient
from app.dependencies import get_db, verify_api_key
from app.models.schemas import APIResponse, SearchRequest, SearchResponse

router = APIRouter(prefix="/search", tags=["search"])


@router.post(
    "",
    response_model=APIResponse,
    summary="Semantic search",
    description="Search emails using natural language queries",
)
async def search_emails(
    request: SearchRequest,
    _api_key: str = Depends(verify_api_key),
    db: ChromaDBClient = Depends(get_db),
) -> APIResponse:
    """Perform semantic search across email content."""
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty",
        )

    try:
        query_embedding = embed_text(request.query)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate embedding: {str(e)}",
        )

    results = db.search(
        query_embedding=query_embedding,
        n_results=request.n_results,
        filters=request.filters or None,
    )

    return APIResponse(
        success=True,
        data=SearchResponse(
            query=request.query,
            results=results,
            total=len(results),
        ),
    )
