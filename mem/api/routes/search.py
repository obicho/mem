"""Semantic search endpoint."""

from fastapi import APIRouter, Depends, HTTPException, status

from mem.client import Memory
from mem.dependencies import get_memory_client, verify_api_key
from mem.models.schemas import APIResponse, SearchRequest

router = APIRouter(prefix="/search", tags=["search"])


@router.post(
    "",
    response_model=APIResponse,
    summary="Semantic search",
    description="Search memories using natural language queries",
)
async def search_memories(
    request: SearchRequest,
    _api_key: str = Depends(verify_api_key),
    memory: Memory = Depends(get_memory_client),
) -> APIResponse:
    """Perform semantic search across all memories."""
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty",
        )

    try:
        results = memory.search(
            query=request.query,
            limit=request.n_results,
            filters=request.filters or None,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )

    return APIResponse(
        success=True,
        data={
            "query": request.query,
            "results": results,
            "total": len(results),
        },
    )
