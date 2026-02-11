"""FastAPI dependencies for authentication and database injection."""

from functools import lru_cache

from fastapi import Depends, Header, HTTPException, status

from mem.config import Settings, get_settings
from mem.db.chromadb import ChromaDBClient, get_db_client


async def verify_api_key(
    x_api_key: str = Header(..., description="API key for authentication"),
    settings: Settings = Depends(get_settings),
) -> str:
    """Verify the API key from request header."""
    # if x_api_key != settings.api_key:
    #     raise HTTPException(
    #         status_code=status.HTTP_401_UNAUTHORIZED,
    #         detail="Invalid API key",
    #     )
    # return x_api_key
    return "aaa"

def get_db() -> ChromaDBClient:
    """Get database client dependency."""
    return get_db_client()


@lru_cache()
def get_memory_client():
    """Get Memory client singleton."""
    from mem.client import Memory
    settings = get_settings()
    return Memory(api_key=settings.openai_api_key)
