"""Application configuration using Pydantic settings."""

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Required settings
    openai_api_key: str
    api_key: str

    # Optional settings with defaults
    chroma_persist_dir: str = "./chroma_data"
    embedding_model: str = "text-embedding-3-small"

    # API settings
    api_v1_prefix: str = "/api/v1"
    cors_origins: List[str] = ["*"]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
