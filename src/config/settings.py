"""
Centralized application settings.

All configuration is loaded from environment variables (via .env file).
No secrets are hardcoded — every sensitive value reads from env vars.
"""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM Provider ---
    LLM_PROVIDER: str = "groq"              # "groq" (online) or "ollama" (offline)

    # --- Groq (Online) ---
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # --- Ollama (Offline / Local) ---
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    # --- Embedding Model ---
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # --- Qdrant ---
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "document_chunks"

    # --- Ingestion ---
    OCR_ENABLED: bool = True
    OCR_LANGUAGE: str = "eng"

    # --- Server ---
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    # --- Logging ---
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"               # "json" or "console"

    # --- Paths ---
    DATA_DIR: Path = PROJECT_ROOT / "data"
    SAMPLES_DIR: Path = PROJECT_ROOT / "data" / "samples"
    CANONICAL_DIR: Path = PROJECT_ROOT / "data" / "canonical"
    REPORTS_DIR: Path = PROJECT_ROOT / "data" / "reports"
    OUTPUTS_DIR: Path = PROJECT_ROOT / "data" / "outputs"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"

    def ensure_directories(self) -> None:
        """Create all output directories if they don't exist."""
        for d in [
            self.CANONICAL_DIR,
            self.REPORTS_DIR,
            self.OUTPUTS_DIR,
            self.LOGS_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton — call this everywhere instead of constructing Settings()."""
    settings = Settings()
    settings.ensure_directories()
    return settings
