"""
Document Delta & Grounded Chat — FastAPI Application

Entry point for the API server. Routes will be added as each
pipeline component (ingest, delta, chat) is built.
"""

from fastapi import FastAPI

from src.config.settings import get_settings


settings = get_settings()

app = FastAPI(
    title="Document Delta & Grounded Chat",
    description="Compute structured deltas between document revisions and chat with grounded citations.",
    version="0.1.0",
)


@app.get("/health")
async def health_check():
    """Basic liveness probe."""
    model = (
        settings.GROQ_MODEL
        if settings.LLM_PROVIDER == "groq"
        else settings.OLLAMA_MODEL
    )
    return {
        "status": "healthy",
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": model,
    }
