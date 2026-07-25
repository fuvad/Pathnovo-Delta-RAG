"""
Document Delta & Grounded Chat — FastAPI Application

Endpoints:
    - GET  /health          : Liveness probe
    - POST /ingest          : Upload PDF/document file -> Ingest to Canonical JSON
    - POST /delta           : Compare two ingested documents -> Delta Report
    - POST /chat            : Grounded RAG Chat query with citations
"""

import shutil
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel, Field

from src.config.settings import get_settings
from src.config.logging import setup_logging, bind_request_id, get_logger
from src.ingest.pdf_native import NativePDFAdapter
from src.ingest.pdf_scanned import ScannedPDFAdapter
from src.canonical.io import save_canonical, load_canonical
from src.canonical.model import Document
from src.delta.engine import DeltaEngine
from src.delta.report import generate_report
from src.chat.index import QdrantIndexer
from src.chat.answer import GroundedChat
from src.observability.tracing import RequestTrace

setup_logging()
logger = get_logger(__name__)
settings = get_settings()

app = FastAPI(
    title="Pathnovo — Document Delta & Grounded Chat API",
    description="Upload technical drawings/PDFs, compute delta reports, and query grounded chat with citations.",
    version="1.0.0",
)

# Active chat sessions in memory
chat_sessions: dict[str, GroundedChat] = {}


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class IngestResponse(BaseModel):
    pid: str
    source_format: str
    source_filename: str
    total_pages: int
    total_elements: int
    canonical_json_path: str


class DeltaRequest(BaseModel):
    pid_a: str = Field(..., example="export_gas_902", description="PID of base document (old revision)")
    pid_b: str = Field(..., example="lift_gas_901", description="PID of revised document (new revision)")


class DeltaResponse(BaseModel):
    pid_a: str
    pid_b: str
    summary: dict
    report_markdown_path: str
    report_json_path: str


class ChatRequest(BaseModel):
    pid_a: str = Field(..., example="export_gas_902", description="PID of base document")
    pid_b: str = Field(..., example="lift_gas_901", description="PID of revised document")
    question: str = Field(..., example="What changed in the equipment tag and service?", description="Question to ask")
    top_k: int = Field(10, description="Max context chunks to retrieve")


class ChatResponse(BaseModel):
    question: str
    answer: str
    model: str
    provider: str
    usage: dict
    retrieved_count: int
    delta_context_count: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health_check():
    """Liveness probe."""
    model = settings.GROQ_MODEL if settings.LLM_PROVIDER == "groq" else settings.OLLAMA_MODEL
    return {
        "status": "healthy",
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": model,
        "qdrant_host": f"{settings.QDRANT_HOST}:{settings.QDRANT_PORT}",
    }


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
@app.post("/api/ingest", response_model=IngestResponse, tags=["Ingestion"], include_in_schema=False)
@app.post("/api/v1/ingest", response_model=IngestResponse, tags=["Ingestion"], include_in_schema=False)
async def ingest_document(
    file: UploadFile = File(..., description="PDF or document file to upload"),
    pid: Optional[str] = Form(None, description="Unique persistent identifier (defaults to filename if omitted)"),
    adapter_type: str = Form("native", description="Adapter type: 'native' or 'scanned'"),
):
    """Upload a PDF file and extract its canonical document representation.

    Note: Our architecture uses layout-aware element extraction (Document -> Page -> Element).
    Arbitrary text chunking parameters (chunk_size, chunking_method) are not used because
    elements are structurally classified (Equipment, Pipe, Instrument, Note, Table).

    The extracted canonical representation will be saved to `data/canonical/{pid}.json`.
    """
    trace = RequestTrace()
    bind_request_id(trace.request_id)

    # Derive PID from filename if not provided
    clean_pid = pid.strip() if pid and pid.strip() else Path(file.filename).stem.replace(" ", "_").replace("&", "and")

    # Save uploaded file to temp samples directory
    samples_dir = settings.SAMPLES_DIR
    samples_dir.mkdir(parents=True, exist_ok=True)
    temp_file_path = samples_dir / f"{clean_pid}_{file.filename}"

    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if adapter_type.lower() == "scanned":
            adapter = ScannedPDFAdapter()
        else:
            adapter = NativePDFAdapter()

        with trace.span("ingest", pid=clean_pid, adapter=adapter_type):
            doc = adapter.ingest(temp_file_path, clean_pid)

        # Save canonical JSON to data/canonical/{clean_pid}.json
        canonical_path = save_canonical(doc)
        trace.finish()
        trace.save()

        return IngestResponse(
            pid=doc.id,
            source_format=doc.source_format,
            source_filename=file.filename,
            total_pages=doc.page_count,
            total_elements=doc.total_elements,
            canonical_json_path=str(canonical_path),
        )

    except Exception as e:
        logger.error("ingestion_endpoint_failed", pid=clean_pid, error=str(e))
        trace.finish(status="error", error=str(e))
        trace.save()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {type(e).__name__} - {str(e)}")


@app.post("/delta", response_model=DeltaResponse, tags=["Delta Engine"])
async def compute_document_delta(request: DeltaRequest):
    """Compute structured delta between two previously ingested document revisions.

    Requires both `pid_a` and `pid_b` to be ingested first.
    Generates Markdown and JSON delta reports in `data/reports/`.
    """
    trace = RequestTrace()
    bind_request_id(trace.request_id)

    path_a = settings.CANONICAL_DIR / f"{request.pid_a}.json"
    path_b = settings.CANONICAL_DIR / f"{request.pid_b}.json"

    if not path_a.exists():
        raise HTTPException(status_code=404, detail=f"Document '{request.pid_a}' not found. Please ingest it first via /ingest.")
    if not path_b.exists():
        raise HTTPException(status_code=404, detail=f"Document '{request.pid_b}' not found. Please ingest it first via /ingest.")

    try:
        doc_a = load_canonical(path_a)
        doc_b = load_canonical(path_b)

        engine = DeltaEngine()
        with trace.span("delta", pid_a=request.pid_a, pid_b=request.pid_b):
            deltas = engine.compute_delta(doc_a, doc_b)

        with trace.span("report"):
            report = generate_report(request.pid_a, request.pid_b, deltas)

        md_path, json_path = report.save()

        # Index both documents and delta report into Qdrant in background for chat
        indexer = QdrantIndexer()
        with trace.span("index"):
            indexer.index_document(doc_a)
            indexer.index_document(doc_b)

        # Store session in memory for chat queries
        session_key = f"{request.pid_a}_vs_{request.pid_b}"
        chat = GroundedChat()
        chat.load_delta(request.pid_a, request.pid_b, deltas)
        chat_sessions[session_key] = chat

        trace.finish()
        trace.save()

        return DeltaResponse(
            pid_a=request.pid_a,
            pid_b=request.pid_b,
            summary=report.to_dict()["summary"],
            report_markdown_path=str(md_path),
            report_json_path=str(json_path),
        )

    except Exception as e:
        trace.finish(status="error", error=str(e))
        trace.save()
        raise HTTPException(status_code=500, detail=f"Delta computation failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse, tags=["Grounded Chat"])
async def chat_with_documents(request: ChatRequest):
    """Ask a question about two document revisions and their delta report.

    Retrieves grounded context from Qdrant and LLM, enforcing exact citations.
    """
    trace = RequestTrace()
    bind_request_id(trace.request_id)

    session_key = f"{request.pid_a}_vs_{request.pid_b}"
    chat = chat_sessions.get(session_key)

    if not chat:
        # Load canonical docs and compute delta on the fly if session expired
        path_a = settings.CANONICAL_DIR / f"{request.pid_a}.json"
        path_b = settings.CANONICAL_DIR / f"{request.pid_b}.json"

        if not path_a.exists() or not path_b.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Session or documents for '{request.pid_a}' vs '{request.pid_b}' not found. Run /delta first.",
            )

        doc_a = load_canonical(path_a)
        doc_b = load_canonical(path_b)
        engine = DeltaEngine()
        deltas = engine.compute_delta(doc_a, doc_b)

        chat = GroundedChat()
        chat.load_delta(request.pid_a, request.pid_b, deltas)
        chat_sessions[session_key] = chat

    try:
        with trace.span("chat", question=request.question):
            result = chat.ask(request.question, top_k=request.top_k)

        trace.record_llm(
            provider=result["provider"],
            model=result["model"],
            prompt_tokens=result["usage"].get("prompt_tokens", 0),
            completion_tokens=result["usage"].get("completion_tokens", 0),
            total_tokens=result["usage"].get("total_tokens", 0),
        )
        trace.finish()
        trace.save()

        return ChatResponse(
            question=result["question"],
            answer=result["answer"],
            model=result["model"],
            provider=result["provider"],
            usage=result["usage"],
            retrieved_count=result["retrieved_count"],
            delta_context_count=result["delta_context_count"],
        )

    except Exception as e:
        trace.finish(status="error", error=str(e))
        trace.save()
        raise HTTPException(status_code=500, detail=f"Chat query failed: {str(e)}")
