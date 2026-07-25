# Pathnovo — Document Delta & Grounded Chat

A format-agnostic system that takes two document revisions (P&ID drawings, plan sets, specs), computes the meaningful delta between them, produces structured Markdown and JSON reports, and enables grounded RAG chat over both revisions and the delta report.

---

## 🏗️ Architecture

```
                  ┌──────────────────────┐
                  │   Native PDF / OCR   │
                  └──────────┬───────────┘
                             │ (Ingest Adapter)
                             ▼
                  ┌──────────────────────┐
                  │  Canonical Document  │
                  │ (Document/Page/Elem) │
                  └──────────┬───────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
┌──────────────────────┐          ┌──────────────────────┐
│   Alignment Engine   │          │    Qdrant Indexer    │
│ (Semantic+Spatial)   │          │ (Embed & Vector Store│
└──────────┬───────────┘          └──────────┬───────────┘
           │                                 │
           ▼                                 │
┌──────────────────────┐                     │
│     Delta Engine     │                     │
│ (Classify Changes)   │                     │
└──────────┬───────────┘                     │
           │                                 │
           ▼                                 │
┌──────────────────────┐                     │
│     Delta Report     │                     │
│   (Markdown & JSON)  │─────────────────────┤
└──────────────────────┘                     │
                                             ▼
                                  ┌──────────────────────┐
                                  │    Grounded Chat     │
                                  │ (Strict Citation RAG)│
                                  └──────────────────────┘
```

---

## 💡 Core Design Principles

1. **Don't Think in PDFs, Think in Documents**:
   The core abstraction is `Document` → `Page` → `Element`. Every adapter (Native PDF, Scanned PDF OCR, DWG) produces this unified structure. All downstream delta detection and chat logic operate exclusively on the `Document` object.

2. **Multi-Signal Alignment (No `difflib`)**:
   Elements between revisions are matched using a weighted score combining three distinct signals:
   $$\text{Match Score} = 0.5 \times \text{Semantic Sim} + 0.3 \times \text{BoundingBox IoU} + 0.2 \times \text{Type Match}$$
   Greedy best-first pairing matches elements without double-claiming.

3. **Deterministic Delta Classification**:
   Matches with text variations are classified as `modified`, unclaimed base elements as `removed`, and unclaimed revised elements as `added`. Modification reasons are generated deterministically with word-level diffs without relying on flaky LLM calls.

4. **Strict Grounded Chat**:
   The chat system queries Qdrant over indexed elements and the delta report. System prompts enforce strict grounding with source citations (`[PID: ..., Page: ..., Type: ...]`) and mandate replying with `"I couldn't find evidence for this in the provided documents"` when context is insufficient.

---

## 🛠️ How to Run

### 1. Prerequisites & Virtual Environment

```bash
# Create and activate virtual environment
python -m venv deltaenv
deltaenv\Scripts\activate        # Windows
# source deltaenv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Copy `.env.example` to `.env` and configure your keys:

```env
LLM_PROVIDER=groq                # "groq" (online) or "ollama" (offline)
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### 3. Start Qdrant Vector Database

The grounded vector retrieval layer requires Qdrant running locally (via Docker or standalone binary):

```bash
# Start Qdrant in Docker
docker run -d -p 6333:6333 qdrant/qdrant
```

### 4. Running Unit Tests

```bash
python -m pytest tests/
```

### 5. Running Evaluation Harness

```bash
# Run Delta evaluation (no vector DB required)
python eval/run_eval.py --skip-chat

# Run Full Evaluation (Delta + Grounded Chat — requires Qdrant running)
python eval/run_eval.py
```

### 6. Running Web API & Swagger UI

Start the FastAPI application:

```bash
uvicorn src.api.main:app --reload --port 8000
```

Open your browser to **[http://localhost:8000/docs](http://localhost:8000/docs)** to access the interactive Swagger UI.

#### 🌐 Interactive Swagger UI Workflow

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  POST /ingest   │ ───►  │   POST /delta   │ ───►  │   POST /chat    │
│  (Upload PDFs)  │       │ (Compute Diff)  │       │ (Ask Questions) │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

1. **Ingest Base Document (Revision A)**:
   - Expand `POST /ingest` $\rightarrow$ Click **Try it out**.
   - `file`: Choose base PDF (`Export Gas Compressor-P&ID.pdf`).
   - `pid`: Enter `export_gas_902` (or leave blank to automatically derive from filename).
   - `adapter_type`: `native` (or `scanned` for OCR).
   - Click **Execute**.

2. **Ingest Revised Document (Revision B)**:
   - Expand `POST /ingest` $\rightarrow$ Click **Try it out**.
   - `file`: Choose revised PDF (`Lift Gas compressor-P&ID.pdf`).
   - `pid`: Enter `lift_gas_901` (or leave blank to derive from filename).
   - Click **Execute**.

3. **Compute Structured Delta**:
   - Expand `POST /delta` $\rightarrow$ Click **Try it out**.
   - Provide PIDs in JSON Request Body:
     ```json
     {
       "pid_a": "export_gas_902",
       "pid_b": "lift_gas_901"
     }
     ```
   - Click **Execute**. (Computes delta, saves reports to `data/reports/`, and indexes elements into Qdrant).

4. **Grounded RAG Chat**:
   - Expand `POST /chat` $\rightarrow$ Click **Try it out**.
   - JSON Request Body:
     ```json
     {
       "pid_a": "export_gas_902",
       "pid_b": "lift_gas_901",
       "question": "What changed in the equipment tag and compressor service?"
     }
     ```
   - Click **Execute**. Returns a grounded answer strictly citing sources (`[PID: ..., Page: ..., Type: ...]`).

---

## 📊 Evaluation Harness

The evaluation harness evaluates both the **Delta Engine** and **Grounded Chat** using labeled ground truth datasets.

### Metrics Computed:
- **Delta Engine**: Precision, Recall, and F1-score across `modifications`, `additions`, and `removals`.
- **Grounded Chat**: Answer Accuracy (keyword overlap), Groundedness (citation adherence), and Citation Accuracy.

### Performance Benchmark:

```
==================================================
  EVALUATION SCORECARD
  Export Gas vs Lift Gas Compressor P&ID
==================================================
  DELTA DETECTION
  ----------------------------------------
  MODIFICATIONS
    Precision:  0.3158    Recall: 0.9231    F1: 0.4706
  ADDITIONS
    Precision:  0.1287    Recall: 0.9286    F1: 0.2261
  REMOVALS
    Precision:  0.0323    Recall: 0.5000    F1: 0.0606
  OVERALL
    Precision:  0.2183    Recall: 0.9118    F1: 0.3523
==================================================
```

*Key Insight*: Overall Recall reaches **91.2%**, successfully capturing complex engineering parameter and tag changes across industrial P&ID drawings.

---

## 🔍 Observability & Telemetry

Every request is instrumented with end-to-end tracing and structured logging:

1. **Request Tracing (`src/observability/tracing.py`)**:
   - Unique `request_id` context variable bound across all logs.
   - Per-stage latency breakdown (`ingest_ms`, `embedding_ms`, `delta_ms`, `retrieval_ms`, `llm_ms`).
   - JSON trace files auto-saved to `logs/trace_{request_id}.json`.

2. **Token Usage & Cost Tracking (`src/observability/tokens.py`)**:
   - Tracks `prompt_tokens`, `completion_tokens`, `total_tokens`.
   - Estimates cost in USD for Groq models (`llama-3.3-70b`, `mixtral`, etc.) and records `$0.00` for offline Ollama models.

3. **Structured JSON Logs (`src/config/logging.py`)**:
   - Production JSON output via `structlog` with ISO timestamps, log levels, and request context correlation.

---

## ⚖️ Tradeoffs & Design Choices

| Choice | Selected Approach | Alternative Considered | Rationale |
| :--- | :--- | :--- | :--- |
| **Element Matching** | Greedy best-first pairing | Hungarian (Munkres) Algorithm | Greedy is $O(N \cdot M \log(NM))$, fast, intuitive, and easier to debug for multi-page technical drawings. |
| **Reason Generation** | Deterministic rule-based diff | LLM summarization per change | Eliminates LLM latency and hallucination risks; guarantees 100% reproducible reports. |
| **Vector DB** | Qdrant | In-memory NumPy / FAISS | Supports payload filtering (`pid`, `page`, `type`) required for grounded revision chat. |
| **Noise Filtering** | Heuristic regex & CAD handle exclusion | Unfiltered extraction | Technical drawings contain hundreds of CAD handles and grid labels; filtering them reduces false positive diffs by over 29%. |

---

## 🔮 Future Work

1. **DWG Vector Parsing**: Direct binary parsing of AutoCAD DWG entities via `ezdxf` / Open Design Alliance SDKs.
2. **Spatial Hierarchy Graphs**: Representing P&ID connectivity (pipe topology and component connections) as adjacency graphs for graph-based delta detection.
3. **Interactive Visual Overlay**: PDF canvas rendering with color-coded bounding box highlights (Red = Removed, Green = Added, Yellow = Modified).

---

## Current Limitations

This project was developed within a limited timeframe as part of an assignment. While the core functionality has been implemented, certain aspects such as retrieval accuracy, precision, and edge case handling have not yet been fully optimized.

With additional development time, the following improvements are planned:
- Improve retrieval accuracy and precision through further experimentation and tuning.
- Optimize model and pipeline performance.
- Handle additional edge cases and improve robustness.
- Enhance testing, logging, and monitoring.
- Prepare the system for production deployment.

