# Document Delta & Grounded Chat

A system that takes two document revisions, computes the meaningful delta between them, produces a delta report, and lets a user chat with both documents and that report.

## Supported Formats

| Format       | Status      |
|-------------|-------------|
| Native PDF  | ✅ Supported |
| Scanned PDF | 🔧 In Progress |
| DWG         | 🔲 Stubbed   |

## Quick Start

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv deltaenv
deltaenv/Scripts/activate      # Windows
# source deltaenv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy env template and fill in your API keys
cp .env.example .env
```

### 3. Run

```bash
# Start the server
make run

# Run chat interface
make chat

# Run evaluation
make eval
```

## Project Structure

```
Pathnovo/
├── README.md                 # This file
├── .env.example              # Required env vars (NO real keys)
├── requirements.txt          # Python dependencies
│
├── src/
│   ├── ingest/               # Format adapters → canonical representation
│   │   ├── base.py           # FormatAdapter interface
│   │   ├── pdf_native.py     # Native PDF extractor
│   │   ├── pdf_scanned.py    # OCR + layout for scanned PDFs
│   │   └── dwg.py            # DWG stub behind the same seam
│   ├── canonical/            # Format-agnostic document model
│   │   └── model.py          # Pydantic models for canonical repr
│   ├── delta/                # Delta computation engine
│   ├── chat/                 # Grounded chat with RAG
│   └── observability/        # Tracing, logging, metrics
│
├── data/
│   └── samples/              # Document pairs with provenance notes
│
├── eval/
│   └── datasets/             # Labeled pairs + Q&A ground truth
│
└── tests/                    # Unit and integration tests
```

## Design Decisions

*To be updated as the project progresses.*

## Trade-offs & Cuts

*To be updated as the project progresses.*

## What's Next

*To be updated as the project progresses.*
