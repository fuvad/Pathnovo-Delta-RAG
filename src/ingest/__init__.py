# Ingestion — Format adapters that normalize documents into canonical representation
from src.ingest.base import FormatAdapter
from src.ingest.classifier import classify_text
from src.ingest.pdf_native import NativePDFAdapter

__all__ = [
    "FormatAdapter",
    "classify_text",
    "NativePDFAdapter",
]
