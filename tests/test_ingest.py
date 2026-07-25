"""Unit tests for PDF Ingestion and Regex Classifier."""

from pathlib import Path
from src.ingest.classifier import classify_text
from src.ingest.pdf_native import NativePDFAdapter, _is_noise
from src.canonical.model import ElementType


def test_classifier_rules():
    """Test regex classification rules for technical elements."""
    assert classify_text("26-KA-902") == ElementType.EQUIPMENT
    assert classify_text("PIT 9023") == ElementType.INSTRUMENT
    assert classify_text("NOTE 19") == ElementType.NOTE
    assert classify_text("6\"-PV-26-9017-FC11S-38") == ElementType.PIPE
    assert classify_text("XV-100") == ElementType.VALVE
    assert classify_text("1835 kW") == ElementType.TABLE
    assert classify_text("3RD STAGE HP GAS EXPORT COMPRESSOR") == ElementType.TITLE


def test_noise_filter():
    """Test noise filtering for CAD handles, grid labels, and standalone chars."""
    assert _is_noise("A") is True
    assert _is_noise("12") is True
    assert _is_noise("26GT9135") is True
    assert _is_noise("300#") is True
    assert _is_noise("N3207") is True
    assert _is_noise("26-KA-902") is False
    assert _is_noise("COMPRESSOR CASING DRAIN") is False


def test_native_pdf_adapter_can_handle():
    """Test PDF adapter file handling check."""
    adapter = NativePDFAdapter()
    sample_pdf = Path("data/samples/Export Gas Compressor-P&ID.pdf")

    if sample_pdf.exists():
        assert adapter.can_handle(sample_pdf) is True

    assert adapter.can_handle(Path("invalid_path.xyz")) is False


def test_native_pdf_ingest_sample():
    """Test native PDF ingestion on actual sample file."""
    sample_pdf = Path("data/samples/Export Gas Compressor-P&ID.pdf")
    if not sample_pdf.exists():
        return

    adapter = NativePDFAdapter()
    doc = adapter.ingest(sample_pdf, pid="test_pdf")

    assert doc.id == "test_pdf"
    assert doc.page_count == 1
    assert doc.total_elements > 0
    assert any(e.type == ElementType.EQUIPMENT for e in doc.all_elements())
