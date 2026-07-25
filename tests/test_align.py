"""Unit tests for Alignment Engine."""

from src.canonical.model import Document, Element, ElementType, Page, BoundingBox
from src.delta.align import AlignmentEngine, _normalize_text


def test_text_normalization():
    """Test text normalization for change comparison."""
    assert _normalize_text("LL : 50") == _normalize_text("LL :50")
    assert _normalize_text("SP= 225.4") == _normalize_text("SP = 225.4")
    assert _normalize_text("PROVISION FOR\nOIL") == _normalize_text("PROVISION FOR OIL")


def test_alignment_engine_matching():
    """Test multi-signal matching between two documents."""
    el1_a = Element(
        type=ElementType.EQUIPMENT,
        text="26-KA-902",
        bbox=BoundingBox(x0=0.1, y0=0.1, x1=0.3, y1=0.2),
        page_number=1,
    )
    el2_a = Element(
        type=ElementType.NOTE,
        text="NOTE 19",
        bbox=BoundingBox(x0=0.1, y0=0.5, x1=0.3, y1=0.6),
        page_number=1,
    )
    doc_a = Document(
        id="doc_a",
        source_format="native_pdf",
        pages=[Page(page_number=1, width=100.0, height=100.0, elements=[el1_a, el2_a])],
    )

    el1_b = Element(
        type=ElementType.EQUIPMENT,
        text="26-KA-901",  # Modified tag
        bbox=BoundingBox(x0=0.1, y0=0.1, x1=0.3, y1=0.2),
        page_number=1,
    )
    el2_b = Element(
        type=ElementType.NOTE,
        text="NOTE 19",  # Unchanged
        bbox=BoundingBox(x0=0.1, y0=0.5, x1=0.3, y1=0.6),
        page_number=1,
    )
    el3_b = Element(
        type=ElementType.INSTRUMENT,
        text="PIT 9759",  # Added element
        bbox=BoundingBox(x0=0.7, y0=0.7, x1=0.8, y1=0.8),
        page_number=1,
    )
    doc_b = Document(
        id="doc_b",
        source_format="native_pdf",
        pages=[Page(page_number=1, width=100.0, height=100.0, elements=[el1_b, el2_b, el3_b])],
    )

    aligner = AlignmentEngine()
    matches = aligner.align(doc_a, doc_b)

    statuses = [m.status for m in matches]
    assert "modified" in statuses
    assert "unchanged" in statuses
    assert "added" in statuses
