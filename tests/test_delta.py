"""Unit tests for Delta Engine and Delta Report generation."""

from src.canonical.model import Document, Element, ElementType, Page, BoundingBox
from src.delta.engine import DeltaEngine, DeltaEntry
from src.delta.report import DeltaReport, generate_report


def test_delta_engine_classification():
    """Test classification of additions, removals, and modifications."""
    el1_a = Element(
        type=ElementType.TABLE,
        text="1835 kW",
        bbox=BoundingBox(x0=0.1, y0=0.1, x1=0.2, y1=0.2),
        page_number=1,
    )
    el2_a = Element(
        type=ElementType.NOTE,
        text="OLD DCN 100",
        bbox=BoundingBox(x0=0.5, y0=0.5, x1=0.6, y1=0.6),
        page_number=1,
    )
    doc_a = Document(
        id="rev_a",
        source_format="native_pdf",
        pages=[Page(page_number=1, width=100.0, height=100.0, elements=[el1_a, el2_a])],
    )

    el1_b = Element(
        type=ElementType.TABLE,
        text="776 kW",
        bbox=BoundingBox(x0=0.1, y0=0.1, x1=0.2, y1=0.2),
        page_number=1,
    )
    el3_b = Element(
        type=ElementType.INSTRUMENT,
        text="PDIT 9757",
        bbox=BoundingBox(x0=0.8, y0=0.8, x1=0.9, y1=0.9),
        page_number=1,
    )
    doc_b = Document(
        id="rev_b",
        source_format="native_pdf",
        pages=[Page(page_number=1, width=100.0, height=100.0, elements=[el1_b, el3_b])],
    )

    engine = DeltaEngine()
    deltas = engine.compute_delta(doc_a, doc_b)

    changes = {d.change: d for d in deltas}
    assert "modified" in changes
    assert "removed" in changes
    assert "added" in changes

    mod_entry = changes["modified"]
    assert mod_entry.old_text == "1835 kW"
    assert mod_entry.new_text == "776 kW"


def test_report_generation(tmp_path):
    """Test generating Markdown and JSON delta reports."""
    delta_entry = DeltaEntry(
        change="modified",
        page=1,
        element_type="equipment",
        old_text="26-KA-902",
        new_text="26-KA-901",
        confidence=0.95,
        reason="Equipment changed",
    )

    report = DeltaReport(pid_a="rev_a", pid_b="rev_b", deltas=[delta_entry])
    md_content = report.to_markdown()
    json_dict = report.to_dict()

    assert "# Delta Report" in md_content
    assert "26-KA-902" in md_content
    assert json_dict["summary"]["total_changes"] == 1

    md_path, json_path = report.save(output_dir=tmp_path)
    assert md_path.exists()
    assert json_path.exists()
