"""Unit tests for Canonical Model and I/O serialization."""

import pytest
from src.canonical.model import (
    BoundingBox,
    Document,
    Element,
    ElementType,
    Page,
)
from src.canonical.io import document_to_dict, save_canonical, load_canonical


def test_bounding_box_iou():
    """Test IoU calculation for BoundingBox."""
    bbox1 = BoundingBox(x0=0.0, y0=0.0, x1=0.5, y1=0.5)
    bbox2 = BoundingBox(x0=0.25, y0=0.25, x1=0.75, y1=0.75)

    # Intersection: [0.25, 0.25] to [0.5, 0.5] -> area = 0.25 * 0.25 = 0.0625
    # Union: area1 (0.25) + area2 (0.25) - intersection (0.0625) = 0.4375
    # IoU: 0.0625 / 0.4375 = 1/7 ~= 0.142857
    assert round(bbox1.iou(bbox2), 4) == 0.1429
    assert bbox1.iou(bbox1) == 1.0


def test_canonical_document_structure():
    """Test Document building and element aggregation."""
    el1 = Element(
        type=ElementType.EQUIPMENT,
        text="26-KA-902",
        bbox=BoundingBox(x0=0.1, y0=0.1, x1=0.3, y1=0.2),
        page_number=1,
    )
    el2 = Element(
        type=ElementType.NOTE,
        text="NOTE 19",
        bbox=BoundingBox(x0=0.1, y0=0.5, x1=0.3, y1=0.6),
        page_number=1,
    )
    page1 = Page(page_number=1, width=1000.0, height=800.0, elements=[el1, el2])
    doc = Document(id="doc_test", source_format="native_pdf", pages=[page1])

    assert doc.page_count == 1
    assert doc.total_elements == 2
    assert len(doc.all_elements()) == 2
    assert doc.elements_by_type(ElementType.EQUIPMENT) == [el1]


def test_canonical_serialization(tmp_path):
    """Test JSON serialization and deserialization."""
    el = Element(
        type=ElementType.INSTRUMENT,
        text="PIT 9023",
        bbox=BoundingBox(x0=0.2, y0=0.3, x1=0.4, y1=0.5),
        page_number=1,
    )
    doc = Document(
        id="ser_test",
        source_format="native_pdf",
        pages=[Page(page_number=1, width=100.0, height=100.0, elements=[el])],
    )

    doc_dict = document_to_dict(doc)
    assert doc_dict["id"] == "ser_test"
    assert doc_dict["pages"][0]["elements"][0]["type"] == "instrument"

    saved_path = save_canonical(doc, output_dir=tmp_path)
    assert saved_path.exists()

    loaded_doc = load_canonical(saved_path)
    assert loaded_doc.id == doc.id
    assert loaded_doc.total_elements == 1
