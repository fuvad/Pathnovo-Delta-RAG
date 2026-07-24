"""
Canonical JSON I/O — serialize Documents to JSON and load them back.

This is how the canonical representation is persisted. The JSON becomes
the single source of truth that the delta engine and chat layer consume.
"""

import json
from pathlib import Path
from typing import Optional
from src.canonical.model import (
    BoundingBox,
    Document,
    Element,
    ElementType,
    Page,
)
from src.config.settings import get_settings
from src.config.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Serialize: Document → JSON
# ---------------------------------------------------------------------------

# Convert to dict first coz json.dump cannot directly serialize custom classes
def document_to_dict(doc: Document) -> dict:
    """Convert a Document to a clean JSON-serializable dict.

    Output format:
        {
            "id": "pid_a",
            "source_format": "native_pdf",
            "source_filename": "drawing_rev_a.pdf",
            "metadata": { ... },
            "pages": [
                {
                    "page": 1,
                    "width": 612.0,
                    "height": 792.0,
                    "elements": [
                        {
                            "id": "a1b2c3d4e5f6",
                            "type": "equipment",
                            "text": "26-KA-902",
                            "bbox": [0.1, 0.2, 0.3, 0.4],
                            "confidence": 1.0,
                            "metadata": { "font_size": 12.0 }
                        }
                    ]
                }
            ]
        }
    """
    return {
        "id": doc.id,
        "source_format": doc.source_format,
        "source_filename": doc.source_filename,
        "metadata": doc.metadata,
        "pages": [
            _page_to_dict(page) for page in doc.pages
        ],
    }


def _page_to_dict(page: Page) -> dict:
    """Convert a Page to a clean dict."""
    return {
        "page": page.page_number,
        "width": page.width,
        "height": page.height,
        "elements": [
            _element_to_dict(el) for el in page.elements
        ],
    }


def _element_to_dict(el: Element) -> dict:
    """Convert an Element to a clean dict."""
    result = {
        "id": el.id,
        "type": el.type.value.lower(),
        "text": el.text,
        "bbox": _bbox_to_list(el.bbox),
        "confidence": el.confidence,
    }
    if el.metadata:
        result["metadata"] = el.metadata
    return result


def _bbox_to_list(bbox: Optional[BoundingBox]) -> Optional[list[float]]:
    """Convert BoundingBox to [x0, y0, x1, y1] list."""
    if bbox is None:
        return None
    return [bbox.x0, bbox.y0, bbox.x1, bbox.y1]


# ---------------------------------------------------------------------------
# Save: Document → JSON file on disk
# ---------------------------------------------------------------------------

def save_canonical(doc: Document, output_dir: Optional[Path] = None) -> Path:
    """Save a Document as canonical JSON to disk.

    Args:
        doc: The canonical Document to save.
        output_dir: Directory to write to (defaults to data/canonical/).

    Returns:
        Path to the written JSON file.
    """
    if output_dir is None:
        output_dir = get_settings().CANONICAL_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{doc.id}.json"     # eg. data/canonical/pid_a.json

    data = document_to_dict(doc)

    with open(output_path, "w", encoding="utf-8") as f:     # dict -> JSON
        json.dump(data, f, indent=2, ensure_ascii=False)    # Writes dict into the file

    logger.info(
        "canonical_saved",
        pid=doc.id,
        path=str(output_path),
        pages=doc.page_count,
        elements=doc.total_elements,
    )

    return output_path


# ---------------------------------------------------------------------------
# Load: JSON file → Document
# ---------------------------------------------------------------------------

def load_canonical(file_path: Path) -> Document:
    """Load a canonical JSON file back into a Document.

    Args:
        file_path: Path to the JSON file.

    Returns:
        A fully reconstructed Document.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pages: list[Page] = []
    for page_data in data.get("pages", []):
        elements: list[Element] = []
        for el_data in page_data.get("elements", []):
            bbox = None
            if el_data.get("bbox") is not None:
                blist = el_data["bbox"]
                bbox = BoundingBox(x0=blist[0], y0=blist[1], x1=blist[2], y1=blist[3])

            element = Element(
                id=el_data.get("id", ""),
                type=ElementType(el_data["type"].upper()),
                text=el_data.get("text", ""),
                bbox=bbox,
                page_number=page_data["page"],
                confidence=el_data.get("confidence", 1.0),
                metadata=el_data.get("metadata", {}),
            )
            elements.append(element)

        page = Page(
            page_number=page_data["page"],
            width=page_data.get("width", 0.0),
            height=page_data.get("height", 0.0),
            elements=elements,
        )
        pages.append(page)

    doc = Document(
        id=data["id"],
        source_format=data.get("source_format", "unknown"),
        source_filename=data.get("source_filename", ""),
        pages=pages,
        metadata=data.get("metadata", {}),
    )

    logger.info(
        "canonical_loaded",
        pid=doc.id,
        path=str(file_path),
        pages=doc.page_count,
        elements=doc.total_elements,
    )

    return doc
