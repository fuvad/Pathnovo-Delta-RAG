"""
Canonical Document Representation.

This is the format-agnostic intermediate model that every ingestion adapter
outputs. The delta engine, chat layer, and report generator all consume this
representation — they never see PDFs, scans, or DWGs directly.

Design principle: Think in Documents, not in file formats.
"""

from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Element types — the vocabulary of things we can find on a page
# ---------------------------------------------------------------------------

class ElementType(str, Enum):
    """Classification of a document element.

    Every element extracted from any format is assigned one of these types.
    The delta engine uses these to align and classify changes.
    """
    TEXT = "TEXT"
    TITLE = "TITLE"
    NOTE = "NOTE"
    EQUIPMENT = "EQUIPMENT"
    INSTRUMENT = "INSTRUMENT"
    PIPE = "PIPE"
    VALVE = "VALVE"
    TABLE = "TABLE"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Bounding box — spatial location on a page
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    """Axis-aligned bounding box in normalized coordinates (0.0–1.0).

    Using normalized coordinates makes the representation resolution-
    independent: a box at (0.5, 0.5) is always the center of the page
    regardless of whether the sources have different dimensions.
    """
    x0: float = Field(..., ge=0.0, le=1.0, description="Left edge (normalized)")
    y0: float = Field(..., ge=0.0, le=1.0, description="Top edge (normalized)")
    x1: float = Field(..., ge=0.0, le=1.0, description="Right edge (normalized)")
    y1: float = Field(..., ge=0.0, le=1.0, description="Bottom edge (normalized)")

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def iou(self, other: "BoundingBox") -> float:
        """Intersection over Union — used by the delta engine for spatial matching."""
        inter_x0 = max(self.x0, other.x0)
        inter_y0 = max(self.y0, other.y0)
        inter_x1 = min(self.x1, other.x1)
        inter_y1 = min(self.y1, other.y1)

        inter_area = max(0, inter_x1 - inter_x0) * max(0, inter_y1 - inter_y0)
        union_area = self.area + other.area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0
        # 1 -> same, 0 -> no overlap


# ---------------------------------------------------------------------------
# Element — a single extractable item on a page
# ---------------------------------------------------------------------------

class Element(BaseModel):       # smallest unit in the doc
    """One extractable item from a document page.

    This is the atomic unit of the canonical representation. Every piece
    of content — a text block, a title, an equipment tag, a table cell —
    becomes an Element with a type, location, and confidence score.
    """
    id: str = Field(
        default_factory=lambda: uuid4().hex[:12],
        description="Unique element identifier",
    )
    type: ElementType = Field(
        default=ElementType.UNKNOWN,
        description="Semantic classification of this element",
    )
    text: str = Field(
        default="",
        description="Extracted text content",
    )
    bbox: Optional[BoundingBox] = Field(
        default=None,
        description="Spatial location on the page (normalized coordinates)",
    )
    page_number: int = Field(
        ...,
        ge=1,
        description="1-indexed page this element belongs to",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Extraction confidence (1.0 = native text, lower = OCR/heuristic)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Format-specific extras (font, layer, block name, etc.)",
    )


# ---------------------------------------------------------------------------
# Page — a single page/sheet in the document
# ---------------------------------------------------------------------------

class Page(BaseModel):
    """One page (or sheet) of a document, containing its extracted elements."""
    page_number: int = Field(..., ge=1, description="1-indexed page number")
    width: float = Field(default=0.0, description="Original page width in points")
    height: float = Field(default=0.0, description="Original page height in points")
    elements: list[Element] = Field(default_factory=list)

    @property
    def element_count(self) -> int:
        return len(self.elements)

    def elements_by_type(self, element_type: ElementType) -> list[Element]:
        """Filter elements by type — useful for delta alignment."""
        return [e for e in self.elements if e.type == element_type]


# ---------------------------------------------------------------------------
# Document — the top-level canonical representation
# ---------------------------------------------------------------------------

class Document(BaseModel):      # represents the entire document
    """The complete canonical representation of one document revision.

    This is what every ingestion adapter produces. The rest of the pipeline
    (delta engine, chat retrieval, report generation) works exclusively
    with this model.

    Attributes:
        id: The PID — persistent identifier for this document revision.
        source_format: Original file format (e.g., "native_pdf", "scanned_pdf", "dwg").
        source_filename: Original filename for traceability.
        pages: Ordered list of pages, each containing extracted elements.
        metadata: Any additional document-level info (revision label, title, etc.).
    """
    id: str = Field(
        ...,
        description="PID — persistent identifier for this document revision",
    )
    source_format: str = Field(
        ...,
        description="Original format: 'native_pdf', 'scanned_pdf', or 'dwg'",
    )
    source_filename: str = Field(
        default="",
        description="Original filename for traceability",
    )
    pages: list[Page] = Field(default_factory=list)  
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Document-level metadata (revision, title, author, etc.)",
    )

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def total_elements(self) -> int:
        return sum(p.element_count for p in self.pages)

    def all_elements(self) -> list[Element]:
        """Flat list of every element across all pages."""
        return [e for p in self.pages for e in p.elements]

    def elements_by_type(self, element_type: ElementType) -> list[Element]:
        """All elements of a given type across the entire document."""
        return [e for p in self.pages for e in p.elements_by_type(element_type)]

    def get_page(self, page_number: int) -> Optional[Page]:
        """Get a page by its 1-indexed page number."""
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None
