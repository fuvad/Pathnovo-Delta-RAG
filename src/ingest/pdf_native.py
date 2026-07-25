"""
Native PDF Adapter — extracts text, bounding boxes, and font info using PyMuPDF.

Handles born-digital PDFs with an extractable text/vector layer.
Blocks are extracted, then classified via regex into canonical ElementTypes.
"""

import re
from pathlib import Path
import fitz  # PyMuPDF
from src.canonical.model import (
    BoundingBox,
    Document,
    Element,
    Page,
)
from src.ingest.base import FormatAdapter
from src.ingest.classifier import classify_text
from src.config.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Noise filter — skips elements that add no semantic value
# ---------------------------------------------------------------------------

# Grid/border labels: single letters A-J or single numbers 1-12
_GRID_LABEL_RE = re.compile(r"^[A-J]$|^\d{1,2}$", re.IGNORECASE)

# Standalone punctuation or symbols
_PUNCTUATION_RE = re.compile(r"^[/*.,\-\\_|#&()\"\']+$")

# Bare pressure class (just "300#", "150#", "2500#" with nothing else)
_BARE_CLASS_RE = re.compile(r"^\d+#$")

# Nozzle IDs (N3207, N4601, O3207, O4207) — useful for piping but noise for delta
_NOZZLE_RE = re.compile(r"^[NO]\d{4}$", re.IGNORECASE)

# CAD drawing graphic handles / internal object IDs (26GT9135, 26CB9811, 43GT9052, 26BL9031)
_CAD_HANDLE_RE = re.compile(r"^\d{2}[A-Z]{2}\d{4}[A-Z]?$", re.IGNORECASE)


def _is_noise(text: str) -> bool:
    """Return True if the text block is noise that should be skipped.

    Noise elements create hundreds of false positive matches because
    both documents contain similar short codes that match each other
    without representing meaningful changes.
    """
    stripped = text.strip()

    # Too short — single chars, "26", "RS", "*"
    if len(stripped) <= 2:
        return True

    # Check each line — if multi-line, only noise if ALL lines are noise
    lines = [l.strip() for l in stripped.split("\n") if l.strip()]

    # Single line checks
    if len(lines) == 1:
        line = lines[0]
        if _GRID_LABEL_RE.match(line):
            return True
        if _PUNCTUATION_RE.match(line):
            return True
        if _BARE_CLASS_RE.match(line):
            return True
        if _NOZZLE_RE.match(line):
            return True
        if _CAD_HANDLE_RE.match(line):
            return True

    # Multi-line: noise if it's ONLY nozzle IDs, grid labels, or CAD handles
    if len(lines) > 1 and all(
        _NOZZLE_RE.match(l) or _GRID_LABEL_RE.match(l) or _CAD_HANDLE_RE.match(l) or len(l) <= 2
        for l in lines
    ):
        return True

    return False


class NativePDFAdapter(FormatAdapter):
    """Ingestion adapter for native (born-digital) PDFs."""

    @property
    def format_name(self) -> str:
        return "native_pdf"

    def can_handle(self, file_path: Path) -> bool:
        """Check if the file is a PDF with extractable text."""
        if file_path.suffix.lower() != ".pdf":
            return False

        try:
            doc = fitz.open(str(file_path))
            # Sample the first few pages — if any have real text, it's native
            for page_num in range(min(3, len(doc))):
                page = doc[page_num]
                text = page.get_text("text").strip()
                if len(text) > 50:  # enough text to be native, not just a header
                    doc.close()
                    return True
            doc.close()
            return False
        except Exception:
            return False

    def ingest(self, file_path: Path, pid: str) -> Document:
        """Extract blocks, words, bboxes, and font sizes from a native PDF.

        Args:
            file_path: Path to the PDF file.
            pid: Persistent identifier for this document revision.

        Returns:
            A canonical Document with classified elements.
        """
        logger.info("ingesting_native_pdf", pid=pid, file=str(file_path))

        doc = fitz.open(str(file_path))
        pages: list[Page] = []

        for page_idx in range(len(doc)):
            fitz_page = doc[page_idx]
            page_number = page_idx + 1
            page_width = fitz_page.rect.width
            page_height = fitz_page.rect.height

            # --- Extract blocks with dict mode for font info ---
            raw_dict = fitz_page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            blocks = raw_dict.get("blocks", [])

            elements: list[Element] = []
            font_sizes: list[float] = []

            # First pass: collect font size of every piece of text in the page to compute average font size
            for block in blocks:
                if block.get("type") != 0:  # 0 = text block, 1 = image
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):      # span is the smallest piece of text with same formatting
                        if span.get("text", "").strip():
                            font_sizes.append(span.get("size", 0.0))

            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0.0

            # Second pass: build elements from text blocks
            for block in blocks:
                if block.get("type") != 0:
                    continue

                # Combine all spans in the block into one text
                block_text_parts: list[str] = []
                block_font_sizes: list[float] = []

                for line in block.get("lines", []):
                    line_text_parts: list[str] = []
                    for span in line.get("spans", []):
                        span_text = span.get("text", "")
                        if span_text.strip():
                            line_text_parts.append(span_text)   # Join each span
                            block_font_sizes.append(span.get("size", 0.0))  # Join each font size
                    if line_text_parts:
                        block_text_parts.append(" ".join(line_text_parts))  # Join all the spans

                block_text = "\n".join(block_text_parts).strip()    # Join all lines

                if not block_text:
                    continue

                # --- Noise filter: skip elements that add no semantic value ---
                if _is_noise(block_text):
                    continue

                # Block bounding box from PyMuPDF (x0, y0, x1, y1)
                bx0, by0, bx1, by1 = block["bbox"]

                # Normalize to 0.0–1.0
                bbox = BoundingBox(
                    x0=round(bx0 / page_width, 6) if page_width else 0.0,
                    y0=round(by0 / page_height, 6) if page_height else 0.0,
                    x1=round(bx1 / page_width, 6) if page_width else 0.0,
                    y1=round(by1 / page_height, 6) if page_height else 0.0,
                )

                # Dominant font size for this block -> for title detection
                block_font = max(block_font_sizes) if block_font_sizes else 0.0

                # Classify the block
                element_type = classify_text(
                    text=block_text,
                    font_size=block_font,
                    page_avg_font=avg_font_size,
                )

                element = Element(
                    type=element_type,
                    text=block_text,
                    bbox=bbox,
                    page_number=page_number,
                    confidence=1.0,  # native text = full confidence
                    metadata={
                        "font_size": block_font,
                        "source": "pymupdf_block",
                    },
                )
                elements.append(element)

            page = Page(
                page_number=page_number,
                width=page_width,
                height=page_height,
                elements=elements,
            )
            pages.append(page)

            logger.debug(
                "page_extracted",
                pid=pid,
                page=page_number,
                elements=len(elements),
            )

        doc.close()

        document = Document(
            id=pid,
            source_format=self.format_name,
            source_filename=file_path.name,
            pages=pages,
            metadata={
                "total_pages": len(pages),
                "total_elements": sum(p.element_count for p in pages),
            },
        )

        logger.info(
            "ingestion_complete",
            pid=pid,
            pages=document.page_count,
            elements=document.total_elements,
        )

        return document
