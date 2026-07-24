"""
Scanned PDF Adapter — renders pages to images, runs OCR, builds canonical output.

Pipeline:  PDF → Image (PyMuPDF pixmap) → OCR (Tesseract) → Canonical Document

Handles raster/image PDFs with no reliable text layer — scans, photographs,
print-and-scan documents. Requires Tesseract to be installed on the system.

Install Tesseract:
    Windows:  https://github.com/UB-Mannheim/tesseract/wiki
    Linux:    sudo apt install tesseract-ocr
    Mac:      brew install tesseract
"""

from pathlib import Path
from io import BytesIO  # in-memory file (file that exists only in RAM)
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
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

# DPI for rendering PDF pages to images — higher = better OCR, slower
RENDER_DPI = 300    #Dots Per Inch. Higher DPI means more pixels are used to represent the same physical page


class ScannedPDFAdapter(FormatAdapter):
    """Ingestion adapter for scanned (image-based) PDFs."""

    @property
    def format_name(self) -> str:
        return "scanned_pdf"

    def can_handle(self, file_path: Path) -> bool:
        """Check if the file is a PDF with little or no extractable text (i.e., scanned)."""
        if file_path.suffix.lower() != ".pdf":
            return False

        try:
            doc = fitz.open(str(file_path))
            total_text = 0
            pages_checked = min(3, len(doc))

            for page_num in range(pages_checked):
                page = doc[page_num]
                text = page.get_text("text").strip()
                total_text += len(text)

            doc.close()

            # If very little native text across sampled pages → likely scanned pdf
            avg_text_per_page = total_text / pages_checked if pages_checked else 0
            return avg_text_per_page < 50
        except Exception:
            return False

    def ingest(self, file_path: Path, pid: str) -> Document:
        """Render each page as an image, run OCR, and build canonical Document.

        Args:
            file_path: Path to the scanned PDF file.
            pid: Persistent identifier for this document revision.

        Returns:
            A canonical Document with OCR-extracted elements.
        """
        logger.info("ingesting_scanned_pdf", pid=pid, file=str(file_path))

        doc = fitz.open(str(file_path))
        pages: list[Page] = []

        for page_idx in range(len(doc)):
            fitz_page = doc[page_idx]
            page_number = page_idx + 1
            page_width = fitz_page.rect.width
            page_height = fitz_page.rect.height

            # --- Step 1: Render page to image ---
            pil_image = self._render_page(fitz_page)
            img_width, img_height = pil_image.size

            # --- Step 2: OCR with bounding box data ---
            ocr_data = pytesseract.image_to_data(
                pil_image,
                output_type=pytesseract.Output.DICT,
            )

            # --- Step 3: Group words into blocks and build elements ---
            elements = self._build_elements(
                ocr_data=ocr_data,
                img_width=img_width,
                img_height=img_height,
                page_number=page_number,
            )

            page = Page(
                page_number=page_number,
                width=page_width,
                height=page_height,
                elements=elements,
            )
            pages.append(page)

            logger.debug(
                "page_ocr_complete",
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
                "ocr_dpi": RENDER_DPI,
            },
        )

        logger.info(
            "ingestion_complete",
            pid=pid,
            pages=document.page_count,
            elements=document.total_elements,
            format="scanned_pdf",
        )

        return document

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _render_page(self, fitz_page: fitz.Page) -> Image.Image:
        """
        Render a PDF page to a PIL Image at high DPI for OCR.
        input: One PDF page (fitz.Page)
        output: One image (PIL.Image)
        """
        zoom = RENDER_DPI / 72  # 72 is PDF default DPI
        matrix = fitz.Matrix(zoom, zoom)    # scaling matrix
        pixmap = fitz_page.get_pixmap(matrix=matrix)   # rendered page as pixmap (Pixmap is PyMuPDF's own image object)

        # Convert PyMuPDF pixmap → PIL Image
        img_data = pixmap.tobytes("png")    # Convert the pixmap to PNG bytes
        pil_image = Image.open(BytesIO(img_data))   # Create a PIL image
        return pil_image    # return the image

    def _build_elements(
        self,
        ocr_data: dict,
        img_width: int,
        img_height: int,
        page_number: int,
    ) -> list[Element]:
        """Group OCR words by Tesseract block number and build Elements.

        Tesseract assigns each word a block_num → we group by that to
        reconstruct text blocks, then classify and normalize bboxes.
        """
        # Group words by block_num
        blocks: dict[int, list[dict]] = {}
        n_words = len(ocr_data.get("text", []))

        for i in range(n_words):
            text = ocr_data["text"][i].strip()  # read one OCR word
            conf = int(ocr_data["conf"][i])

            # Skip empty or very low confidence words
            if not text or conf < 30:
                continue

            block_num = ocr_data["block_num"][i]
            word_info = {       # dict for one block
                "text": text,
                "conf": conf,
                "left": ocr_data["left"][i],
                "top": ocr_data["top"][i],
                "width": ocr_data["width"][i],
                "height": ocr_data["height"][i],
            }

            if block_num not in blocks:
                blocks[block_num] = []
            blocks[block_num].append(word_info)     # group by block

        # Build an Element for each block
        elements: list[Element] = []
        for block_num, words in blocks.items():
            if not words:
                continue

            # Combine text in one block
            block_text = " ".join(w["text"] for w in words)

            # Compute one bounding box (union of all words in the block)
            min_left = min(w["left"] for w in words)
            min_top = min(w["top"] for w in words)
            max_right = max(w["left"] + w["width"] for w in words)
            max_bottom = max(w["top"] + w["height"] for w in words)

            # Normalize bbox to 0.0–1.0 using image dimensions
            bbox = BoundingBox(
                x0=round(min_left / img_width, 6) if img_width else 0.0,
                y0=round(min_top / img_height, 6) if img_height else 0.0,
                x1=round(max_right / img_width, 6) if img_width else 0.0,
                y1=round(max_bottom / img_height, 6) if img_height else 0.0,
            )

            # Average confidence across words in this block
            avg_conf = sum(w["conf"] for w in words) / len(words)
            confidence = round(avg_conf / 100.0, 3)  # normalize to 0.0–1.0

            # Classify using the same regex classifier
            element_type = classify_text(block_text)

            element = Element(
                type=element_type,
                text=block_text,
                bbox=bbox,
                page_number=page_number,
                confidence=confidence,  # OCR confidence, not 1.0
                metadata={
                    "source": "tesseract_ocr",
                    "word_count": len(words),
                    "ocr_dpi": RENDER_DPI,
                },
            )
            elements.append(element)

        return elements
