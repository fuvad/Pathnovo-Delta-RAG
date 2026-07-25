"""
Element classifier — regex-based classification of extracted text blocks.

Kept separate from the adapter so it can be reused across formats
(native PDF, scanned PDF after OCR, etc.) and tested independently.
"""

import re
from src.canonical.model import ElementType


# ---------------------------------------------------------------------------
# Compiled regex patterns — order matters (first match wins)
# ---------------------------------------------------------------------------

# Equipment tags: 26-KA-902, P-101A, V-200, TK-3001, HX-100, 26-CX-9021
# Also matches multi-line blocks where first line is tag + second is service
_EQUIPMENT_RE = re.compile(
    r"^\d{0,3}[-]?[A-Z]{1,4}[-][A-Z]{0,3}[-]?\d{2,5}[A-Z]?$",
    re.IGNORECASE,
)

# Multi-line equipment: "26-KA-902\n3RD STAGE HP GAS EXPORT COMPRESSOR"
_EQUIPMENT_MULTILINE_RE = re.compile(
    r"^\d{0,3}[-]?[A-Z]{1,4}[-][A-Z]{0,3}[-]?\d{2,5}[A-Z]?\s*\n",
    re.IGNORECASE,
)

# Instrument tags (ISA standard): PIT-9023, FIC-101, LT-200, PSV-1001
_INSTRUMENT_RE = re.compile(
    r"^[A-Z]{2,4}[-\s]?\d{3,5}[A-Z]?$",
    re.IGNORECASE,
)

# Notes: NOTE 19, NOTE:, REF NOTE 3, NOTE (GENERAL)
_NOTE_RE = re.compile(
    r"(?:^NOTE\s*[\d:()]|^REF\.?\s*NOTE|^NOTES?\b)",
    re.IGNORECASE,
)

# Pipe specs: 6"-CS-1001, 2"-SS-304, 8"-P-2001
_PIPE_RE = re.compile(
    r'^\d{1,2}["\u2033]?\s*[-]?\s*[A-Z]{1,4}\s*[-]\s*\d{2,5}',
    re.IGNORECASE,
)

# Valve tags: XV-100, HV-200, CV-301, PSV-9027A, 26-FV-9038
_VALVE_RE = re.compile(
    r"^(?:\d{1,3}[-])?[A-Z]{1,3}V[-\s]?\d{2,5}[A-Z]?$",
    re.IGNORECASE,
)

# Table values: numbers with units — 1835 kW, 150 mm, 25.4 °C, 100 m³/h
_TABLE_VALUE_RE = re.compile(
    r"^\d+[\.,]?\d*\s*(?:kW|MW|HP|mm|m|kg|bar|psi|°[CF]|RPM|m[³³]/h|L/min|GPM|%|kPa|MPa)\b",
    re.IGNORECASE,
)

# Title heuristic: ALL CAPS text with optional numbers/symbols
_TITLE_RE = re.compile(
    r"^[A-Z0-9][A-Z0-9\s&,\-/]{8,}$",
)


# ---------------------------------------------------------------------------
# Classification function
# ---------------------------------------------------------------------------

def classify_text(text: str, font_size: float = 0.0, page_avg_font: float = 0.0) -> ElementType:
    """Classify a text block into an ElementType using regex patterns.

    Args:
        text: The extracted text (stripped).
        font_size: Font size of this block (points).
        page_avg_font: Average font size on the page — titles are usually larger.

    Returns:
        The best-matching ElementType.
    """
    stripped = text.strip()

    if not stripped:
        return ElementType.UNKNOWN

    # For multi-line blocks, also check just the first line
    first_line = stripped.split("\n")[0].strip()

    # --- Order matters: more specific patterns first ---

    # Notes (check early — "NOTE 19" could false-match instrument)
    if _NOTE_RE.search(stripped):
        return ElementType.NOTE

    # Valve tags (XV-100, PSV-9027A, 26-FV-9038) — check BEFORE general equipment
    if _VALVE_RE.match(first_line):
        return ElementType.VALVE

    # Equipment tags — "26-KA-902", "P-101A", "V-200"
    if _EQUIPMENT_RE.match(first_line):
        return ElementType.EQUIPMENT

    # Multi-line equipment: "26-KA-902\n3RD STAGE HP GAS EXPORT COMPRESSOR"
    if _EQUIPMENT_MULTILINE_RE.match(stripped):
        return ElementType.EQUIPMENT

    # Instrument tags — also check multi-line: "PSV\n9027A" or "PIT\n9023"
    if _INSTRUMENT_RE.match(first_line):
        return ElementType.INSTRUMENT

    # Multi-line instrument: "PSV\n9027A\n26" or "PIT\n9023\n26"
    if len(first_line) <= 4 and re.match(r"^[A-Z]{2,4}$", first_line):
        lines = [l.strip() for l in stripped.split("\n") if l.strip()]
        if len(lines) >= 2 and re.match(r"^\d{3,5}[A-Z]?$", lines[1]):
            return ElementType.INSTRUMENT

    # Pipe specs
    if _PIPE_RE.match(stripped):
        return ElementType.PIPE

    # Valve tags
    if _VALVE_RE.match(stripped):
        return ElementType.VALVE

    # Table values (numbers with units)
    if _TABLE_VALUE_RE.match(stripped):
        return ElementType.TABLE

    # Titles: ALL CAPS + significantly larger font than page average
    if font_size > 0 and page_avg_font > 0 and font_size >= page_avg_font * 1.3:
        if _TITLE_RE.match(stripped):
            return ElementType.TITLE

    # Titles: ALL CAPS even without font info, if long enough
    if _TITLE_RE.match(stripped) and len(stripped) > 15:
        return ElementType.TITLE

    return ElementType.TEXT
