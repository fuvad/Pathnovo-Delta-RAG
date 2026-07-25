"""
Delta Engine — classifies aligned element pairs into structured changes.

Takes MatchResults from the Alignment Engine and produces a list of
DeltaEntry objects, each with:
    - change type:  added / removed / modified / unchanged
    - page & bbox:  where on the document
    - old & new:    the text before and after
    - confidence:   how confident we are in this classification
    - reason:       human-readable description of what changed
"""

from dataclasses import dataclass, field
from src.canonical.model import Document, Element
from src.delta.align import AlignmentEngine, MatchResult
from src.config.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Delta entry — one structured change
# ---------------------------------------------------------------------------

@dataclass
class DeltaEntry:
    """One detected change between two document revisions."""
    change: str                         # "added" | "removed" | "modified" | "unchanged"
    page: int                           # page number where the change is
    element_type: str                   # element type (equipment, note, etc.)
    old_text: str = ""                  # text from Doc A (base)
    new_text: str = ""                  # text from Doc B (revised)
    bbox: list[float] = field(default_factory=list)   # [x0, y0, x1, y1]
    confidence: float = 0.0            # 0.0–1.0 how confident in this change
    reason: str = ""                    # human-readable explanation

    def to_dict(self) -> dict:
        """JSON-serializable output matching the assignment spec."""
        result = {
            "change": self.change.capitalize(),
            "page": self.page,
            "type": self.element_type,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
        }

        if self.old_text:
            result["old"] = self.old_text
        if self.new_text:
            result["new"] = self.new_text
        if self.bbox:
            result["bbox"] = self.bbox

        return result


# ---------------------------------------------------------------------------
# Delta Engine
# ---------------------------------------------------------------------------

class DeltaEngine:
    """Computes a structured delta between two canonical Documents.

    Uses the AlignmentEngine for matching, then classifies each
    match into a DeltaEntry with confidence and reasoning.
    """

    def __init__(self):
        self.aligner = AlignmentEngine()

    def compute_delta(
        self,
        doc_a: Document,
        doc_b: Document,
    ) -> list[DeltaEntry]:
        """Compute the full structured delta between two document revisions.

        Args:
            doc_a: The base document (older revision).
            doc_b: The revised document (newer revision).

        Returns:
            List of DeltaEntry objects — each is one detected change.
        """
        logger.info(
            "computing_delta",
            pid_a=doc_a.id,
            pid_b=doc_b.id,
        )

        # Step 1: Align elements between the two documents
        matches = self.aligner.align(doc_a, doc_b)

        # Step 2: Classify each match into a DeltaEntry
        deltas: list[DeltaEntry] = []
        for match in matches:
            entry = self._classify_match(match)
            deltas.append(entry)

        # Sort by page, then by change priority (added/removed first)
        change_priority = {"removed": 0, "added": 1, "modified": 2, "unchanged": 3}
        deltas.sort(key=lambda d: (d.page, change_priority.get(d.change, 4)))

        # Log summary
        counts = {}
        for d in deltas:
            counts[d.change] = counts.get(d.change, 0) + 1

        logger.info(
            "delta_computed",
            pid_a=doc_a.id,
            pid_b=doc_b.id,
            total_changes=len(deltas),
            **counts,
        )

        return deltas

    def compute_delta_changes_only(
        self,
        doc_a: Document,
        doc_b: Document,
    ) -> list[DeltaEntry]:
        """Compute delta but return only actual changes (no unchanged)."""
        all_deltas = self.compute_delta(doc_a, doc_b)
        return [d for d in all_deltas if d.change != "unchanged"]

    # -------------------------------------------------------------------
    # Classification logic
    # -------------------------------------------------------------------

    def _classify_match(self, match: MatchResult) -> DeltaEntry:
        """Convert a MatchResult into a classified DeltaEntry."""

        status = match.status  # added / deleted / modified / unchanged

        if status == "added":
            return self._make_added(match)
        elif status == "deleted":
            return self._make_removed(match)
        elif status == "modified":
            return self._make_modified(match)
        else:
            return self._make_unchanged(match)

    def _make_added(self, match: MatchResult) -> DeltaEntry:
        """Element exists in Doc B but not Doc A → added."""
        eb = match.element_b
        return DeltaEntry(
            change="added",
            page=eb.page_number,
            element_type=eb.type.value.lower(),
            old_text="",
            new_text=eb.text,
            bbox=self._get_bbox(eb),
            confidence=eb.confidence,
            reason=f"New {eb.type.value.lower()} added: \"{eb.text[:80]}\"",
        )

    def _make_removed(self, match: MatchResult) -> DeltaEntry:
        """Element exists in Doc A but not Doc B → removed."""
        ea = match.element_a
        return DeltaEntry(
            change="removed",
            page=ea.page_number,
            element_type=ea.type.value.lower(),
            old_text=ea.text,
            new_text="",
            bbox=self._get_bbox(ea),
            confidence=ea.confidence,
            reason=f"{ea.type.value.lower().capitalize()} removed: \"{ea.text[:80]}\"",
        )

    def _make_modified(self, match: MatchResult) -> DeltaEntry:
        """Element exists in both but text differs → modified."""
        ea = match.element_a
        eb = match.element_b

        # Confidence from the alignment match score
        confidence = match.combined_score

        # Generate a human-readable reason
        reason = self._generate_reason(ea, eb)

        # Use the bbox from Doc B (the newer version)
        bbox = self._get_bbox(eb)

        return DeltaEntry(
            change="modified",
            page=eb.page_number,
            element_type=eb.type.value.lower(),
            old_text=ea.text,
            new_text=eb.text,
            bbox=bbox,
            confidence=round(confidence, 4),
            reason=reason,
        )

    def _make_unchanged(self, match: MatchResult) -> DeltaEntry:
        """Element is the same in both → unchanged."""
        ea = match.element_a
        return DeltaEntry(
            change="unchanged",
            page=ea.page_number,
            element_type=ea.type.value.lower(),
            old_text=ea.text,
            new_text=ea.text,
            bbox=self._get_bbox(ea),
            confidence=match.combined_score,
            reason="No change detected",
        )

    # -------------------------------------------------------------------
    # Reason generation — deterministic, no LLM needed
    # -------------------------------------------------------------------

    @staticmethod
    def _generate_reason(ea: Element, eb: Element) -> str:
        """Generate a human-readable reason for a modification.

        Uses deterministic logic, not LLM — keeps the delta engine
        reproducible and fast.
        """
        type_name = eb.type.value.lower()

        # Type changed
        if ea.type != eb.type:
            return (
                f"Element reclassified from {ea.type.value.lower()} "
                f"to {eb.type.value.lower()}"
            )

        # Same type — describe what changed
        old = ea.text.strip()
        new = eb.text.strip()

        # Short texts — likely tags, labels, values
        if len(old) < 50 and len(new) < 50:
            return f"{type_name.capitalize()} changed from \"{old}\" to \"{new}\""

        # Longer texts — note the change
        if len(new) > len(old) * 1.5:
            return f"{type_name.capitalize()} expanded (text added)"
        elif len(old) > len(new) * 1.5:
            return f"{type_name.capitalize()} shortened (text removed)"
        else:
            return f"{type_name.capitalize()} text modified"

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _get_bbox(el: Element) -> list[float]:
        """Extract bbox as a flat list, or empty list if no bbox."""
        if el.bbox:
            return [el.bbox.x0, el.bbox.y0, el.bbox.x1, el.bbox.y1]
        return []
