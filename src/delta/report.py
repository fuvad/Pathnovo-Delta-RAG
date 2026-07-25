"""
Delta Report — generates human-readable Markdown and machine-parseable JSON.

Takes a list of DeltaEntry objects and produces:
    1. Markdown report  → saved to data/reports/{pid_a}_vs_{pid_b}.md
    2. JSON report      → saved to data/reports/{pid_a}_vs_{pid_b}.json

The report is itself a retrievable source for the chat layer.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from src.delta.engine import DeltaEntry
from src.config.settings import get_settings
from src.config.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Report data structure
# ---------------------------------------------------------------------------

class DeltaReport:
    """Holds the full delta report data and generates output formats."""

    def __init__(
        self,
        pid_a: str,
        pid_b: str,
        deltas: list[DeltaEntry],
    ):
        self.pid_a = pid_a
        self.pid_b = pid_b
        self.deltas = deltas
        self.timestamp = datetime.now(timezone.utc).isoformat()

        # Separate changes from unchanged
        self.changes = [d for d in deltas if d.change != "unchanged"]
        self.unchanged = [d for d in deltas if d.change == "unchanged"]

        # Count by change type (counts how many of each change type exist)
        self.counts: dict[str, int] = {}
        for d in deltas:
            self.counts[d.change] = self.counts.get(d.change, 0) + 1

        # Group changes by page (groups changes by which page they appear on)
        self.by_page: dict[int, list[DeltaEntry]] = defaultdict(list)
        for d in self.changes:
            self.by_page[d.page].append(d)

    # -------------------------------------------------------------------
    # JSON output
    # -------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Full report as a JSON-serializable dict."""
        return {
            "pid_a": self.pid_a,
            "pid_b": self.pid_b,
            "timestamp": self.timestamp,
            "summary": {
                "total_elements": len(self.deltas),
                "total_changes": len(self.changes),
                **self.counts,
            },
            "changes": [d.to_dict() for d in self.changes],
        }

    def save_json(self, output_dir: Path | None = None) -> Path:
        """Save the report as JSON."""
        if output_dir is None:
            output_dir = get_settings().REPORTS_DIR

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self.pid_a}_vs_{self.pid_b}.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info("report_json_saved", path=str(path))
        return path

    # -------------------------------------------------------------------
    # Markdown output
    # -------------------------------------------------------------------

    def to_markdown(self) -> str:
        """Generate the full Markdown report."""
        lines: list[str] = []

        # Header
        lines.append(f"# Delta Report: `{self.pid_a}` → `{self.pid_b}`")
        lines.append("")
        lines.append(f"*Generated: {self.timestamp}*")
        lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append("| Change Type | Count |")
        lines.append("|-------------|-------|")

        for change_type in ["modified", "added", "removed", "unchanged"]:
            count = self.counts.get(change_type, 0)
            emoji = _CHANGE_EMOJI.get(change_type, "")
            lines.append(f"| {emoji} {change_type.capitalize()} | {count} |")

        lines.append("")
        lines.append(f"**Total changes: {len(self.changes)}** (out of {len(self.deltas)} elements)")
        lines.append("")
        lines.append("---")
        lines.append("")

        # No changes
        if not self.changes:
            lines.append("*No changes detected between the two revisions.*")
            return "\n".join(lines)

        # Per-page breakdown
        for page_num in sorted(self.by_page.keys()):
            page_deltas = self.by_page[page_num]
            lines.append(f"## Page {page_num}")
            lines.append("")

            for d in page_deltas:
                lines.extend(self._render_change(d))
                lines.append("")

            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def save_markdown(self, output_dir: Path | None = None) -> Path:
        """Save the report as Markdown."""
        if output_dir is None:
            output_dir = get_settings().REPORTS_DIR

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self.pid_a}_vs_{self.pid_b}.md"

        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown())

        logger.info("report_markdown_saved", path=str(path))
        return path

    # -------------------------------------------------------------------
    # Save both formats at once
    # -------------------------------------------------------------------

    def save(self, output_dir: Path | None = None) -> tuple[Path, Path]:
        """Save both Markdown and JSON reports. Returns (md_path, json_path)."""
        md_path = self.save_markdown(output_dir)
        json_path = self.save_json(output_dir)
        return md_path, json_path

    # -------------------------------------------------------------------
    # Markdown rendering helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _render_change(d: DeltaEntry) -> list[str]:
        """Render one DeltaEntry as Markdown lines."""
        lines: list[str] = []
        emoji = _CHANGE_EMOJI.get(d.change, "")
        type_label = d.element_type.capitalize()

        lines.append(f"### {emoji} {d.change.capitalize()} — {type_label}")
        lines.append("")

        if d.change == "modified":
            lines.append(f"**Old:** `{d.old_text}`")
            lines.append("")
            lines.append("↓")
            lines.append("")
            lines.append(f"**New:** `{d.new_text}`")
        elif d.change == "added":
            lines.append(f"**Added:** `{d.new_text}`")
        elif d.change == "removed":
            lines.append(f"**Removed:** `{d.old_text}`")

        lines.append("")
        lines.append(f"- **Confidence:** {d.confidence}")
        lines.append(f"- **Reason:** {d.reason}")

        if d.bbox:
            bbox_str = ", ".join(f"{v:.4f}" for v in d.bbox)
            lines.append(f"- **BBox:** [{bbox_str}]")

        return lines


# Emoji mapping for Markdown readability
_CHANGE_EMOJI = {
    "modified": "✏️",
    "added": "➕",
    "removed": "🗑️",
    "unchanged": "✅",
}


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def generate_report(
    pid_a: str,
    pid_b: str,
    deltas: list[DeltaEntry],
    output_dir: Path | None = None,
) -> DeltaReport:
    """Create a DeltaReport and save both formats to disk.

    Args:
        pid_a: PID of the base document.
        pid_b: PID of the revised document.
        deltas: List of DeltaEntry from the DeltaEngine.
        output_dir: Where to save (defaults to data/reports/).

    Returns:
        The DeltaReport object (already saved).
    """
    report = DeltaReport(pid_a=pid_a, pid_b=pid_b, deltas=deltas)
    md_path, json_path = report.save(output_dir)

    logger.info(
        "report_generated",
        pid_a=pid_a,
        pid_b=pid_b,
        changes=len(report.changes),
        md=str(md_path),
        json=str(json_path),
    )

    return report
