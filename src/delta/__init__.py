# Delta — Structured change detection between document revisions
from src.delta.align import AlignmentEngine, MatchResult
from src.delta.engine import DeltaEngine, DeltaEntry
from src.delta.report import DeltaReport, generate_report

__all__ = [
    "AlignmentEngine",
    "MatchResult",
    "DeltaEngine",
    "DeltaEntry",
    "DeltaReport",
    "generate_report",
]
