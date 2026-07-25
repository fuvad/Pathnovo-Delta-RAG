# Delta — Structured change detection between document revisions
from src.delta.align import AlignmentEngine, MatchResult
from src.delta.engine import DeltaEngine, DeltaEntry

__all__ = [
    "AlignmentEngine",
    "MatchResult",
    "DeltaEngine",
    "DeltaEntry",
]
