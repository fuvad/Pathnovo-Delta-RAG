"""
FormatAdapter — the interface every ingestion adapter implements.

This is the seam that decouples file formats from the rest of the pipeline.
Adding a new format means writing one class that implements this interface.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from src.canonical.model import Document


class FormatAdapter(ABC):       # can't instantiate this
    """Abstract base class for all format adapters.

    Contract:
        - Accept raw bytes or a file path
        - Detect whether this adapter can handle the input
        - Convert to a canonical Document
    """

    @abstractmethod
    def can_handle(self, file_path: Path) -> bool:
        """Return True if this adapter can process the given file."""
        ...

    @abstractmethod
    def ingest(self, file_path: Path, pid: str) -> Document:
        """Read the file and convert it into a canonical Document.

        Args:
            file_path: Path to the source file.
            pid: The persistent identifier for this document revision.

        Returns:
            A fully populated canonical Document.
        """
        ...

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Human-readable name of the format this adapter handles."""
        ...
