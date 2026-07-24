# Canonical — Format-agnostic document representation
from src.canonical.model import (
    BoundingBox,
    Document,
    Element,
    ElementType,
    Page,
)
from src.canonical.io import (
    document_to_dict,
    save_canonical,
    load_canonical,
)

__all__ = [
    "BoundingBox",
    "Document",
    "Element",
    "ElementType",
    "Page",
    "document_to_dict",
    "save_canonical",
    "load_canonical",
]
