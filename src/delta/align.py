"""
Alignment Engine — matches elements between two document revisions.

This is NOT difflib. For every element, we find the best match using
a combined score of three signals:

    match_score = w_semantic * semantic_similarity
                + w_spatial  * spatial_overlap
                + w_type     * type_match

Unmatched elements in Doc A → deleted
Unmatched elements in Doc B → added
Matched elements with text differences → modified
Matched elements with no differences → unchanged
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from src.canonical.model import Document, Element, ElementType
from src.config.settings import get_settings
from src.config.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Scoring weights — configurable, must sum to 1.0
# ---------------------------------------------------------------------------
W_SEMANTIC = 0.5    # How similar is the text? 50%
W_SPATIAL = 0.3     # Where the element appears on the page? 30%
W_TYPE = 0.2        # Whether the element types match? 20%

# Minimum combined score to consider two elements a match
MATCH_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# Match result
# ---------------------------------------------------------------------------

class MatchResult:
    """One match between an element in Doc A and an element in Doc B."""

    def __init__(
        self,
        element_a: Element | None,
        element_b: Element | None,
        semantic_score: float = 0.0,
        spatial_score: float = 0.0,
        type_score: float = 0.0,
    ):
        self.element_a = element_a
        self.element_b = element_b
        self.semantic_score = semantic_score
        self.spatial_score = spatial_score
        self.type_score = type_score

    @property    # behaves like a variable even though Python is executing a function
    def combined_score(self) -> float:
        return (
            W_SEMANTIC * self.semantic_score
            + W_SPATIAL * self.spatial_score
            + W_TYPE * self.type_score
        )

    @property
    def status(self) -> str:
        """Classify this match as added / deleted / modified / unchanged."""
        if self.element_a is None and self.element_b is not None:
            return "added"
        if self.element_a is not None and self.element_b is None:
            return "deleted"
        if self.element_a is not None and self.element_b is not None:
            if self.element_a.text.strip() == self.element_b.text.strip():
                return "unchanged"
            return "modified"
        return "unknown"

    def to_dict(self) -> dict:      #object -> python dict
        return {
            "status": self.status,      #status property
            "semantic_score": round(self.semantic_score, 4),
            "spatial_score": round(self.spatial_score, 4),
            "type_score": round(self.type_score, 4),
            "combined_score": round(self.combined_score, 4),
            "element_a": _element_summary(self.element_a),
            "element_b": _element_summary(self.element_b),
        }


def _element_summary(el: Element | None) -> dict | None:
    """Compact summary of an element for the match result."""
    if el is None:
        return None
    return {
        "id": el.id,
        "type": el.type.value.lower(),
        "text": el.text[:100],
        "page": el.page_number,
        "bbox": [el.bbox.x0, el.bbox.y0, el.bbox.x1, el.bbox.y1] if el.bbox else None,
    }


# ---------------------------------------------------------------------------
# Alignment Engine
# ---------------------------------------------------------------------------

class AlignmentEngine:
    """Aligns elements between two canonical Documents using multi-signal matching."""

    def __init__(self):
        settings = get_settings()
        logger.info("loading_embedding_model", model=settings.EMBEDDING_MODEL)
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

    def align(      # Main orchestration method
        self,
        doc_a: Document,
        doc_b: Document,
    ) -> list[MatchResult]:
        """Align elements between Doc A (base) and Doc B (revised).

        Args:
            doc_a: The base document (older revision).
            doc_b: The revised document (newer revision).

        Returns:
            List of MatchResults — each is an added/deleted/modified/unchanged.
        """
        logger.info(
            "aligning_documents",
            pid_a=doc_a.id,
            pid_b=doc_b.id,
            elements_a=doc_a.total_elements,
            elements_b=doc_b.total_elements,
        )

        elements_a = doc_a.all_elements()
        elements_b = doc_b.all_elements()

        if not elements_a and not elements_b:
            return []

        # All of B is added
        if not elements_a:
            return [
                MatchResult(element_a=None, element_b=eb)
                for eb in elements_b
            ]

        # All of A is deleted
        if not elements_b:
            return [
                MatchResult(element_a=ea, element_b=None)
                for ea in elements_a
            ]

        # --- Compute pairwise scores (all matrices stored, no recomputation later) ---
        score_matrices = self._compute_score_matrix(elements_a, elements_b)

        # --- Greedy matching: best match first, no double-claiming ---
        matches = self._greedy_match(elements_a, elements_b, score_matrices)

        # Log summary
        status_counts = {}
        for m in matches:   # matches -> (unchanged, modified, added, deleted,...)
            status_counts[m.status] = status_counts.get(m.status, 0) + 1

        logger.info(
            "alignment_complete",
            pid_a=doc_a.id,
            pid_b=doc_b.id,
            total_matches=len(matches),
            **status_counts,    # Unpack dict
        )

        return matches

    # -------------------------------------------------------------------
    # Score computation
    # -------------------------------------------------------------------

    def _compute_score_matrix(
        self,
        elements_a: list[Element],
        elements_b: list[Element],
    ) -> dict[str, np.ndarray]:
        """Compute all pairwise score matrices between element pairs.

        Returns:
            Dict with four (len_a x len_b) matrices:
                "semantic"  — cosine similarity of text embeddings
                "spatial"   — bounding box IoU
                "type"      — 1.0 if same type, else 0.0
                "combined"  — weighted sum of the above
        """
        # --- Semantic scores (batch embed + cosine similarity) ---
        texts_a = [el.text for el in elements_a]
        texts_b = [el.text for el in elements_b]

        # Batch Embedding
        emb_a = self.embedder.encode(texts_a, show_progress_bar=False)
        emb_b = self.embedder.encode(texts_b, show_progress_bar=False)

        # Normalize for cosine similarity
        emb_a_norm = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-9)
        emb_b_norm = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-9)
        semantic_matrix = emb_a_norm @ emb_b_norm.T     # cosine similarity matrix (@ -> matrix multi)
        semantic_matrix = np.clip(semantic_matrix, 0, 1)  # keep in 0–1

        len_a = len(elements_a)
        len_b = len(elements_b)

        # --- Spatial + Type scores (pairwise loop) ---
        spatial_matrix = np.zeros((len_a, len_b))
        type_matrix = np.zeros((len_a, len_b))

        for i, ea in enumerate(elements_a):
            for j, eb in enumerate(elements_b):
                # Spatial: IoU between bounding boxes
                if ea.bbox and eb.bbox:
                    spatial_matrix[i][j] = ea.bbox.iou(eb.bbox)

                # Type: 1.0 if same type, 0.0 if different
                if ea.type == eb.type:
                    type_matrix[i][j] = 1.0

        # --- Combined score ---
        combined = (
            W_SEMANTIC * semantic_matrix
            + W_SPATIAL * spatial_matrix
            + W_TYPE * type_matrix
        )

        return {
            "semantic": semantic_matrix,
            "spatial": spatial_matrix,
            "type": type_matrix,
            "combined": combined,
        }

    # -------------------------------------------------------------------
    # Greedy matching
    # -------------------------------------------------------------------

    def _greedy_match(
        self,
        elements_a: list[Element],
        elements_b: list[Element],
        score_matrices: dict[str, np.ndarray],
    ) -> list[MatchResult]:
        """Greedy best-first matching — highest score pair first, no repeats.

        Steps:
            1. Flatten the combined matrix into (score, i, j) triples
            2. Sort descending by score
            3. Greedily claim the best pair if neither i nor j is already claimed
            4. Look up individual scores directly from stored matrices
            5. Unclaimed A elements → deleted
            6. Unclaimed B elements → added
        """
        combined = score_matrices["combined"]
        semantic_matrix = score_matrices["semantic"]
        spatial_matrix = score_matrices["spatial"]
        type_matrix = score_matrices["type"]

        len_a, len_b = combined.shape
        claimed_a: set[int] = set()
        claimed_b: set[int] = set()
        matches: list[MatchResult] = []

        # Build sorted list of all (score, i, j)
        pairs = []
        for i in range(len_a):
            for j in range(len_b):
                pairs.append((combined[i][j], i, j))   # List of tuples of (combined score, row, col)

        pairs.sort(key=lambda x: x[0], reverse=True)   # highest score first

        # Greedy claim
        for score, i, j in pairs:
            if score < MATCH_THRESHOLD:
                break   # no more good matches
            if i in claimed_a or j in claimed_b:
                continue    # already matched

            # Direct lookup — no algebra, no recomputation
            match = MatchResult(
                element_a=elements_a[i],
                element_b=elements_b[j],
                semantic_score=round(float(semantic_matrix[i][j]), 4),
                spatial_score=round(float(spatial_matrix[i][j]), 4),
                type_score=float(type_matrix[i][j]),
            )
            matches.append(match)
            claimed_a.add(i)
            claimed_b.add(j)

        # Unclaimed A → deleted  (Old element never found a match)
        for i in range(len_a):
            if i not in claimed_a:
                matches.append(MatchResult(element_a=elements_a[i], element_b=None))

        # Unclaimed B → added  (New element never found a match)
        for j in range(len_b):
            if j not in claimed_b:
                matches.append(MatchResult(element_a=None, element_b=elements_b[j]))

        return matches
