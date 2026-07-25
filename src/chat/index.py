"""
Qdrant Indexer — embeds document elements and indexes them into Qdrant.

Pipeline:  Canonical Document → Embed each element → Upsert into Qdrant

Each element becomes a point in Qdrant with:
    - vector:   sentence-transformer embedding of the element text
    - payload:  pid, page, type, bbox, text, confidence
"""

from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer
from src.canonical.model import Document, Element
from src.config.settings import get_settings
from src.config.logging import get_logger

logger = get_logger(__name__)


class QdrantIndexer:
    """Indexes canonical Document elements into Qdrant for retrieval."""

    def __init__(self):
        settings = get_settings()

        # Connect to Qdrant Server
        self.client = QdrantClient( 
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )
        self.collection_name = settings.QDRANT_COLLECTION

        # Load embedding model
        logger.info("loading_embedding_model", model=settings.EMBEDDING_MODEL)
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.vector_size = self.embedder.get_sentence_embedding_dimension()

        # Ensure collection exists
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it doesn't exist."""
        collections = [c.name for c in self.client.get_collections().collections]

        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,   # compare vectors using Cosine Similarity
                ),
            )
            logger.info(
                "collection_created",
                collection=self.collection_name,
                vector_size=self.vector_size,
            )
        else:
            logger.info(
                "collection_exists",
                collection=self.collection_name,
            )

    def index_document(self, doc: Document) -> int:
        """Index all elements of a canonical Document into Qdrant.

        Args:
            doc: The canonical Document to index.

        Returns:
            Number of points indexed. (No. of elements stored in Qdrant)
        """
        logger.info(
            "indexing_document",
            pid=doc.id,
            elements=doc.total_elements,
        )

        elements = doc.all_elements()

        if not elements:
            logger.warning("no_elements_to_index", pid=doc.id)
            return 0

        # Filter out elements with empty text
        valid_elements = [el for el in elements if el.text.strip()]

        if not valid_elements:
            logger.warning("no_valid_elements", pid=doc.id)
            return 0

        # Embed all texts in one batch
        texts = [el.text for el in valid_elements]      # extract text
        embeddings = self.embedder.encode(texts, show_progress_bar=False)   # Generate batch embeddings

        # Build Qdrant points
        points = []
        for el, embedding in zip(valid_elements, embeddings):
            point = PointStruct(    # One row inside Qdrant DB (vector + metadata)
                id=uuid4().hex,
                vector=embedding.tolist(),    # Qdrant expects a plain Python list, so we convert
                payload=self._element_to_payload(el, doc.id),   # stores useful metadata
            )
            points.append(point)

        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(     # sends the batch to Qdrant
                collection_name=self.collection_name,
                points=batch,
            )

        logger.info(
            "indexing_complete",
            pid=doc.id,
            points_indexed=len(points),
        )

        return len(points)

    def delete_document(self, pid: str) -> None:
        """Remove all points for a given PID from the collection."""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[      # Every condition inside this list must be true
                    FieldCondition(     # Look at one field in the payload
                        key="pid",      # which field
                        match=MatchValue(value=pid),
                    )
                ]
            ),
        )
        logger.info("document_deleted", pid=pid)

    def search(
        self,
        query: str,
        pid: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search for similar elements by query text.

        Args:
            query: The search query.
            pid: Optional PID filter — search only within one document.
            limit: Max number of results.

        Returns:
            List of dicts with score, text, and metadata.
        """
        query_embedding = self.embedder.encode(query).tolist()  # Embeds query

        # Optional PID filter
        query_filter = None     # By default Search every document
        if pid:     # if the user only want one pid
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="pid",
                        match=MatchValue(value=pid),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        return [    # Python dict
            {
                "score": hit.score,       # cosine similarity score
                "text": hit.payload.get("text", ""),
                "pid": hit.payload.get("pid", ""),
                "page": hit.payload.get("page", 0),
                "type": hit.payload.get("type", ""),
                "bbox": hit.payload.get("bbox", None),    # Location on page
                "confidence": hit.payload.get("confidence", 0.0),    # OCR confidence
            }
            for hit in results.points
        ]

    def get_collection_info(self) -> dict:
        """Get info about the current collection."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "vector_size": info.config.params.vectors.size,
        }

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    @staticmethod    # This function doesn't use anything from the class
    def _element_to_payload(el: Element, pid: str) -> dict:     # Create payload for every element that gets stored in Qdrant
        """Convert an Element to a Qdrant payload dict."""
        payload = {
            "pid": pid,
            "page": el.page_number,
            "type": el.type.value.lower(),
            "text": el.text,
            "confidence": el.confidence,
        }

        # Flatten bbox into payload
        if el.bbox:
            payload["bbox"] = [el.bbox.x0, el.bbox.y0, el.bbox.x1, el.bbox.y1]

        # Include element metadata
        if el.metadata:
            payload["element_metadata"] = el.metadata

        return payload
