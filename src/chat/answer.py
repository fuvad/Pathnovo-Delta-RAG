"""
Grounded Answer — retrieves context, calls LLM, enforces citations.

Every answer is grounded to specific source content:
    - PID + page + element for document content
    - Delta report entry for change-related questions

If no supporting evidence is found, the system says so honestly.
"""

import json
from src.chat.index import QdrantIndexer
from src.chat.llm import LLMClient
from src.delta.engine import DeltaEntry
from src.config.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# System prompt — the grounding contract
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a document analysis assistant. You answer questions about engineering documents and their changes.

RULES — follow these strictly:
1. Use ONLY the retrieved context below to answer. Do NOT use outside knowledge.
2. Always cite your sources using this format: [PID: <pid>, Page: <page>, Type: <type>]
3. For change-related questions, cite the delta report entry.
4. If the context does not contain enough information to answer, say: "I couldn't find evidence for this in the provided documents."
5. Be precise and factual. Prefer short, direct answers over long explanations.
6. When listing changes, include the old and new values.
"""


# ---------------------------------------------------------------------------
# Grounded Chat
# ---------------------------------------------------------------------------

class GroundedChat:
    """Chat interface grounded in two documents and their delta report."""

    def __init__(self):
        self.indexer = QdrantIndexer()
        self.llm = LLMClient()
        self.delta_entries: list[DeltaEntry] = []
        self.pid_a: str = ""
        self.pid_b: str = ""

    def load_delta(self, pid_a: str, pid_b: str, deltas: list[DeltaEntry]) -> None:
        """Store delta entries for context retrieval."""
        self.pid_a = pid_a
        self.pid_b = pid_b
        self.delta_entries = deltas

        # Index delta report entries into Qdrant for semantic search
        self._index_delta_entries(deltas)

        logger.info(
            "delta_loaded_for_chat",
            pid_a=pid_a,
            pid_b=pid_b,
            delta_count=len(deltas),
        )

    def ask(self, question: str, top_k: int = 10) -> dict:
        """Answer a question grounded in the documents and delta report.

        Args:
            question: The user's question.
            top_k: Number of context chunks to retrieve.

        Returns:
            Dict with: answer, citations, usage, retrieved_context.
        """
        logger.info("chat_question", question=question)

        # Step 1: Retrieve relevant context from Qdrant
        retrieved = self.indexer.search(query=question, limit=top_k)

        # Step 2: Find relevant delta entries
        delta_context = self._find_relevant_deltas(question)

        # Step 3: Build the context string for the LLM
        context = self._build_context(retrieved, delta_context)

        # Step 4: Build the user prompt
        user_prompt = f"""RETRIEVED CONTEXT:
{context}

QUESTION: {question}

Answer the question using ONLY the context above. Cite sources with [PID: <pid>, Page: <page>, Type: <type>]."""

        # Step 5: Call LLM
        llm_response = self.llm.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        result = {
            "question": question,
            "answer": llm_response["answer"],
            "model": llm_response["model"],
            "provider": llm_response["provider"],
            "usage": llm_response["usage"],
            "retrieved_count": len(retrieved),
            "delta_context_count": len(delta_context),
        }

        logger.info(
            "chat_answer",
            question=question[:80],
            answer_length=len(llm_response["answer"]),
            retrieved=len(retrieved),
            **llm_response["usage"],
        )

        return result

    # -------------------------------------------------------------------
    # Context building
    # -------------------------------------------------------------------

    def _build_context(
        self,
        retrieved: list[dict],
        delta_context: list[dict],
    ) -> str:
        """Build the context string from retrieved chunks and delta entries."""
        parts: list[str] = []

        # Document content from Qdrant
        if retrieved:
            parts.append("=== DOCUMENT CONTENT ===")
            for i, hit in enumerate(retrieved, 1):
                bbox_str = ""
                if hit.get("bbox"):
                    bbox_str = f", BBox: {hit['bbox']}"
                parts.append(
                    f"[{i}] PID: {hit['pid']}, Page: {hit['page']}, "
                    f"Type: {hit['type']}{bbox_str}\n"
                    f"    Text: {hit['text']}"
                )

        # Delta report entries
        if delta_context:
            parts.append("\n=== DELTA REPORT (Changes between revisions) ===")
            for i, entry in enumerate(delta_context, 1):
                change_str = f"[Δ{i}] {entry['change']} — {entry['type']} on Page {entry['page']}"
                if entry.get("old"):
                    change_str += f"\n    Old: {entry['old']}"
                if entry.get("new"):
                    change_str += f"\n    New: {entry['new']}"
                if entry.get("reason"):
                    change_str += f"\n    Reason: {entry['reason']}"
                change_str += f"\n    Confidence: {entry['confidence']}"
                parts.append(change_str)

        if not parts:
            return "No relevant context found in the documents or delta report."

        return "\n\n".join(parts)

    def _find_relevant_deltas(self, question: str) -> list[dict]:
        """Find delta entries relevant to the question using keyword matching.

        Simple keyword approach — the delta report is typically small enough
        that we can include all changes for change-related questions, or
        filter by type/page keywords.
        """
        q_lower = question.lower()

        # If asking about changes broadly, return all non-unchanged entries
        change_keywords = ["change", "differ", "modif", "add", "remov", "delet", "new", "delta"]
        is_change_question = any(kw in q_lower for kw in change_keywords)

        relevant: list[dict] = []

        for d in self.delta_entries:
            if d.change == "unchanged":
                continue

            entry_dict = d.to_dict()

            # Include if it's a change question
            if is_change_question:
                relevant.append(entry_dict)
                continue

            # Include if question mentions the element type
            if d.element_type in q_lower:
                relevant.append(entry_dict)
                continue

            # Include if question mentions text content
            if d.old_text and d.old_text.lower() in q_lower:
                relevant.append(entry_dict)
                continue
            if d.new_text and d.new_text.lower() in q_lower:
                relevant.append(entry_dict)
                continue

            # Include if question mentions the page
            if f"page {d.page}" in q_lower:
                relevant.append(entry_dict)
                continue

        return relevant

    # -------------------------------------------------------------------
    # Delta indexing into Qdrant
    # -------------------------------------------------------------------

    def _index_delta_entries(self, deltas: list[DeltaEntry]) -> None:
        """Index delta report entries into Qdrant so they're searchable.

        Each non-unchanged delta entry becomes a searchable chunk with
        source="delta_report" in the metadata.
        """
        from uuid import uuid4
        from qdrant_client.models import PointStruct

        points = []
        changes = [d for d in deltas if d.change != "unchanged"]

        if not changes:
            return

        # Build text representations for embedding
        texts = []
        for d in changes:
            text = f"{d.change} {d.element_type}: "
            if d.old_text and d.new_text:
                text += f"changed from '{d.old_text}' to '{d.new_text}'"
            elif d.new_text:
                text += f"added '{d.new_text}'"
            elif d.old_text:
                text += f"removed '{d.old_text}'"
            if d.reason:
                text += f". {d.reason}"
            texts.append(text)

        # Embed all at once
        embeddings = self.indexer.embedder.encode(texts, show_progress_bar=False)

        for text, embedding, d in zip(texts, embeddings, changes):
            point = PointStruct(
                id=uuid4().hex,
                vector=embedding.tolist(),
                payload={
                    "pid": f"{self.pid_a}_vs_{self.pid_b}",
                    "page": d.page,
                    "type": d.element_type,
                    "text": text,
                    "confidence": d.confidence,
                    "source": "delta_report",
                    "change": d.change,
                    "old_text": d.old_text,
                    "new_text": d.new_text,
                    "reason": d.reason,
                },
            )
            points.append(point)

        # Upsert
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.indexer.client.upsert(
                collection_name=self.indexer.collection_name,
                points=points[i : i + batch_size],
            )

        logger.info(
            "delta_entries_indexed",
            count=len(points),
        )
