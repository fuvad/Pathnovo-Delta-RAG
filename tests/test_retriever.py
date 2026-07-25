"""Unit tests for Retriever and Grounded Chat context construction."""

from unittest.mock import MagicMock
from src.chat.answer import GroundedChat, SYSTEM_PROMPT
from src.delta.engine import DeltaEntry


def test_system_prompt_grounding_rules():
    """Verify system prompt enforces citation and evidence rules."""
    assert "Use ONLY the retrieved context" in SYSTEM_PROMPT
    assert "I couldn't find evidence" in SYSTEM_PROMPT
    assert "[PID: <pid>, Page: <page>, Type: <type>]" in SYSTEM_PROMPT


def test_chat_context_building():
    """Test context string building from Qdrant hits and Delta entries."""
    chat = GroundedChat.__new__(GroundedChat)
    chat.pid_a = "doc_a"
    chat.pid_b = "doc_b"

    retrieved = [
        {
            "pid": "doc_a",
            "page": 1,
            "type": "equipment",
            "text": "26-KA-902",
            "bbox": [0.1, 0.1, 0.2, 0.2],
        }
    ]

    delta_entry = DeltaEntry(
        change="modified",
        page=1,
        element_type="equipment",
        old_text="26-KA-902",
        new_text="26-KA-901",
        confidence=0.96,
        reason="Tag changed",
    )

    context = chat._build_context(retrieved, [delta_entry.to_dict()])

    assert "=== DOCUMENT CONTENT ===" in context
    assert "26-KA-902" in context
    assert "=== DELTA REPORT" in context
    assert "26-KA-901" in context


def test_find_relevant_deltas():
    """Test vector-based delta entry retrieval for chat queries."""
    chat = GroundedChat.__new__(GroundedChat)
    chat.pid_a = "doc_a"
    chat.pid_b = "doc_b"
    chat.indexer = MagicMock()

    chat.indexer.search.return_value = [
        {
            "change": "modified",
            "page": 1,
            "type": "equipment",
            "old": "26-KA-902",
            "new": "26-KA-901",
            "reason": "Equipment changed",
            "confidence": 0.96,
        }
    ]

    results = chat._find_relevant_deltas("What changed in the equipment tag?", top_k=15)
    assert len(results) == 1
    assert results[0]["old"] == "26-KA-902"
    assert results[0]["new"] == "26-KA-901"
    chat.indexer.search.assert_called_once_with(
        query="What changed in the equipment tag?",
        source="delta_report",
        limit=15,
    )
