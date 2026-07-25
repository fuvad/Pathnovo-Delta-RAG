"""
Evaluation Metrics — Precision, Recall, F1 for delta; Accuracy, Groundedness for chat.

Delta Metrics:
    Compares the engine's detected changes against ground truth.
    A detected change is "correct" if it matches a ground truth entry on:
        - Change type (modified/added/removed)
        - Fuzzy text match (old and/or new text)

Chat Metrics:
    - Answer Accuracy: do expected keywords appear in the answer?
    - Groundedness: does the answer cite sources?
    - Citation Accuracy: are citations formatted correctly?
"""

from difflib import SequenceMatcher
from src.config.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Text matching helper
# ---------------------------------------------------------------------------

def fuzzy_match(text_a: str, text_b: str, threshold: float = 0.6) -> bool:
    """Check if two text strings are similar enough to be considered a match.

    Uses SequenceMatcher ratio for fuzzy matching. This handles minor
    whitespace/formatting differences between ground truth and extracted text.
    """
    if not text_a or not text_b:
        return False

    a = text_a.strip().lower()
    b = text_b.strip().lower()

    # Exact match
    if a == b:
        return True

    # Substring containment
    if a in b or b in a:
        return True

    # Fuzzy ratio
    ratio = SequenceMatcher(None, a, b).ratio()
    return ratio >= threshold


# ---------------------------------------------------------------------------
# Delta Metrics
# ---------------------------------------------------------------------------

class DeltaMetrics:
    """Compute Precision, Recall, F1 for delta detection."""

    def __init__(
        self,
        predicted: list[dict],
        ground_truth_modifications: list[dict],
        ground_truth_additions: list[dict],
        ground_truth_removals: list[dict],
    ):
        self.predicted = predicted
        self.gt_modifications = ground_truth_modifications
        self.gt_additions = ground_truth_additions
        self.gt_removals = ground_truth_removals

        # Separate predicted by change type
        self.pred_modified = [p for p in predicted if p.get("change", "").lower() == "modified"]
        self.pred_added = [p for p in predicted if p.get("change", "").lower() == "added"]
        self.pred_removed = [p for p in predicted if p.get("change", "").lower() == "removed"]

    def compute(self) -> dict:
        """Compute per-category and overall metrics."""

        # --- Modifications ---
        mod_tp, mod_fp, mod_fn = self._match_modifications()

        # --- Additions ---
        add_tp, add_fp, add_fn = self._match_additions()

        # --- Removals ---
        rem_tp, rem_fp, rem_fn = self._match_removals()

        # --- Overall ---
        total_tp = mod_tp + add_tp + rem_tp
        total_fp = mod_fp + add_fp + rem_fp
        total_fn = mod_fn + add_fn + rem_fn

        overall_p, overall_r, overall_f1 = self._prf(total_tp, total_fp, total_fn)

        results = {
            "modifications": {
                "true_positives": mod_tp,
                "false_positives": mod_fp,
                "false_negatives": mod_fn,
                **dict(zip(["precision", "recall", "f1"], self._prf(mod_tp, mod_fp, mod_fn))),
            },
            "additions": {
                "true_positives": add_tp,
                "false_positives": add_fp,
                "false_negatives": add_fn,
                **dict(zip(["precision", "recall", "f1"], self._prf(add_tp, add_fp, add_fn))),
            },
            "removals": {
                "true_positives": rem_tp,
                "false_positives": rem_fp,
                "false_negatives": rem_fn,
                **dict(zip(["precision", "recall", "f1"], self._prf(rem_tp, rem_fp, rem_fn))),
            },
            "overall": {
                "true_positives": total_tp,
                "false_positives": total_fp,
                "false_negatives": total_fn,
                "precision": overall_p,
                "recall": overall_r,
                "f1": overall_f1,
            },
        }

        return results

    def _match_modifications(self) -> tuple[int, int, int]:
        """Match predicted modifications against ground truth."""
        matched_gt = set()
        tp = 0

        for pred in self.pred_modified:
            pred_old = pred.get("old", "")
            pred_new = pred.get("new", "")

            for i, gt in enumerate(self.gt_modifications):
                if i in matched_gt:
                    continue

                gt_old = gt.get("old", "")
                gt_new = gt.get("new", "")

                # Match if old and new texts fuzzy-match
                old_match = fuzzy_match(pred_old, gt_old)
                new_match = fuzzy_match(pred_new, gt_new)

                if old_match or new_match:
                    tp += 1
                    matched_gt.add(i)
                    break

        fp = len(self.pred_modified) - tp
        fn = len(self.gt_modifications) - len(matched_gt)

        return tp, fp, fn

    def _match_additions(self) -> tuple[int, int, int]:
        """Match predicted additions against ground truth."""
        matched_gt = set()
        tp = 0

        for pred in self.pred_added:
            pred_text = pred.get("new", "")

            for i, gt in enumerate(self.gt_additions):
                if i in matched_gt:
                    continue

                gt_text = gt.get("text", "")
                if fuzzy_match(pred_text, gt_text):
                    tp += 1
                    matched_gt.add(i)
                    break

        fp = len(self.pred_added) - tp
        fn = len(self.gt_additions) - len(matched_gt)

        return tp, fp, fn

    def _match_removals(self) -> tuple[int, int, int]:
        """Match predicted removals against ground truth."""
        matched_gt = set()
        tp = 0

        for pred in self.pred_removed:
            pred_text = pred.get("old", "")

            for i, gt in enumerate(self.gt_removals):
                if i in matched_gt:
                    continue

                gt_text = gt.get("text", "")
                if fuzzy_match(pred_text, gt_text):
                    tp += 1
                    matched_gt.add(i)
                    break

        fp = len(self.pred_removed) - tp
        fn = len(self.gt_removals) - len(matched_gt)

        return tp, fp, fn

    @staticmethod
    def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        """Compute Precision, Recall, F1."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return round(precision, 4), round(recall, 4), round(f1, 4)


# ---------------------------------------------------------------------------
# Chat Metrics
# ---------------------------------------------------------------------------

class ChatMetrics:
    """Compute Accuracy, Groundedness, Citation Accuracy for chat answers."""

    def __init__(self, results: list[dict]):
        """
        Args:
            results: List of dicts with:
                - question
                - answer (from LLM)
                - expected_answer
                - expected_citations (list of keywords that should appear)
        """
        self.results = results

    def compute(self) -> dict:
        """Compute all chat metrics."""
        if not self.results:
            return {"accuracy": 0.0, "groundedness": 0.0, "citation_accuracy": 0.0}

        accuracy_scores = []
        groundedness_scores = []
        citation_scores = []

        per_question = []

        for r in self.results:
            answer = r.get("answer", "").lower()
            expected = r.get("expected_answer", "").lower()
            expected_citations = r.get("expected_citations", [])

            # --- Accuracy: keyword overlap ---
            acc = self._keyword_accuracy(answer, expected)
            accuracy_scores.append(acc)

            # --- Groundedness: does it cite sources? ---
            grounded = self._check_groundedness(r.get("answer", ""))
            groundedness_scores.append(grounded)

            # --- Citation accuracy: expected keywords in answer ---
            cit_acc = self._citation_accuracy(answer, expected_citations)
            citation_scores.append(cit_acc)

            per_question.append({
                "question": r.get("question", ""),
                "accuracy": round(acc, 4),
                "groundedness": grounded,
                "citation_accuracy": round(cit_acc, 4),
            })

        return {
            "accuracy": round(sum(accuracy_scores) / len(accuracy_scores), 4),
            "groundedness": round(sum(groundedness_scores) / len(groundedness_scores), 4),
            "citation_accuracy": round(sum(citation_scores) / len(citation_scores), 4),
            "per_question": per_question,
        }

    @staticmethod
    def _keyword_accuracy(answer: str, expected: str) -> float:
        """What fraction of expected keywords appear in the answer?"""
        expected_words = set(expected.split())
        # Filter out common stop words
        stop_words = {"the", "a", "an", "is", "was", "to", "from", "in", "of", "and", "or", "for"}
        keywords = expected_words - stop_words
        if not keywords:
            return 1.0

        hits = sum(1 for kw in keywords if kw in answer)
        return hits / len(keywords)

    @staticmethod
    def _check_groundedness(answer: str) -> float:
        """Check if the answer contains citation markers like [PID: ...]."""
        # Check for citation patterns
        if "[PID:" in answer or "[pid:" in answer.lower():
            return 1.0
        # Check for "I couldn't find evidence" — honest refusal is grounded
        if "couldn't find evidence" in answer.lower():
            return 1.0
        # Partial credit if it references page numbers or element types
        if "page" in answer.lower() and any(t in answer.lower() for t in ["equipment", "instrument", "note", "valve"]):
            return 0.5
        return 0.0

    @staticmethod
    def _citation_accuracy(answer: str, expected_citations: list[str]) -> float:
        """What fraction of expected citation keywords appear in the answer?"""
        if not expected_citations:
            return 1.0

        hits = sum(1 for c in expected_citations if c.lower() in answer)
        return hits / len(expected_citations)
