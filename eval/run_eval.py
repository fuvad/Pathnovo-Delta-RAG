"""
Evaluation Runner — runs the full pipeline on test pairs and scores results.

Pipeline per pair:
    1. Ingest both PDFs → Canonical Documents
    2. Run Delta Engine → detected changes
    3. Score delta against ground truth → P/R/F1
    4. Index documents + delta into Qdrant
    5. Run chat questions → answers
    6. Score chat answers → Accuracy/Groundedness

Output:
    Pretty-printed scorecard + JSON results file
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest.pdf_native import NativePDFAdapter
from src.delta.engine import DeltaEngine
from src.delta.report import generate_report
from src.chat.answer import GroundedChat
from src.chat.index import QdrantIndexer
from src.observability.tracing import RequestTrace
from src.config.logging import setup_logging, bind_request_id, get_logger
from eval.metrics import DeltaMetrics, ChatMetrics

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EVAL_DIR = Path(__file__).parent
DATASETS_DIR = EVAL_DIR / "datasets"
RESULTS_DIR = EVAL_DIR / "results"


def load_ground_truth() -> dict:
    """Load the ground truth dataset."""
    gt_path = DATASETS_DIR / "ground_truth.json"
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_delta_eval(pair: dict, trace: RequestTrace) -> dict:
    """Run delta evaluation for one document pair.

    Returns:
        Dict with delta metrics and detected changes.
    """
    pdf_a = Path(pair["pdf_a"])
    pdf_b = Path(pair["pdf_b"])
    pid_a = pair["pid_a"]
    pid_b = pair["pid_b"]

    # --- Ingest ---
    adapter = NativePDFAdapter()

    with trace.span("ingest_a", pid=pid_a):
        doc_a = adapter.ingest(pdf_a, pid_a)

    with trace.span("ingest_b", pid=pid_b):
        doc_b = adapter.ingest(pdf_b, pid_b)

    logger.info(
        "ingestion_complete",
        elements_a=doc_a.total_elements,
        elements_b=doc_b.total_elements,
    )

    # --- Delta ---
    engine = DeltaEngine()

    with trace.span("delta"):
        deltas = engine.compute_delta(doc_a, doc_b)

    # Generate report
    with trace.span("report"):
        report = generate_report(pid_a, pid_b, deltas)

    # Convert deltas to dicts for scoring
    predicted = [d.to_dict() for d in deltas if d.change != "unchanged"]

    # --- Score ---
    gt = pair["ground_truth"]
    delta_metrics = DeltaMetrics(
        predicted=predicted,
        ground_truth_modifications=gt["expected_modifications"],
        ground_truth_additions=gt["expected_additions"],
        ground_truth_removals=gt["expected_removals"],
    )

    scores = delta_metrics.compute()

    return {
        "pair_id": pair["pair_id"],
        "name": pair["name"],
        "elements_a": doc_a.total_elements,
        "elements_b": doc_b.total_elements,
        "predicted_changes": len(predicted),
        "delta_scores": scores,
        "doc_a": doc_a,
        "doc_b": doc_b,
        "deltas": deltas,
        "report": report,
    }


def run_chat_eval(
    pair: dict,
    doc_a,
    doc_b,
    deltas: list,
    trace: RequestTrace,
) -> dict:
    """Run chat evaluation for one document pair.

    Returns:
        Dict with chat metrics and per-question results.
    """
    pid_a = pair["pid_a"]
    pid_b = pair["pid_b"]
    questions = pair.get("questions", [])

    if not questions:
        return {"chat_scores": {}, "answers": []}

    # --- Index documents ---
    indexer = QdrantIndexer()

    with trace.span("index_a"):
        indexer.index_document(doc_a)

    with trace.span("index_b"):
        indexer.index_document(doc_b)

    # --- Setup chat ---
    chat = GroundedChat()
    chat.load_delta(pid_a, pid_b, deltas)

    # --- Ask questions ---
    answers = []
    for q in questions:
        with trace.span("chat_question", question_id=q["id"]):
            try:
                result = chat.ask(q["question"])
                answers.append({
                    "question": q["question"],
                    "answer": result["answer"],
                    "expected_answer": q["expected_answer"],
                    "expected_citations": q["expected_citations"],
                    "usage": result["usage"],
                })
            except Exception as e:
                logger.error("chat_question_failed", question=q["question"], error=str(e))
                answers.append({
                    "question": q["question"],
                    "answer": f"ERROR: {str(e)}",
                    "expected_answer": q["expected_answer"],
                    "expected_citations": q["expected_citations"],
                })

    # --- Score ---
    chat_metrics = ChatMetrics(answers)
    scores = chat_metrics.compute()

    return {
        "chat_scores": scores,
        "answers": answers,
    }


def print_scorecard(delta_result: dict, chat_result: dict) -> str:
    """Generate the pretty-printed scorecard."""
    lines = []
    lines.append("")
    lines.append("=" * 50)
    lines.append(f"  EVALUATION SCORECARD")
    lines.append(f"  {delta_result['name']}")
    lines.append("=" * 50)

    lines.append("")
    lines.append("  DELTA DETECTION")
    lines.append("  " + "-" * 40)

    ds = delta_result["delta_scores"]

    for category in ["modifications", "additions", "removals"]:
        cat_scores = ds.get(category, {})
        lines.append(f"  {category.upper()}")
        lines.append(f"    Precision:  {cat_scores.get('precision', 0):.4f}")
        lines.append(f"    Recall:     {cat_scores.get('recall', 0):.4f}")
        lines.append(f"    F1:         {cat_scores.get('f1', 0):.4f}")
        lines.append(f"    TP: {cat_scores.get('true_positives', 0)}  "
                      f"FP: {cat_scores.get('false_positives', 0)}  "
                      f"FN: {cat_scores.get('false_negatives', 0)}")
        lines.append("")

    overall = ds.get("overall", {})
    lines.append("  OVERALL")
    lines.append(f"    Precision:  {overall.get('precision', 0):.4f}")
    lines.append(f"    Recall:     {overall.get('recall', 0):.4f}")
    lines.append(f"    F1:         {overall.get('f1', 0):.4f}")

    lines.append("")
    lines.append("=" * 50)
    lines.append("  CHAT (RAG)")
    lines.append("  " + "-" * 40)

    cs = chat_result.get("chat_scores", {})
    if cs:
        lines.append(f"    Accuracy:           {cs.get('accuracy', 0):.4f}")
        lines.append(f"    Groundedness:       {cs.get('groundedness', 0):.4f}")
        lines.append(f"    Citation Accuracy:  {cs.get('citation_accuracy', 0):.4f}")
    else:
        lines.append("    (chat eval skipped)")

    lines.append("")
    lines.append("=" * 50)

    scorecard = "\n".join(lines)
    return scorecard


def run_eval(skip_chat: bool = False) -> None:
    """Run the full evaluation pipeline."""
    setup_logging()
    trace = RequestTrace()
    bind_request_id(trace.request_id)

    logger.info("eval_started")

    # Load ground truth
    gt = load_ground_truth()
    pairs = gt["pairs"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for pair in pairs:
        logger.info("evaluating_pair", pair_id=pair["pair_id"], name=pair["name"])

        # --- Delta eval ---
        delta_result = run_delta_eval(pair, trace)

        # --- Chat eval (optional) ---
        chat_result = {"chat_scores": {}, "answers": []}
        if not skip_chat:
            try:
                chat_result = run_chat_eval(
                    pair=pair,
                    doc_a=delta_result["doc_a"],
                    doc_b=delta_result["doc_b"],
                    deltas=delta_result["deltas"],
                    trace=trace,
                )
            except Exception as e:
                logger.error("chat_eval_failed", error=str(e))

        # --- Print scorecard ---
        scorecard = print_scorecard(delta_result, chat_result)
        print(scorecard)

        # Store results (without non-serializable objects)
        pair_result = {
            "pair_id": pair["pair_id"],
            "name": pair["name"],
            "elements_a": delta_result["elements_a"],
            "elements_b": delta_result["elements_b"],
            "predicted_changes": delta_result["predicted_changes"],
            "delta_scores": delta_result["delta_scores"],
            "chat_scores": chat_result.get("chat_scores", {}),
        }
        all_results.append(pair_result)

    # --- Save results ---
    trace.finish()
    trace.save()

    results_path = RESULTS_DIR / f"eval_{trace.request_id}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": trace.request_id,
            "results": all_results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {results_path}")
    print(f"Trace saved to: logs/trace_{trace.request_id}.json")


if __name__ == "__main__":
    # Pass --skip-chat to skip the chat evaluation (runs only delta)
    skip_chat = "--skip-chat" in sys.argv
    run_eval(skip_chat=skip_chat)
