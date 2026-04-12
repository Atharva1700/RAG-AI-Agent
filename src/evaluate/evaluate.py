"""
evaluate.py
Phase 4: RAG quality evaluation pipeline.

Resume connection (Uber):
'Improved retrieval precision by 35% and accelerated
pipeline evaluation cycles by 30%'
This file measures those improvements.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from src.pipeline.rag_pipeline import (
    load_vectorstore, build_llm, build_rag_chain, query_rag
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ── Test questions with expected source categories ────────────────────
# expected_categories: which ChromaDB categories SHOULD be retrieved
EVAL_QUESTIONS = [
    {
        "question": "What is the cancellation fee for Uber rides?",
        "expected_categories": ["billing", "driver_policy"],
        "key_terms": ["cancellation", "fee", "driver"]
    },
    {
        "question": "How often do Uber drivers get paid?",
        "expected_categories": ["driver_policy"],
        "key_terms": ["payment", "weekly", "tuesday", "direct deposit"]
    },
    {
        "question": "What safety features does Uber have in the app?",
        "expected_categories": ["safety"],
        "key_terms": ["safety", "emergency", "ridecheck", "gps"]
    },
    {
        "question": "How long does Uber keep my personal data?",
        "expected_categories": ["privacy"],
        "key_terms": ["retention", "data", "deleted", "years"]
    },
    {
        "question": "What rating must Uber Eats drivers maintain?",
        "expected_categories": ["delivery"],
        "key_terms": ["rating", "4.5", "delivery", "completion"]
    },
    {
        "question": "Does Uber provide insurance coverage for drivers?",
        "expected_categories": ["safety", "driver_policy"],
        "key_terms": ["insurance", "liability", "coverage", "million"]
    },
]


def compute_retrieval_precision(sources: list[dict], expected_categories: list[str]) -> float:
    """
    Precision = relevant retrieved chunks / total retrieved chunks.
    A chunk is relevant if its category matches expected_categories.
    """
    if not sources:
        return 0.0
    relevant = sum(
        1 for s in sources
        if s.get("category") in expected_categories
    )
    return round(relevant / len(sources), 3)


def compute_groundedness(answer: str, sources: list[dict]) -> float:
    """
    Groundedness = fraction of answer words found in retrieved context.
    Measures whether the answer is grounded in the retrieved chunks
    rather than hallucinated from model weights.
    """
    if not answer or not sources:
        return 0.0

    # Combine all retrieved context into one string
    context = " ".join(s.get("snippet", "") for s in sources).lower()
    stop_words = {'the','a','an','is','in','to','of','and','or','for',
                  'with','that','this','it','are','be','was','by','at'}

    answer_words = [
        w.lower().strip(".,?!\"'") for w in answer.split()
        if w.lower() not in stop_words and len(w) > 3
    ]
    if not answer_words:
        return 0.0

    grounded = sum(1 for w in answer_words if w in context)
    return round(grounded / len(answer_words), 3)


def compute_answer_relevance(answer: str, question: str, key_terms: list[str]) -> float:
    """
    Answer relevance = fraction of key terms from the question
    that appear in the answer.
    """
    if not answer:
        return 0.0
    answer_lower = answer.lower()
    found = sum(1 for t in key_terms if t.lower() in answer_lower)
    return round(found / len(key_terms), 3) if key_terms else 0.0


def run_evaluation() -> dict:
    """Run all evaluation questions and compute metrics."""
    logger.info("Starting RAG evaluation pipeline...")

    vs    = load_vectorstore()
    llm   = build_llm()
    chain = build_rag_chain(vs, llm)

    results = []
    for item in EVAL_QUESTIONS:
        logger.info(f"Evaluating: {item['question'][:50]}...")
        result = query_rag(chain, item["question"], verbose=False)

        precision  = compute_retrieval_precision(
            result["sources"], item["expected_categories"]
        )
        groundedness = compute_groundedness(
            result["answer"], result["sources"]
        )
        relevance = compute_answer_relevance(
            result["answer"], item["question"], item["key_terms"]
        )

        eval_result = {
            "question":            item["question"],
            "answer":              result["answer"],
            "latency_ms":          result["latency_ms"],
            "retrieval_precision": precision,
            "groundedness":        groundedness,
            "answer_relevance":    relevance,
            "sources":             [s["source"] for s in result["sources"]]
        }
        results.append(eval_result)
        logger.info(
            f"  Precision={precision:.2f} | "
            f"Groundedness={groundedness:.2f} | "
            f"Relevance={relevance:.2f} | "
            f"Latency={result['latency_ms']}ms"
        )

    # ── Aggregate report ─────────────────────────────────────────────
    avg_precision    = round(sum(r["retrieval_precision"] for r in results) / len(results), 3)
    avg_groundedness = round(sum(r["groundedness"]        for r in results) / len(results), 3)
    avg_relevance    = round(sum(r["answer_relevance"]    for r in results) / len(results), 3)
    avg_latency      = round(sum(r["latency_ms"]          for r in results) / len(results), 1)

    report = {
        "timestamp":          datetime.now().isoformat(),
        "num_questions":      len(results),
        "aggregate_metrics":  {
            "avg_retrieval_precision": avg_precision,
            "avg_groundedness":        avg_groundedness,
            "avg_answer_relevance":    avg_relevance,
            "avg_latency_ms":          avg_latency
        },
        "individual_results": results
    }

    # Save report
    report_path = "logs/evaluation_report.json"
    Path("logs").mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  RAG EVALUATION PIPELINE")
    print("="*55)

    report = run_evaluation()
    agg    = report["aggregate_metrics"]

    print("\n  AGGREGATE METRICS")
    print("  -----------------")
    print(f"  Avg Retrieval Precision : {agg['avg_retrieval_precision']:.1%}")
    print(f"  Avg Groundedness        : {agg['avg_groundedness']:.1%}")
    print(f"  Avg Answer Relevance    : {agg['avg_answer_relevance']:.1%}")
    print(f"  Avg Latency             : {agg['avg_latency_ms']} ms")
    print(f"\n  Report saved: logs/evaluation_report.json")
    print("="*55)