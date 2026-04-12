"""
monitor.py
Phase 5: Production monitoring and quality alerting.

Logs every RAG query with metrics to SQLite.
Detects retrieval quality degradation.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

DB_PATH = "./logs/rag_monitor.db"

# Quality thresholds - alert if metrics drop below these
THRESHOLDS = {
    "retrieval_precision": 0.50,   # alert if below 50%
    "groundedness":        0.40,   # alert if below 40%
    "answer_relevance":    0.30,   # alert if below 30%
    "latency_ms":          5000.0  # alert if above 5 seconds
}


def init_db() -> None:
    """Create the monitoring database and table if they do not exist."""
    Path("logs").mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rag_queries (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp           TEXT    NOT NULL,
            question            TEXT    NOT NULL,
            answer              TEXT,
            sources             TEXT,
            latency_ms          REAL,
            retrieval_precision REAL,
            groundedness        REAL,
            answer_relevance    REAL,
            alert_triggered     INTEGER DEFAULT 0,
            alert_reason        TEXT
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"Monitoring DB initialized: {DB_PATH}")


def log_query(
    question:            str,
    answer:              str,
    sources:             list[dict],
    latency_ms:          float,
    retrieval_precision: float = 0.0,
    groundedness:        float = 0.0,
    answer_relevance:    float = 0.0
) -> int:
    """
    Log a RAG query + metrics to SQLite.
    Returns the row ID of the inserted record.
    """
    # Check if any metric violates threshold
    alert_triggered = False
    alert_reasons   = []

    if retrieval_precision < THRESHOLDS["retrieval_precision"]:
        alert_reasons.append(f"low_precision:{retrieval_precision:.2f}")
        alert_triggered = True
    if groundedness < THRESHOLDS["groundedness"]:
        alert_reasons.append(f"low_groundedness:{groundedness:.2f}")
        alert_triggered = True
    if answer_relevance < THRESHOLDS["answer_relevance"]:
        alert_reasons.append(f"low_relevance:{answer_relevance:.2f}")
        alert_triggered = True
    if latency_ms > THRESHOLDS["latency_ms"]:
        alert_reasons.append(f"high_latency:{latency_ms:.0f}ms")
        alert_triggered = True

    if alert_triggered:
        logger.warning(f"QUALITY ALERT: {', '.join(alert_reasons)}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        """
        INSERT INTO rag_queries (
            timestamp, question, answer, sources, latency_ms,
            retrieval_precision, groundedness, answer_relevance,
            alert_triggered, alert_reason
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (
            datetime.now().isoformat(),
            question,
            answer,
            json.dumps(sources),
            latency_ms,
            retrieval_precision,
            groundedness,
            answer_relevance,
            int(alert_triggered),
            ", ".join(alert_reasons) if alert_reasons else None
        )
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_health_report(last_n: int = 10) -> dict:
    """
    Compute health metrics over the last N queries.
    Returns a summary dict for dashboards or API endpoints.
    """
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """
        SELECT latency_ms, retrieval_precision, groundedness,
               answer_relevance, alert_triggered
        FROM rag_queries
        ORDER BY id DESC
        LIMIT ?
        """,
        (last_n,)
    ).fetchall()
    conn.close()

    if not rows:
        return {"status": "no_data", "total_queries_analyzed": 0}

    avg_latency    = round(sum(r[0] for r in rows) / len(rows), 1)
    avg_precision  = round(sum(r[1] for r in rows) / len(rows), 3)
    avg_groundedness = round(sum(r[2] for r in rows) / len(rows), 3)
    avg_relevance  = round(sum(r[3] for r in rows) / len(rows), 3)
    alert_rate     = round(sum(r[4] for r in rows) / len(rows), 3)

    status = "healthy"
    if avg_precision < THRESHOLDS["retrieval_precision"]:
        status = "degraded"
    if avg_latency > THRESHOLDS["latency_ms"]:
        status = "degraded"

    return {
        "status":                  status,
        "total_queries_analyzed":  len(rows),
        "avg_latency_ms":          avg_latency,
        "avg_retrieval_precision": avg_precision,
        "avg_groundedness":        avg_groundedness,
        "avg_answer_relevance":    avg_relevance,
        "alert_rate":              alert_rate,
        "thresholds":              THRESHOLDS
    }


if __name__ == "__main__":
    # Initialize DB
    init_db()

    # Simulate logging some queries
    print("\nSimulating query logs...")

    test_logs = [
        {"question": "What is the cancellation fee?",
         "answer": "The cancellation fee is $5",
         "sources": [{"source": "billing", "category": "billing", "snippet": "..."}],
         "latency_ms": 450, "precision": 0.75, "groundedness": 0.80, "relevance": 0.90},
        {"question": "How do drivers get paid?",
         "answer": "Drivers get paid weekly every Tuesday",
         "sources": [{"source": "driver_policy", "category": "driver_policy", "snippet": "..."}],
         "latency_ms": 380, "precision": 1.0, "groundedness": 0.85, "relevance": 0.95},
        {"question": "What is Uber's refund policy?",
         "answer": "Refunds are processed within 3-5 business days",
         "sources": [{"source": "billing", "category": "billing", "snippet": "..."}],
         "latency_ms": 520, "precision": 0.50, "groundedness": 0.70, "relevance": 0.80},
    ]

    for log in test_logs:
        row_id = log_query(
            question=log["question"], answer=log["answer"],
            sources=log["sources"],   latency_ms=log["latency_ms"],
            retrieval_precision=log["precision"],
            groundedness=log["groundedness"],
            answer_relevance=log["relevance"]
        )
        print(f"  Logged query ID={row_id}: {log['question'][:40]}...")

    # Print health report
    health = get_health_report(last_n=10)
    print("\n" + "="*50)
    print("  MONITORING HEALTH REPORT")
    print("="*50)
    print(f"  Status              : {health['status'].upper()}")
    print(f"  Queries analyzed    : {health['total_queries_analyzed']}")
    print(f"  Avg latency         : {health['avg_latency_ms']} ms")
    print(f"  Avg precision       : {health['avg_retrieval_precision']:.1%}")
    print(f"  Avg groundedness    : {health['avg_groundedness']:.1%}")
    print(f"  Avg relevance       : {health['avg_answer_relevance']:.1%}")
    print(f"  Alert rate          : {health['alert_rate']:.1%}")
    print("="*50)