"""Metric helpers (shared with the v1 protocol)."""
from __future__ import annotations

from agentic_rag_v2.eval.metrics import (answer_metrics, evidence_precision,
                                        evidence_recall, normalize_answer, summarize)


def test_normalize_is_squad_like() -> None:
    assert normalize_answer("The  Netherlands!") == "netherlands"


def test_em_and_f1() -> None:
    exact = answer_metrics("Right-hand", "right-hand")
    assert exact["em"] == 1.0 and exact["f1"] == 1.0

    partial = answer_metrics("right-hand batsman", "right-hand")
    assert partial["em"] == 0.0
    assert 0.0 < partial["f1"] < 1.0
    assert partial["containment"] == 1.0


def test_aliases_are_accepted() -> None:
    metrics = answer_metrics("USA", "United States", ["USA", "US"])
    assert metrics["em"] == 1.0


def test_evidence_recall_and_precision() -> None:
    assert evidence_recall(["a", "b"], ["a", "b"]) == 1.0
    assert evidence_recall(["a"], ["a", "b"]) == 0.5
    assert evidence_recall([], ["a"]) == 0.0
    assert evidence_precision(["a", "x"], ["a"]) == 0.5


def test_summarize_averages_records() -> None:
    records = [
        {"em": 1.0, "f1": 1.0, "containment": 1.0, "evidence_recall_final": 1.0,
         "evidence_recall_retrieved": 1.0, "rounds": 1, "llm_calls": 4, "latency_ms": 100},
        {"em": 0.0, "f1": 0.5, "containment": 1.0, "evidence_recall_final": 0.5,
         "evidence_recall_retrieved": 0.5, "rounds": 2, "llm_calls": 6, "latency_ms": 300},
    ]
    stats = summarize(records)
    assert stats["n"] == 2
    assert stats["em"] == 0.5
    assert stats["avg_llm_calls"] == 5.0
