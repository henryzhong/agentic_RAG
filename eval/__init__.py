"""Evaluation: metrics, sampling, runner (reuses the v1 benchmark + gold protocol)."""
from __future__ import annotations

from agentic_rag_v2.eval.metrics import answer_metrics, evidence_recall, summarize
from agentic_rag_v2.eval.runner import SYSTEMS, build_runner, score_record

__all__ = ["answer_metrics", "evidence_recall", "summarize", "score_record", "SYSTEMS",
           "build_runner"]
