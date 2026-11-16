"""Agentic RAG v2 — a staged, auditable retrieval pipeline.

Architecture: two LLM calls + local intelligence + controlled adaptive retrieval.

    analyze      Model A (local classifier)  -> RouteSignal  (intent / multi-hop / hop / complexity)
    plan         DeepSeek #1                 -> SearchPlan   (keep | rewrite | graph, queries, rounds)
    search       local multi-route retrieval -> Evidence      (bm25 | E5 dense | title | graph | graph-hop,
                                                              GraphRAG on demand, CoverageJudge decides
                                                              extra rounds)
    rerank       local cross-encoder         -> ranked evidence
    answer       DeepSeek #2                 -> sufficiency + answer + citations

Every stage is independently testable and swappable (`pipeline.PipelineOptions`),
so ablations are configuration, not forks of the code.
"""
from __future__ import annotations

__version__ = "2.0.0"
