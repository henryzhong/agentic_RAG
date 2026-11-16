"""Retrieval package: lexical index, entity graph, hybrid facade."""
from __future__ import annotations

from agentic_rag_v2.retrieval.engine import HybridRetriever, index_stats
from agentic_rag_v2.retrieval.lexical import LexicalIndex
from agentic_rag_v2.retrieval.text import extract_entities, tokenize

__all__ = ["HybridRetriever", "LexicalIndex", "index_stats", "extract_entities", "tokenize"]
