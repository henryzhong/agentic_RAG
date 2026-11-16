"""Shared test fixtures.

Unit tests never touch the network: stages receive scripted fakes. The only
heavy fixture is `real_retriever`, which builds the actual corpus index once per
session (and is skipped when the corpus is unavailable).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_rag_v2.data import Chunk  # noqa: E402
from agentic_rag_v2.retrieval import HybridRetriever  # noqa: E402


@pytest.fixture(scope="session")
def chunks() -> list[Chunk]:
    """Small synthetic corpus with one obvious title match and one bridge entity."""
    raw = [
        ("c1", "d1", "Kent County Cricket Club in 2011",
         "In 2011 Kent County Cricket Club competed in Division Two of the County Championship. "
         "Pakistan fast bowler Wahab Riaz joined Kent as their overseas player in June."),
        ("c2", "d2", "Wahab Riaz",
         "Wahab Riaz is a Pakistani cricketer. He is a left-arm fast bowler and a right-hand batsman."),
        ("c3", "d3", "Ada Lovelace",
         "Ada Lovelace worked with Charles Babbage on the Analytical Engine."),
        ("c4", "d4", "Charles Babbage",
         "Charles Babbage designed the Analytical Engine in the nineteenth century."),
        ("c5", "d5", "Unrelated Article",
         "The weather in Lisbon is mild and the tram network is old."),
        ("c6", "d6", "Stefan Nemanja",
         "Stefan Nemanja was followed by his son Stefan the First-Crowned."),
        ("c7", "d7", "Stefan the First-Crowned",
         "Stefan the First-Crowned was the son of Stefan Nemanja and became king in 1217."),
        ("c8", "d8", "Analytical Engine",
         "The Analytical Engine was a proposed mechanical general-purpose computer."),
    ]
    return [Chunk(chunk_id=c, doc_id=d, title=t, text=x) for c, d, t, x in raw]


@pytest.fixture(scope="session")
def retriever(chunks: list[Chunk]) -> HybridRetriever:
    return HybridRetriever(chunks)


@pytest.fixture(scope="session")
def real_retriever() -> HybridRetriever:
    from agentic_rag_v2.data import corpus_path, load_chunks

    if not corpus_path().exists():
        pytest.skip(f"corpus unavailable at {corpus_path()}")
    return HybridRetriever(load_chunks())
