"""Stages 4 and 5: local reranking and grounded answering."""
from __future__ import annotations

from agentic_rag_v2.stages.answer import (answer, answer_direct, build_evidence_block,
                                          normalize_answer)
from agentic_rag_v2.stages.rerank import rerank
from agentic_rag_v2.tests.fakes import BrokenReranker, FakeReranker, ScriptedClient
from agentic_rag_v2.types import Evidence, Passage


def _evidence() -> Evidence:
    ev = Evidence()
    ev.add(Passage(chunk_id="c1", doc_id="d1", title="title c1",
                   text="first hop passage about the club", score=0.9))
    ev.add(Passage(chunk_id="c2", doc_id="d2", title="title c2",
                   text="the second hop passage holds the answer", score=0.8))
    ev.add(Passage(chunk_id="c3", doc_id="d3", title="title c3",
                   text="a distractor about another club", score=0.7))
    return ev


# -------------------------------------------------------------------- rerank
def test_local_rerank_promotes_the_answer_passage() -> None:
    reranker = FakeReranker(key="second hop")
    result = rerank("q", _evidence(), reranker=reranker)
    assert result.source == "local"
    assert result.order[0] == "c2"
    assert set(result.order) == {"c1", "c2", "c3"}
    assert 0.0 < result.scores["c2"] < 1.0     # sigmoid of the raw logit
    assert result.scores["c1"] < result.scores["c2"]
    assert reranker.calls == 1


def test_local_rerank_falls_back_when_model_is_unavailable() -> None:
    result = rerank("q", _evidence(), reranker=BrokenReranker())
    assert result.source == "score"
    assert result.order == ["c1", "c2", "c3"]


def test_score_strategy_keeps_retrieval_order() -> None:
    result = rerank("q", _evidence(), reranker=FakeReranker(), strategy="score")
    assert result.source == "score"
    assert result.order == ["c1", "c2", "c3"]


def test_llm_rerank_ablation_and_its_failure_mode() -> None:
    client = ScriptedClient({"rerank": {"ranking": [2, 1], "reason": "bridge first"}})
    result = rerank("q", _evidence(), client=client, strategy="llm")
    assert result.source == "llm" and result.order[:2] == ["c2", "c1"]
    assert result.llm_calls == 1

    failing = ScriptedClient({"rerank": RuntimeError("timeout")})
    fallback = rerank("q", _evidence(), client=failing, strategy="llm")
    assert fallback.source == "score" and fallback.order == ["c1", "c2", "c3"]


# -------------------------------------------------------------------- answer
def test_build_evidence_block_caps_per_document() -> None:
    ev = Evidence()
    for i in range(4):
        ev.add(Passage(chunk_id=f"c{i}", doc_id="same", title="t", text="x" * 10,
                       score=1.0 - i * 0.1))
    block, num_map = build_evidence_block(ev)
    assert len(num_map) == 2             # two chunks per document max
    assert block.startswith("[1] (t)")


def test_build_evidence_block_follows_rerank_order() -> None:
    block, num_map = build_evidence_block(_evidence(), ["c2", "c1", "c3"])
    assert num_map[1] == "c2"
    assert block.splitlines()[0].startswith("[1] (title c2)")


def test_answer_maps_citations_and_sufficiency() -> None:
    client = ScriptedClient({"answer": {"sufficient": True, "answer": "the Championship",
                                        "evidence_ids": [1], "confidence": 0.7,
                                        "relevant_facts": ["club competed in the Championship"],
                                        "missing": ""}})
    result = answer("q", _evidence(), client, order=["c2", "c1", "c3"])
    assert result.answer == "Championship"       # leading article stripped
    assert result.cited_docs == ["d2"]
    assert result.sufficient is True
    assert result.confidence == 0.7
    assert result.facts == ["club competed in the Championship"]


def test_answer_reports_insufficient_evidence() -> None:
    client = ScriptedClient({"answer": {"sufficient": False, "answer": "UNKNOWN",
                                        "evidence_ids": [], "missing": "the founding year"}})
    result = answer("q", _evidence(), client)
    assert result.answer == "UNKNOWN"
    assert result.sufficient is False
    assert result.missing == "the founding year"
    assert result.cited_docs == []


def test_answer_handles_bad_ids_and_errors() -> None:
    client = ScriptedClient({"answer": {"answer": "x", "evidence_ids": ["nope", 77]}})
    assert answer("q", _evidence(), client).cited_docs == []

    failing = ScriptedClient({"answer": RuntimeError("boom")})
    assert answer("q", _evidence(), failing).answer == "UNKNOWN"


def test_answer_with_empty_evidence_skips_the_call() -> None:
    client = ScriptedClient({})
    result = answer("q", Evidence(), client)
    assert result.answer == "UNKNOWN"
    assert result.sufficient is False
    assert client.calls == []


def test_normalize_answer_variants() -> None:
    assert normalize_answer("13-15 February 1942") == "15 February 1942"
    assert normalize_answer(" the Netherlands. ") == "Netherlands"
    assert normalize_answer("") == ""


def test_answer_direct_uses_no_evidence() -> None:
    client = ScriptedClient({"direct": {"answer": "Paris", "confidence": 0.4}})
    result = answer_direct("capital of France?", client)
    assert result.answer == "Paris"
    assert result.block_ids == {}
