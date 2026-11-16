"""Contract tests for the stage-to-stage data objects."""
from __future__ import annotations

from agentic_rag_v2.types import (Evidence, Passage, RouteSignal, SearchMode,
                                  SearchPlan, StageBudget)


def _p(cid: str, score: float, doc: str | None = None) -> Passage:
    return Passage(chunk_id=cid, doc_id=doc or cid, title=f"t-{cid}", text="x", score=score)


def test_evidence_is_ordered_deterministically() -> None:
    ev = Evidence()
    ev.add(_p("b", 0.5))
    ev.add(_p("a", 0.5))
    ev.add(_p("c", 0.9))
    assert [p.chunk_id for p in ev.ordered()] == ["c", "a", "b"]
    assert [p.chunk_id for p in ev.ordered()] == [p.chunk_id for p in ev.ordered()]


def test_evidence_keeps_better_duplicate() -> None:
    ev = Evidence()
    assert ev.add(_p("a", 0.2)) is True
    assert ev.add(_p("a", 0.1)) is False
    assert ev.passages["a"].score == 0.2
    assert ev.add(_p("a", 0.7)) is True
    assert ev.passages["a"].score == 0.7


def test_evidence_cap_drops_the_weakest() -> None:
    ev = Evidence()
    for i, score in enumerate([0.1, 0.9, 0.4, 0.7]):
        ev.add(_p(f"c{i}", score))
    ev.cap(2)
    assert [p.chunk_id for p in ev.ordered()] == ["c1", "c3"]


def test_merge_reports_additions_only() -> None:
    ev = Evidence()
    ev.add(_p("a", 0.5))
    assert ev.merge([_p("a", 0.4), _p("b", 0.3)]) == 1
    assert len(ev) == 2


def test_plan_queries_per_mode() -> None:
    keep = SearchPlan(mode=SearchMode.KEEP, queries=["ignored", "also ignored"])
    assert keep.search_queries("the question?") == ["the question?"]

    graph = SearchPlan(mode=SearchMode.GRAPH, queries=["q?", "bridge"], graph=True)
    assert graph.search_queries("unused") == ["q?", "bridge"]

    rewrite = SearchPlan(mode=SearchMode.REWRITE, queries=["a", "b", "a", " "])
    assert rewrite.search_queries("unused") == ["a", "b"]

    empty = SearchPlan(mode=SearchMode.REWRITE, queries=[])
    assert empty.search_queries("fallback") == ["fallback"]


def test_search_mode_coerce() -> None:
    assert SearchMode.coerce("GRAPH") is SearchMode.GRAPH
    assert SearchMode.coerce("nonsense") is SearchMode.KEEP
    assert SearchMode.coerce(None) is SearchMode.KEEP


def test_route_signal_describe_marks_unavailable() -> None:
    assert "unavailable" in RouteSignal(available=False, error="boom").describe()
    text = RouteSignal(available=True, intent="multi_hop", multihop_prob=0.8,
                       hop_bucket="3+", complexity="high").describe()
    assert "intent=multi_hop" in text and "hop=3+" in text


def test_stage_budget_serialises() -> None:
    budget = StageBudget(rounds=2, top_k=8, answer_top_n=24, source="model_a")
    assert budget.to_dict() == {"rounds": 2, "top_k": 8,
                                "answer_top_n": 24, "source": "model_a"}


def test_plan_serialises() -> None:
    plan = SearchPlan(mode=SearchMode.GRAPH, queries=["q1"], graph=True,
                      rounds=2, top_k=6, reason="bridge", source="llm")
    assert plan.to_dict() == {"mode": "graph", "queries": ["q1"], "graph": True,
                              "rounds": 2, "top_k": 6, "reason": "bridge",
                              "source": "llm"}
