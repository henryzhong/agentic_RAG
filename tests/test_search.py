"""Stage 3: multi-route parallel retrieval, adaptive rounds and the local judge."""
from __future__ import annotations

from agentic_rag_v2.models.judge import CoverageJudge, LocalJudgeAdapter, StubJudge
from agentic_rag_v2.stages.search import run_search
from agentic_rag_v2.tests.fakes import RecordingJudge
from agentic_rag_v2.types import SearchMode, SearchPlan


def _plan(mode: SearchMode = SearchMode.KEEP, *, rounds: int = 3, top_k: int = 4,
          queries: list[str] | None = None, graph: bool | None = None) -> SearchPlan:
    return SearchPlan(mode=mode, queries=queries or ["Wahab Riaz"],
                      graph=bool(graph) if graph is not None else mode is SearchMode.GRAPH,
                      rounds=rounds, top_k=top_k)


def _bridge_corpus():
    from agentic_rag_v2.data import Chunk
    from agentic_rag_v2.retrieval import HybridRetriever

    chunks = [
        Chunk("x1", "d1", "Stefan Nemanja",
              "Stefan Nemanja ruled Serbia and was followed by his son Stefan the First-Crowned."),
        Chunk("x2", "d2", "Serbia",
              "Stefan Nemanja was a Serbian prince of the Vukanovic dynasty."),
        Chunk("x3", "d3", "Stefan the First-Crowned",
              "Stefan the First-Crowned became king in 1217."),
    ]
    return HybridRetriever(chunks)


# ------------------------------------------------------------------ stopping
def test_stub_judge_stops_after_one_round(retriever) -> None:
    result = run_search("Who is Wahab Riaz?", _plan(rounds=3), retriever, StubJudge())
    assert result.rounds == 1
    assert len(result.evidence) > 0
    assert result.llm_calls == 0


def test_judge_sufficient_stops_early(retriever) -> None:
    judge = RecordingJudge([{"sufficient": True, "reason": "enough"}])
    result = run_search("Who is Wahab Riaz?", _plan(rounds=3), retriever, judge)
    assert result.rounds == 1
    assert result.verdicts[0].sufficient is True
    assert len(judge.seen) == 1
    assert result.llm_calls == 1          # this judge is LLM-backed


def test_next_query_triggers_second_round(retriever) -> None:
    judge = RecordingJudge([
        {"sufficient": False, "next_query": "Kent County Cricket Club overseas player 2011"},
        {"sufficient": True, "reason": "chained"},
    ])
    result = run_search("Who was the overseas player for Kent in 2011?",
                        _plan(rounds=3, queries=["Kent County Cricket Club 2011"]),
                        retriever, judge)
    assert result.rounds == 2
    assert result.queries[1] == ["Kent County Cricket Club overseas player 2011"]
    assert result.llm_calls == 2


def test_round_budget_is_respected(retriever) -> None:
    judge = RecordingJudge([{"sufficient": False, "next_query": q}
                            for q in ["ada lovelace", "charles babbage", "analytical engine"]])
    result = run_search("Wahab Riaz", _plan(rounds=2), retriever, judge)
    assert result.rounds == 2
    assert len(result.verdicts) == 1      # no judge call after the last round


def test_local_judge_adapter_is_used_without_llm(retriever) -> None:
    seen: list[int] = []

    def fn(question, evidence):
        seen.append(len(evidence))
        return {"sufficient": False, "next_query": "Wahab Riaz Pakistan"}

    result = run_search("Who is Wahab Riaz?", _plan(rounds=2), retriever, LocalJudgeAdapter(fn))
    assert result.rounds == 2
    assert result.llm_calls == 0
    assert result.verdicts[0].source == "jev"
    assert seen


# -------------------------------------------------------------------- routes
def test_graph_route_reaches_the_bridge_passage() -> None:
    retriever = _bridge_corpus()
    plan = _plan(SearchMode.GRAPH, rounds=1, top_k=2, queries=["Stefan Nemanja"])
    result = run_search("Who was Stefan Nemanja followed by?", plan, retriever)
    assert "x3" in result.evidence.passages
    routes = result.trace[0]["routes"]
    assert routes.get("graph", 0) > 0 or routes.get("hop", 0) > 0


def test_keep_route_does_not_touch_the_graph() -> None:
    retriever = _bridge_corpus()
    plan = _plan(SearchMode.KEEP, rounds=1, top_k=2, queries=["Stefan Nemanja"])
    result = run_search("Who was Stefan Nemanja followed by?", plan, retriever)
    assert result.trace[0]["routes"].get("graph") is None
    assert result.trace[0]["routes"].get("hop") is None


def test_parallel_and_sequential_give_identical_evidence(retriever) -> None:
    def run(parallel: bool):
        plan = _plan(SearchMode.GRAPH, rounds=2, top_k=4,
                     queries=["Ada Lovelace", "Charles Babbage"])
        judge = RecordingJudge([{"sufficient": False, "next_query": "Analytical Engine"}])
        return run_search("q", plan, retriever, judge, parallel=parallel)

    a, b = run(True), run(False)
    assert a.evidence.to_dict() == b.evidence.to_dict()
    assert a.rounds == b.rounds == 2


def test_pool_cap_is_enforced() -> None:
    from agentic_rag_v2.data import Chunk
    from agentic_rag_v2.retrieval import HybridRetriever

    # BM25 IDF must stay positive: "alpha" appears in 4 of 12 passages
    chunks = [Chunk(f"y{i}", f"d{i}", f"Doc {i}",
                    f"alpha passage {i}" if i < 4 else f"filler passage {i}")
              for i in range(12)]
    retriever = HybridRetriever(chunks)
    result = run_search("alpha", _plan(SearchMode.GRAPH, rounds=2, top_k=5, queries=["alpha"]),
                        retriever, StubJudge(), pool_cap=3)
    assert len(result.evidence) == 3


# --------------------------------------------------------------- local judge
def test_coverage_judge_accepts_fully_covered_evidence(retriever) -> None:
    question = "Who is the Pakistani cricketer Wahab Riaz?"
    result = run_search(question, _plan(rounds=1, top_k=6, queries=["Wahab Riaz Pakistani cricketer"]),
                        retriever)
    verdict = CoverageJudge(retriever).judge(question, result.evidence)
    assert verdict.signals["token_coverage"] > 0.5
    assert verdict.source == "coverage"
    assert 0.0 <= verdict.score <= 1.0


def test_coverage_judge_asks_for_a_follow_up_when_a_concept_is_missing(retriever) -> None:
    question = "Which coach did Ada Lovelace hire for the Analytical Engine orchestra in Lisbon?"
    result = run_search(question, _plan(rounds=1, top_k=4, queries=["Ada Lovelace"]), retriever)
    verdict = CoverageJudge(retriever).judge(question, result.evidence)
    assert verdict.sufficient is False
    assert verdict.next_query                    # deterministic follow-up query
    assert verdict.missing


def test_bridge_detection_triggers_a_follow_up_round() -> None:
    from agentic_rag_v2.data import Chunk
    from agentic_rag_v2.retrieval import HybridRetriever
    from agentic_rag_v2.types import Evidence, Passage

    chunks = [
        Chunk("z1", "zd1", "Renegade (song)",
              "Renegade is a song recorded by the American rock band Styx."),
        Chunk("z2", "zd2", "Styx (band)",
              "Styx is an American rock band from Chicago formed in 1972."),
    ]
    retriever = HybridRetriever(chunks)
    judge = CoverageJudge(retriever)
    question = "Which city is the band that recorded the song Renegade from?"

    only_first_hop = Evidence()
    only_first_hop.add(Passage("z1", "zd1", "Renegade (song)", chunks[0].text, 1.0))
    verdict = judge.judge(question, only_first_hop, multihop=True)
    assert verdict.signals["bridge_entity"] == "styx"
    assert verdict.signals["bridge_resolved"] is False
    assert verdict.sufficient is False
    assert "styx" in verdict.next_query

    both_hops = Evidence()
    both_hops.add(Passage("z1", "zd1", "Renegade (song)", chunks[0].text, 1.0))
    both_hops.add(Passage("z2", "zd2", "Styx (band)", chunks[1].text, 0.9))
    resolved = judge.judge(question, both_hops, multihop=True)
    assert resolved.signals["bridge_resolved"] is True


def test_bridge_rule_is_ignored_for_single_hop() -> None:
    from agentic_rag_v2.data import Chunk
    from agentic_rag_v2.retrieval import HybridRetriever
    from agentic_rag_v2.types import Evidence, Passage

    chunks = [Chunk("z1", "zd1", "Renegade (song)",
                    "Renegade is a song recorded by the American rock band Styx.")]
    retriever = HybridRetriever(chunks)
    ev = Evidence()
    ev.add(Passage("z1", "zd1", "Renegade (song)", chunks[0].text, 1.0))
    verdict = CoverageJudge(retriever).judge("What is Renegade?", ev, multihop=False)
    assert verdict.signals["bridge_resolved"] is True


def test_coverage_judge_without_evidence_is_never_sufficient() -> None:
    from agentic_rag_v2.types import Evidence

    verdict = CoverageJudge().judge("anything?", Evidence())
    assert verdict.sufficient is False
    assert verdict.next_query == ""
