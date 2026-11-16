"""End-to-end pipeline wiring with scripted LLM replies (no network)."""
from __future__ import annotations

from agentic_rag_v2.pipeline import AgenticRAG, PipelineOptions
from agentic_rag_v2.tests.fakes import FakeAnalyzer, FakeReranker, ScriptedClient
from agentic_rag_v2.types import SearchMode

QUESTION = "Who was Stefan Nemanja followed by?"


def _client() -> ScriptedClient:
    return ScriptedClient({
        "plan": {"mode": "graph", "queries": [QUESTION], "graph": True,
                 "rounds": 2, "top_k": 6, "reason": "bridge question"},
        "judge": {"sufficient": True, "missing": "", "next_query": "", "reason": "chain found"},
        "rerank": {"ranking": [1, 2, 3], "reason": "bridge passage first"},
        "answer": {"sufficient": True, "answer": "Stefan the First-Crowned",
                   "evidence_ids": [1], "confidence": 0.8,
                   "relevant_facts": ["Nemanja was followed by his son"], "missing": ""},
        "direct": {"answer": "Stefan the First-Crowned", "confidence": 0.5},
    })


def _rag(retriever, client, *, reranker=None, **option_kw) -> AgenticRAG:
    # unit tests stay offline: the dense route is exercised by its own tests with a
    # fake semantic retriever, never by loading multilingual-e5-large
    option_kw.setdefault("use_dense", False)
    strategy = option_kw.get("rerank", "score")
    if reranker is None and strategy == "local":
        reranker = FakeReranker(key="nomatch")
    return AgenticRAG(client=client, retriever=retriever, analyzer=FakeAnalyzer(),
                      reranker=reranker, options=PipelineOptions(**option_kw))


def test_full_pipeline_uses_exactly_two_llm_calls(retriever) -> None:
    client = _client()
    result = _rag(retriever, client).answer(QUESTION)

    assert result.answer == "Stefan the First-Crowned"
    assert result.plan.mode is SearchMode.GRAPH
    assert result.rounds == 1                    # local judge stopped after round 1
    assert result.llm_calls == 2                 # plan + answer, nothing else
    assert client.calls == ["plan", "answer"]
    assert [s["stage"] for s in result.trace] == ["analyze", "plan", "search",
                                                  "rerank", "answer"]
    assert result.verdicts[0].source == "coverage"
    record = result.to_record()
    assert record["mode"] == "graph" and record["route"]["available"] is True
    assert record["pred"] == result.answer


def test_pipeline_without_judge_keeps_one_round(retriever) -> None:
    client = _client()
    result = _rag(retriever, client, use_judge=False).answer(QUESTION)
    assert result.rounds == 1
    assert result.verdicts == []
    assert result.llm_calls == 2
    assert "judge" not in client.calls


def test_llm_judge_arm_costs_one_extra_call(retriever) -> None:
    client = _client()
    result = _rag(retriever, client, judge_kind="llm").answer(QUESTION)
    assert result.llm_calls == 3
    assert "judge" in client.calls
    assert result.verdicts[0].source == "llm"


def test_score_rerank_keeps_retrieval_order(retriever) -> None:
    client = _client()
    result = _rag(retriever, client, rerank="score").answer(QUESTION)
    stage = next(s for s in result.trace if s["stage"] == "rerank")
    assert stage["source"] == "score"
    assert result.ranking == [p.chunk_id for p in result.evidence.ordered()]
    assert result.llm_calls == 2


def test_cross_encoder_rerank_arm_is_selectable(retriever) -> None:
    client = _client()
    result = _rag(retriever, client, rerank="local").answer(QUESTION)
    stage = next(s for s in result.trace if s["stage"] == "rerank")
    assert stage["source"] == "local"
    assert result.llm_calls == 2


def test_kg_relations_arm_injects_cached_triples(retriever) -> None:
    from agentic_rag_v2.retrieval.relations import RelationStore
    from agentic_rag_v2.stages.answer import build_evidence_block

    store = RelationStore.__new__(RelationStore)
    store.cache = {"c6": [{"head": "Stefan Nemanja", "relation": "followed by",
                           "tail": "Stefan the First-Crowned"}]}
    store.loaded, store.error, store.extract = True, None, False
    store.path = "memory"
    evidence = __import__("agentic_rag_v2.types", fromlist=["Evidence"]).Evidence()
    from agentic_rag_v2.types import Passage
    evidence.add(Passage("c6", "d6", "Stefan Nemanja", "text", 1.0))
    _, num_map = build_evidence_block(evidence)
    lines = store.prompt_lines(evidence, [num_map[1]])
    assert "KNOWLEDGE-GRAPH RELATIONS" in lines
    assert "Stefan Nemanja --followed by--> Stefan the First-Crowned" in lines


def test_llm_rerank_arm_is_selectable(retriever) -> None:
    client = _client()
    result = _rag(retriever, client, rerank="llm").answer(QUESTION)
    stage = next(s for s in result.trace if s["stage"] == "rerank")
    assert stage["source"] == "llm"
    assert result.llm_calls == 3


def test_forced_mode_skips_planner_call(retriever) -> None:
    client = _client()
    result = _rag(retriever, client, force_mode=SearchMode.KEEP).answer(QUESTION)
    assert result.plan.mode is SearchMode.KEEP
    assert result.plan.source == "forced"
    assert "plan" not in client.calls
    assert result.llm_calls == 1               # only the answer call


def test_planner_fallback_still_answers(retriever) -> None:
    responses = _client().responses
    responses["plan"] = RuntimeError("planner down")
    client = ScriptedClient(responses)
    result = _rag(retriever, client).answer(QUESTION)
    assert result.plan.source == "fallback"
    assert result.answer == "Stefan the First-Crowned"


def test_model_a_cannot_be_switched_off(retriever) -> None:
    from dataclasses import fields

    assert "use_model_a" not in {f.name for f in fields(PipelineOptions)}


def test_broken_model_a_blocks_construction(retriever) -> None:
    from agentic_rag_v2.tests.fakes import BrokenAnalyzer

    before = BrokenAnalyzer.calls
    try:
        AgenticRAG(client=_client(), retriever=retriever, analyzer=BrokenAnalyzer(),
                   reranker=FakeReranker(), options=PipelineOptions(use_dense=False))
    except RuntimeError as e:
        assert "could not be loaded" in str(e)
    else:  # pragma: no cover
        raise AssertionError("a failing Model A must stop the pipeline")
    assert BrokenAnalyzer.calls == before       # never fell back to a neutral signal


def test_every_question_passes_through_model_a(retriever) -> None:
    analyzer = FakeAnalyzer()
    rag = AgenticRAG(client=_client(), retriever=retriever, analyzer=analyzer,
                     reranker=FakeReranker(),
                     options=PipelineOptions(use_judge=False, use_dense=False))
    rag.answer(QUESTION)
    rag.answer_baseline(QUESTION)
    assert analyzer.questions == [QUESTION, QUESTION]
    trace = next(s for s in rag.answer(QUESTION).trace if s["stage"] == "analyze")
    assert trace["route"]["available"] is True and trace["budget"]["rounds"] >= 1


def test_dense_route_is_used_when_semantic_is_injected(retriever) -> None:
    from agentic_rag_v2.tests.fakes import FakeSemantic

    client = _client()
    semantic = FakeSemantic({"stefan nemanja": "c6", "son": "c7"})
    rag = AgenticRAG(client=client, retriever=retriever, analyzer=FakeAnalyzer(),
                     reranker=FakeReranker(), semantic=semantic,
                     options=PipelineOptions(use_dense=True))
    result = rag.answer(QUESTION)
    search_stage = next(s for s in result.trace if s["stage"] == "search")
    assert "dense" in search_stage["rounds_trace"][0]["routes"]
    assert rag.manifest()["semantic"]["vectors"] > 0
    assert result.answer == "Stefan the First-Crowned"


def test_reranker_failure_degrades_to_score_order(retriever) -> None:
    from agentic_rag_v2.tests.fakes import BrokenReranker

    result = _rag(retriever, _client(), reranker=BrokenReranker()).answer(QUESTION)
    stage = next(s for s in result.trace if s["stage"] == "rerank")
    assert stage["source"] == "score"
    assert result.answer == "Stefan the First-Crowned"


def test_baseline_has_no_retrieval(retriever) -> None:
    client = _client()
    result = _rag(retriever, client).answer_baseline(QUESTION)
    assert result.answer == "Stefan the First-Crowned"
    assert result.evidence.doc_ids() == []
    assert result.llm_calls == 1
    assert client.calls == ["direct"]


def test_manifest_describes_the_run(retriever) -> None:
    manifest = _rag(retriever, _client()).manifest()
    assert manifest["prompt_version"]
    assert manifest["index"]["chunks"] == len(retriever.chunks)
    assert manifest["judge"] == "CoverageJudge"
    assert manifest["reranker"] is None               # score order is the default
    assert manifest["options"]["rerank"] == "score"
    assert manifest["relations"] is None
