"""Stage 1 (analyze / budget) and stage 2 (plan) behaviour."""
from __future__ import annotations

from agentic_rag_v2.stages.analyze import analyze, budget_for
from agentic_rag_v2.stages.plan import coerce, heuristic, make_plan
from agentic_rag_v2.tests.fakes import FakeAnalyzer, ScriptedClient
from agentic_rag_v2.types import RouteSignal, SearchMode, StageBudget


def _signal(**kw) -> RouteSignal:
    base = dict(available=True, multihop=True, multihop_prob=0.9,
                hop_bucket="2", complexity="medium")
    base.update(kw)
    return RouteSignal(**base)


def _budget(rounds: int = 3, top_k: int = 8) -> StageBudget:
    return StageBudget(rounds=rounds, top_k=top_k, answer_top_n=24, source="model_a")


# ------------------------------------------------------------------ analyze
def test_analyze_returns_signal_and_budget() -> None:
    signal, budget = analyze("q", FakeAnalyzer(hop="2"))
    assert signal.available and signal.hop_bucket == "2"
    assert budget.source == "model_a"
    assert budget.rounds >= 1


def test_budget_shrinks_for_single_hop() -> None:
    budget = budget_for(_signal(hop_bucket="1", multihop_prob=0.9))
    assert budget.rounds == 1
    assert budget.top_k == 6


def test_budget_expands_for_hard_multi_hop() -> None:
    budget = budget_for(_signal(hop_bucket="3+", complexity="high"))
    assert budget.rounds == 4           # 3 + complexity bonus, at the MAX_ROUNDS cap
    assert budget.top_k == 12


def test_budget_collapses_when_classifier_is_confident_it_is_not_multihop() -> None:
    budget = budget_for(_signal(hop_bucket="3+", multihop_prob=0.1))
    assert budget.rounds == 1


def test_budget_defaults_without_model_a() -> None:
    budget = budget_for(RouteSignal(available=False, error="missing"))
    assert (budget.rounds, budget.top_k, budget.source) == (2, 8, "default")


# --------------------------------------------------------------------- plan
def test_plan_accepts_valid_plan() -> None:
    client = ScriptedClient({"plan": {"mode": "rewrite",
                                      "queries": ["stefan nemanja successor",
                                                  "serbia ruler after stefan nemanja"],
                                      "graph": False, "rounds": 2, "top_k": 8,
                                      "reason": "ambiguous naming"}})
    plan = make_plan("Who followed Stefan Nemanja?", _signal(), _budget(), client)
    assert plan.mode is SearchMode.REWRITE
    assert plan.source == "llm"
    assert plan.graph is False
    assert plan.queries[0] == "stefan nemanja successor"


def test_plan_keep_mode_uses_the_question_verbatim() -> None:
    client = ScriptedClient({"plan": {"mode": "keep", "queries": ["Something Else"],
                                      "graph": True, "rounds": 1, "top_k": 8}})
    plan = make_plan("Who is Ada Lovelace?", _signal(), _budget(), client)
    assert plan.mode is SearchMode.KEEP
    assert plan.queries == ["Who is Ada Lovelace?"]
    assert plan.graph is False          # a verbatim single-hop query never needs the graph


def test_plan_graph_mode_forces_graph_route() -> None:
    client = ScriptedClient({"plan": {"mode": "graph", "queries": ["q"],
                                      "graph": False, "rounds": 2}})
    plan = make_plan("q", _signal(), _budget(), client)
    assert plan.graph is True


def test_plan_clamps_rounds_and_top_k() -> None:
    client = ScriptedClient({"plan": {"mode": "graph", "queries": ["q"], "graph": True,
                                      "rounds": 99, "top_k": 99}})
    plan = make_plan("q", _signal(), _budget(rounds=2, top_k=8), client)
    assert plan.rounds == 2             # budget binds
    assert plan.top_k == 12             # budget.top_k + 4


def test_plan_graph_mode_cannot_go_below_the_model_a_budget() -> None:
    client = ScriptedClient({"plan": {"mode": "graph", "queries": ["q"], "graph": True,
                                      "rounds": 1, "top_k": 6}})
    plan = make_plan("q", _signal(), _budget(rounds=3), client)
    assert plan.rounds == 3             # local rounds are free; the budget is a floor


def test_plan_rejects_invalid_mode_and_clamps_low_top_k() -> None:
    client = ScriptedClient({"plan": {"mode": "teleport", "queries": ["q"], "top_k": 1}})
    plan = make_plan("q", _signal(), _budget(), client)
    assert plan.mode is SearchMode.KEEP
    assert plan.top_k == 4              # MIN_TOP_K


def test_plan_falls_back_on_planner_error() -> None:
    client = ScriptedClient({"plan": RuntimeError("boom")})
    plan = make_plan("q", _signal(), _budget(), client)
    assert plan.source == "fallback"
    assert plan.mode is SearchMode.GRAPH and plan.graph is True
    assert "planner error" in plan.reason


def test_plan_without_llm_is_deterministic() -> None:
    plan = make_plan("q", _signal(multihop=False, hop_bucket="1"), _budget(), None)
    assert plan.source == "fallback"
    assert plan.mode is SearchMode.KEEP and plan.graph is False


def test_force_mode_overrides_planner() -> None:
    client = ScriptedClient({"plan": {"mode": "graph", "queries": ["q"], "graph": True}})
    plan = make_plan("q", _signal(), _budget(), client, force_mode=SearchMode.KEEP)
    assert plan.source == "forced"
    assert plan.mode is SearchMode.KEEP and plan.queries == ["q"]
    assert plan.graph is False
    assert client.calls == []           # no wasted LLM call


def test_coerce_drops_duplicate_and_question_queries() -> None:
    plan = coerce({"mode": "rewrite", "queries": ["a", "question?", "b", "b", "c", "d"]},
                  "question?", _budget())
    assert plan.queries == ["a", "b", "c"]      # capped at PLAN_MAX_QUERIES


def test_heuristic_marks_graph_for_multihop() -> None:
    plan = heuristic("q", _signal(), _budget())
    assert plan.mode is SearchMode.GRAPH and plan.graph is True
