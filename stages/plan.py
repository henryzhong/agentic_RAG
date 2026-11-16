"""Stage 2 — plan: one DeepSeek call delivers route + rewrite + search plan.

The model proposes; validated code disposes. Model A's budget caps always bind,
so a chatty planner cannot inflate cost, and a malformed reply falls back to a
deterministic heuristic instead of aborting the run.
"""
from __future__ import annotations

from agentic_rag_v2.config import MAX_ROUNDS, MAX_TOP_K, MIN_TOP_K, PLAN_MAX_QUERIES
from agentic_rag_v2.llm import ChatClient
from agentic_rag_v2.prompts import PLAN_SYSTEM
from agentic_rag_v2.types import RouteSignal, SearchMode, SearchPlan, StageBudget


def make_plan(question: str, signal: RouteSignal, budget: StageBudget,
              client: ChatClient | None = None, *, use_llm: bool = True,
              force_mode: SearchMode | None = None) -> SearchPlan:
    if force_mode is not None:
        base = heuristic(question, signal, budget)
        base.mode = force_mode
        base.reason = f"forced by config ({force_mode.value})"
        base.source = "forced"
        if force_mode is SearchMode.KEEP:
            base.queries, base.graph = [question], False
        elif force_mode is SearchMode.GRAPH:
            base.graph = True
        return base
    if client is None or not use_llm:
        return heuristic(question, signal, budget)

    prompt = (f"QUESTION:\n{question}\n\n"
              f"ROUTE SIGNAL (small model):\n{signal.describe()}\n\n"
              f"BUDGET (hard caps enforced downstream):\n"
              f"rounds<={budget.rounds}, top_k<={budget.top_k}")
    try:
        obj = client.json_call(
            [{"role": "system", "content": PLAN_SYSTEM}, {"role": "user", "content": prompt}],
            max_tokens=400, tag="plan")
    except Exception as e:  # noqa: BLE001 - fall through to the heuristic
        fallback = heuristic(question, signal, budget)
        fallback.reason = f"planner error: {type(e).__name__}: {e}"
        return fallback

    return coerce(obj, question, budget, source="llm")


def heuristic(question: str, signal: RouteSignal, budget: StageBudget) -> SearchPlan:
    """Deterministic plan used without an LLM (ablation arm and error path)."""
    multi = signal.available and (signal.multihop or signal.hop_bucket in ("2", "3+"))
    return SearchPlan(
        mode=SearchMode.GRAPH if multi else SearchMode.KEEP,
        queries=[question],
        graph=multi,
        rounds=budget.rounds,
        top_k=budget.top_k,
        reason="heuristic: graph for multi-hop, keep otherwise",
        source="fallback",
    )


def coerce(obj: dict, question: str, budget: StageBudget, *, source: str = "llm") -> SearchPlan:
    """Validate the model's plan against the schema and the budget."""
    mode = SearchMode.coerce(obj.get("mode"), default=None) or SearchMode.KEEP

    queries: list[str] = []
    for raw in obj.get("queries") or []:
        q = str(raw or "").strip()
        if q and q != question and q not in queries:
            queries.append(q)
        if len(queries) >= PLAN_MAX_QUERIES:
            break
    if mode is SearchMode.KEEP:
        queries = [question]
    elif not queries:
        queries = [question]

    graph = bool(obj.get("graph"))
    if mode is SearchMode.GRAPH:
        graph = True
    elif mode is SearchMode.KEEP:
        graph = False

    rounds = _int(obj.get("rounds"), budget.rounds)
    if mode is SearchMode.GRAPH:
        # chaining is expected: extra retrieval rounds are local (no API call), so the
        # Model A budget is a floor here. Simple questions are protected by the hop=1
        # budget and by the judge stopping as soon as the chain is covered.
        rounds = max(rounds, budget.rounds)
    rounds = max(1, min(rounds, budget.rounds, MAX_ROUNDS))
    top_k = _int(obj.get("top_k"), budget.top_k)
    top_k = max(MIN_TOP_K, min(top_k, min(budget.top_k + 4, MAX_TOP_K)))

    return SearchPlan(mode=mode, queries=queries, graph=graph, rounds=rounds,
                      top_k=top_k, reason=str(obj.get("reason") or ""), source=source)


def _int(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
