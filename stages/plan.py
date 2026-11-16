"""Stage 2 — plan: one DeepSeek call delivers route + rewrite + search plan.

The model proposes; validated code disposes. Model A's budget caps always bind,
so a chatty planner cannot inflate cost, and a malformed reply falls back to a
deterministic heuristic instead of aborting the run.
"""
from __future__ import annotations

import re

from agentic_rag_v2.config import MAX_ROUNDS, MAX_TOP_K, MIN_TOP_K, PLAN_MAX_QUERIES
from agentic_rag_v2.llm import ChatClient
from agentic_rag_v2.prompts import PLAN_SYSTEM
from agentic_rag_v2.types import RouteSignal, SearchMode, SearchPlan, StageBudget

# Comparison / intersection questions need both entities' passages and a different
# answer protocol, and they measured 0.40 EM vs 0.61 for the rest
# (docs/PLAN_2026-11-16.md §1.6). The patterns are deliberately narrow: a false
# positive only changes the answer-prompt appendix, so precision matters more
# than recall.
_COMPARE_PATTERNS = (
    re.compile(r"\bin common\b", re.I),
    re.compile(r"\bcommon to (both|the two|them)\b", re.I),
    # "what is in both X and Y" asks for a shared attribute; "which actor starred in
    # both X and Y" asks for one entity, which is a different answer protocol
    re.compile(r"\bwhat\b[^?]*\bin both\b", re.I),
    re.compile(r"\bwhat\b[^?]*\b(share|shares|shared|have in common)\b", re.I),
    re.compile(r"\bwho\b[^?]*\b(more|less|older|younger|taller|shorter|bigger|larger"
               r"|higher|lower|better|worse|earlier|later)\b", re.I),
    re.compile(r"\bwhich\b[^?]*\b(more|less|older|younger|taller|shorter|bigger|larger"
               r"|higher|lower|better|worse|earlier|later)\b", re.I),
)

# "shares a border with ..." is a spatial chain, not an attribute comparison, and it
# is common in MuSiQue's 3-hop questions.
_BORDER = re.compile(r"\bshares? (a |the )?(land )?border", re.I)


def is_comparison(question: str) -> bool:
    """True for questions that compare two entities or ask what they share."""
    text = question or ""
    if _BORDER.search(text):
        return False
    return any(p.search(text) for p in _COMPARE_PATTERNS)


def _apply_compare(plan: SearchPlan, question: str, use_compare: bool) -> SearchPlan:
    """Force the comparison route when the question shape says so.

    Deterministic on purpose: the answer protocol for these questions differs, and
    a planner that forgets to say "compare" would silently fall back to the
    single-entity instructions. `graph` goes on because both entities' documents
    are needed; rounds stay whatever the budget allowed.
    """
    if not use_compare or not is_comparison(question):
        return plan
    plan.mode = SearchMode.COMPARE
    plan.graph = True
    marker = "comparison question: both entities needed"
    plan.reason = f"{plan.reason} | {marker}" if plan.reason else marker
    return plan


def make_plan(question: str, signal: RouteSignal, budget: StageBudget,
              client: ChatClient | None = None, *, use_llm: bool = True,
              force_mode: SearchMode | None = None,
              use_compare: bool = True) -> SearchPlan:
    if force_mode is not None:
        base = heuristic(question, signal, budget)
        base.mode = force_mode
        base.reason = f"forced by config ({force_mode.value})"
        base.source = "forced"
        if force_mode is SearchMode.KEEP:
            base.queries, base.graph = [question], False
        elif force_mode in (SearchMode.GRAPH, SearchMode.COMPARE):
            base.graph = True
        return base
    if client is None or not use_llm:
        return _apply_compare(heuristic(question, signal, budget), question, use_compare)

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
        return _apply_compare(fallback, question, use_compare)

    return _apply_compare(coerce(obj, question, budget, source="llm"), question, use_compare)


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
    if mode in (SearchMode.GRAPH, SearchMode.COMPARE):
        graph = True
    elif mode is SearchMode.KEEP:
        graph = False

    rounds = _int(obj.get("rounds"), budget.rounds)
    if mode in (SearchMode.GRAPH, SearchMode.COMPARE):
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
