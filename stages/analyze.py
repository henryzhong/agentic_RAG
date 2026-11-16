"""Stage 1 — analyze: Model A signal -> deterministic search budget."""
from __future__ import annotations

from agentic_rag_v2.config import (ANSWER_TOP_N, DEFAULT_TOP_K, MAX_ROUNDS,
                                   MAX_TOP_K, MIN_TOP_K, BudgetPolicy)
from agentic_rag_v2.types import RouteSignal, StageBudget


def analyze(question: str, analyzer, policy: BudgetPolicy | None = None) -> tuple[RouteSignal, StageBudget]:
    """Run the local classifier and translate it into hard limits.

    The classifier is a hard dependency (the pipeline refuses to start without a
    loadable checkpoint), so `budget_for` only needs a defensive default for the
    `available=False` signal used by unit tests of the budget policy itself.
    """
    policy = policy or BudgetPolicy()
    signal = analyzer.predict(question)
    return signal, budget_for(signal, policy)


def budget_for(signal: RouteSignal, policy: BudgetPolicy | None = None) -> StageBudget:
    policy = policy or BudgetPolicy()
    hop = signal.hop_bucket if signal.available else "2"
    rounds = policy.round_cap_by_hop.get(hop, 2)
    top_k = policy.top_k_by_hop.get(hop, DEFAULT_TOP_K)

    if signal.available and signal.complexity == "high":
        rounds += policy.high_complexity_bonus
        top_k += 2
    if signal.available and signal.multihop_prob < policy.low_multihop_prob:
        rounds = 1

    return StageBudget(
        rounds=max(1, min(rounds, MAX_ROUNDS)),
        top_k=max(MIN_TOP_K, min(top_k, MAX_TOP_K)),
        answer_top_n=ANSWER_TOP_N,
        source="model_a" if signal.available else "default",
    )
