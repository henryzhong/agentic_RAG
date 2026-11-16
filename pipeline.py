"""Pipeline orchestrator: two LLM calls, local intelligence in between.

    analyze(question)          Model A (local)      -> RouteSignal + StageBudget
    plan(question, signal)     DeepSeek #1          -> SearchPlan (route/rewrite/graph)
    run_search(plan)           local retrieval      -> Evidence (+ local verdicts)
    rerank(evidence)           local cross-encoder  -> RerankResult
    answer(evidence)           DeepSeek #2          -> sufficiency + answer + citations

`PipelineOptions` is the ablation surface: every stage can be switched off or
forced, which is how the evaluation compares variants without forking code.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from agentic_rag_v2 import config
from agentic_rag_v2.llm import ArkClient, ChatClient
from agentic_rag_v2.models.judge import CoverageJudge, EvidenceJudge, LLMJudge
from agentic_rag_v2.models.model_a import ModelAAnalyzer
from agentic_rag_v2.models.reranker import LocalReranker
from agentic_rag_v2.retrieval import HybridRetriever
from agentic_rag_v2.retrieval.vector import SemanticRetriever, build_index
from agentic_rag_v2.stages.analyze import analyze
from agentic_rag_v2.stages.answer import answer, answer_direct
from agentic_rag_v2.stages.plan import make_plan
from agentic_rag_v2.stages.rerank import rerank
from agentic_rag_v2.stages.search import run_search
from agentic_rag_v2.types import (Evidence, PipelineResult, SearchMode,
                                  SearchPlan, StageBudget)


@dataclass
class PipelineOptions:
    # Model A is not switchable: every question goes through stage 1 (see README
    # §4.1). Only stages *after* the signal are ablation-tunable.
    use_planner_llm: bool = True    # stage 2 route + rewrite + plan
    use_judge: bool = True          # stage 3 adaptive stopping
    judge_kind: str = "coverage"    # coverage | llm
    use_dense: bool = True          # stage 3 semantic (E5) route
    use_relations: bool = False     # KG relations in the answer prompt: measured -3pt EM,
                                    # see README §9.3 (the model answers the related entity)
    embed_backend: str | None = None  # local | vllm | None (None -> config default)
    rerank: str = "score"           # score | local | llm. Measured: cross-encoder
                                    # ordering improves gold-doc rank@k but costs ~7pt
                                    # EM on multi-hop answer selection (README §9.3)
    answer_prompt: str = "v3"       # v3 (answer candidates + verifier) | v2 | v1
    use_compare: bool = True        # deterministic comparison-question routing
    force_mode: SearchMode | None = None
    max_rounds: int | None = None
    answer_top_n: int = config.ANSWER_TOP_N
    answer_max_chars: int = config.ANSWER_MAX_CHARS
    pool_cap: int = config.POOL_CAP
    parallel_routes: bool = True


class AgenticRAG:
    """One instance per process; the heavy objects (index, models) are built once."""

    def __init__(self, *, client: ChatClient | None = None,
                 retriever: HybridRetriever | None = None,
                 analyzer=None, judge: EvidenceJudge | None = None,
                 reranker: LocalReranker | None = None, semantic=None,
                 options: PipelineOptions | None = None, chunks=None):
        self.options = options or PipelineOptions()
        self.client = client or ArkClient()
        if retriever is None:
            from agentic_rag_v2.data import load_chunks
            chunks = chunks if chunks is not None else load_chunks()
            retriever = HybridRetriever(chunks)
        self.retriever = retriever
        if analyzer is None:
            analyzer = ModelAAnalyzer()
        analyzer.require()
        self.analyzer = analyzer

        self.semantic_error: str | None = None
        self.semantic_stats: dict = {}
        if semantic is not None:
            self.semantic = semantic
        elif self.options.use_dense:
            self.semantic = self._build_semantic()
        else:
            self.semantic = None

        self.relations = None
        if self.options.use_relations:
            from agentic_rag_v2.retrieval.relations import get_store
            self.relations = get_store()

        if judge is not None:
            self.judge = judge
        elif not self.options.use_judge:
            self.judge = None
        elif self.options.judge_kind == "llm":
            self.judge = LLMJudge(self.client)
        else:
            self.judge = CoverageJudge(self.retriever)

        if reranker is not None:
            self.reranker = reranker
        elif self.options.rerank == "local":
            self.reranker = LocalReranker()
        else:
            self.reranker = None

    # -------------------------------------------------------------- semantic
    def _build_semantic(self) -> "SemanticRetriever | None":
        """Embed the corpus once (cached on disk) and wrap it as a retrieval tool."""
        from agentic_rag_v2.models.embedder import build_embedder

        backend = self.options.embed_backend or config.EMBED_BACKEND
        try:
            embedder = build_embedder(backend)
            index, stats = build_index(self.retriever.chunks, embedder)
            self.semantic_stats: dict = stats
            return SemanticRetriever(index, embedder)
        except Exception as e:  # noqa: BLE001 - semantic retrieval is an optimisation
            self.semantic_error = f"{type(e).__name__}: {e}"
            self.semantic_stats = {"error": self.semantic_error}
            return None

    # ------------------------------------------------------------------ main
    def answer(self, question: str) -> PipelineResult:
        t_start = time.perf_counter()
        trace: list[dict] = []

        signal, budget = analyze(question, self.analyzer)
        if self.options.max_rounds is not None:
            budget.rounds = max(1, min(budget.rounds, self.options.max_rounds))
        trace.append({"stage": "analyze", "route": signal.to_dict(), "budget": budget.to_dict()})

        plan = make_plan(question, signal, budget, self.client,
                         use_llm=self.options.use_planner_llm,
                         force_mode=self.options.force_mode,
                         use_compare=self.options.use_compare)
        trace.append({"stage": "plan", "plan": plan.to_dict()})

        search = run_search(question, plan, self.retriever, self.judge,
                            pool_cap=self.options.pool_cap,
                            parallel=self.options.parallel_routes,
                            semantic=self.semantic)
        trace.append({"stage": "search", "rounds": search.rounds,
                      "n_evidence": len(search.evidence), "rounds_trace": search.trace})

        ranking = rerank(question, search.evidence, reranker=self.reranker,
                         client=self.client, strategy=self.options.rerank)
        trace.append({"stage": "rerank", "order": ranking.order[:12],
                      "source": ranking.source, "reason": ranking.reason})

        result = answer(question, search.evidence, self.client, order=ranking.order,
                        top_n=self.options.answer_top_n,
                        max_chars=self.options.answer_max_chars,
                        retriever=self.retriever,
                        relations=self.relations,
                        prompt=self.options.answer_prompt,
                        mode=plan.mode)
        trace.append({"stage": "answer", "answer": result.to_dict()})

        llm_calls = (1 if plan.source == "llm" else 0) + search.llm_calls + ranking.llm_calls + 1
        if llm_calls > config.LLM_CALLS_PER_QUESTION:
            trace.append({"stage": "budget",
                          "warning": f"{llm_calls} LLM calls > budget "
                                     f"{config.LLM_CALLS_PER_QUESTION}"})
        return PipelineResult(
            question=question,
            answer=result.answer,
            cited_docs=result.cited_docs,
            evidence=search.evidence,
            route=signal,
            budget=budget,
            plan=plan,
            verdicts=search.verdicts,
            ranking=ranking.order,
            answer_result=result,
            trace=trace,
            rounds=search.rounds,
            llm_calls=llm_calls,
            latency_ms=(time.perf_counter() - t_start) * 1000,
        )

    def answer_baseline(self, question: str) -> PipelineResult:
        """No-retrieval lower bound (same result contract as `answer`).

        Model A still runs: the lower bound must be attributable, and no arm is
        allowed to skip stage 1 (otherwise the arms are not comparable).
        """
        t0 = time.perf_counter()
        signal, budget = analyze(question, self.analyzer)
        result = answer_direct(question, self.client)
        plan = SearchPlan(mode=SearchMode.KEEP, queries=[question], rounds=0, top_k=0,
                          reason="baseline: no retrieval", source="baseline")
        return PipelineResult(
            question=question,
            answer=result.answer,
            cited_docs=[],
            evidence=Evidence(),
            route=signal,
            budget=StageBudget(rounds=0, top_k=0, answer_top_n=0, source="baseline"),
            plan=plan,
            answer_result=result,
            trace=[{"stage": "analyze", "route": signal.to_dict(),
                    "budget": budget.to_dict()},
                   {"stage": "direct", "answer": result.to_dict()}],
            rounds=0,
            llm_calls=1,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    # ------------------------------------------------------------- manifest
    def manifest(self) -> dict:
        from agentic_rag_v2.retrieval.engine import index_stats

        return {
            "prompt_version": config.PROMPT_VERSION,
            "model": getattr(self.client, "model", type(self.client).__name__),
            "llm_calls_per_question": config.LLM_CALLS_PER_QUESTION,
            "options": {k: (v.value if isinstance(v, SearchMode) else v)
                        for k, v in self.options.__dict__.items()},
            "index": index_stats(self.retriever),
            "model_a": {"ok": getattr(self.analyzer, "ok", None),
                        "error": getattr(self.analyzer, "error", None),
                        "checkpoint": getattr(self.analyzer, "checkpoint_name", None),
                        "path": getattr(self.analyzer, "checkpoint", None)},
            "judge": type(self.judge).__name__ if self.judge else None,
            "reranker": self.reranker.describe() if self.reranker else None,
            "semantic": (self.semantic.describe() if self.semantic else
                         {"disabled": True, "error": self.semantic_error}),
            "relations": self.relations.describe() if self.relations else None,
        }
