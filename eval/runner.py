"""Multi-system evaluation runner (same protocol as the v1 harness).

Every system sees the same questions, the same shared corpus and the same gold
file; only the pipeline options differ. Records are written per system so paired,
per-question comparisons (McNemar) stay possible.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from agentic_rag_v2 import config
from agentic_rag_v2.config import RESULTS, SEED
from agentic_rag_v2.data import load_chunks, load_gold, load_questions
from agentic_rag_v2.eval.metrics import answer_metrics, evidence_recall, summarize
from agentic_rag_v2.llm import ArkClient
from agentic_rag_v2.models.model_a import ModelAAnalyzer
from agentic_rag_v2.models.reranker import LocalReranker
from agentic_rag_v2.pipeline import AgenticRAG, PipelineOptions
from agentic_rag_v2.retrieval import HybridRetriever
from agentic_rag_v2.retrieval.engine import index_stats
from agentic_rag_v2.types import PipelineResult, SearchMode


@dataclass(frozen=True)
class SystemSpec:
    name: str
    options: PipelineOptions
    baseline: bool = False


SYSTEMS: dict[str, SystemSpec] = {
    # full pipeline: 2 LLM calls + local judge + local rerank
    "v2_full": SystemSpec("v2_full", PipelineOptions()),
    # one stage switched off at a time (paired against v2_full in the same batch)
    "v2_no_dense": SystemSpec("v2_no_dense", PipelineOptions(use_dense=False)),
    # same pipeline, E5 served by the vLLM container instead of in-process
    "v2_vllm": SystemSpec("v2_vllm", PipelineOptions(embed_backend="vllm")),
    "v2_no_rerank": SystemSpec("v2_no_rerank", PipelineOptions(rerank="score")),
    "v2_ce_rerank": SystemSpec("v2_ce_rerank", PipelineOptions(rerank="local")),
    "v2_ce_rerank_kg": SystemSpec("v2_ce_rerank_kg",
                                  PipelineOptions(rerank="local", use_relations=True)),
    # diagnosis arms: which v1 component explains the EM gap?
    "v2_v1prompt": SystemSpec("v2_v1prompt", PipelineOptions(answer_prompt="v1")),
    "v2_no_rel": SystemSpec("v2_no_rel", PipelineOptions(use_relations=False)),
    "v2_kg": SystemSpec("v2_kg", PipelineOptions(use_relations=True)),
    "v2_rounds4": SystemSpec("v2_rounds4", PipelineOptions(max_rounds=4)),
    "v2_llm_rerank": SystemSpec("v2_llm_rerank", PipelineOptions(rerank="llm")),
    "v2_no_judge": SystemSpec("v2_no_judge", PipelineOptions(use_judge=False)),
    "v2_llm_judge": SystemSpec("v2_llm_judge", PipelineOptions(judge_kind="llm")),
    # P1 answer-side 2x2 (docs/PLAN_2026-11-16.md §5): prompt/verifier x compare route
    "v2_llm_v2prompt": SystemSpec("v2_llm_v2prompt",
                                  PipelineOptions(judge_kind="llm", answer_prompt="v2",
                                                  use_compare=False)),
    "v2_llm_v2prompt_compare": SystemSpec("v2_llm_v2prompt_compare",
                                          PipelineOptions(judge_kind="llm", answer_prompt="v2")),
    "v2_llm_v3prompt": SystemSpec("v2_llm_v3prompt",
                                  PipelineOptions(judge_kind="llm", use_compare=False)),
    # does the cross-encoder ordering, or the controller, explain v1's answer-side
    # advantage? score order + LLM controller = exactly v1's combination.
    "v2_llm_judge_norerank": SystemSpec("v2_llm_judge_norerank",
                                        PipelineOptions(judge_kind="llm", rerank="score")),
    "v2_llm_judge_kg": SystemSpec("v2_llm_judge_kg",
                                  PipelineOptions(judge_kind="llm", rerank="score",
                                                  use_relations=True)),
    # 排序 A/B 的另一档：LLM judge + cross-encoder 顺序（= 批次 v2final/v2_llm_judge）
    "v2_ce_rerank_llmjudge": SystemSpec("v2_ce_rerank_llmjudge",
                                        PipelineOptions(judge_kind="llm", rerank="local")),
    "v2_no_planner": SystemSpec("v2_no_planner", PipelineOptions(use_planner_llm=False)),
    # forced route: isolates the planner's decision from the rest of the system
    "v2_keep": SystemSpec("v2_keep", PipelineOptions(force_mode=SearchMode.KEEP)),
    # no-retrieval lower bound
    "v2_direct": SystemSpec("v2_direct", PipelineOptions(), baseline=True),
}

SYSTEM_ORDER = ["v2_direct", "v2_full", "v2_no_dense", "v2_vllm", "v2_no_rerank",
                "v2_ce_rerank", "v2_ce_rerank_kg", "v2_no_rel", "v2_kg", "v2_v1prompt",
                "v2_rounds4", "v2_llm_rerank",
                "v2_no_judge", "v2_llm_judge", "v2_llm_v2prompt", "v2_llm_v2prompt_compare",
                "v2_llm_v3prompt", "v2_llm_judge_norerank",
                "v2_llm_judge_kg", "v2_ce_rerank_llmjudge",
                "v2_no_planner", "v2_keep"]


def build_runner(rag: AgenticRAG, spec: SystemSpec):
    if spec.baseline:
        return rag.answer_baseline
    return rag.answer


def score_record(q: dict, gold: dict, result: PipelineResult) -> dict:
    gold_rec = gold[q["id"]]
    metrics = answer_metrics(result.answer, gold_rec["answer"], gold_rec.get("aliases"))
    record = {
        "qid": q["id"],
        "dataset": gold_rec.get("dataset") or q.get("source"),
        "qtype": gold_rec.get("qtype"),
        "n_hops": gold_rec.get("n_hops"),
        "gold": gold_rec["answer"],
        "evidence_recall_retrieved": round(
            evidence_recall(result.evidence.doc_ids(), gold_rec["gold_doc_ids"]), 4),
        "evidence_recall_final": round(
            evidence_recall(result.cited_docs, gold_rec["gold_doc_ids"]), 4),
        **metrics,
    }
    record.update(result.to_record())
    return record


def _make_rag(client, retriever, analyzer, reranker, options: PipelineOptions,
              semantic=None) -> AgenticRAG:
    return AgenticRAG(client=client, retriever=retriever, analyzer=analyzer,
                      reranker=reranker, semantic=semantic, options=options)


def _build_semantics(names: list[str], retriever: HybridRetriever) -> dict:
    """One semantic index per backend, shared by every arm that needs it.

    Building the index (or loading its cache) is the expensive part; doing it once
    per batch also keeps all arms on byte-identical passage vectors.
    """
    from agentic_rag_v2 import config
    from agentic_rag_v2.models.embedder import build_embedder
    from agentic_rag_v2.retrieval.vector import SemanticRetriever, build_index

    backends = {(SYSTEMS[n].options.embed_backend or config.EMBED_BACKEND)
                for n in names
                if SYSTEMS[n].options.use_dense and not SYSTEMS[n].baseline}
    built: dict = {}
    for backend in sorted(backends):
        try:
            embedder = build_embedder(backend)
            index, stats = build_index(retriever.chunks, embedder)
            built[backend] = SemanticRetriever(index, embedder)
            print(f"[semantic:{backend}] {stats}", flush=True)
        except Exception as e:  # noqa: BLE001 - fall back to lexical-only arms
            print(f"[semantic:{backend}] UNAVAILABLE {type(e).__name__}: {e}", flush=True)
            built[backend] = None
    return built


def run_suite(names: list[str], questions: list[dict], gold: dict, *,
              workers: int = 4, tag: str = "run", use_cache: bool = True) -> dict:
    for name in names:
        if name not in SYSTEMS:
            raise SystemExit(f"unknown system {name}; known: {SYSTEM_ORDER}")

    outdir = RESULTS / tag
    outdir.mkdir(parents=True, exist_ok=True)

    client = ArkClient(use_cache=use_cache)
    t_index = time.perf_counter()
    retriever = HybridRetriever(load_chunks())
    index_seconds = round(time.perf_counter() - t_index, 1)
    analyzer = ModelAAnalyzer()
    analyzer.require()          # Model A is mandatory; no arm may run without it
    reranker = (LocalReranker()
                if any(SYSTEMS[n].options.rerank == "local" for n in names) else None)
    semantics = _build_semantics(names, retriever)

    manifest = {
        "tag": tag, "systems": names, "n_questions": len(questions), "workers": workers,
        "seed": SEED, "index_seconds": index_seconds,
        "index": index_stats(retriever),
        "model_a": {"ok": analyzer.ok, "error": analyzer.error,
                    "checkpoint": analyzer.checkpoint_name},
        "reranker": reranker.describe() if reranker else None,
        "semantic": {k: (v.describe() if v else None) for k, v in semantics.items()},
        "model": client.model,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(json.dumps(manifest, indent=1, ensure_ascii=False), flush=True)

    summary: list[dict] = []
    for name in names:
        spec = SYSTEMS[name]
        backend = spec.options.embed_backend or config.EMBED_BACKEND
        semantic = semantics.get(backend) if (spec.options.use_dense and not spec.baseline) else None
        rag = _make_rag(client, retriever, analyzer, reranker, spec.options, semantic)
        runner = build_runner(rag, spec)

        t0 = time.time()
        records: list[dict] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(runner, q["question"]): q for q in questions}
            for i, fut in enumerate(as_completed(futures), 1):
                q = futures[fut]
                try:
                    records.append(score_record(q, gold, fut.result()))
                except Exception:  # noqa: BLE001 - one bad question must not kill the run
                    traceback.print_exc()
                if i % 10 == 0:
                    print(f"  [{name}] {i}/{len(questions)}", flush=True)
        records.sort(key=lambda r: r["qid"])
        with (outdir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        stats = summarize(records)
        stats["wall_seconds"] = round(time.time() - t0, 1)
        summary.append({"system": name, **stats})
        print(f"[{name}] EM={stats.get('em')} F1={stats.get('f1')} "
              f"rec_ret={stats.get('evidence_recall_retrieved')} "
              f"calls={stats.get('avg_llm_calls')} lat={stats.get('avg_latency_ms')}ms "
              f"wall={stats['wall_seconds']}s", flush=True)

    manifest["usage"] = client.usage.snapshot()
    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    payload = {"manifest": manifest, "summary": summary}
    (outdir / "summary.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                                         encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="v2_full")
    ap.add_argument("--datasets", default="hotpotqa_50,musique_50")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--qids", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="v2run")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)

    names = [s.strip() for s in args.systems.split(",") if s.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    qids = [q.strip() for q in args.qids.split(",") if q.strip()] or None
    questions = load_questions(datasets, args.n, qids)
    gold = load_gold()
    payload = run_suite(names, questions, gold, workers=args.workers,
                        tag=args.tag, use_cache=not args.no_cache)
    print("\n" + json.dumps(payload["summary"], indent=1, ensure_ascii=False))



