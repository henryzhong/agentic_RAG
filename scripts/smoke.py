"""Live smoke run: a handful of questions through the full pipeline.

    python agentic_rag_v2/scripts/smoke.py --n 4
    python agentic_rag_v2/scripts/smoke.py --qids hp-001,hp-007 --trace
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# local weights are already cached; a hub HEAD request must never block a run
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_rag_v2.data import load_gold, load_questions  # noqa: E402
from agentic_rag_v2.eval.metrics import em  # noqa: E402
from agentic_rag_v2.pipeline import AgenticRAG, PipelineOptions  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="hotpotqa_50")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--qids", default="")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--no-llm", action="store_true", help="skip decide/rerank/judge LLM stages")
    ap.add_argument("--judge", default="coverage", choices=["coverage", "llm"],
                    help="coverage = 2-call default; llm = v1-style assess (best EM)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    qids = [q.strip() for q in args.qids.split(",") if q.strip()] or None
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    questions = load_questions(datasets, args.n, qids)
    gold = load_gold()

    options = PipelineOptions(use_planner_llm=not args.no_llm,
                              use_judge=not args.no_llm,
                              judge_kind=args.judge,
                              rerank="score" if args.no_llm else PipelineOptions().rerank)
    rag = AgenticRAG(options=options)
    print(json.dumps(rag.manifest(), indent=1, ensure_ascii=False), flush=True)

    hits = 0
    for q in questions:
        result = rag.answer(q["question"])
        record = result.to_record()
        hit = em(record["pred"], gold[q["id"]]["answer"])
        hits += hit
        if args.json:
            print(json.dumps({"qid": q["id"], "em": hit, **record}, ensure_ascii=False))
            continue
        print(f"\n=== {q['id']}  EM={hit:.0f}  mode={result.plan.mode.value} "
              f"rounds={result.rounds} calls={result.llm_calls} "
              f"lat={result.latency_ms:.0f}ms")
        print(f"Q   : {q['question']}")
        print(f"A   : {result.answer!r}   (gold {gold[q['id']]['answer']!r})")
        print(f"cites: {result.cited_docs}  pool={len(result.evidence)}  "
              f"recall={record['retrieved_docs'] and len(set(record['retrieved_docs']) & set(gold[q['id']]['gold_doc_ids']))}/{len(gold[q['id']]['gold_doc_ids'])}")
        if args.trace:
            print(json.dumps(record["trace"], indent=1, ensure_ascii=False))
    print(f"\nsmoke EM {hits}/{len(questions)}")


if __name__ == "__main__":
    main()
