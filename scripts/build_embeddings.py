"""Build (or verify) the semantic index for the shared corpus.

    python agentic_rag_v2/scripts/build_embeddings.py                 # local E5, cached
    python agentic_rag_v2/scripts/build_embeddings.py --force          # rebuild
    python agentic_rag_v2/scripts/build_embeddings.py --backend vllm   # use the server
    python agentic_rag_v2/scripts/build_embeddings.py --verify         # dense vs bm25 recall

The first build embeds every corpus chunk once (~10 s on the 4070 Ti Super) and
caches it under `agentic_rag_v2/work/embeddings/<slug>.npz`; every later run
loads the cache instead.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_rag_v2.config import EMBED_BACKEND, e5_model_dir  # noqa: E402
from agentic_rag_v2.data import load_chunks, load_gold, load_questions  # noqa: E402
from agentic_rag_v2.models.embedder import build_embedder  # noqa: E402
from agentic_rag_v2.retrieval import HybridRetriever  # noqa: E402
from agentic_rag_v2.retrieval.vector import SemanticRetriever, build_index  # noqa: E402


def verify(semantic: SemanticRetriever, retriever: HybridRetriever, *,
           datasets: list[str], n: int) -> dict:
    """Dense vs BM25 doc recall@k — retrieval quality independent of the answerer."""
    gold = load_gold()
    questions = load_questions(datasets, n)
    out: dict = {"n": len(questions)}
    for k in (4, 8, 12):
        dense_hits = bm25_hits = 0.0
        for q in questions:
            gold_docs = set(gold[q["id"]]["gold_doc_ids"])
            dense = semantic.search(q["question"], k=k)
            bm25 = retriever.bm25_search(q["question"], k=k)
            dense_hits += len({retriever.by_id[c].doc_id for c, _ in dense} & gold_docs) / len(gold_docs)
            bm25_hits += len({retriever.by_id[c].doc_id for c, _ in bm25} & gold_docs) / len(gold_docs)
        out[f"dense_recall@{k}"] = round(dense_hits / len(questions), 4)
        out[f"bm25_recall@{k}"] = round(bm25_hits / len(questions), 4)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=EMBED_BACKEND, choices=["local", "vllm"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--datasets", default="hotpotqa_50,musique_50")
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args(argv)

    model_dir = e5_model_dir()
    print(json.dumps({"backend": args.backend,
                      "model_dir": str(model_dir) if model_dir else None}, indent=1))
    if args.backend == "local" and model_dir is None:
        raise SystemExit("multilingual-e5-large not found; set RAGV2_E5_DIR")

    chunks = load_chunks()
    embedder = build_embedder(args.backend)
    index, stats = build_index(chunks, embedder, force=args.force, progress=True)
    print(json.dumps(stats, indent=1, ensure_ascii=False))

    if args.verify:
        retriever = HybridRetriever(chunks)
        semantic = SemanticRetriever(index, embedder)
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        print(json.dumps(verify(semantic, retriever, datasets=datasets, n=args.n),
                         indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
