"""Score Model A checkpoints on the 100 evaluation questions — honestly.

    python agentic_rag_v2/scripts/eval_model_a_holdout.py
    python agentic_rag_v2/scripts/eval_model_a_holdout.py --checkpoints E5,E6

Labels come from `air/data/processed_clean/eval_holdout.jsonl`, i.e. the AIR label
records of the evaluation questions **after they were removed from every training
split** (see scripts/make_clean_split.py). A checkpoint trained on
`processed/` (E5 and earlier) has seen these questions; E6 has not. Running both
through the same path makes the difference visible instead of assumed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HOLDOUT = ROOT / "air" / "data" / "processed_clean" / "eval_holdout.jsonl"
AIR_SRC = ROOT / "air" / "src"
sys.path.insert(0, str(AIR_SRC))

from common import COMPLEXITY, HOP_BUCKETS, INTENTS  # noqa: E402


def load_holdout() -> list[dict]:
    if not HOLDOUT.exists():
        raise SystemExit(f"missing {HOLDOUT}; run scripts/make_clean_split.py first")
    return [json.loads(line) for line in
            HOLDOUT.read_text(encoding="utf-8").splitlines() if line.strip()]


def predict(analyzer, question: str) -> dict:
    signal = analyzer.predict(question)
    return {"intent": signal.intent, "intents": signal.intents,
            "complexity": signal.complexity, "multihop": signal.multihop,
            "multihop_prob": signal.multihop_prob, "hop": signal.hop_bucket}


def score(rows: list[dict], preds: list[dict]) -> dict:
    n = len(rows)
    hop_ok = sum(1 for r, p in zip(rows, preds) if p["hop"] == r["hop_count_bucket"])
    cx_ok = sum(1 for r, p in zip(rows, preds) if p["complexity"] == r["complexity"])
    mh_ok = sum(1 for r, p in zip(rows, preds)
                if bool(p["multihop"]) == bool(r["requires_multi_hop"]))
    intent_ok = 0
    intent_total = 0
    for r, p in zip(rows, preds):
        gold = set(r["intents"])
        pred = {k for k, v in (p["intents"] or {}).items() if v >= 0.5}
        intent_ok += len(gold & pred)
        intent_total += len(gold) + len(pred)
    intent_f1 = (2 * intent_ok / intent_total) if intent_total else 0.0
    return {
        "n": n,
        "hop_acc": round(hop_ok / n, 4),
        "complexity_acc": round(cx_ok / n, 4),
        "multihop_acc": round(mh_ok / n, 4),
        "intent_f1": round(intent_f1, 4),
        "mean": round((hop_ok + cx_ok + mh_ok) / (3 * n), 4),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default="E5,E6")
    ap.add_argument("--tag", default="model_a_holdout")
    args = ap.parse_args(argv)

    rows = load_holdout()
    print(f"holdout questions: {len(rows)}")
    print("gold label distribution:", {
        "hop": dict(Counter(r["hop_count_bucket"] for r in rows)),
        "complexity": dict(Counter(r["complexity"] for r in rows)),
        "multihop": dict(Counter(bool(r["requires_multi_hop"]) for r in rows)),
    })

    from agentic_rag_v2.models.model_a import ModelAAnalyzer

    results: dict[str, dict] = {}
    for name in [c.strip() for c in args.checkpoints.split(",") if c.strip()]:
        analyzer = ModelAAnalyzer(checkpoint=name)
        if not analyzer.ok:
            print(f"{name}: UNAVAILABLE {analyzer.error}")
            continue
        preds = [predict(analyzer, r["question"]) for r in rows]
        results[name] = score(rows, preds)
        print(f"{name:>4}  " + "  ".join(
            f"{k}={v}" for k, v in results[name].items() if k != "n"))

    out = ROOT / "agentic_rag_v2" / "results" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps({"n": len(rows), "checkpoints": results,
                    "note": "labels from air/data/processed_clean/eval_holdout.jsonl "
                            "(eval questions removed from all training splits); "
                            "E5 and earlier saw these questions, E6 did not"},
                   indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", out / "summary.json")


if __name__ == "__main__":
    main()
