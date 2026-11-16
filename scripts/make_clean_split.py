"""Build a leakage-free AIR dataset for Model A, given the v2 eval questions.

    python agentic_rag_v2/scripts/make_clean_split.py [--dry-run]

Why this exists
---------------
`air/src/build_dataset.py` samples HotpotQA **dev** and MuSiQue **dev** and then
splits them by a hash of the sample id. The v2 benchmark (and the v1 one) samples
*the same two dev files*, so 83/100 evaluation questions ended up inside Model A's
`train.jsonl` — see `scripts/audit_leakage.py`. Any Model A accuracy measured on
those questions is memorisation, not generalisation.

What it does
------------
Copy `air/data/processed/` to `air/data/processed_clean/` and, for every split
file, drop samples whose question matches an evaluation question:
  * exact match after `air/src/common.py::norm_question`, or
  * near duplicate (token Jaccard >= --jaccard, default 0.85)
The dropped samples are written to `eval_holdout.jsonl` (labels kept) so Model A
can be scored on genuinely unseen questions.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmark"
SRC = ROOT / "air" / "data" / "processed"
DST = ROOT / "air" / "data" / "processed_clean"
SPLITS = ("train.jsonl", "validation.jsonl", "test.jsonl",
          "hard_negatives.jsonl", "ood_test.jsonl")


def norm_question(q: str) -> str:
    q = (q or "").lower().strip()
    q = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def tokens(q: str) -> set[str]:
    return set(norm_question(q).split())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_eval_questions() -> list[dict]:
    out: list[dict] = []
    for name in ("hotpotqa_50", "musique_50"):
        for line in (BENCH / f"{name}.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out.append({"qid": rec["id"], "question": rec["question"],
                            "norm": norm_question(rec["question"]),
                            "tokens": tokens(rec["question"])})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jaccard", type=float, default=0.85)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--dst", default=str(DST))
    args = ap.parse_args(argv)

    src, dst = Path(args.src), Path(args.dst)
    eval_qs = load_eval_questions()
    exact = {e["norm"]: e for e in eval_qs}
    print(f"eval questions: {len(eval_qs)}  src={src}  dst={dst}  jaccard>={args.jaccard}")

    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    removed: list[dict] = []
    stats: dict[str, tuple[int, int]] = {}
    for name in SPLITS:
        path = src / name
        if not path.exists():
            continue
        kept, dropped = [], []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            q = norm_question(rec.get("question", ""))
            if not q:
                kept.append(rec)
                continue
            hit = exact.get(q)
            if hit is None:
                qt = tokens(rec.get("question", ""))
                for e in eval_qs:
                    if jaccard(qt, e["tokens"]) >= args.jaccard:
                        hit = e
                        break
            if hit is None:
                kept.append(rec)
            else:
                dropped.append((rec, hit["qid"]))
        stats[name] = (len(kept), len(dropped))
        removed.extend({"qid": qid, "split": name, **rec} for rec, qid in dropped)
        print(f"  {name:<24} keep={len(kept):>6}  drop={len(dropped):>3}")
        if not args.dry_run:
            (dst / name).write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n",
                encoding="utf-8")

    # non-split files (decomposition etc.) are copied verbatim on purpose: Model A
    # trains from train/validation/hard_negatives only, so `decomposition_train.jsonl`
    # keeps its 58/100 eval overlap without leaking into the classifier

    if not args.dry_run:
        for path in src.iterdir():
            if path.name in SPLITS or path.is_dir():
                continue
            shutil.copy2(path, dst / path.name)

        # one label record per eval question, for honest held-out scoring
        by_qid: dict[str, dict] = {}
        for rec in removed:
            by_qid.setdefault(rec["qid"], rec)
        missing = [e["qid"] for e in eval_qs if e["qid"] not in by_qid]
        out = dst / "eval_holdout.jsonl"
        out.write_text("\n".join(json.dumps(by_qid[q], ensure_ascii=False)
                                 for q in sorted(by_qid)) + "\n", encoding="utf-8")
        print(f"\nwrote {out} ({len(by_qid)} of {len(eval_qs)} eval questions had labels)")
        if missing:
            print(f"  no label record for: {missing}")

    total_keep = sum(k for k, _ in stats.values())
    total_drop = sum(d for _, d in stats.values())
    print(f"\ntotal keep={total_keep}  total dropped={total_drop}")


if __name__ == "__main__":
    main()
