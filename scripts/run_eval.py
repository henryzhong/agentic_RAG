"""CLI entry point for the v2 evaluation suite.

    python agentic_rag_v2/scripts/run_eval.py --systems v2_full --n 10 --tag v2smoke
    python agentic_rag_v2/scripts/run_eval.py \
        --systems v2_full,v2_no_rerank,v2_no_judge,v2_no_planner --tag v2ab
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# local weights are already cached; a hub HEAD request must never block a run
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agentic_rag_v2.eval.runner import main  # noqa: E402

if __name__ == "__main__":
    main()
