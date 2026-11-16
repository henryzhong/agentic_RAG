"""Central configuration for Agentic RAG v2.

Everything that changes a result (model id, endpoints, prompt version, budgets,
caps) lives here so a run manifest can describe itself.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT = Path(__file__).resolve().parent            # .../agentic_rag_v2
ROOT = PROJECT.parent                                # parent workspace directory

WORK = PROJECT / "work"                              # v2 caches / ledgers
CACHE_DIR = WORK / "llm_cache"
EMBED_CACHE = WORK / "embeddings"
RESULTS = PROJECT / "results"
BENCH = ROOT / "benchmark"

# v1 artifacts we deliberately reuse instead of rebuilding:
#   corpus_chunks.jsonl (8070 shared chunks, anti-cheating corpus)
#   gold.json           (answers + gold doc ids, read only by the evaluator)
LEGACY_WORK = ROOT / "agentic_rag" / "work"
AIR_CONFIG = ROOT / "air" / "configs" / "ark.json"
AIR_SRC = ROOT / "air" / "src"
MODEL_A_DIR = ROOT / "air" / "checkpoints" / "model_a"

for _p in (WORK, CACHE_DIR, EMBED_CACHE, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42

# --------------------------------------------------------------- data roots
def data_root() -> Path:
    """Where corpus/gold live: v2's own data dir wins, else the v1 artifacts."""
    env = os.environ.get("RAGV2_DATA_ROOT")
    if env:
        return Path(env)
    local = PROJECT / "data"
    if (local / "corpus_chunks.jsonl").exists():
        return local
    return LEGACY_WORK


def corpus_path() -> Path:
    return data_root() / "corpus_chunks.jsonl"


def gold_path() -> Path:
    return data_root() / "gold.json"


# ------------------------------------------------------------------- LLM
DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
DEFAULT_MODEL = "deepseek-v4.1-flash"
MAX_CONCURRENCY = int(os.environ.get("ARK_MAX_CONCURRENCY", "4"))


@dataclass(frozen=True)
class ArkSettings:
    api_key: str
    base_url: str
    model: str
    reasoning_effort: str = "none"


def ark_settings(model: str | None = None) -> ArkSettings:
    cfg: dict = {}
    if AIR_CONFIG.exists():
        cfg = json.loads(AIR_CONFIG.read_text(encoding="utf-8"))
    key = os.environ.get("ARK_API_KEY") or cfg.get("api_key", "")
    base = os.environ.get("ARK_BASE_URL") or cfg.get("base_url") or DEFAULT_BASE_URL
    name = model or os.environ.get("ARK_MODEL") or cfg.get("model") or DEFAULT_MODEL
    effort = os.environ.get("ARK_EFFORT") or "none"
    if not key:
        raise RuntimeError("ARK_API_KEY / air/configs/ark.json missing api_key")
    return ArkSettings(api_key=key, base_url=base.rstrip("/"), model=name, reasoning_effort=effort)


# ---------------------------------------------------------------- Model A
# E6 was trained on a leakage-free split (air/data/processed_clean): the 100 eval
# questions were removed from every split first. E5 and earlier saw 83 of them.
MODEL_A_CHECKPOINT = os.environ.get("RAGV2_MODEL_A", "E6")

# ------------------------------------------------------------- retrieval
BM25_K = 8                # first-stage candidates per query
GRAPH_HOPS = 2            # entity-graph expansion depth
GRAPH_FANOUT = 6          # neighbours kept per expanded entity
POOL_CAP = 30             # hard cap on pooled evidence per question

# ----------------------------------------------------------------- budgets
MAX_ROUNDS = 4            # retrieval rounds in the adaptive loop
LLM_CALLS_PER_QUESTION = 2  # plan + answer; the ablation arms may add one more
DEFAULT_TOP_K = 8
MIN_TOP_K, MAX_TOP_K = 4, 16
PLAN_MAX_QUERIES = 3
ANSWER_TOP_N = 30         # passages shown to the answer model (v1-proven capacity)
ANSWER_MAX_CHARS = 40_000
JUDGE_DIGEST_TOP = 12
JUDGE_DIGEST_CHARS = 420

# --------------------------------------------------------------- local rerank
# Local cross-encoder (ONNX or transformers weights). Defaults to the reranker
# shipped with the hoprag harness; any BGE cross-encoder directory with the same
# layout can be dropped in via RAGV2_RERANKER_DIR.
RERANKER_DIR = Path(os.environ.get("RAGV2_RERANKER_DIR")
                    or (ROOT / "hoprag" / "models" / "reranker"))
RERANK_MAX_LEN = 512
RERANK_BATCH = 32

# ------------------------------------------------------ semantic (E5 vector)
# multilingual-e5-large: 1024-d, requires the "query: " / "passage: " prefixes.
# Two interchangeable backends behind one interface:
#   local -> transformers + mean pooling, in-process on the GPU (~10 ms/query)
#   vllm  -> OpenAI-compatible /v1/embeddings served from the vllm image (see README)
E5_CANDIDATE_DIRS = (
    Path(os.environ["RAGV2_E5_DIR"]) if os.environ.get("RAGV2_E5_DIR") else None,
    ROOT / "models" / "multilingual-e5-large",
    # optional second location of the same weights (e.g. a mounted model volume);
    # set RAGV2_E5_DIR_VLLM to enable it without hard-coding a local path
    Path(os.environ["RAGV2_E5_DIR_VLLM"]) if os.environ.get("RAGV2_E5_DIR_VLLM") else None,
    )
EMBED_BACKEND = os.environ.get("RAGV2_EMBED_BACKEND", "local").lower()   # local | vllm
EMBED_ENDPOINT = os.environ.get("RAGV2_EMBED_ENDPOINT", "http://127.0.0.1:18000/v1")
EMBED_MODEL_NAME = os.environ.get("RAGV2_EMBED_MODEL", "multilingual-e5-large")
EMBED_MAX_LEN = 512
EMBED_BATCH = 32
EMBED_DIM = 1024
DENSE_K_MULTIPLIER = 2      # dense candidates per top_k, like the bm25 stage


def e5_model_dir() -> Path | None:
    """First existing candidate directory, or None when the model is absent."""
    for candidate in E5_CANDIDATE_DIRS:
        if candidate and (candidate / "config.json").exists():
            return candidate
    return None


# ------------------------------------------------------- local coverage judge# A question is considered covered when its content words and its named entities
# are present in the pooled evidence AND at least one passage actually discusses
# the question (best single-passage coverage).
COVERAGE_TOKEN_THRESHOLD = 0.72
COVERAGE_ENTITY_THRESHOLD = 0.50
COVERAGE_BEST_PASSAGE_THRESHOLD = 0.50

# --------------------------------------------------------------- judging
# The local judge only runs when another round is still affordable inside the
# plan; a one-round question is answered straight away.

PROMPT_VERSION = "v3.0"


@dataclass(frozen=True)
class BudgetPolicy:
    """Deterministic caps derived from Model A's signal.

    Model A proposes; these caps dispose. v1 measured that letting the router
    raise the round budget alone saturated and changed nothing, so the caps here
    are intentionally asymmetric: a confident single-hop prediction is the only
    thing that *reduces* work, everything else gets the default ceiling.
    """

    round_cap_by_hop: dict[str, int] = field(
        default_factory=lambda: {"1": 1, "2": 2, "3+": 3})
    top_k_by_hop: dict[str, int] = field(
        default_factory=lambda: {"1": 6, "2": 8, "3+": 10})
    low_multihop_prob: float = 0.35
    high_complexity_bonus: int = 1
