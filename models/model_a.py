"""Model A: the trained multi-task classifier (intent / complexity / multi-hop / hop).

The weights are reused from `air/checkpoints/model_a/<checkpoint>/best.pt`; this
module only adapts its raw logits into the pipeline's `RouteSignal` contract.
"""
from __future__ import annotations

import os
import sys
import threading
import time

# The trained weights are already cached locally: a hub HEAD request must never
# block (or slow down) a run. Set before transformers is imported anywhere.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from agentic_rag_v2.config import AIR_SRC, MODEL_A_CHECKPOINT, MODEL_A_DIR
from agentic_rag_v2.types import RouteSignal


class ModelAAnalyzer:
    """Wraps `air/src/model_a.EncoderClassifier` behind a stable interface.

    Model A is a required component of the pipeline: if the checkpoint cannot be
    loaded, construction raises instead of silently degrading to a neutral signal
    (a silent fallback makes runs incomparable without anyone noticing).
    """

    def __init__(self, checkpoint: str | None = None):
        self.ok = False
        self.error: str | None = None
        checkpoint = checkpoint or MODEL_A_CHECKPOINT
        self.checkpoint_name = checkpoint
        self.checkpoint = str(MODEL_A_DIR / checkpoint / "best.pt")
        self._lock = threading.Lock()
        self._cache: dict[str, RouteSignal] = {}
        try:
            sys.path.insert(0, str(AIR_SRC))
            import torch
            from transformers import AutoTokenizer

            import model_a as ma
            from common import COMPLEXITY, HOP_BUCKETS, INTENTS

            self._torch = torch
            self._ma = ma
            self._intents = INTENTS
            self._complexity = COMPLEXITY
            self._hop_buckets = HOP_BUCKETS
            ck = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            self._tasks = ck["tasks"]
            self._model = ma.EncoderClassifier(self._tasks, frozen=ck["frozen"])
            self._model.load_state_dict(ck["state_dict"])
            self._device = ma.device()
            self._model.to(self._device).eval()
            self._tok = AutoTokenizer.from_pretrained(ma.MODEL_NAME)
            self.ok = True
        except Exception as e:  # noqa: BLE001 - surfaced by `require()` below
            self.error = f"{type(e).__name__}: {e}"

    def require(self) -> None:
        """Fail fast: the pipeline refuses to run without a working Model A."""
        if not self.ok:
            raise RuntimeError(
                f"Model A checkpoint '{self.checkpoint_name}' could not be loaded "
                f"({self.checkpoint}): {self.error}")

    def predict(self, question: str) -> RouteSignal:
        if not self.ok:
            return RouteSignal(available=False, error=self.error)
        cached = self._cache.get(question)
        if cached is not None:
            return cached
        signal = self._predict(question)
        self._cache[question] = signal
        return signal

    def _predict(self, question: str) -> RouteSignal:
        torch = self._torch
        enc = self._tok([question], truncation=True, padding=True,
                        max_length=self._ma.MAX_LEN, return_tensors="pt")
        t0 = time.perf_counter()
        with self._lock, torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=self._device.type == "cuda"):
            logits = self._model(enc["input_ids"].to(self._device),
                                 enc["attention_mask"].to(self._device))
        latency = (time.perf_counter() - t0) * 1000

        out = RouteSignal(available=True, latency_ms=round(latency, 2))
        if "multihop" in self._tasks:
            p = torch.softmax(logits["multihop"][0].float(), -1).cpu().numpy()
            out.multihop_prob = round(float(p[1]), 4)
            out.multihop = bool(p[1] >= 0.5)
        if "complexity" in self._tasks:
            p = torch.softmax(logits["complexity"][0].float(), -1).cpu().numpy()
            out.complexity = self._complexity[int(p.argmax())]
        if "hop" in self._tasks:
            p = torch.softmax(logits["hop"][0].float(), -1).cpu().numpy()
            out.hop_bucket = str(self._hop_buckets[int(p.argmax())])
        if "intent" in self._tasks:
            p = torch.sigmoid(logits["intent"][0].float()).cpu().numpy()
            ranked = sorted(zip(self._intents, p), key=lambda kv: (-kv[1], kv[0]))[:2]
            out.intents = {k: round(float(v), 4) for k, v in ranked}
            out.intent = ranked[0][0]
        return out
