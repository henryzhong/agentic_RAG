"""Local cross-encoder reranker (CPU).

Loads, in order of preference:
  1. an ONNX cross-encoder directory (`*.onnx` + tokenizer) — the reranker shipped
     with the hoprag harness (ms-marco MiniLM family);
  2. transformers weights for a BGE / MiniLM cross-encoder directory.

Either way the interface is `rerank(query, passages) -> list[float]`, higher is
better. If the model cannot be loaded the pipeline falls back to retrieval-score
order and records why — reranking is an optimisation, never a dependency.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

# local weights only — never let a hub lookup block the reranker
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from agentic_rag_v2.config import RERANK_BATCH, RERANK_MAX_LEN, RERANKER_DIR


class LocalReranker:
    def __init__(self, model_dir: str | Path | None = None, *,
                 max_len: int = RERANK_MAX_LEN, batch: int = RERANK_BATCH):
        self.available = False
        self.error: str | None = None
        self.kind: str | None = None
        self.model_dir = str(model_dir or RERANKER_DIR)
        self.max_len = max_len
        self.batch = batch
        self._lock = threading.Lock()
        try:
            path = Path(self.model_dir)
            if not path.exists():
                raise FileNotFoundError(path)
            onnx_files = sorted(path.glob("*.onnx"))
            if onnx_files:
                self._init_onnx(path, onnx_files[0])
            else:
                self._init_torch(path)
            self.available = True
        except Exception as e:  # noqa: BLE001 - degrade to score order
            self.error = f"{type(e).__name__}: {e}"

    def _init_onnx(self, path: Path, onnx_file: Path) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(str(path))
        opts = ort.SessionOptions()
        # evaluation runs several questions concurrently, so stay modest here to
        # avoid thread oversubscription (the model is tiny and CPU-bound)
        opts.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(str(onnx_file), sess_options=opts,
                                         providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.sess.get_inputs()]
        self.kind = "onnx"

    def _init_torch(self, path: Path) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(path), torch_dtype=torch.float32).eval()
        self.torch = torch
        self.kind = "torch"

    # ------------------------------------------------------------------ score
    def rerank(self, query: str, passages: list[str]) -> list[float] | None:
        if not self.available:
            return None
        if not passages:
            return []
        scores: list[float] = []
        for i in range(0, len(passages), self.batch):
            chunk = passages[i:i + self.batch]
            enc = self.tok([query] * len(chunk), chunk, padding=True, truncation=True,
                           max_length=self.max_len, return_tensors="np")
            if self.kind == "onnx":
                inputs = {k: v.astype(np.int64) for k, v in enc.items()
                          if k in self.input_names}
                with self._lock:
                    out = self.sess.run(None, inputs)
                scores.extend(np.asarray(out[0]).reshape(-1).tolist())
            else:
                with self._lock, self.torch.no_grad():
                    ids = self.torch.as_tensor(enc["input_ids"], dtype=self.torch.long)
                    mask = self.torch.as_tensor(enc["attention_mask"], dtype=self.torch.long)
                    logits = self.model(input_ids=ids, attention_mask=mask).logits
                scores.extend(logits.reshape(-1).tolist())
        return scores

    def describe(self) -> dict:
        return {"available": self.available, "kind": self.kind,
                "model_dir": self.model_dir, "error": self.error}
