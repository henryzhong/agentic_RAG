"""Volcengine Ark chat client: caching, retries, accounting, request ledger.

The client is the only place that talks to the network. Stages receive it as a
`ChatClient` protocol, so tests drive the whole pipeline with a scripted fake.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from typing import Protocol

import httpx

from agentic_rag_v2.config import (CACHE_DIR, MAX_CONCURRENCY, WORK,
                                   ArkSettings, ark_settings)

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class ChatClient(Protocol):
    """Minimal surface the pipeline depends on."""

    def chat(self, messages: list[dict], *, max_tokens: int = 1024,
             temperature: float | None = 0.0, tag: str = "") -> str: ...

    def json_call(self, messages: list[dict], *, max_tokens: int = 1024,
                  tag: str = "") -> dict: ...


class Usage:
    def __init__(self) -> None:
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reasoning_tokens = 0
        self.latency_ms = 0.0
        self.cache_hits = 0
        self.errors = 0
        self._lock = threading.Lock()

    def add(self, usage: dict, latency_ms: float, cached: bool = False) -> None:
        with self._lock:
            self.calls += 1
            if cached:
                self.cache_hits += 1
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)
            det = usage.get("completion_tokens_details") or {}
            self.reasoning_tokens += int(det.get("reasoning_tokens") or 0)
            self.latency_ms += latency_ms

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "llm_calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "llm_latency_ms": round(self.latency_ms, 1),
                "cache_hits": self.cache_hits,
                "llm_errors": self.errors,
            }


def _cache_key(model: str, messages: list[dict], params: dict) -> str:
    blob = json.dumps({"m": model, "msgs": messages, "p": params},
                      ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


class ArkClient:
    """Chat completions client.

    `model` can differ per instance so a cheaper/faster model can later be
    plugged into the rerank or summarize stage without touching stage code.
    """

    def __init__(self, model: str | None = None, *, timeout: float = 90.0,
                 max_retries: int = 3, use_cache: bool = True,
                 settings: ArkSettings | None = None):
        cfg = settings or ark_settings(model)
        self.settings = cfg
        self.model = cfg.model
        self.base_url = cfg.base_url
        self.max_retries = max_retries
        self.use_cache = use_cache
        self.usage = Usage()
        self.ledger_path = WORK / "request_ledger.jsonl"
        self._ledger_lock = threading.Lock()
        self._sem = threading.BoundedSemaphore(MAX_CONCURRENCY)
        self._client = httpx.Client(
            timeout=timeout,
            headers={"Authorization": f"Bearer {cfg.api_key}",
                     "Content-Type": "application/json"},
        )

    # ------------------------------------------------------------------ core
    def chat(self, messages: list[dict], *, max_tokens: int = 1024,
             temperature: float | None = 0.0, tag: str = "") -> str:
        params: dict = {"max_tokens": max_tokens, "reasoning_effort": self.settings.reasoning_effort}
        if temperature is not None:
            params["temperature"] = temperature
        key = _cache_key(self.model, messages, params)
        cache_file = CACHE_DIR / f"{key}.json"
        if self.use_cache and cache_file.exists():
            rec = json.loads(cache_file.read_text(encoding="utf-8"))
            self.usage.add(rec.get("usage", {}), 0.0, cached=True)
            return rec["text"]

        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "reasoning_effort": self.settings.reasoning_effort, "stream": False}
        if temperature is not None:
            body["temperature"] = temperature

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                with self._sem:
                    r = self._client.post(f"{self.base_url}/chat/completions", json=body)
                dt = (time.perf_counter() - t0) * 1000
                if r.status_code in RETRY_STATUS:
                    raise httpx.HTTPStatusError(
                        f"retryable {r.status_code}: {r.text[:200]}",
                        request=r.request, response=r)
                r.raise_for_status()
                data = r.json()
                text = data["choices"][0]["message"].get("content") or ""
                usage = data.get("usage") or {}
                self.usage.add(usage, dt)
                if self.use_cache:
                    cache_file.write_text(
                        json.dumps({"text": text, "usage": usage}, ensure_ascii=False),
                        encoding="utf-8")
                with self._ledger_lock:
                    with self.ledger_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({
                            "ts": time.time(), "tag": tag, "model": data.get("model"),
                            "endpoint": self.base_url, "prompt_hash": key,
                            "prompt_tokens": usage.get("prompt_tokens"),
                            "completion_tokens": usage.get("completion_tokens"),
                            "latency_ms": round(dt, 1), "attempt": attempt,
                        }, ensure_ascii=False) + "\n")
                return text
            except Exception as e:  # noqa: BLE001 - retry any transport/HTTP error
                last_err = e
                if attempt == self.max_retries - 1:
                    break
                time.sleep(min(2 ** attempt * 0.8, 12))
        with self.usage._lock:
            self.usage.errors += 1
        raise RuntimeError(f"ark call failed tag={tag}: {last_err}")

    def json_call(self, messages: list[dict], *, max_tokens: int = 1024,
                  tag: str = "") -> dict:
        raw = self.chat(messages, max_tokens=max_tokens, tag=tag, temperature=0.0)
        return parse_json(raw)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ArkClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def parse_json(text: str) -> dict:
    """Tolerant JSON extraction: fenced blocks, leading prose, trailing commas."""
    if not text:
        raise ValueError("empty model output")
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    if start >= 0:
        depth, in_str, esc = 0, False, False
        for i, ch in enumerate(t[start:], start):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        cand = re.sub(r",\s*([}\]])", r"\1", t[start:i + 1])
                        return json.loads(cand)
    raise ValueError(f"no JSON object in model output: {text[:300]}")
