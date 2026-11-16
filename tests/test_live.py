"""Live integration tests (real Ark API + real corpus).

Skipped unless RAGV2_LIVE=1:

    $env:RAGV2_LIVE=1; python -m pytest agentic_rag_v2/tests/test_live.py -q
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RAGV2_LIVE") != "1",
                                reason="set RAGV2_LIVE=1 to run live API tests")


@pytest.fixture(scope="module")
def rag():
    from agentic_rag_v2.pipeline import AgenticRAG

    return AgenticRAG()


def test_live_pipeline_answers_a_bridge_question(rag) -> None:
    result = rag.answer("Who was the overseas player for Kent in June 2011?")
    assert result.answer and result.answer.upper() != "UNKNOWN"
    assert result.evidence
    stages = [s["stage"] for s in result.trace]
    assert stages == ["analyze", "plan", "search", "rerank", "answer"]
    assert result.llm_calls == 2           # the two-call budget holds end to end


def test_live_semantic_route_is_active(rag) -> None:
    manifest = rag.manifest()
    assert manifest["semantic"].get("embedder", {}).get("kind") == "e5", manifest["semantic"]
    result = rag.answer("Which city is the band that recorded the song Renegade from?")
    routes = result.trace[2]["rounds_trace"][0]["routes"]
    assert routes.get("dense", 0) > 0, routes


def test_live_model_a_signal_is_available(rag) -> None:
    assert rag.analyzer.ok, rag.analyzer.error
    signal = rag.analyzer.predict("Who is the child of the person who followed Tihomir of Serbia?")
    assert 0.0 <= signal.multihop_prob <= 1.0
    assert signal.hop_bucket in {"1", "2", "3+"}


@pytest.mark.skipif(os.environ.get("RAGV2_VLLM_ENDPOINT") is None,
                    reason="set RAGV2_VLLM_ENDPOINT=http://127.0.0.1:18000/v1")
def test_live_vllm_backend_matches_local() -> None:
    """The container backend must return the same vectors as the in-process model."""
    import numpy as np

    from agentic_rag_v2.models.embedder import LocalE5Embedder, VLLMEmbedder

    texts = ["Wahab Riaz is a Pakistani cricketer and a right-hand batsman.",
             "In 2011 Kent County Cricket Club competed in Division Two."]
    remote = VLLMEmbedder(os.environ["RAGV2_VLLM_ENDPOINT"])
    assert remote.probe()["ok"], remote.last_error
    local = LocalE5Embedder()
    a = remote.embed_passages(texts)
    b = local.embed_passages(texts)
    assert a.shape == b.shape
    assert all(float(np.dot(x, y)) > 0.99 for x, y in zip(a, b))
