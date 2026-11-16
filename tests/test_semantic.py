"""Semantic retrieval: E5 prefixes, vector index, dense route, cache round-trip."""
from __future__ import annotations

import numpy as np
import pytest

from agentic_rag_v2.models.embedder import (CachedEmbedder, cosine, format_passage,
                                           format_query)
from agentic_rag_v2.retrieval.vector import (SemanticRetriever, VectorIndex, build_index,
                                             cache_slug, corpus_fingerprint)
from agentic_rag_v2.stages.search import ROUTE_WEIGHTS, _fuse, _normalise, run_search
from agentic_rag_v2.tests.fakes import FakeEmbedder
from agentic_rag_v2.types import SearchMode, SearchPlan


# ------------------------------------------------------------------ embedder
def test_e5_prefixes_are_part_of_the_contract() -> None:
    assert format_query("  Who is Ada?  ") == "query: Who is Ada?"
    assert format_passage(" Ada Lovelace ") == "passage: Ada Lovelace"


def test_fake_embedder_is_normalised_and_stable() -> None:
    embedder = FakeEmbedder()
    a = embedder.embed_queries(["who is ada lovelace"])
    b = embedder.embed_queries(["who is ada lovelace"])
    assert a.shape == (1, 16)
    assert np.allclose(np.linalg.norm(a[0]), 1.0)
    assert np.allclose(a, b)


def test_cached_embedder_avoids_repeated_forward_passes() -> None:
    inner = FakeEmbedder()
    cached = CachedEmbedder(inner)
    cached.embed_queries(["q1", "q2"])
    cached.embed_queries(["q2", "q3"])
    assert inner.calls == [["q1", "q2"], ["q3"]]
    assert len(cached.describe()) >= 4


def test_cosine_helper() -> None:
    assert cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


# --------------------------------------------------------------- vector index
def _index(vectors: list[list[float]], ids: list[str]) -> VectorIndex:
    matrix = np.asarray(vectors, dtype=np.float32)
    matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9, None)
    return VectorIndex(ids=ids, vectors=matrix)


def test_vector_search_orders_by_cosine_and_is_deterministic() -> None:
    index = _index([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], ["a", "b", "c"])
    hits = index.search(np.array([1.0, 0.0]), k=2)
    assert [cid for cid, _ in hits] == ["a", "b"]
    assert hits[0][1] > hits[1][1]
    assert index.search(np.array([1.0, 0.0]), k=2) == hits


def test_vector_search_respects_exclude() -> None:
    index = _index([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], ["a", "b", "c"])
    hits = index.search(np.array([1.0, 0.0]), k=3, exclude={"a"})
    assert [cid for cid, _ in hits] == ["b", "c"]


def test_vector_search_handles_zero_vector() -> None:
    index = _index([[1.0, 0.0]], ["a"])
    assert index.search(np.zeros(2, dtype=np.float32), k=1) == []


def test_vector_index_round_trip(tmp_path) -> None:
    index = _index([[1.0, 0.0], [0.0, 1.0]], ["a", "b"])
    index.meta = {"chunks": 2, "dim": 2}
    index.save(tmp_path / "idx")
    loaded = VectorIndex.load(tmp_path / "idx")
    assert loaded.ids == ["a", "b"]
    assert np.allclose(loaded.vectors, index.vectors)
    assert loaded.meta["dim"] == 2


def test_index_rejects_a_cache_without_its_sidecar(tmp_path) -> None:
    """The ids live in the JSON sidecar, so a lone .npz is not a usable cache.

    This replaces the old `allow_pickle=True` recovery path: a cache file that
    cannot be validated is rebuilt, never partially trusted.
    """
    from agentic_rag_v2.data import Chunk

    corpus = [Chunk("c1", "d1", "A", "alpha"), Chunk("c2", "d2", "B", "beta")]
    embedder = FakeEmbedder()
    target = tmp_path / "nojson"
    build_index(corpus, embedder, cache_path=target)
    (tmp_path / "nojson.json").unlink()

    with pytest.raises(ValueError):
        VectorIndex.load(target)

    rebuilt, stats = build_index(corpus, embedder, cache_path=target)
    assert stats["cached"] is False
    assert rebuilt.ids == ["c1", "c2"]


def test_build_index_reuses_the_cache(tmp_path) -> None:
    from agentic_rag_v2.data import Chunk

    corpus = [Chunk("c1", "d1", "A", "alpha text"), Chunk("c2", "d2", "B", "beta text")]
    embedder = FakeEmbedder()
    target = tmp_path / "tiny"
    first, stats_first = build_index(corpus, embedder, cache_path=target)
    second, stats_second = build_index(corpus, embedder, cache_path=target)
    assert stats_first["cached"] is False and stats_second["cached"] is True
    assert len(first) == len(second) == 2
    assert first.meta["corpus_fingerprint"] == corpus_fingerprint(corpus)


def test_cache_slug_changes_with_corpus() -> None:
    from agentic_rag_v2.data import Chunk

    embedder = FakeEmbedder()
    a = [Chunk("c1", "d1", "A", "alpha")]
    b = [Chunk("c1", "d1", "A", "alpha beta")]
    assert cache_slug(embedder, a) != cache_slug(embedder, b)


def test_corpus_fingerprint_detects_same_length_edits() -> None:
    """Regression: the fingerprint only hashed chunk ids + text lengths, so an edit
    that kept the length ('alpha' -> 'omega') silently reused stale vectors."""
    from agentic_rag_v2.data import Chunk

    a = [Chunk("c1", "d1", "A", "alpha")]
    b = [Chunk("c1", "d1", "A", "omega")]
    assert corpus_fingerprint(a) != corpus_fingerprint(b)


def test_cache_slug_changes_with_the_vllm_endpoint() -> None:
    """Two servers can serve the same model name with different weights; the endpoint
    has to be part of the key or one cache silently serves both."""
    from agentic_rag_v2.data import Chunk
    from agentic_rag_v2.models.embedder import VLLMEmbedder

    corpus = [Chunk("c1", "d1", "A", "alpha")]
    a = VLLMEmbedder("http://127.0.0.1:18000/v1", model="e5")
    b = VLLMEmbedder("http://127.0.0.1:18001/v1", model="e5")
    assert cache_slug(a, corpus) != cache_slug(b, corpus)


def test_index_rejects_a_cache_built_for_another_corpus(tmp_path) -> None:
    """A cache file must not be usable when the corpus it was built from changed."""
    from agentic_rag_v2.data import Chunk

    embedder = FakeEmbedder()
    first = [Chunk("c1", "d1", "A", "alpha"), Chunk("c2", "d2", "B", "beta")]
    index, _ = build_index(first, embedder, cache_path=tmp_path / "shared")

    other = [Chunk("c1", "d1", "A", "alpha"), Chunk("c3", "d3", "C", "gamma")]
    rebuilt, stats = build_index(other, embedder, cache_path=tmp_path / "shared")
    assert stats["cached"] is False                   # fingerprint mismatch -> rebuild
    assert rebuilt.ids == ["c1", "c3"]
    assert rebuilt.meta["corpus_fingerprint"] == corpus_fingerprint(other)


def test_index_validates_dimension_on_load(tmp_path) -> None:
    """A cache whose dimension does not match the embedder must be rejected."""
    from agentic_rag_v2.data import Chunk

    corpus = [Chunk("c1", "d1", "A", "alpha")]
    build_index(corpus, FakeEmbedder(dim=8), cache_path=tmp_path / "dimmed")
    _, stats = build_index(corpus, FakeEmbedder(dim=16), cache_path=tmp_path / "dimmed")
    assert stats["cached"] is False
    assert stats["dim"] == 16


def test_ids_are_stored_without_pickle(tmp_path) -> None:
    """The sidecar JSON carries the ids, so loading never needs allow_pickle."""
    ids = ["c1", "c2"]
    index = VectorIndex(ids=ids,
                        vectors=np.ones((2, 4), dtype=np.float32) / 2.0,
                        meta={"chunks": 2, "dim": 4})
    index.save(tmp_path / "nopickle")
    assert (tmp_path / "nopickle.json").exists()
    loaded = VectorIndex.load(tmp_path / "nopickle")
    assert loaded.ids == ids


# ------------------------------------------------------------------- retrieval
def test_semantic_retriever_batches_queries(chunks, tmp_path) -> None:
    embedder = FakeEmbedder()
    index, _ = build_index(chunks, embedder, cache_path=tmp_path / "fixture")
    semantic = SemanticRetriever(index, embedder)
    vectors = semantic.embed_many(["ada lovelace", "ada lovelace", "charles babbage"])
    assert set(vectors) == {"ada lovelace", "charles babbage"}
    hits = semantic.search("ada lovelace analytical engine", k=3)
    assert hits and all(isinstance(score, float) for _, score in hits)
    assert "embedder" in semantic.describe()


def test_embed_many_keys_are_the_original_queries(chunks, tmp_path) -> None:
    """Regression: a trailing space in a question used to raise KeyError."""
    embedder = FakeEmbedder()
    index, _ = build_index(chunks, embedder, cache_path=tmp_path / "fixture")
    semantic = SemanticRetriever(index, embedder)
    queries = ["Which species is the most numerous? ", "  ada lovelace  "]
    vectors = semantic.embed_many(queries)
    for query in queries:
        assert query in vectors                      # lookup with the raw string
    assert len(vectors) == 2


def test_dense_route_survives_an_unbatched_query(retriever) -> None:
    """Empty dense_vectors must fall back to encoding, not raise."""
    from agentic_rag_v2.tests.fakes import FakeSemantic

    plan = SearchPlan(mode=SearchMode.KEEP, queries=["Ada Lovelace "], graph=False,
                      rounds=1, top_k=3)
    result = run_search("Ada Lovelace ", plan, retriever, semantic=FakeSemantic({"ada": "c3"}))
    assert "c3" in result.evidence.passages


# --------------------------------------------------------------- dense route
def test_dense_route_contributes_to_the_pool(retriever) -> None:
    from agentic_rag_v2.tests.fakes import FakeSemantic

    plan = SearchPlan(mode=SearchMode.KEEP, queries=["Who is Ada Lovelace?"],
                      graph=False, rounds=1, top_k=3)
    semantic = FakeSemantic({"ada": "c3"})
    result = run_search("Who is Ada Lovelace?", plan, retriever, semantic=semantic)
    assert "c3" in result.evidence.passages
    assert "dense" in result.trace[0]["routes"]
    assert result.trace[0]["routes"]["dense"] >= 1


def test_no_dense_route_without_semantic(retriever) -> None:
    plan = SearchPlan(mode=SearchMode.KEEP, queries=["Who is Ada Lovelace?"],
                      graph=False, rounds=1, top_k=3)
    result = run_search("Who is Ada Lovelace?", plan, retriever)
    assert "dense" not in result.trace[0]["routes"]


def test_dense_scores_are_minmax_normalised_and_others_max() -> None:
    hits = [("q", "a", 0.80), ("q", "b", 0.70), ("q", "c", 0.60)]
    dense = _normalise("dense", hits)
    assert dense["a"] == pytest.approx(1.0)
    assert dense["c"] == pytest.approx(0.0)
    bm25 = _normalise("bm25", hits)
    assert bm25["a"] == pytest.approx(1.0)
    assert bm25["c"] == pytest.approx(0.6 / 0.8)


def test_fusion_sums_weighted_routes() -> None:
    results = {
        "bm25": [("q", "a", 2.0), ("q", "b", 1.0)],
        "dense": [("q", "b", 0.9), ("q", "c", 0.7)],
    }
    fused, meta = _fuse(results, exclude=set())
    assert fused["b"] == pytest.approx(ROUTE_WEIGHTS["bm25"] * 0.5 + ROUTE_WEIGHTS["dense"] * 1.0)
    assert meta["b"]["routes"] == ["dense", "bm25"]      # strongest weight first
    assert meta["c"]["routes"] == ["dense"]


def test_fusion_does_not_double_count_the_same_route() -> None:
    """A route that returns a chunk for two of the plan's queries votes once.

    Regression: the loop used to iterate every (query, chunk) hit, so a chunk
    found by 3 queries of the same route got 3x the route weight — one retriever
    could outvote the other routes by simply being asked more often.
    """
    one_query = {"bm25": [("q1", "a", 2.0), ("q1", "b", 1.0)]}
    two_queries = {"bm25": [("q1", "a", 2.0), ("q1", "b", 1.0),
                            ("q2", "a", 2.0), ("q2", "b", 1.0)]}
    fused_one, _ = _fuse(one_query, exclude=set())
    fused_two, _ = _fuse(two_queries, exclude=set())
    assert fused_two["a"] == pytest.approx(fused_one["a"])
    assert fused_two["b"] == pytest.approx(fused_one["b"])
    assert fused_two["a"] <= ROUTE_WEIGHTS["bm25"] + 1e-9


def test_fusion_takes_the_best_score_per_chunk_within_a_route() -> None:
    hits = [("q1", "a", 1.0), ("q2", "a", 3.0), ("q2", "b", 2.0)]
    fused, _ = _fuse({"bm25": hits}, exclude=set())
    assert fused["a"] == pytest.approx(ROUTE_WEIGHTS["bm25"] * 1.0)   # max of 1.0/3.0


def test_fusion_keeps_the_query_that_scored_best() -> None:
    hits = [("weak query", "a", 1.0), ("strong query", "a", 3.0)]
    _, meta = _fuse({"bm25": hits}, exclude=set())
    assert meta["a"]["query"] == "strong query"


def test_fusion_excludes_seen_chunks() -> None:
    results = {"bm25": [("q", "a", 2.0), ("q", "b", 1.0)]}
    fused, meta = _fuse(results, exclude={"a"})
    assert "a" not in fused and "b" in fused
    assert "a" not in meta
