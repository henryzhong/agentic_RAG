"""Retrieval layer: BM25, entity extraction, graph determinism, fusion."""
from __future__ import annotations

from agentic_rag_v2.retrieval import HybridRetriever
from agentic_rag_v2.retrieval.graph import EntityGraph
from agentic_rag_v2.retrieval.text import extract_entities, tokenize


def test_tokenize_drops_stopwords() -> None:
    assert tokenize("Who is the child of the person?") == ["child", "person"]


def test_extract_entities_keeps_titles_and_years() -> None:
    ents = extract_entities("Wahab Riaz joined Kent in June 2011.", title="Wahab Riaz")
    assert "wahab riaz" in ents
    assert "2011" in ents
    assert "kent" in ents
    assert all("the" not in e for e in ents)


def test_bm25_prefers_title_match(retriever: HybridRetriever) -> None:
    hits = retriever.bm25_search("Wahab Riaz Pakistani cricketer batsman", k=3)
    assert hits and hits[0][0] == "c2"


def test_graph_expansion_is_deterministic(chunks) -> None:
    first = EntityGraph(chunks)
    second = EntityGraph(chunks)
    seeds = [("charles babbage", 3.0)]
    ranked_a, scores_a, paths_a = first.expand(seeds)
    ranked_b, scores_b, paths_b = second.expand(seeds)
    assert ranked_a == ranked_b
    assert scores_a == scores_b
    assert paths_a == paths_b
    assert ranked_a, "expansion should reach the Babbage/Lovelace passages"
    assert [cid for cid, _ in ranked_a][0] in {"c3", "c4", "c8"}


def test_hybrid_anchors_exact_title(retriever: HybridRetriever) -> None:
    ranked, trace = retriever.hybrid("Kent County Cricket Club in 2011", k=5, use_graph=False)
    assert "c1" in [cid for cid, _ in ranked]
    assert trace["title_hits"]


def test_graph_route_reaches_bridge_passage(retriever: HybridRetriever) -> None:
    ranked, trace = retriever.hybrid("Who was Stefan Nemanja followed by?", k=6, use_graph=True)
    ids = [cid for cid, _ in ranked]
    assert trace["seeds"], "entity linking should find the named person"
    assert "c6" in ids or "c7" in ids


def test_shared_entities_finds_bridge() -> None:
    from agentic_rag_v2.data import Chunk

    chunks = [
        Chunk("x1", "d1", "A", "Stefan Nemanja ruled Serbia."),
        Chunk("x2", "d2", "B", "Stefan Nemanja was followed by his son."),
    ]
    graph = EntityGraph(chunks)
    shared = graph.shared_entities(["x1", "x2"])
    assert any(name == "stefan nemanja" and ids == [1, 2] for name, ids in shared)


def test_neighbours_of_chunks_excludes_input(retriever: HybridRetriever) -> None:
    nbrs = retriever.neighbours_of_chunks(["c2"], limit=5)
    assert all(cid != "c2" for cid, _ in nbrs)
