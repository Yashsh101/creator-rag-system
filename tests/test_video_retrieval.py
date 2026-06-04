"""Tests for the video chat retrieval path: A/B filter fix, BM25, RRF, balancing."""

import app.services.rag_graph as rag


class _FakeDoc:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata


class _FakeVectorstore:
    """Minimal Chroma stand-in that honors a {'label': X} metadata filter."""

    def __init__(self, docs):
        self._docs = docs

    def similarity_search_with_score(self, query, k, filter=None):
        docs = self._docs
        if filter:
            label = filter.get("label")
            docs = [doc for doc in docs if doc.metadata.get("label") == label]
        return [(doc, 0.1 * (i + 1)) for i, doc in enumerate(docs[:k])]


def _seed_session(session_id="s1"):
    records = [
        {"label": "A", "video_id": "yt_real", "chunk_index": 0, "creator": "Alice", "platform": "youtube",
         "content": "Strong hook about morning routines and productivity habits."},
        {"label": "A", "video_id": "yt_real", "chunk_index": 1, "creator": "Alice", "platform": "youtube",
         "content": "Call to action subscribe for more productivity tips."},
        {"label": "B", "video_id": "ig_real", "chunk_index": 0, "creator": "Bob", "platform": "instagram",
         "content": "Reel hook about quick recipes and cooking shortcuts."},
        {"label": "B", "video_id": "ig_real", "chunk_index": 1, "creator": "Bob", "platform": "instagram",
         "content": "Engagement driven by trending audio and fast cuts."},
    ]
    rag._SESSION_CHUNKS[session_id] = records
    docs = [
        _FakeDoc(
            rec["content"],
            {"label": rec["label"], "video_id": rec["video_id"], "chunk_index": rec["chunk_index"],
             "creator": rec["creator"], "platform": rec["platform"]},
        )
        for rec in records
    ]
    rag._SESSION_VECTORSTORES[session_id] = _FakeVectorstore(docs)
    return session_id


def test_single_video_filter_uses_label_not_raw_id():
    """Regression: 'Video A' must filter on display label, returning only A chunks."""
    sid = _seed_session("single")
    state = {"session_id": sid, "current_query": "Summarize Video A hook"}
    out = rag.retrieve_node(state)
    labels = {chunk["label"] for chunk in out["retrieved_chunks"]}
    assert labels == {"A"}
    assert out["retrieved_chunks"], "filter must not return empty (the old bug)"


def test_comparison_query_returns_both_videos():
    """Balanced retrieval: a generic comparison must include evidence from A and B."""
    sid = _seed_session("compare")
    state = {"session_id": sid, "current_query": "Why did one video outperform the other?"}
    out = rag.retrieve_node(state)
    labels = {chunk["label"] for chunk in out["retrieved_chunks"]}
    assert labels == {"A", "B"}


def test_bm25_search_scopes_to_label():
    sid = _seed_session("bm25")
    hits = rag._bm25_search(sid, "productivity habits", candidate_k=5, label="A")
    assert hits and all(hit["label"] == "A" for hit in hits)


def test_rrf_fuses_and_dedupes():
    dense = [{"label": "A", "chunk_index": 0, "video_id": "x", "content": "c", "snippet": "c", "score": 0.9}]
    lexical = [{"label": "A", "chunk_index": 0, "video_id": "x", "content": "c", "snippet": "c", "score": 5.0},
               {"label": "B", "chunk_index": 0, "video_id": "y", "content": "d", "snippet": "d", "score": 1.0}]
    fused = rag._reciprocal_rank_fusion(dense, lexical, k=60, top_n=10)
    keys = {(c["label"], c["chunk_index"]) for c in fused}
    assert keys == {("A", 0), ("B", 0)}  # deduped A, kept B
    # A appears in both lists at rank 0 => highest fused score.
    assert fused[0]["label"] == "A"


def test_missing_vectorstore_returns_empty():
    out = rag.retrieve_node({"session_id": "nope", "current_query": "hi"})
    assert out["retrieved_chunks"] == []


def test_rerank_node_lexical_orders_by_relevance(monkeypatch):
    import app.services.rag_graph as rag
    from app.core.config import settings

    monkeypatch.setattr(settings, "reranker_provider", "local")
    chunks = [
        {"label": "A", "chunk_index": 0, "video_id": "x", "content": "unrelated filler text", "snippet": "", "score": 0.016},
        {"label": "B", "chunk_index": 0, "video_id": "y", "content": "morning routine productivity habits", "snippet": "", "score": 0.016},
    ]
    out = rag.rerank_node({"current_query": "productivity habits", "retrieved_chunks": chunks})
    assert out["retrieved_chunks"][0]["label"] == "B"  # lexical overlap wins despite lower base score


def test_rerank_node_empty_passthrough():
    import app.services.rag_graph as rag
    out = rag.rerank_node({"current_query": "", "retrieved_chunks": []})
    assert out["retrieved_chunks"] == []


def test_vectorstore_factory_falls_back_to_chroma(monkeypatch):
    """qdrant backend without a URL must fall back to in-process Chroma."""
    import app.services.rag_graph as rag
    from app.core.config import settings as cfg
    from langchain_core.documents import Document

    monkeypatch.setattr(cfg, "video_vectorstore_backend", "qdrant")
    monkeypatch.setattr(cfg, "qdrant_url", "")  # forces fallback

    captured = {}

    class _FakeChroma:
        @classmethod
        def from_documents(cls, documents, embedding, collection_name, persist_directory=None):
            captured["used"] = "chroma"
            return object()

    import langchain_community.vectorstores as vs
    monkeypatch.setattr(vs, "Chroma", _FakeChroma)

    docs = [Document(page_content="hi", metadata={"label": "A"})]
    rag._build_session_vectorstore(docs, embeddings=object(), collection_name="t")
    assert captured["used"] == "chroma"


def test_build_qdrant_returns_none_without_url(monkeypatch):
    import app.services.rag_graph as rag
    from app.core.config import settings as cfg
    monkeypatch.setattr(cfg, "qdrant_url", "")
    assert rag._build_qdrant([], embeddings=object(), collection_name="t") is None
