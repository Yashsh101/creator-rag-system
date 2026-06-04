"""Unit tests for pure rag_graph helpers (no network/LLM)."""

import app.services.rag_graph as rag


def test_route_query_classifies_metadata_vs_content():
    assert rag.route_query_node({"current_query": "What is the engagement rate?"})["query_type"] == "metadata"
    assert rag.route_query_node({"current_query": "How many followers does the creator have?"})["query_type"] == "metadata"
    assert rag.route_query_node({"current_query": "Compare the hooks in the first 5 seconds"})["query_type"] == "content"


def test_route_decision_maps_query_type():
    assert rag._route_decision({"query_type": "metadata"}) == "metadata"
    assert rag._route_decision({"query_type": "content"}) == "content"


def test_engagement_rate_zero_view_safe():
    assert rag._engagement_rate(10, 5, 0) == round(15 / 1 * 100, 4)
    assert rag._engagement_rate(50, 50, 1000) == 10.0


def test_compute_metrics_node_does_not_fabricate():
    state = {"video_metadata": {
        "A": {"views": 1000, "likes": 80, "comments": 20},
        "B": {"views": 0, "likes": 0, "comments": 0},
    }}
    out = rag.compute_metrics_node(state)
    assert out["engagement_rates"]["A"] == 10.0
    assert out["engagement_rates"]["B"] == 0.0


def test_query_video_filter_detects_label():
    assert rag._query_video_filter("Summarize Video A") == "A"
    assert rag._query_video_filter("What about video b?") == "B"
    assert rag._query_video_filter("Compare both videos") is None


def test_build_documents_tags_real_id_and_label():
    state = {
        "video_metadata": {
            "A": {"video_id": "yt_real", "creator": "Alice", "platform": "youtube", "views": 1, "likes": 1, "comments": 1, "engagement_rate": 1.0, "hook": "hi"},
            "B": {"video_id": "ig_real", "creator": "Bob", "platform": "instagram", "views": 1, "likes": 1, "comments": 1, "engagement_rate": 1.0, "hook": "yo"},
        },
        "transcripts": {"A": "alpha transcript", "B": "beta transcript"},
    }
    docs = rag._build_documents(state)
    assert len(docs) == 2
    by_label = {doc.metadata["label"]: doc for doc in docs}
    assert by_label["A"].metadata["video_id"] == "yt_real"
    assert by_label["B"].metadata["video_id"] == "ig_real"


def test_build_documents_skips_empty_transcript():
    state = {
        "video_metadata": {"A": {"video_id": "x"}, "B": {"video_id": "y"}},
        "transcripts": {"A": "content here", "B": "   "},
    }
    docs = rag._build_documents(state)
    assert {doc.metadata["label"] for doc in docs} == {"A"}


def test_extract_citations_dedupes():
    chunks = [
        {"video_id": "x", "label": "A", "chunk_index": 0, "snippet": "s"},
        {"video_id": "x", "label": "A", "chunk_index": 0, "snippet": "s"},
        {"video_id": "y", "label": "B", "chunk_index": 1, "snippet": "t"},
    ]
    citations = rag._extract_citations(chunks)
    labels = {(c["label"], c["chunk_index"]) for c in citations}
    assert labels == {("A", 0), ("B", 1)}
    assert citations[0]["citation_label"] == "Video A · chunk 0"


def test_build_generation_context_includes_sections():
    state = {
        "video_metadata": {"A": {"views": 1}},
        "engagement_rates": {"A": 1.0},
        "retrieved_chunks": [{"label": "A", "chunk_index": 0, "content": "hook text"}],
    }
    ctx = rag._build_generation_context(state)
    assert "VIDEO_METADATA" in ctx and "ENGAGEMENT_RATES" in ctx and "RETRIEVED_CHUNKS" in ctx
    assert "Video A · chunk 0" in ctx


def test_safe_collection_name_normalizes():
    assert rag._safe_collection_name("abc/def?123") == "abc_def_123"
    assert rag._safe_collection_name("") == "default"


def test_session_labels_distinct_order():
    rag._SESSION_CHUNKS["u"] = [
        {"label": "A"}, {"label": "A"}, {"label": "B"},
    ]
    assert rag._session_labels("u") == ["A", "B"]
