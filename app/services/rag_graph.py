import json
import logging
import re
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from app.services.checkpointer import build_chat_checkpointer
from langgraph.graph import END, START, StateGraph, add_messages

from app.core.config import settings
from app.ingest.instagram import get_instagram_data
from app.ingest.models import VideoMetadata
from app.ingest.youtube import get_youtube_data
from app.utils.text import count_tokens_approx

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a social media analytics AI. Answer questions about Video A and Video B using transcript chunks and metadata. "
    "Always cite sources as [Video A · chunk N] or [Video B · chunk N]. Be specific, data-driven, and actionable. "
    "Never fabricate metrics."
)

METADATA_KEYWORDS = {
    "views",
    "likes",
    "comments",
    "comment",
    "creator",
    "followers",
    "follower",
    "hashtag",
    "hashtags",
    "upload",
    "uploaded",
    "duration",
    "engagement",
    "rate",
    "metrics",
    "metadata",
}

CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 60
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " "]


class VideoRAGState(TypedDict):
    session_id: str
    video_urls: dict[str, str]
    video_metadata: dict[str, dict[str, Any]]
    transcripts: dict[str, str]
    engagement_rates: dict[str, float]
    vectorstore_ready: bool
    messages: Annotated[list[BaseMessage], add_messages]
    current_query: str
    query_type: str
    retrieved_chunks: list[dict[str, Any]]
    answer: str
    citations: list[dict[str, Any]]


_SESSION_VECTORSTORES: dict[str, Any] = {}
# Per-session raw chunk records {label, video_id, chunk_index, text, metadata} for BM25.
_SESSION_CHUNKS: dict[str, list[dict[str, Any]]] = {}


def extract_videos_node(state: VideoRAGState) -> dict[str, Any]:
    """Extracts metadata and transcripts for the required YouTube and Instagram inputs."""
    video_urls = state.get("video_urls", {})
    youtube_url = video_urls.get("A", "")
    instagram_url = video_urls.get("B", "")
    video_a = get_youtube_data(url=youtube_url)
    video_b = get_instagram_data(url=instagram_url)
    video_a.label = "A"
    video_b.label = "B"
    metadata = {"A": _metadata_to_dict(video_a), "B": _metadata_to_dict(video_b)}
    return {
        "video_metadata": metadata,
        "transcripts": {"A": video_a.transcript or "", "B": video_b.transcript or ""},
        "engagement_rates": {
            "A": _engagement_rate(video_a.likes, video_a.comments, video_a.views),
            "B": _engagement_rate(video_b.likes, video_b.comments, video_b.views),
        },
        "vectorstore_ready": False,
        "retrieved_chunks": [],
        "citations": [],
        "answer": "",
    }


def compute_metrics_node(state: VideoRAGState) -> dict[str, Any]:
    """Computes engagement rates from extracted metadata without fabricating missing values."""
    metadata = state.get("video_metadata", {})
    engagement_rates: dict[str, float] = {}
    updated_metadata: dict[str, dict[str, Any]] = {}
    for video_id, values in metadata.items():
        views = _safe_int(values.get("views"))
        likes = _safe_int(values.get("likes"))
        comments = _safe_int(values.get("comments"))
        rate = _engagement_rate(likes=likes, comments=comments, views=views)
        updated = {**values, "engagement_rate": rate}
        updated_metadata[video_id] = updated
        engagement_rates[video_id] = rate
    return {"video_metadata": updated_metadata, "engagement_rates": engagement_rates}


def build_vectorstore_node(state: VideoRAGState) -> dict[str, Any]:
    """Builds an in-memory Chroma collection for the session with video-tagged chunks."""
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    session_id = state["session_id"]
    documents = _build_documents(state)
    if not documents:
        logger.warning("vectorstore_empty", extra={"event": "vectorstore_empty", "session_id": session_id})
        return {"vectorstore_ready": False}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=count_tokens_approx,
        separators=CHUNK_SEPARATORS,
    )
    chunks = splitter.split_documents(documents)
    chunk_indexes: dict[str, int] = {}
    chunk_records: list[dict[str, Any]] = []
    for chunk in chunks:
        label = str(chunk.metadata.get("label") or chunk.metadata.get("video_id", "unknown"))
        chunk_index = chunk_indexes.get(label, 0)
        chunk_indexes[label] = chunk_index + 1
        chunk.metadata["chunk_index"] = chunk_index
        # Citation labels use the human display label (A/B), not the raw platform id.
        chunk.metadata["citation_label"] = f"Video {label} · chunk {chunk_index}"
        chunk_records.append(
            {
                "label": label,
                "video_id": str(chunk.metadata.get("video_id", "unknown")),
                "chunk_index": chunk_index,
                "creator": str(chunk.metadata.get("creator", "")),
                "platform": str(chunk.metadata.get("platform", "")),
                "content": chunk.page_content,
            }
        )
    _SESSION_CHUNKS[session_id] = chunk_records

    collection_name = f"session_{_safe_collection_name(session_id)}"
    embeddings = OpenAIEmbeddings(model=settings.openai_embedding_model, api_key=settings.openai_api_key)
    vectorstore = _build_session_vectorstore(chunks, embeddings, collection_name)
    _SESSION_VECTORSTORES[session_id] = vectorstore
    return {"vectorstore_ready": True}


def _build_session_vectorstore(chunks: list[Document], embeddings: Any, collection_name: str) -> Any:
    """Builds the session vector store from the configured backend.

    - ``chroma`` (default): in-process, zero-infra; optional persist dir survives restart.
    - ``qdrant``: shared server for multi-worker scale; partitioned per session collection.
    Falls back to in-process Chroma if the scale backend is misconfigured so a
    demo never breaks on infra.
    """
    backend = (settings.video_vectorstore_backend or "chroma").lower()
    if backend == "qdrant":
        store = _build_qdrant(chunks, embeddings, collection_name)
        if store is not None:
            return store
        logger.warning("qdrant_unavailable_fallback_chroma", extra={"event": "qdrant_unavailable_fallback_chroma"})

    from langchain_community.vectorstores import Chroma

    persist_dir = settings.video_chroma_persist_dir or None
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )


def _build_qdrant(chunks: list[Document], embeddings: Any, collection_name: str) -> Any | None:
    """Builds a Qdrant-backed store for horizontal scale; None if unavailable."""
    if not settings.qdrant_url:
        return None
    try:
        from langchain_qdrant import QdrantVectorStore

        return QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
    except Exception:
        logger.warning("qdrant_init_failed", exc_info=True, extra={"event": "qdrant_init_failed"})
        return None


def route_query_node(state: VideoRAGState) -> dict[str, str]:
    """Classifies the current query as metadata lookup or transcript-content retrieval."""
    query = state.get("current_query", "").lower()
    query_words = set(re.findall(r"[a-z0-9_]+", query))
    query_type = "metadata" if query_words & METADATA_KEYWORDS else "content"
    return {"query_type": query_type}


def retrieve_node(state: VideoRAGState) -> dict[str, list[dict[str, Any]]]:
    """Hybrid (dense + BM25) retrieval with per-video balancing for comparison queries.

    - Single-video queries ("summarize Video A") filter to that video's label.
    - Comparison/general queries retrieve a balanced slice from BOTH videos so the
      model always sees evidence from each side, then fuse dense + lexical ranks
      via Reciprocal Rank Fusion (RRF) and finally re-rank by fused score.
    """
    session_id = state["session_id"]
    vectorstore = _SESSION_VECTORSTORES.get(session_id)
    if vectorstore is None:
        logger.warning("vectorstore_missing", extra={"event": "vectorstore_missing", "session_id": session_id})
        return {"retrieved_chunks": []}

    query = state.get("current_query", "")
    top_k = settings.video_retrieval_top_k or settings.retrieval_top_k or 6
    candidate_k = max(settings.video_hybrid_candidate_k, top_k)
    video_filter = _query_video_filter(query)  # returns display label "A"/"B" or None
    available_labels = _session_labels(session_id)

    if video_filter is not None and video_filter in available_labels:
        # Single-video question: scope retrieval to that label.
        chunks = _hybrid_search(session_id, vectorstore, query, candidate_k, label=video_filter)
        return {"retrieved_chunks": chunks[:top_k]}

    if len(available_labels) >= 2:
        # Comparison/general question: balanced retrieval from each video, then merge.
        per_video = max(1, top_k // len(available_labels))
        merged: list[dict[str, Any]] = []
        for label in available_labels:
            merged.extend(_hybrid_search(session_id, vectorstore, query, candidate_k, label=label)[:per_video])
        # Backfill remaining slots from a global hybrid pass to use the full budget.
        if len(merged) < top_k:
            seen = {(c["label"], c["chunk_index"]) for c in merged}
            for chunk in _hybrid_search(session_id, vectorstore, query, candidate_k, label=None):
                if (chunk["label"], chunk["chunk_index"]) not in seen:
                    merged.append(chunk)
                    if len(merged) >= top_k:
                        break
        return {"retrieved_chunks": merged[:top_k]}

    chunks = _hybrid_search(session_id, vectorstore, query, candidate_k, label=None)
    return {"retrieved_chunks": chunks[:top_k]}


def _hybrid_search(
    session_id: str,
    vectorstore: Any,
    query: str,
    candidate_k: int,
    label: str | None,
) -> list[dict[str, Any]]:
    """Runs dense + BM25 retrieval and fuses them with Reciprocal Rank Fusion."""
    # Dense: filter on the display label (A/B), which is what _build_documents stores.
    search_kwargs: dict[str, Any] = {"k": candidate_k}
    if label is not None:
        search_kwargs["filter"] = {"label": label}
    try:
        dense_hits = vectorstore.similarity_search_with_score(query=query, **search_kwargs)
    except Exception:
        logger.warning("dense_search_failed", exc_info=True, extra={"event": "dense_search_failed", "session_id": session_id})
        dense_hits = []
    dense = [_serialize_chroma_result(document, score) for document, score in dense_hits]

    lexical = _bm25_search(session_id, query, candidate_k, label=label)
    if not lexical:
        return dense
    return _reciprocal_rank_fusion(dense, lexical, k=settings.video_rrf_k, top_n=candidate_k)


def _bm25_search(session_id: str, query: str, candidate_k: int, label: str | None) -> list[dict[str, Any]]:
    """Lexical BM25 ranking over the session's chunk corpus (small, in-memory)."""
    records = _SESSION_CHUNKS.get(session_id, [])
    if label is not None:
        records = [record for record in records if record.get("label") == label]
    query_terms = _tokenize(query)
    if not records or not query_terms:
        return []
    try:
        from rank_bm25 import BM25Okapi

        corpus = [_tokenize(record["content"]) for record in records]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_terms)
    except Exception:
        logger.warning("bm25_search_failed", exc_info=True, extra={"event": "bm25_search_failed", "session_id": session_id})
        return []
    ranked = sorted(zip(records, scores), key=lambda pair: pair[1], reverse=True)[:candidate_k]
    results: list[dict[str, Any]] = []
    for record, score in ranked:
        results.append(
            {
                "video_id": record["video_id"],
                "label": record["label"],
                "chunk_index": record["chunk_index"],
                "creator": record.get("creator", ""),
                "platform": record.get("platform", ""),
                "score": float(score),
                "snippet": record["content"][:120],
                "content": record["content"],
            }
        )
    return results


def _reciprocal_rank_fusion(
    dense: list[dict[str, Any]],
    lexical: list[dict[str, Any]],
    k: int,
    top_n: int,
) -> list[dict[str, Any]]:
    """Fuses two ranked lists by RRF: score = sum(1 / (k + rank)). No tuning needed."""
    fused: dict[tuple[str, int], dict[str, Any]] = {}
    for ranking in (dense, lexical):
        for rank, chunk in enumerate(ranking):
            key = (str(chunk.get("label")), _safe_int(chunk.get("chunk_index")))
            entry = fused.get(key)
            if entry is None:
                entry = {**chunk, "score": 0.0}
                fused[key] = entry
            entry["score"] += 1.0 / (k + rank + 1)
    return sorted(fused.values(), key=lambda chunk: chunk["score"], reverse=True)[:top_n]


def _session_labels(session_id: str) -> list[str]:
    """Returns the distinct display labels (A/B) present in the session corpus."""
    seen: list[str] = []
    for record in _SESSION_CHUNKS.get(session_id, []):
        label = record.get("label")
        if label and label not in seen:
            seen.append(label)
    return seen


def _tokenize(text: str) -> list[str]:
    """Lowercased alphanumeric tokenization shared by BM25 indexing and querying."""
    return re.findall(r"[a-z0-9_]+", text.lower())


def rerank_node(state: VideoRAGState) -> dict[str, list[dict[str, Any]]]:
    """Re-ranks retrieved chunks with a cross-encoder (Cohere) or lexical fallback.

    Operates on the dict-shaped chunks produced by retrieve_node. Cohere is used
    only when reranker_provider="cohere" and a key is set; any failure degrades
    gracefully to lexical re-scoring so a live demo never breaks on a provider.
    """
    chunks = state.get("retrieved_chunks", [])
    query = state.get("current_query", "")
    if not chunks or not query:
        return {"retrieved_chunks": chunks}
    top_k = settings.video_retrieval_top_k or settings.retrieval_top_k or 6

    if settings.reranking_enabled and settings.reranker_provider == "cohere" and settings.cohere_api_key:
        reranked = _cohere_rerank_chunks(query, chunks, top_k)
        if reranked is not None:
            return {"retrieved_chunks": reranked}

    return {"retrieved_chunks": _lexical_rerank_chunks(query, chunks, top_k)}


def _cohere_rerank_chunks(query: str, chunks: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]] | None:
    """True cross-encoder rerank via Cohere; returns None on any failure."""
    try:
        import cohere

        client = cohere.Client(settings.cohere_api_key)
        documents = [str(chunk.get("content", "")) for chunk in chunks]
        response = client.rerank(query=query, documents=documents, top_n=min(top_k, len(documents)), model=settings.cohere_rerank_model)
        ranked: list[dict[str, Any]] = []
        for item in response.results:
            chunk = {**chunks[item.index], "score": float(item.relevance_score)}
            ranked.append(chunk)
        return ranked
    except Exception:
        logger.warning("video_cohere_rerank_failed", exc_info=True, extra={"event": "video_cohere_rerank_failed"})
        return None


def _lexical_rerank_chunks(query: str, chunks: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Deterministic lexical re-scoring blending fused score with term overlap."""
    query_terms = set(_tokenize(query))
    if not query_terms:
        return chunks[:top_k]
    scored: list[dict[str, Any]] = []
    for chunk in chunks:
        text = str(chunk.get("content", ""))
        overlap = len(query_terms & set(_tokenize(text))) / max(1, len(query_terms))
        phrase_bonus = 0.05 if query.lower() in text.lower() else 0.0
        base = float(chunk.get("score", 0.0))
        scored.append({**chunk, "score": min(1.0, 0.65 * base + 0.35 * overlap + phrase_bonus)})
    return sorted(scored, key=lambda chunk: chunk["score"], reverse=True)[:top_k]


async def generate_node(state: VideoRAGState) -> dict[str, Any]:
    """Generates a streamed LangGraph answer grounded in metadata and retrieved chunks."""
    from langchain_openai import ChatOpenAI

    query = state.get("current_query", "")
    retrieved_chunks = state.get("retrieved_chunks", [])
    citations = _extract_citations(retrieved_chunks)
    context = _build_generation_context(state)
    llm = ChatOpenAI(model=settings.openai_chat_model, temperature=0, streaming=True, api_key=settings.openai_api_key)
    messages = _build_prompt_messages(state=state, query=query, context=context)
    response = await llm.ainvoke(messages)
    answer = str(response.content or "").strip()
    if citations and not _answer_contains_any_citation(answer, citations):
        answer = f"{answer}\n\nSources: " + ", ".join(str(citation.get("citation_label") or citation.get("label")) for citation in citations)
    return {"answer": answer, "citations": citations, "messages": [AIMessage(content=answer)]}


def build_ingest_graph() -> Any:
    """Builds the ingestion graph without persistent chat checkpointing."""
    graph = StateGraph(VideoRAGState)
    graph.add_node("extract_videos", extract_videos_node)
    graph.add_node("compute_metrics", compute_metrics_node)
    graph.add_node("build_vectorstore", build_vectorstore_node)
    graph.add_edge(START, "extract_videos")
    graph.add_edge("extract_videos", "compute_metrics")
    graph.add_edge("compute_metrics", "build_vectorstore")
    graph.add_edge("build_vectorstore", END)
    return graph.compile()


def build_chat_graph() -> Any:
    """Builds the checkpointed chat graph with metadata/content routing."""
    graph = StateGraph(VideoRAGState)
    graph.add_node("route_query", route_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate", generate_node)
    graph.add_edge(START, "route_query")
    graph.add_conditional_edges("route_query", _route_decision, {"metadata": "generate", "content": "retrieve"})
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate")
    graph.add_edge("generate", END)
    return graph.compile(checkpointer=build_chat_checkpointer())


def get_ingest_graph() -> Any:
    """Returns the module-level ingestion graph singleton."""
    return _ingest_graph


def get_chat_graph() -> Any:
    """Returns the module-level chat graph singleton."""
    return _chat_graph


def _route_decision(state: VideoRAGState) -> Literal["metadata", "content"]:
    """Returns the conditional route selected by route_query_node."""
    return "metadata" if state.get("query_type") == "metadata" else "content"


def _build_documents(state: VideoRAGState) -> list[Document]:
    """Converts transcripts and metadata into Chroma-ready LangChain documents."""
    metadata = state.get("video_metadata", {})
    transcripts = state.get("transcripts", {})
    documents: list[Document] = []
    for video_id in ("A", "B"):
        values = metadata.get(video_id, {})
        transcript = transcripts.get(video_id, "").strip()
        if not transcript:
            continue
        content = "\n".join(
            [
                f"Video {video_id}",
                f"Creator: {values.get('creator', '')}",
                f"Platform: {values.get('platform', '')}",
                f"Views: {values.get('views', 0)}",
                f"Likes: {values.get('likes', 0)}",
                f"Comments: {values.get('comments', 0)}",
                f"Engagement Rate: {values.get('engagement_rate', 0.0)}",
                f"Hook: {values.get('hook', '')}",
                "",
                transcript,
            ]
        )
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "video_id": str(values.get("video_id") or video_id),
                    "label": video_id,
                    "creator": str(values.get("creator") or ""),
                    "platform": str(values.get("platform") or ""),
                },
            )
        )
    return documents


def _build_generation_context(state: VideoRAGState) -> str:
    """Builds model context from dynamic metadata and retrieved transcript chunks."""
    metadata_json = json.dumps(state.get("video_metadata", {}), ensure_ascii=False, indent=2)
    rates_json = json.dumps(state.get("engagement_rates", {}), ensure_ascii=False, indent=2)
    chunk_blocks = []
    for chunk in state.get("retrieved_chunks", []):
        label = chunk.get("label") or chunk.get("video_id", "unknown")
        chunk_blocks.append(f"[Video {label} · chunk {chunk.get('chunk_index', 0)}]\n{chunk.get('content', '')}")
    chunks_text = "\n\n".join(chunk_blocks) if chunk_blocks else "No transcript chunks retrieved for this query."
    return f"VIDEO_METADATA:\n{metadata_json}\n\nENGAGEMENT_RATES:\n{rates_json}\n\nRETRIEVED_CHUNKS:\n{chunks_text}"


def _build_prompt_messages(state: VideoRAGState, query: str, context: str) -> list[BaseMessage]:
    """Builds the full chat prompt with prior message history and current context."""
    history = [message for message in state.get("messages", []) if not isinstance(message, SystemMessage)]
    has_current_query = any(isinstance(message, HumanMessage) and message.content == query for message in history[-2:])
    user_message = HumanMessage(content=f"Question: {query}\n\nUse this dynamic context:\n{context}")
    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT), *history]
    if not has_current_query:
        messages.append(user_message)
    else:
        messages[-1] = user_message
    return messages


def _serialize_chroma_result(document: Document, score: float) -> dict[str, Any]:
    """Serializes a Chroma search hit into citation-ready chunk data."""
    metadata = document.metadata or {}
    return {
        "video_id": metadata.get("video_id", "unknown"),
        "label": metadata.get("label", metadata.get("video_id", "unknown")),
        "chunk_index": _safe_int(metadata.get("chunk_index")),
        "creator": metadata.get("creator", ""),
        "platform": metadata.get("platform", ""),
        "score": float(score),
        "snippet": document.page_content[:120],
        "content": document.page_content,
    }


def _extract_citations(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extracts unique per-chunk source citations from retrieved chunks."""
    citations: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for chunk in chunks:
        video_id = str(chunk.get("video_id", "unknown"))
        label = str(chunk.get("label") or video_id)
        chunk_index = _safe_int(chunk.get("chunk_index"))
        key = (video_id, chunk_index)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "video_id": video_id,
                "label": label,
                "chunk_index": chunk_index,
                "citation_label": f"Video {label} · chunk {chunk_index}",
                "snippet": str(chunk.get("snippet") or ""),
            }
        )
    return citations


def _query_video_filter(query: str) -> str | None:
    """Returns a video metadata filter when the query explicitly names Video A or Video B."""
    normalized = query.lower()
    if "video a" in normalized:
        return "A"
    if "video b" in normalized:
        return "B"
    return None


def _answer_contains_any_citation(answer: str, citations: list[dict[str, Any]]) -> bool:
    """Checks whether the generated answer already contains one of the retrieved citation labels."""
    return any(str(citation.get("citation_label") or citation.get("label")) in answer for citation in citations)


def _metadata_to_dict(metadata: VideoMetadata) -> dict[str, Any]:
    """Converts VideoMetadata into a serializable dict with stable default values."""
    return metadata.model_dump()


def _engagement_rate(likes: int, comments: int, views: int) -> float:
    """Computes engagement percentage with zero-view protection."""
    return round((likes + comments) / max(views, 1) * 100, 4)


def _safe_int(value: Any) -> int:
    """Converts metadata numeric values to int with a zero fallback."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_collection_name(session_id: str) -> str:
    """Normalizes session IDs into Chroma-safe collection names."""
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", session_id).strip("_")
    return normalized[:48] or "default"


_ingest_graph = build_ingest_graph()
_chat_graph = build_chat_graph()
