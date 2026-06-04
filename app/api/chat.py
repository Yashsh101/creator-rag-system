import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.core.auth import AuthContext, require_auth
from app.ingest.router import _SESSION_STATES
from app.services.rag_graph import VideoRAGState, get_chat_graph

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    query: str = Field(min_length=3, max_length=4000)


@router.post("/api/chat/{session_id}")
async def stream_chat(
    session_id: str,
    payload: ChatRequest,
    auth: AuthContext = Depends(require_auth),
) -> StreamingResponse:
    """Streams LangGraph chat events for a completed video-ingestion session."""
    if session_id not in _SESSION_STATES:
        raise HTTPException(status_code=404, detail="Session not found")

    return StreamingResponse(
        _generate_events(session_id=session_id, query=payload.query, auth=auth),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


async def _generate_events(session_id: str, query: str, auth: AuthContext) -> AsyncGenerator[str, None]:
    """Yields SSE events from LangGraph astream_events(version='v2')."""
    graph = get_chat_graph()
    config = {"configurable": {"thread_id": session_id}}
    try:
        await _seed_state_if_needed(graph=graph, session_id=session_id, config=config)
        graph_input: VideoRAGState = {
            **_SESSION_STATES[session_id],
            "current_query": query,
            "messages": [HumanMessage(content=query)],
        }
        async for event in graph.astream_events(graph_input, config=config, version="v2"):
            event_type = event.get("event")
            event_name = event.get("name")
            event_data = event.get("data") or {}
            if event_type == "on_chat_model_stream":
                token = _extract_token(event_data)
                if token:
                    yield _sse({"type": "token", "content": token})
            if event_type == "on_chain_end" and event_name == "generate":
                citations = _extract_citations(event_data)
                if citations:
                    yield _sse({"type": "citations", "data": citations})
        yield "data: [DONE]\n\n"
    except Exception as exc:
        logger.exception(
            "chat_stream_failed",
            extra={"event": "chat_stream_failed", "session_id": session_id, "user_id": auth.user_id},
        )
        yield _sse({"type": "error", "message": str(exc)})


async def _seed_state_if_needed(graph: Any, session_id: str, config: dict[str, Any]) -> None:
    """Seeds the checkpointed chat graph with the ingest state on the first turn."""
    snapshot = graph.get_state(config)
    if getattr(snapshot, "values", None):
        return
    await graph.aupdate_state(config, _SESSION_STATES[session_id])


def _extract_token(event_data: dict[str, Any]) -> str:
    """Extracts streamed token text from a LangChain chat model stream event."""
    chunk = event_data.get("chunk")
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
    return ""


def _extract_citations(event_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extracts citation payloads from the completed generate node event."""
    output = event_data.get("output") or {}
    if isinstance(output, dict):
        citations = output.get("citations") or []
        if isinstance(citations, list):
            return [citation for citation in citations if isinstance(citation, dict)]
    return []


def _sse(data: dict[str, Any]) -> str:
    """Serializes one server-sent event payload."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
