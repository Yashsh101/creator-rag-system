import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.auth import AuthContext, require_auth
from app.db.session import get_db
from app.ingest.instagram import get_instagram_data, parse_instagram_id
from app.ingest.models import IngestJobResponse, IngestResult, IngestStatus, VideoIngestRequest, VideoIngestResponse, VideoMetadata, VideoMetadataPublic, VideoPairIngestRequest
from app.ingest.youtube import get_youtube_data, parse_youtube_id
from app.services.ingestion_service import IngestionService
from app.services.video_ingestion_job_service import VideoIngestionJobService
from app.services.rag_graph import VideoRAGState, build_vectorstore_node, compute_metrics_node

router = APIRouter()
logger = logging.getLogger(__name__)
ingestion_service = IngestionService()
video_ingestion_service = VideoIngestionJobService()
_SESSION_STATES: dict[str, VideoRAGState] = {}
_INGEST_JOBS: dict[str, IngestJobResponse] = {}
_VIDEO_CACHE: dict[str, VideoMetadata] = {}
_video_store: dict[str, VideoMetadata] = {}
_latest_session_id: str | None = None


@router.post("/ingest", response_model=IngestJobResponse)
def ingest_videos(
    payload: VideoPairIngestRequest,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_auth),
) -> IngestJobResponse:
    """Creates an async pair-ingest job for video processing."""
    del auth

    # Create video ingestion job using the job service
    job = video_ingestion_service.create_video_pair_job(db, payload)

    # Return job response
    job_response = IngestJobResponse(
        job_id=job.id,
        session_id=payload.session_id,
        status=job.status
    )

    logger.info(
        "video_pair_ingest_job_created",
        extra={
            "event": "video_pair_ingest_job_created",
            "job_id": job.id,
            "session_id": payload.session_id
        }
    )

    return job_response


@router.get("/ingest/{job_id}", response_model=IngestJobResponse)
def get_ingest_job(job_id: str, db: Session = Depends(get_db), auth: AuthContext = Depends(require_auth)) -> IngestJobResponse:
    """Returns the current pair-ingest job snapshot for polling clients."""
    del auth
    job = db.get(IngestionJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingest job not found")

    # For video pair jobs, we don't store session_id in the job, so we'll need to
    # look it up from our session state cache or return a placeholder
    # In a more complete implementation, we'd store session_id in the job
    session_id = "unknown"  # Placeholder - in reality we'd track this better

    job_response = IngestJobResponse(
        job_id=job.id,
        session_id=session_id,
        status=job.status
    )
    return job_response



@router.post("/videos/ingest", response_model=VideoIngestResponse)
def ingest_video(
    payload: VideoIngestRequest,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_auth),
) -> VideoIngestResponse:
    """Ingests a video transcript into the existing RAG document store."""
    platform = _detect_platform(payload.url)
    metadata = _fetch_video_metadata(url=payload.url, video_id=payload.video_id, platform=platform)
    if not metadata.transcript.strip() and not _has_stored_transcript():
        raise HTTPException(status_code=422, detail="Both transcripts are empty")

    document, version, chunk_count = ingestion_service.ingest_text_with_acl(
        db=db,
        filename=f"{metadata.platform}-{metadata.video_id}.txt",
        text=_format_video_text(metadata),
        source_type="video",
        metadata=_chunk_metadata(metadata),
        owner_id=auth.user_id,
        visibility="private",
        allowed_user_ids=[auth.user_id],
        allowed_group_ids=auth.groups,
    )
    _store_legacy_metadata(metadata)
    return VideoIngestResponse(
        document_id=document.id,
        document_version_id=version.id,
        filename=document.filename,
        chunk_count=chunk_count,
        status=version.status,
        metadata=metadata,
    )


@router.get("/videos/metadata", response_model=dict[str, VideoMetadataPublic])
def get_video_metadata(auth: AuthContext = Depends(require_auth)) -> dict[str, VideoMetadataPublic]:
    """Returns stored video metadata without transcripts."""
    del auth
    if _video_store:
        return {video_id: VideoMetadataPublic.model_validate(metadata) for video_id, metadata in _video_store.items()}
    if _latest_session_id is None or _latest_session_id not in _SESSION_STATES:
        return {}
    metadata = _SESSION_STATES[_latest_session_id].get("video_metadata", {})
    return {video_id: VideoMetadataPublic.model_validate(values) for video_id, values in metadata.items()}


def _initial_state(payload: VideoPairIngestRequest) -> VideoRAGState:
    """Builds the initial state for the LangGraph ingest pipeline."""
    return {
        "session_id": payload.session_id,
        "video_urls": {"A": payload.url_a, "B": payload.url_b},
        "video_metadata": {},
        "transcripts": {},
        "engagement_rates": {},
        "vectorstore_ready": False,
        "messages": [],
        "current_query": "",
        "query_type": "",
        "retrieved_chunks": [],
        "answer": "",
        "citations": [],
    }


def _detect_platform(url: str) -> str:
    """Detects supported video platform from URL."""
    normalized_url = url.lower()
    if "youtube.com" in normalized_url or "youtu.be" in normalized_url:
        return "youtube"
    return "instagram"


def _fetch_video_metadata(url: str, video_id: str, platform: str) -> VideoMetadata:
    """Routes video metadata extraction to the platform adapter."""
    del video_id
    if platform == "youtube":
        return get_youtube_data(url=url)
    if platform == "instagram":
        return get_instagram_data(url=url)
    raise HTTPException(status_code=422, detail="Unsupported video platform")


def _has_stored_transcript() -> bool:
    """Checks whether any stored video has usable transcript text."""
    return any(metadata.transcript.strip() for metadata in _video_store.values()) or any(
        transcript.strip()
        for state in _SESSION_STATES.values()
        for transcript in state.get("transcripts", {}).values()
    )


def _store_legacy_metadata(metadata: VideoMetadata) -> None:
    """Stores legacy single-video ingest metadata in the session state cache."""
    global _latest_session_id
    session_id = "legacy"
    state = _SESSION_STATES.get(
        session_id,
        {
            "session_id": session_id,
            "video_urls": {},
            "video_metadata": {},
            "transcripts": {},
            "engagement_rates": {},
            "vectorstore_ready": False,
            "messages": [],
            "current_query": "",
            "query_type": "",
            "retrieved_chunks": [],
            "answer": "",
            "citations": [],
        },
    )
    video_metadata = {**state.get("video_metadata", {}), metadata.video_id: metadata.model_dump()}
    transcripts = {**state.get("transcripts", {}), metadata.video_id: metadata.transcript}
    engagement_rates = {**state.get("engagement_rates", {}), metadata.video_id: metadata.engagement_rate}
    _video_store[metadata.video_id] = metadata
    _SESSION_STATES[session_id] = {
        **state,
        "video_metadata": video_metadata,
        "transcripts": transcripts,
        "engagement_rates": engagement_rates,
    }
    _latest_session_id = session_id


def _chunk_metadata(metadata: VideoMetadata) -> dict:
    """Builds per-chunk video metadata for retrieval context."""
    return {
        "video_id": metadata.video_id,
        "platform": metadata.platform,
        "creator": metadata.creator,
        "engagement_rate": metadata.engagement_rate,
        "views": metadata.views,
        "likes": metadata.likes,
    }


def _format_video_text(metadata: VideoMetadata) -> str:
    """Formats video metadata and transcript into retrievable source text."""
    return "\n".join(
        [
            f"Video ID: {metadata.video_id}",
            f"Platform: {metadata.platform}",
            f"URL: {metadata.url}",
            f"Creator: {metadata.creator}",
            f"Views: {metadata.views}",
            f"Likes: {metadata.likes}",
            f"Comments: {metadata.comments}",
            f"Engagement Rate: {metadata.engagement_rate}",
            f"Upload Date: {metadata.upload_date}",
            f"Duration Seconds: {metadata.duration}",
            f"Hook: {metadata.hook}",
            "",
            "Transcript:",
            metadata.transcript,
        ]
    )
