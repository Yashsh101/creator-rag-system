import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import SessionLocal
from app.ingest.instagram import get_instagram_data
from app.ingest.models import (
    VideoMetadata,
    VideoMetadataPublic,
    VideoPairIngestRequest,
    IngestResult,
    IngestStatus
)
from app.ingest.youtube import get_youtube_data
from app.models.document import IngestionJob
from app.services.ingestion_service import IngestionService
from app.services.rag_graph import VideoRAGState, build_vectorstore_node, compute_metrics_node
from app.utils.hash import sha256_bytes

logger = logging.getLogger(__name__)


class VideoIngestionJobService:
    def __init__(self, ingestion_service: Optional[IngestionService] = None):
        self.ingestion_service = ingestion_service or IngestionService()
        # Memory-based session states (matching existing implementation)
        self._SESSION_STATES: dict[str, VideoRAGState] = {}
        self._VIDEO_CACHE: dict[str, VideoMetadata] = {}
        self._latest_session_id: str | None = None

    def create_video_pair_job(
        self,
        db: Session,
        payload: VideoPairIngestRequest,
        trace_id: str | None = None,
    ) -> IngestionJob:
        """Creates a video pair ingestion job (YouTube + Instagram)"""

        # Serialize the payload to store it as a "file"
        payload_json = payload.model_dump_json()
        payload_bytes = payload_json.encode('utf-8')
        content_hash = sha256_bytes(payload_bytes)

        # Store the payload in the storage backend (similar to how PDF service works)
        raw_file_uri = self.ingestion_service.storage_backend.put_bytes(
            key=f"video_pair/{content_hash}/{payload.session_id}.json",
            content=payload_bytes,
            content_type="application/json",
        )

        job = IngestionJob(
            id=uuid.uuid4().hex,
            trace_id=trace_id,
            filename=f"video_pair_{payload.session_id}.json",
            content_type="application/json",
            raw_file_uri=raw_file_uri,
            storage_backend=self.ingestion_service.storage_backend.name,
            content_hash=content_hash,
            file_size_bytes=len(payload_bytes),
            status="queued",
            max_retries=3,
        )

        # Store session_id in allowed_user_ids for easy retrieval (not ideal but works for now)
        # Better approach would be to add a session_id column to IngestionJob model
        job.allowed_user_ids = [payload.session_id] if payload.session_id else []
        job.allowed_group_ids = []  # Keep empty

        db.add(job)
        db.commit()
        db.refresh(job)

        logger.info(
            "video_pair_ingestion_job_created",
            extra={
                "event": "video_pair_ingestion_job_created",
                "job_id": job.id,
                "session_id": payload.session_id,
                "url_a": payload.url_a,
                "url_b": payload.url_b
            }
        )

        return job

    def process_job(self, job_id: str, worker_id: str | None = None) -> None:
        """Process a video pair ingestion job"""
        worker_id = worker_id or f"video-worker-{uuid.uuid4().hex[:8]}"
        db = SessionLocal()

        try:
            job = db.get(IngestionJob, job_id)
            if job is None:
                logger.error("video_ingestion_job_missing", extra={"event": "video_ingestion_job_missing"})
                return

            # Update job status to processing
            job.status = "processing"
            job.started_at = datetime.now(timezone.utc)
            job.last_attempt_at = datetime.now(timezone.utc)
            job.locked_at = datetime.now(timezone.utc)
            job.locked_by = worker_id
            db.commit()

            try:
                # Extract the payload from the job's stored file
                payload = self._extract_payload_from_job(job)
                if payload is None:
                    raise ValueError("Could not extract payload from job")

                # Process the video pair ingestion
                self._process_video_pair_job(payload, job_id)

                # Mark job as completed
                job = db.get(IngestionJob, job_id)
                if job is None:
                    return

                job.status = "completed"
                job.completed_at = datetime.now(timezone.utc)
                job.locked_at = None
                job.locked_by = None
                db.commit()

                logger.info(
                    "video_ingestion_job_completed",
                    extra={
                        "event": "video_ingestion_job_completed",
                        "job_id": job.id,
                        "session_id": payload.session_id if payload else 'unknown'
                    }
                )

            except Exception as exc:
                db.rollback()
                failed_job = db.get(IngestionJob, job_id)
                if failed_job is not None:
                    failed_job.retry_count = (failed_job.retry_count or 0) + 1
                    max_retries = failed_job.max_retries or 3
                    failed_job.status = "failed" if failed_job.retry_count >= max_retries else "queued"
                    failed_job.error_message = str(exc)
                    failed_job.failed_at = datetime.now(timezone.utc) if failed_job.status == "failed" else None
                    failed_job.locked_at = None
                    failed_job.locked_by = None
                    db.commit()

                    logger.exception(
                        "video_ingestion_job_failed",
                        extra={
                            "event": "video_ingestion_job_failed",
                            "job_id": job_id,
                            "error": str(exc)
                        }
                    )

        finally:
            db.close()

    def _extract_payload_from_job(self, job: IngestionJob) -> Optional[VideoPairIngestRequest]:
        """Extract the original payload from the job's stored file"""
        try:
            if not job.raw_file_uri:
                logger.error(
                    "missing_raw_file_uri",
                    extra={
                        "event": "missing_raw_file_uri",
                        "job_id": job.id
                    }
                )
                return None

            # Retrieve the payload bytes from storage
            payload_bytes = self.ingestion_service.storage_backend.get_bytes(job.raw_file_uri)
            if not payload_bytes:
                logger.error(
                    "failed_to_retrieve_payload",
                    extra={
                        "event": "failed_to_retrieve_payload",
                        "job_id": job.id,
                        "raw_file_uri": job.raw_file_uri
                    }
                )
                return None

            # Deserialize the payload
            payload_json = payload_bytes.decode('utf-8')
            payload = VideoPairIngestRequest.model_validate_json(payload_json)

            logger.debug(
                "payload_extracted_successfully",
                extra={
                    "event": "payload_extracted_successfully",
                    "job_id": job.id,
                    "session_id": payload.session_id
                }
            )

            return payload

        except Exception as e:
            logger.error(
                "payload_extraction_failed",
                extra={
                    "event": "payload_extraction_failed",
                    "job_id": job.id,
                    "error": str(e)
                }
            )
            return None

    def _process_video_pair_job(self, payload: VideoPairIngestRequest, job_id: str) -> None:
        """Process a video pair ingestion job using the existing logic from the router"""
        try:
            logger.info(
                "processing_video_pair_job",
                extra={
                    "event": "processing_video_pair_job",
                    "job_id": job_id,
                    "session_id": payload.session_id,
                    "url_a": payload.url_a,
                    "url_b": payload.url_b
                }
            )

            # Extract both videos concurrently: I/O-bound network calls, so a thread pool
            # roughly halves ingest latency while preserving independent per-video failure.
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_a = executor.submit(self._ingest_one, label="A", url=payload.url_a, platform="youtube")
                future_b = executor.submit(self._ingest_one, label="B", url=payload.url_b, platform="instagram")
                results = [future_a.result(), future_b.result()]

            ready_results = [result for result in results if result.status == "ready" and result.metadata is not None]
            status_str = "ready" if len(ready_results) == 2 else "partial" if ready_results else "failed"
            error_reason = "; ".join(result.error_reason for result in results if result.error_reason)
            engagement_rates = {
                result.label: result.metadata.engagement_rate
                for result in ready_results
                if result.metadata is not None
            }

            if ready_results:
                self._store_pair_session(payload=payload, results=ready_results)

            logger.info(
                "video_pair_job_processed",
                extra={
                    "event": "video_pair_job_processed",
                    "job_id": job_id,
                    "session_id": payload.session_id,
                    "status": status_str,
                    "ready_results": len(ready_results)
                }
            )

        except Exception as e:
            logger.exception(
                "video_pair_job_processing_failed",
                extra={
                    "event": "video_pair_job_processing_failed",
                    "job_id": job_id,
                    "error": str(e)
                }
            )
            raise

    def recover_stale_jobs(self, db: Session, stale_after_seconds: int = 900) -> int:
        """Recover stale processing jobs"""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=stale_after_seconds)
        jobs = (
            db.query(IngestionJob)
            .filter(IngestionJob.status == "processing", IngestionJob.locked_at.is_not(None), IngestionJob.locked_at < cutoff)
            .all()
        )
        for job in jobs:
            retry_count = job.retry_count or 0
            max_retries = job.max_retries or 3
            job.status = "queued" if retry_count < max_retries else "failed"
            job.error_message = "Recovered stale processing job"
            job.locked_at = None
            job.locked_by = None
            if job.status == "failed":
                job.failed_at = datetime.now(timezone.utc)
        db.commit()
        return len(jobs)

    def claim_next_job(self, db: Session, worker_id: str) -> Optional[IngestionJob]:
        """Claim the next queued video ingestion job"""
        job = (
            db.query(IngestionJob)
            .filter(
                IngestionJob.status == "queued",
                IngestionJob.retry_count < IngestionJob.max_retries,
                IngestionJob.storage_backend == "video_pair"  # Only video pair jobs
            )
            .order_by(IngestionJob.created_at.asc())
            .with_for_update(skip_locked=True)
            .first()
        )
        if job is None:
            return None

        job.status = "processing"
        job.started_at = job.started_at or datetime.now(timezone.utc)
        job.last_attempt_at = datetime.now(timezone.utc)
        job.locked_at = datetime.now(timezone.utc)
        job.locked_by = worker_id
        db.commit()
        return job

    def _store_pair_session(self, payload: VideoPairIngestRequest, results) -> None:
        """Store a processed video pair session (matching existing implementation)"""
        metadata_by_label: dict[str, dict] = {}
        transcripts: dict[str, str] = {}
        for result in results:
            if result.metadata is None:
                continue

            metadata = self._VIDEO_CACHE.get(f"{result.metadata.platform}:{result.metadata.video_id}")
            if metadata is None:
                continue

            metadata_by_label[result.label] = metadata.model_dump()
            transcripts[result.label] = metadata.transcript or ""

        state = self._initial_state(payload)
        state["video_metadata"] = metadata_by_label
        state["transcripts"] = transcripts
        state.update(compute_metrics_node(state))

        try:
            state.update(build_vectorstore_node(state))
        except Exception:
            logger.warning(
                "pair_vectorstore_build_failed",
                extra={
                    "event": "pair_vectorstore_build_failed",
                    "session_id": payload.session_id
                }
            )
            state["vectorstore_ready"] = False

        self._SESSION_STATES[payload.session_id] = state
        self._latest_session_id = payload.session_id

    def _initial_state(self, payload: VideoPairIngestRequest) -> VideoRAGState:
        """Build initial state for LangGraph ingest pipeline"""
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

    def _ingest_one(self, label: str, url: str, platform: str):
        """Ingest single video (matching existing implementation)"""
        try:
            if platform == "youtube":
                from app.ingest.youtube import parse_youtube_id
                video_id = parse_youtube_id(url)
            else:
                from app.ingest.instagram import parse_instagram_id
                video_id = parse_instagram_id(url)

            cache_key = f"{platform}:{video_id}"
            metadata = self._VIDEO_CACHE.get(cache_key)

            if metadata is None:
                if platform == "youtube":
                    metadata = get_youtube_data(url=url)
                else:
                    metadata = get_instagram_data(url=url)

                metadata.label = label
                self._VIDEO_CACHE[cache_key] = metadata
            else:
                metadata = metadata.model_copy(update={"label": label})

            from app.ingest.models import VideoMetadataPublic
            public = VideoMetadataPublic.model_validate(metadata)

            return type('IngestResult', (), {
                'label': label,
                'status': 'ready',
                'metadata': public,
                'error_reason': None
            })()

        except Exception as exc:
            logger.warning(
                "video_ingest_failed",
                extra={
                    "event": "video_ingest_failed",
                    "label": label,
                    "platform": platform
                },
                exc_info=True
            )

            from app.ingest.models import VideoMetadataPublic
            fallback = VideoMetadataPublic(
                video_id="",
                url=url,
                platform=platform,
                label=label,
                error_reason=str(exc),
            )

            return type('IngestResult', (), {
                'label': label,
                'status': 'failed',
                'metadata': fallback,
                'error_reason': str(exc)
            })()

    def get_session_state(self, session_id: str) -> dict | None:
        """Get stored session state for a given session ID"""
        return self._SESSION_STATES.get(session_id)

    def get_latest_session_id(self) -> str | None:
        """Get the most recently processed session ID"""
        return self._latest_session_id