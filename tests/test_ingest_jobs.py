from collections.abc import Callable
import sys
import types
from typing import Any

import pytest

from app.ingest import router as ingest_router
from app.ingest.models import IngestJobResponse, IngestResult, VideoMetadata, VideoMetadataPublic, VideoPairIngestRequest
from app.ingest.youtube import _fetch_transcript


def _metadata(video_id: str, label: str, platform: str = "youtube") -> VideoMetadataPublic:
    """Builds public metadata for ingest job tests."""
    return VideoMetadataPublic(
        video_id=video_id,
        url=f"https://example.com/{video_id}",
        platform=platform,
        transcript="hidden",
        label=label,
        views=100,
        likes=9,
        comments=1,
        engagement_rate=10.0,
        hook="opening hook",
    )


def test_youtube_transcript_fetch_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retries transient transcript provider failures before succeeding."""
    attempts = {"count": 0}

    class FakeYouTubeTranscriptApi:
        @staticmethod
        def get_transcript(video_id: str) -> list[dict[str, Any]]:
            return flaky_get_transcript(video_id)

    def flaky_get_transcript(video_id: str) -> list[dict[str, Any]]:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("temporary")
        return [{"text": f"ok {video_id}"}]

    fake_module = types.SimpleNamespace(YouTubeTranscriptApi=FakeYouTubeTranscriptApi)
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", fake_module)

    assert _fetch_transcript("abc123") == [{"text": "ok abc123"}]
    assert attempts["count"] == 3


def test_ingest_one_uses_real_id_and_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caches metadata by real platform ID and keeps A/B as display labels only."""
    ingest_router._VIDEO_CACHE.clear()
    calls = {"count": 0}

    def fake_youtube(url: str) -> VideoMetadata:
        calls["count"] += 1
        return VideoMetadata(
            video_id="real123",
            url=url,
            platform="youtube",
            transcript="transcript",
            views=100,
            likes=9,
            comments=1,
            engagement_rate=10.0,
        )

    monkeypatch.setattr(ingest_router, "get_youtube_data", fake_youtube)

    first = ingest_router._ingest_one(label="A", url="https://youtu.be/real123", platform="youtube")
    second = ingest_router._ingest_one(label="B", url="https://youtu.be/real123", platform="youtube")

    assert first.status == "ready"
    assert first.metadata is not None
    assert first.metadata.video_id == "real123"
    assert first.metadata.label == "A"
    assert second.metadata is not None
    assert second.metadata.label == "B"
    assert calls["count"] == 1


def test_ingest_job_partial_when_one_video_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keeps successful video data when the paired video fails."""
    job_id = "job-partial"
    payload = VideoPairIngestRequest(
        url_a="https://youtu.be/real123",
        url_b="https://www.instagram.com/reel/fail123/",
        session_id="session-partial",
    )
    ingest_router._INGEST_JOBS[job_id] = IngestJobResponse(job_id=job_id, session_id=payload.session_id, status="queued")

    def fake_ingest_one(label: str, url: str, platform: str) -> IngestResult:
        if label == "A":
            return IngestResult(label="A", status="ready", metadata=_metadata("real123", "A"), error_reason="")
        return IngestResult(label="B", status="failed", metadata=_metadata("", "B", "instagram"), error_reason="private reel")

    monkeypatch.setattr(ingest_router, "_ingest_one", fake_ingest_one)
    monkeypatch.setattr(ingest_router, "_store_pair_session", lambda payload, results: None)

    ingest_router._run_ingest_job(job_id, payload)

    job = ingest_router._INGEST_JOBS[job_id]
    assert job.status == "partial"
    assert job.video_a is not None
    assert job.video_a.video_id == "real123"
    assert job.video_b is not None
    assert job.video_b.error_reason == ""
    assert job.error_reason == "private reel"


def test_ingest_job_status_transitions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Transitions queued jobs through processing into ready."""
    job_id = "job-ready"
    payload = VideoPairIngestRequest(
        url_a="https://youtu.be/real123",
        url_b="https://www.instagram.com/reel/ig123/",
        session_id="session-ready",
    )
    ingest_router._INGEST_JOBS[job_id] = IngestJobResponse(job_id=job_id, session_id=payload.session_id, status="queued")
    original_update: Callable[..., None] = ingest_router._update_job
    seen: list[str] = ["queued"]

    def recording_update(job_id: str, **changes: object) -> None:
        original_update(job_id, **changes)
        status = changes.get("status")
        if isinstance(status, str):
            seen.append(status)

    def fake_ingest_one(label: str, url: str, platform: str) -> IngestResult:
        metadata = _metadata("real123" if label == "A" else "ig123", label, platform)
        return IngestResult(label=label, status="ready", metadata=metadata, error_reason="")

    monkeypatch.setattr(ingest_router, "_update_job", recording_update)
    monkeypatch.setattr(ingest_router, "_ingest_one", fake_ingest_one)
    monkeypatch.setattr(ingest_router, "_store_pair_session", lambda payload, results: None)

    ingest_router._run_ingest_job(job_id, payload)

    assert seen == ["queued", "processing", "ready"]
    assert ingest_router._INGEST_JOBS[job_id].status == "ready"
