import pytest
from fastapi import HTTPException

from app.core.auth import AuthContext
from app.ingest import router as video_router
from app.ingest.instagram import _build_metadata as build_instagram_metadata
from app.ingest.instagram import _extract_subtitle_text
from app.ingest.models import VideoMetadata
from app.ingest.router import _chunk_metadata, _detect_platform, _format_video_text, get_video_metadata
from app.ingest.youtube import _build_metadata as build_youtube_metadata
from app.main import app
from app.models.document import DocumentVersion
from app.services.ingestion_service import IngestionService


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value

    def scalar_one(self):
        return self.value


class FakeDB:
    def __init__(self):
        self.added = []
        self.committed = False
        self.rolled_back = False

    def execute(self, statement):
        return FakeScalar(None)

    def scalar(self, statement):
        return 0

    def add(self, item):
        self.added.append(item)

    def flush(self):
        for index, item in enumerate(self.added, start=1):
            if getattr(item, "id", None) is None:
                item.id = index

    def commit(self):
        self.committed = True

    def refresh(self, item):
        return None

    def rollback(self):
        self.rolled_back = True


class FakeEmbeddingService:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1536 for _ in texts]


class FakeStorage:
    name = "local"

    def put_bytes(self, key: str, content: bytes, content_type: str | None = None) -> str:
        return f"local://{key}"


AUTH = AuthContext(user_id="user-1", groups=["default"], role="user")


class FakeVideoPersistService:
    def __init__(self):
        self.calls = []

    def ingest_text_with_acl(self, **kwargs):
        self.calls.append(kwargs)
        document = type("DocumentObj", (), {"id": 7, "filename": kwargs["filename"]})()
        version = type("VersionObj", (), {"id": 8, "status": "completed"})()
        return document, version, 3


def test_youtube_metadata_normalizes_engagement_and_hook():
    metadata = build_youtube_metadata(
        url="https://youtube.com/watch?v=abc",
        video_id="A",
        transcript="Strong opening line " * 40,
        info={
            "view_count": 1000,
            "like_count": 90,
            "comment_count": 10,
            "uploader": "Creator",
            "channel_follower_count": 5000,
            "tags": ["ai", "rag"],
            "upload_date": "20260531",
            "duration": 42,
        },
    )

    assert metadata.platform == "youtube"
    assert metadata.engagement_rate == 10.0
    assert metadata.hook == metadata.transcript[:500]


def test_instagram_metadata_uses_zero_comments():
    metadata = build_instagram_metadata(
        url="https://instagram.com/reel/abc",
        video_id="B",
        transcript="caption transcript",
        hook="caption transcript",
        info={"view_count": 200, "like_count": 20},
    )

    assert metadata.platform == "instagram"
    assert metadata.comments == 0
    assert metadata.engagement_rate == 10.0


def test_instagram_subtitle_extraction_uses_inline_data():
    transcript = _extract_subtitle_text({"automatic_captions": {"en": [{"data": "hello from captions"}]}})

    assert transcript == "hello from captions"


def test_platform_detection_uses_youtube_domains():
    assert _detect_platform("https://youtube.com/watch?v=abc") == "youtube"
    assert _detect_platform("https://youtu.be/abc") == "youtube"
    assert _detect_platform("https://instagram.com/reel/abc") == "instagram"


def test_empty_first_transcript_is_rejected(monkeypatch):
    monkeypatch.setattr(video_router, "_video_store", {})
    monkeypatch.setattr(
        video_router,
        "_fetch_video_metadata",
        lambda url, video_id, platform: VideoMetadata(video_id=video_id, url=url, platform=platform, transcript=""),
    )

    with pytest.raises(HTTPException) as exc:
        video_router.ingest_video(payload=video_router.VideoIngestRequest(url="https://instagram.com/reel/abc", video_id="B"), db=FakeDB(), auth=AUTH)

    assert exc.value.status_code == 422


def test_video_ingest_detects_platform_and_delegates_to_ingestion(monkeypatch):
    service = FakeVideoPersistService()
    monkeypatch.setattr(video_router, "_video_store", {"A": VideoMetadata(video_id="A", url="u", platform="youtube", transcript="existing")})
    monkeypatch.setattr(video_router, "ingestion_service", service)
    monkeypatch.setattr(
        video_router,
        "_fetch_video_metadata",
        lambda url, video_id, platform: VideoMetadata(
            video_id=video_id,
            url=url,
            platform=platform,
            transcript="transcript",
            creator="Creator",
            views=10,
            likes=2,
            engagement_rate=20.0,
        ),
    )

    response = video_router.ingest_video(
        payload=video_router.VideoIngestRequest(url="https://youtu.be/abc", video_id="B"),
        db=FakeDB(),
        auth=AUTH,
    )

    assert response.document_id == 7
    assert response.metadata.platform == "youtube"
    assert video_router._video_store["B"].transcript == "transcript"
    assert service.calls[0]["metadata"]["video_id"] == "B"
    assert service.calls[0]["metadata"]["creator"] == "Creator"


def test_ingestion_service_persists_video_metadata_on_chunks():
    db = FakeDB()
    service = IngestionService(embedding_service=FakeEmbeddingService(), storage_backend=FakeStorage())
    metadata = VideoMetadata(
        video_id="A",
        url="https://youtube.com/watch?v=abc",
        platform="youtube",
        transcript="Revenue expanded because enterprise customers adopted the product. " * 30,
        views=1000,
        likes=80,
        comments=20,
        creator="Creator",
        engagement_rate=10.0,
        hook="Revenue expanded because enterprise customers adopted the product.",
    )

    document, version, chunk_count = service.ingest_text_with_acl(
        db=db,
        filename="youtube-A.txt",
        text=_format_video_text(metadata),
        source_type="video",
        metadata=_chunk_metadata(metadata),
        owner_id="user-1",
        allowed_user_ids=["user-1"],
        allowed_group_ids=["default"],
    )

    assert document.source_type == "video"
    assert document.owner_id == "user-1"
    assert isinstance(version, DocumentVersion)
    assert version.status == "completed"
    assert chunk_count >= 1
    assert db.committed is True
    chunk = next(item for item in db.added if getattr(item, "section_path", None) == "video")
    assert chunk.metadata_json["video_id"] == "A"
    assert chunk.metadata_json["platform"] == "youtube"
    assert chunk.metadata_json["creator"] == "Creator"
    assert chunk.metadata_json["engagement_rate"] == 10.0
    assert chunk.metadata_json["views"] == 1000
    assert chunk.metadata_json["likes"] == 80


def test_video_metadata_endpoint_excludes_transcripts(monkeypatch):
    monkeypatch.setattr(
        video_router,
        "_video_store",
        {"A": VideoMetadata(video_id="A", url="url", platform="youtube", transcript="secret transcript", views=12)},
    )

    response = get_video_metadata()

    assert response["A"].views == 12
    assert not hasattr(response["A"], "transcript")


def test_video_text_includes_metrics_and_transcript():
    metadata = VideoMetadata(video_id="A", url="url", platform="youtube", transcript="transcript", views=12)

    text = _format_video_text(metadata)

    assert "Views: 12" in text
    assert "Transcript:" in text
    assert "transcript" in text


def test_video_routes_are_mounted():
    route_paths = {getattr(route, "path", "") for route in app.routes}

    assert "/api/v1/videos/ingest" in route_paths
    assert "/api/v1/videos/metadata" in route_paths
