from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class VideoMetadata(BaseModel):
    video_id: str
    url: str
    platform: str
    transcript: str
    views: int = 0
    likes: int = 0
    comments: int = 0
    creator: str = ""
    follower_count: int = 0
    hashtags: list[str] = Field(default_factory=list)
    upload_date: str = ""
    duration: int = 0
    engagement_rate: float = 0.0
    hook: str = ""
    label: str = ""
    error_reason: str = ""


class VideoIngestRequest(BaseModel):
    url: str
    video_id: str


class VideoPairIngestRequest(BaseModel):
    url_a: str = Field(min_length=8)
    url_b: str = Field(min_length=8)
    session_id: str = Field(default_factory=lambda: uuid.uuid4().hex, min_length=1, max_length=128)

    @field_validator("url_a")
    @classmethod
    def url_a_must_be_youtube(cls, value: str) -> str:
        """Validates that Video A is a YouTube URL."""
        normalized = value.lower()
        if "youtube.com" not in normalized and "youtu.be" not in normalized:
            raise ValueError("Video A must be a YouTube URL")
        return value

    @field_validator("url_b")
    @classmethod
    def url_b_must_be_instagram(cls, value: str) -> str:
        """Validates that Video B is an Instagram URL."""
        if "instagram.com" not in value.lower():
            raise ValueError("Video B must be an Instagram URL")
        return value


class VideoIngestResponse(BaseModel):
    document_id: int
    document_version_id: int
    filename: str
    chunk_count: int
    status: str
    metadata: VideoMetadata


class VideoPairIngestResponse(BaseModel):
    session_id: str
    video_a: VideoMetadataPublic
    video_b: VideoMetadataPublic
    engagement_rates: dict[str, float]


IngestStatus = Literal["queued", "processing", "partial", "ready", "failed"]


class IngestResult(BaseModel):
    """Represents one video extraction outcome with partial metadata fallback."""
    label: Literal["A", "B"]
    status: Literal["ready", "failed"]
    metadata: VideoMetadataPublic | None = None
    error_reason: str = ""


class IngestJobResponse(BaseModel):
    """Represents a pair-ingest job snapshot for frontend polling."""
    job_id: str
    session_id: str
    status: IngestStatus
    video_a: VideoMetadataPublic | None = None
    video_b: VideoMetadataPublic | None = None
    engagement_rates: dict[str, float] = Field(default_factory=dict)
    results: list[IngestResult] = Field(default_factory=list)
    error_reason: str = ""


class VideoMetadataPublic(BaseModel):
    video_id: str
    url: str
    platform: str
    views: int = 0
    likes: int = 0
    comments: int = 0
    creator: str = ""
    follower_count: int = 0
    hashtags: list[str] = Field(default_factory=list)
    upload_date: str = ""
    duration: int = 0
    engagement_rate: float = 0.0
    hook: str = ""
    label: str = ""
    error_reason: str = ""

    model_config = {"from_attributes": True}
