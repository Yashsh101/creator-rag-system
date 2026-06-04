from typing import Any
from urllib.parse import parse_qs, urlparse

from fastapi import HTTPException
from tenacity import retry, stop_after_attempt, wait_exponential

from app.ingest.models import VideoMetadata

YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be", "www.youtu.be"}


def parse_youtube_id(url: str) -> str:
    """Extracts the canonical YouTube video ID from supported public URL shapes."""
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().split(":")[0]
    if host not in YOUTUBE_HOSTS:
        raise ValueError("Unsupported YouTube host")

    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/")[0]
        if video_id:
            return video_id

    query_id = parse_qs(parsed.query).get("v", [""])[0].strip()
    if query_id:
        return query_id

    parts = [part for part in parsed.path.split("/") if part]
    for marker in ("shorts", "embed", "live"):
        if marker in parts:
            index = parts.index(marker)
            if len(parts) > index + 1 and parts[index + 1]:
                return parts[index + 1]

    raise ValueError("YouTube video ID not found")


def get_youtube_data(url: str, video_id: str | None = None) -> VideoMetadata:
    """Fetches YouTube transcript and public metadata."""
    real_video_id = parse_youtube_id(url) if video_id is None else video_id
    try:
        transcript_items = _fetch_transcript(real_video_id)
        transcript = " ".join(str(item.get("text", "")).strip() for item in transcript_items).strip()
        if not transcript:
            raise HTTPException(status_code=422, detail="Transcript unavailable")

        info = _fetch_youtube_info(url)

        return _build_metadata(url=url, video_id=real_video_id, transcript=transcript, info=info or {})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Transcript unavailable") from exc


@retry(wait=wait_exponential(multiplier=1, min=1, max=4), stop=stop_after_attempt(3), reraise=True)
def _fetch_transcript(video_id: str) -> list[dict[str, Any]]:
    """Fetches YouTube transcript with retry around provider failures."""
    from youtube_transcript_api import YouTubeTranscriptApi

    return YouTubeTranscriptApi.get_transcript(video_id)


@retry(wait=wait_exponential(multiplier=1, min=1, max=4), stop=stop_after_attempt(3), reraise=True)
def _fetch_youtube_info(url: str) -> dict[str, Any]:
    """Fetches YouTube metadata with yt-dlp timeout and retry."""
    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "socket_timeout": 15}) as ydl:
        info = ydl.extract_info(url, download=False)
    return info or {}


def _build_metadata(url: str, video_id: str, transcript: str, info: dict[str, Any]) -> VideoMetadata:
    """Builds normalized YouTube metadata from yt-dlp payload."""
    views = int(info.get("view_count") or 0)
    likes = int(info.get("like_count") or 0)
    comments = int(info.get("comment_count") or 0)
    engagement_rate = round((likes + comments) / max(views, 1) * 100, 4)
    return VideoMetadata(
        video_id=video_id,
        url=url,
        platform="youtube",
        transcript=transcript,
        views=views,
        likes=likes,
        comments=comments,
        creator=str(info.get("uploader") or info.get("channel") or ""),
        follower_count=int(info.get("channel_follower_count") or 0),
        hashtags=[str(item) for item in info.get("tags") or []],
        upload_date=str(info.get("upload_date") or ""),
        duration=int(info.get("duration") or 0),
        engagement_rate=engagement_rate,
        hook=transcript[:500],
    )
