import logging
from typing import Any
from urllib.parse import urlparse

from tenacity import retry, stop_after_attempt, wait_exponential

from app.ingest.models import VideoMetadata

logger = logging.getLogger(__name__)

INSTAGRAM_HOSTS = {"instagram.com", "www.instagram.com", "m.instagram.com"}


def parse_instagram_id(url: str) -> str:
    """Extracts the public Instagram media shortcode from reel, p, or tv URLs."""
    parsed = urlparse(url.strip())
    host = parsed.netloc.lower().split(":")[0]
    if host not in INSTAGRAM_HOSTS:
        raise ValueError("Unsupported Instagram host")
    parts = [part for part in parsed.path.split("/") if part]
    for marker in ("reel", "p", "tv"):
        if marker in parts:
            index = parts.index(marker)
            if len(parts) > index + 1 and parts[index + 1]:
                return parts[index + 1]
    if parts:
        return parts[-1]
    raise ValueError("Instagram media ID not found")


def get_instagram_data(url: str, video_id: str | None = None) -> VideoMetadata:
    """Fetches Instagram metadata and best-effort transcript text."""
    real_video_id = parse_instagram_id(url) if video_id is None else video_id
    try:
        info = _fetch_instagram_info(url)
    except Exception:
        logger.warning(
            "instagram_extraction_failed",
            exc_info=True,
            extra={"event": "instagram_extraction_failed", "video_id": real_video_id},
        )
        return _fallback_metadata(url=url, video_id=real_video_id, error_reason="Instagram extraction failed")

    payload = info or {}
    if not payload:
        logger.warning("instagram_payload_empty", extra={"event": "instagram_payload_empty", "video_id": real_video_id})
    transcript = _extract_subtitle_text(payload).strip() or str(payload.get("description") or "").strip()
    hook = transcript[:500]
    if not transcript:
        logger.warning("instagram_transcript_empty", extra={"event": "instagram_transcript_empty", "video_id": real_video_id})
        hook = "Unknown"

    return _build_metadata(url=url, video_id=real_video_id, transcript=transcript, hook=hook, info=payload)


@retry(wait=wait_exponential(multiplier=1, min=1, max=4), stop=stop_after_attempt(3), reraise=True)
def _fetch_instagram_info(url: str) -> dict[str, Any]:
    """Fetches Instagram metadata with yt-dlp timeout and retry."""
    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True, "writesubtitles": True, "writeautomaticsub": True, "socket_timeout": 15}) as ydl:
        info = ydl.extract_info(url, download=False)
    return info or {}


def _extract_subtitle_text(info: dict[str, Any]) -> str:
    """Extracts inline subtitle text from yt-dlp automatic subtitles when available."""
    subtitles = info.get("automatic_captions") or info.get("subtitles") or {}
    for language_entries in subtitles.values():
        if not isinstance(language_entries, list):
            continue
        for entry in language_entries:
            if not isinstance(entry, dict):
                continue
            data = entry.get("data")
            if isinstance(data, str) and data.strip():
                return data
    return ""


def _build_metadata(url: str, video_id: str, transcript: str, hook: str, info: dict[str, Any]) -> VideoMetadata:
    """Builds normalized Instagram metadata from yt-dlp payload."""
    views = _int_field(info, "view_count", video_id)
    likes = _int_field(info, "like_count", video_id)
    comments = _int_field(info, "comment_count", video_id)
    creator = _string_field(info, ("uploader", "channel", "uploader_id"), video_id)
    follower_count = _get_follower_count(creator)
    if follower_count is None:
        follower_count = _int_field(info, "channel_follower_count", video_id)
    engagement_rate = round((likes + comments) / max(views, 1) * 100, 4)
    return VideoMetadata(
        video_id=video_id,
        url=url,
        platform="instagram",
        transcript=transcript,
        views=views,
        likes=likes,
        comments=comments,
        creator=creator,
        follower_count=follower_count,
        hashtags=_hashtags(info, video_id),
        upload_date=_string_field(info, ("upload_date",), video_id),
        duration=_int_field(info, "duration", video_id),
        engagement_rate=engagement_rate,
        hook=hook,
    )


def _fallback_metadata(url: str, video_id: str, error_reason: str = "") -> VideoMetadata:
    """Returns a safe empty Instagram metadata payload after extraction failure."""
    return VideoMetadata(
        video_id=video_id,
        url=url,
        platform="instagram",
        transcript="",
        views=0,
        likes=0,
        comments=0,
        creator="Unknown",
        follower_count=0,
        hashtags=[],
        upload_date="Unknown",
        duration=0,
        engagement_rate=0.0,
        hook="Unknown",
        error_reason=error_reason,
    )


def _get_follower_count(username: str) -> int | None:
    """Fetches Instagram follower count with an isolated optional Instaloader fallback."""
    if not username or username == "Unknown":
        logger.warning("instagram_follower_username_missing", extra={"event": "instagram_follower_username_missing"})
        return None
    try:
        import instaloader

        loader = instaloader.Instaloader(download_pictures=False, download_videos=False, download_video_thumbnails=False)
        profile = instaloader.Profile.from_username(loader.context, username)
        return int(profile.followers or 0)
    except Exception:
        logger.warning(
            "instagram_follower_fetch_failed",
            exc_info=True,
            extra={"event": "instagram_follower_fetch_failed", "creator": username},
        )
        return None


def _int_field(info: dict[str, Any], key: str, video_id: str) -> int:
    """Reads an integer metadata field with a logged zero fallback."""
    value = info.get(key)
    if value is None:
        logger.warning("instagram_numeric_field_missing", extra={"event": "instagram_numeric_field_missing", "video_id": video_id, "field": key})
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        logger.warning("instagram_numeric_field_invalid", extra={"event": "instagram_numeric_field_invalid", "video_id": video_id, "field": key})
        return 0


def _string_field(info: dict[str, Any], keys: tuple[str, ...], video_id: str) -> str:
    """Reads the first non-empty string metadata field with a logged Unknown fallback."""
    for key in keys:
        value = info.get(key)
        if value:
            return str(value)
    logger.warning("instagram_string_field_missing", extra={"event": "instagram_string_field_missing", "video_id": video_id, "fields": ",".join(keys)})
    return "Unknown"


def _hashtags(info: dict[str, Any], video_id: str) -> list[str]:
    """Reads hashtags from yt-dlp tags with a logged empty-list fallback."""
    tags = info.get("tags")
    if not tags:
        logger.warning("instagram_hashtags_missing", extra={"event": "instagram_hashtags_missing", "video_id": video_id})
        return []
    return [str(item) for item in tags if item]
