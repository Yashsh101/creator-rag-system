import pytest

from app.ingest.instagram import parse_instagram_id
from app.ingest.youtube import parse_youtube_id


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.youtube.com/watch?v=abc123XYZ09", "abc123XYZ09"),
        ("https://youtube.com/watch?v=abc123XYZ09&t=42s", "abc123XYZ09"),
        ("https://youtu.be/abc123XYZ09?si=share", "abc123XYZ09"),
        ("https://www.youtube.com/shorts/abc123XYZ09?feature=share", "abc123XYZ09"),
        ("https://www.youtube.com/embed/abc123XYZ09?start=5", "abc123XYZ09"),
        ("https://m.youtube.com/watch?feature=youtu.be&v=abc123XYZ09", "abc123XYZ09"),
    ],
)
def test_parse_youtube_id_supported_urls(url: str, expected: str) -> None:
    """Parses real YouTube IDs from every supported public URL form."""
    assert parse_youtube_id(url) == expected


def test_parse_youtube_id_rejects_unsupported_host() -> None:
    """Rejects non-YouTube hosts before any external network call."""
    with pytest.raises(ValueError):
        parse_youtube_id("https://evil.example/watch?v=abc123")


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.instagram.com/reel/Cabc123/?igsh=abc", "Cabc123"),
        ("https://instagram.com/p/Cabc123/", "Cabc123"),
        ("https://m.instagram.com/tv/Cabc123/?utm_source=ig_web_copy_link", "Cabc123"),
    ],
)
def test_parse_instagram_id_supported_urls(url: str, expected: str) -> None:
    """Parses real Instagram shortcode IDs from supported public URL forms."""
    assert parse_instagram_id(url) == expected
