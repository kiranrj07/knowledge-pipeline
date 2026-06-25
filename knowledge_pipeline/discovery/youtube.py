"""YouTube transcript client for the comparison layer.

Fetches transcripts for YouTube video URLs returned by SearXNG. Uses
`youtube-transcript-api` which scrapes the auto-generated captions (no API key
required for the public transcript endpoint).

Each transcript is returned as a `YouTubeTranscript` carrying:
- the source URL
- the video ID
- the full transcript text (segments joined with spaces)
- a short summary (first ~500 chars) for inclusion in the synthesis prompt
  when we don't want to bloat the context with full video text

Design notes:
- We never raise on a single failed transcript; we skip and continue. YouTube
  routinely returns `TranscriptsDisabled` or `NoTranscriptFound` for videos
  without captions, and we don't want the whole layer to fail because one
  video is missing captions.
- The transcript list is intentionally capped at `max_videos` to keep the
  synthesis prompt tractable (default 5).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore[import-not-found]
from youtube_transcript_api._errors import (  # type: ignore[import-not-found]
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

logger = logging.getLogger(__name__)


_YT_HOSTS = ("youtube.com", "youtu.be")


@dataclass(frozen=True)
class YouTubeTranscript:
    """A single successfully-fetched YouTube transcript."""

    url: str
    video_id: str
    text: str  # full transcript (segments joined)
    summary: str  # first ~500 chars for prompt inclusion

    @property
    def snippet(self) -> str:
        return self.summary


@dataclass(frozen=True)
class YouTubeFetchError:
    """A single failed YouTube fetch (kept so we can report coverage gaps)."""

    url: str
    video_id: str
    reason: str


def extract_video_id(url: str) -> str | None:
    """Extract the YouTube video ID from a URL.

    Supports common URL forms:
    - https://www.youtube.com/watch?v=ID
    - https://youtu.be/ID
    - https://www.youtube.com/embed/ID
    - https://www.youtube.com/shorts/ID
    """
    # youtu.be/<id>
    m = re.search(r"youtu\.be/([A-Za-z0-9_-]{6,})", url)
    if m:
        return m.group(1)
    # youtube.com/<anything>?v=<id> or &v=<id>
    m = re.search(r"youtube\.com/[^?]*\?(?:[^&]*&)*v=([A-Za-z0-9_-]{6,})", url)
    if m:
        return m.group(1)
    # youtube.com/embed/<id>
    m = re.search(r"youtube\.com/embed/([A-Za-z0-9_-]{6,})", url)
    if m:
        return m.group(1)
    # youtube.com/shorts/<id>
    m = re.search(r"youtube\.com/shorts/([A-Za-z0-9_-]{6,})", url)
    if m:
        return m.group(1)
    return None


def filter_youtube_urls(urls: list[str]) -> list[str]:
    """Return only the YouTube URLs from a mixed list, preserving order, deduped."""
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if not isinstance(u, str) or not u:
            continue
        if any(h in u for h in _YT_HOSTS):
            if u not in seen:
                seen.add(u)
                out.append(u)
    return out


def fetch_transcripts(
    urls: list[str],
    *,
    max_videos: int = 5,
    summary_chars: int = 500,
    languages: tuple[str, ...] = ("en",),
) -> tuple[list[YouTubeTranscript], list[YouTubeFetchError]]:
    """Fetch transcripts for a list of YouTube URLs.

    Returns (transcripts, errors). Skips non-YouTube URLs silently. Caps at
    `max_videos` to keep the synthesis prompt tractable. Returns errors
    (not exceptions) for individual videos that fail — callers can decide
    whether to surface them.
    """
    yt_urls = filter_youtube_urls(urls)[:max_videos]
    transcripts: list[YouTubeTranscript] = []
    errors: list[YouTubeFetchError] = []

    for url in yt_urls:
        video_id = extract_video_id(url)
        if not video_id:
            errors.append(
                YouTubeFetchError(url=url, video_id="", reason="could not parse video id")
            )
            continue
        try:
            # youtube-transcript-api returns a list of {"text", "start", "duration"}.
            segments: list[dict[str, Any]] = list(
                YouTubeTranscriptApi.get_transcript(video_id, languages=list(languages))
            )
        except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable) as e:
            errors.append(
                YouTubeFetchError(url=url, video_id=video_id, reason=type(e).__name__)
            )
            continue
        except Exception as e:  # noqa: BLE001 - last-resort for any transcript error
            logger.warning("YouTube transcript fetch failed for %s: %r", url, e)
            errors.append(
                YouTubeFetchError(url=url, video_id=video_id, reason=f"unexpected: {e!r}")
            )
            continue

        text = " ".join(seg.get("text", "").strip() for seg in segments).strip()
        if not text:
            errors.append(
                YouTubeFetchError(url=url, video_id=video_id, reason="empty transcript")
            )
            continue

        transcripts.append(
            YouTubeTranscript(
                url=url,
                video_id=video_id,
                text=text,
                summary=text[:summary_chars],
            )
        )

    return transcripts, errors


def format_for_prompt(
    transcripts: list[YouTubeTranscript],
    *,
    max_chars_per_video: int = 400,
) -> str:
    """Render a compact list of transcripts for inclusion in a synthesis prompt.

    The model uses this to reason about what YouTube videos already cover so
    the brief's "YouTube coverage delta" section can highlight unique value.
    """
    if not transcripts:
        return "(no YouTube transcripts available)"

    blocks: list[str] = []
    for t in transcripts:
        block = (
            f"- {t.url} (video_id={t.video_id})\n"
            f"  excerpt: {t.summary[:max_chars_per_video]}"
        )
        blocks.append(block)
    return "\n".join(blocks)
