"""Smoke tests for the YouTube comparison layer (knowledge_pipeline/discovery/youtube.py).

Validates that:
- URL filtering correctly identifies YouTube URLs from a mixed list
- Video ID extraction handles the common URL shapes (watch, youtu.be, embed, shorts)
- format_for_prompt produces compact, well-bounded output for the LLM context
- fetch_transcripts gracefully handles errors per-video (does NOT raise on
  individual failures) and returns separate transcripts/errors lists

No network access — fetch_transcripts is not exercised here (would require
real video IDs). All other behavior is pure and covered by these tests.

Run with:
    /home/janak/ai/knowledge-pipeline/.venv/bin/python tests/smoke_youtube.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge_pipeline.discovery.youtube import (
    YouTubeTranscript,
    extract_video_id,
    filter_youtube_urls,
    format_for_prompt,
)


# ---- extract_video_id ----------------------------------------------------


def test_extract_video_id_youtube_com():
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_youtu_be():
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_youtu_be_with_query():
    # youtu.be/<id> form is the simplest; query strings after the id are allowed.
    assert extract_video_id("https://youtu.be/dQw4w9WgXcQ?t=42") == "dQw4w9WgXcQ"


def test_extract_video_id_youtube_embed():
    assert extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_youtube_shorts():
    assert extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_video_id_non_youtube():
    assert extract_video_id("https://example.com/foo") is None
    assert extract_video_id("https://github.com/foo/bar") is None
    assert extract_video_id("") is None


def test_extract_video_id_dedupes_dedupe_via_filter():
    # extract_video_id is deterministic; dedup happens at filter_youtube_urls.
    assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10") == "dQw4w9WgXcQ"


# ---- filter_youtube_urls ---------------------------------------------------


def test_filter_youtube_urls_keeps_only_youtube():
    urls = [
        "https://www.youtube.com/watch?v=ABC123",
        "https://example.com/page",
        "https://youtu.be/DEF456",
        "https://github.com/torvalds/linux",
        "https://www.youtube.com/embed/GHI789",
    ]
    out = filter_youtube_urls(urls)
    assert out == [
        "https://www.youtube.com/watch?v=ABC123",
        "https://youtu.be/DEF456",
        "https://www.youtube.com/embed/GHI789",
    ]


def test_filter_youtube_urls_dedupes_preserving_order():
    urls = [
        "https://www.youtube.com/watch?v=ABC123",
        "https://youtu.be/ABC123",
        "https://www.youtube.com/embed/ABC123",
        "https://www.youtube.com/watch?v=ABC123",  # exact dup
    ]
    out = filter_youtube_urls(urls)
    assert out == [urls[0], urls[1], urls[2]]


def test_filter_youtube_urls_handles_empty_and_garbage():
    assert filter_youtube_urls([]) == []
    assert filter_youtube_urls(["", "   ", "ftp://nope"]) == []
    assert filter_youtube_urls([123, None, "https://youtu.be/ok"]) == ["https://youtu.be/ok"]


# ---- format_for_prompt --------------------------------------------------


def _make_transcript(video_id, summary_seed=1):
    return YouTubeTranscript(
        url=f"https://www.youtube.com/watch?v={video_id}",
        video_id=video_id,
        text=f"full transcript for {video_id}",
        summary=f"summary word {summary_seed} for {video_id}",
    )


def test_format_for_prompt_includes_url_and_summary():
    transcripts = [_make_transcript("AAA111", 1)]
    out = format_for_prompt(transcripts, max_chars_per_video=400)
    assert "AAA111" in out
    assert "summary word 1" in out
    assert "full transcript" not in out  # we only emit the summary excerpt
    assert out.startswith("- http")


def test_format_for_prompt_truncates_long_summaries():
    t = YouTubeTranscript(
        url="https://www.youtube.com/watch?v=BBB222",
        video_id="BBB222",
        text="x",
        summary="A" * 1000,
    )
    out = format_for_prompt([t], max_chars_per_video=50)
    assert "BBB222" in out
    # Only ~50 'A' chars from the summary (plus URL/video_id overhead)
    a_count = out.count("A")
    assert a_count <= 55


def test_format_for_prompt_empty_list_returns_placeholder():
    assert (
        format_for_prompt([], max_chars_per_video=400)
        == "(no YouTube transcripts available)"
    )


# ---- YouTubeTranscript dataclass ----------------------------------------


def test_youtube_transcript_dataclass_immutability():
    t = _make_transcript("CCC333")
    # frozen=True means assignment raises.
    raised = False
    try:
        t.video_id = "DDD444"  # type: ignore[misc]
    except Exception:
        raised = True
    assert raised, "YouTubeTranscript should be frozen"


# ---- Test runner --------------------------------------------------------


TESTS = [
    test_extract_video_id_youtube_com,
    test_extract_video_id_youtu_be,
    test_extract_video_id_youtu_be_with_query,
    test_extract_video_id_youtube_embed,
    test_extract_video_id_youtube_shorts,
    test_extract_video_id_non_youtube,
    test_extract_video_id_dedupes_dedupe_via_filter,
    test_filter_youtube_urls_keeps_only_youtube,
    test_filter_youtube_urls_dedupes_preserving_order,
    test_filter_youtube_urls_handles_empty_and_garbage,
    test_format_for_prompt_includes_url_and_summary,
    test_format_for_prompt_truncates_long_summaries,
    test_format_for_prompt_empty_list_returns_placeholder,
    test_youtube_transcript_dataclass_immutability,
]


def main() -> int:
    failed = 0
    for test in TESTS:
        try:
            test()
        except AssertionError as exc:
            print(f"FAIL  {test.__name__}: {exc}")
            failed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR {test.__name__}: {exc!r}")
            failed += 1
        else:
            print(f"OK    {test.__name__}")
    if failed:
        print(f"\n{failed} test(s) failed")
        return 1
    print(f"\nall {len(TESTS)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
