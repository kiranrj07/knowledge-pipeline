"""Smoke tests for the GitHub client (run directly with the venv python; no pytest).

Usage:
    /home/janak/ai/knowledge-pipeline/.venv/bin/python tests/smoke_github_client.py
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge_pipeline.source_code.github_client import (  # noqa: E402
    FileContents,
    GitHubClient,
    GitHubClientError,
    TOPIC_REPO_MAP,
    _parse_code_match,
    _parse_file_contents,
    match_topic_files,
)


def test_client_construction() -> None:
    c = GitHubClient("ghp_test_token")
    assert c._token == "ghp_test_token"
    assert c._base_url == "https://api.github.com"

    try:
        GitHubClient("")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on empty token")

    # Trailing slash on base_url must be stripped.
    c2 = GitHubClient("tok", base_url="https://api.github.com/")
    assert c2._base_url == "https://api.github.com"


def test_parse_code_match() -> None:
    search_item = {
        "name": "tcp_input.c",
        "path": "net/ipv4/tcp_input.c",
        "sha": "abc123def456",
        "html_url": "https://github.com/torvalds/linux/blob/master/net/ipv4/tcp_input.c",
        "score": 1.0,
        "repository": {"full_name": "torvalds/linux"},
    }
    match = _parse_code_match(search_item)
    assert match.repo_full_name == "torvalds/linux"
    assert match.path == "net/ipv4/tcp_input.c"
    assert match.sha == "abc123def456"
    assert match.score == 1.0
    assert match.html_url is not None

    # Missing fields handled gracefully.
    sparse = _parse_code_match({"name": "x.c", "path": "x.c", "sha": "1", "repository": {}})
    assert sparse.repo_full_name == ""
    assert sparse.html_url is None
    assert sparse.score is None


def test_parse_file_contents_text() -> None:
    sample = "/* TCP input */\nstatic void tcp_rcv_state_process(void) {}\n"
    encoded = base64.b64encode(sample.encode("utf-8")).decode("ascii")
    # GitHub wraps base64 at 60-char lines.
    wrapped = "\n".join(encoded[i : i + 60] for i in range(0, len(encoded), 60))
    body = {
        "name": "tcp_input.c",
        "path": "net/ipv4/tcp_input.c",
        "sha": "deadbeef",
        "size": len(sample),
        "type": "file",
        "encoding": "base64",
        "content": wrapped,
        "html_url": "https://github.com/torvalds/linux/blob/master/net/ipv4/tcp_input.c",
    }
    file = _parse_file_contents("torvalds/linux", body)
    assert file.repo_full_name == "torvalds/linux"
    assert file.path == "net/ipv4/tcp_input.c"
    assert file.sha == "deadbeef"
    assert file.content == sample
    assert file.size == len(sample)


def test_parse_file_contents_binary() -> None:
    binary = bytes([0, 1, 2, 255, 254, 253])
    encoded = base64.b64encode(binary).decode("ascii")
    body = {
        "path": "weird.bin",
        "encoding": "base64",
        "content": encoded,
        "sha": "1",
        "size": len(binary),
    }
    file = _parse_file_contents("o/r", body)
    # Binary decode uses 'replace' -> some chars are replacement / non-ascii.
    assert file.size == len(binary)
    assert any(ord(ch) > 127 for ch in file.content)


def test_parse_file_contents_errors() -> None:
    # Non-base64 encoding raises.
    try:
        _parse_file_contents("o/r", {"path": "p", "encoding": "utf-8", "content": "abc", "sha": "1"})
    except GitHubClientError as e:
        assert "encoding" in str(e)
    else:
        raise AssertionError("expected GitHubClientError on non-base64 encoding")

    # Missing content raises.
    try:
        _parse_file_contents("o/r", {"path": "p", "encoding": "base64", "content": "", "sha": "1"})
    except GitHubClientError as e:
        assert "content" in str(e)
    else:
        raise AssertionError("expected GitHubClientError on missing content")


def test_get_file_contents_input_validation() -> None:
    c = GitHubClient("tok")
    for args in (("no-slash", "path"), ("owner/repo", "")):
        try:
            c.get_file_contents(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {args}")


# ---- TOPIC_REPO_MAP + match_topic_files + tiered find_topic_sources -----


def test_match_topic_known() -> None:
    files = match_topic_files("TCP SYN backlog")
    assert len(files) > 0
    # TCP topic maps to torvalds/linux/net/ipv4/.
    assert all(repo == "torvalds/linux" for repo, _ in files)
    assert any("net/ipv4" in path for _, path in files)


def test_match_topic_specificity_ipv6_beats_socket() -> None:
    # "ipv6" is inserted before "socket" in TOPIC_REPO_MAP; verify the
    # most-specific-first ordering is honored.
    files = match_topic_files("IPv6 socket")
    assert len(files) > 0
    assert any("ipv6" in path for _, path in files), (
        "ipv6 entries should win for 'IPv6 socket'; got: " + repr(files)
    )


def test_match_topic_unknown_returns_empty() -> None:
    assert match_topic_files("quantum entanglement") == []
    assert match_topic_files("zzz_no_match_xyz") == []


def test_match_topic_empty_input() -> None:
    assert match_topic_files("") == []


def test_match_topic_iterates_in_insertion_order() -> None:
    # tcp_syn_backlog is inserted BEFORE tcp in the map; for "tcp_syn_backlog"
    # topic, the more-specific key must win (and yield fewer / more focused files).
    specific = match_topic_files("tcp_syn_backlog")
    generic = match_topic_files("tcp_retransmit_timeout")
    assert specific != generic, (
        "first-match-wins ordering broken: both queries returned the same files"
    )


def test_find_topic_sources_uses_curated_map() -> None:
    """When the topic matches the map, fetch the curated files directly
    (search_code must NOT be called)."""

    class FakeGitHubClient(GitHubClient):
        def __init__(self) -> None:
            super().__init__("ghp_test")
            self.fetched: list[tuple[str, str]] = []
            self.search_calls = 0

        def get_file_contents(
            self, repo_full_name: str, path: str, ref: str | None = None,
        ) -> FileContents:
            self.fetched.append((repo_full_name, path))
            return FileContents(
                repo_full_name=repo_full_name,
                path=path,
                sha="stub",
                size=10,
                content=f"// stub for {repo_full_name}/{path}",
                html_url=f"https://github.com/{repo_full_name}/blob/master/{path}",
            )

        def search_code(self, query: str, max_results: int = 10):  # noqa: ARG002
            self.search_calls += 1
            raise AssertionError(
                "search_code should not be called when topic matches TOPIC_REPO_MAP"
            )

    client = FakeGitHubClient()
    results = client.find_topic_sources("TCP SYN backlog", max_files=4)

    # We fetched at least one curated file, all from torvalds/linux.
    assert 1 <= len(results) <= 4
    assert client.fetched, "expected at least one curated fetch"
    assert all(repo == "torvalds/linux" for repo, _ in client.fetched)
    assert client.search_calls == 0, (
        f"search_code was called {client.search_calls} time(s); expected 0"
    )


def test_find_topic_sources_unknown_topic_uses_search() -> None:
    """For topics that don't match the map, find_topic_sources should
    fall back to search_code. We mock search_code to return empty."""

    class StubClient(GitHubClient):
        def __init__(self) -> None:
            super().__init__("ghp_test")
            self.search_called_with: str | None = None

        def search_code(self, query: str, max_results: int = 10):  # noqa: ARG002
            self.search_called_with = query
            return []  # no matches -> empty results

    client = StubClient()
    results = client.find_topic_sources("quantum entanglement noise", max_files=3)
    assert results == []
    assert client.search_called_with is not None
    assert "quantum entanglement noise" in client.search_called_with
    assert "in:file" in client.search_called_with


def test_topic_repo_map_is_nonempty_and_well_formed() -> None:
    # Sanity check on the map's structure so future edits don't silently
    # break the matcher.
    assert TOPIC_REPO_MAP, "TOPIC_REPO_MAP must not be empty"
    for key, entries in TOPIC_REPO_MAP.items():
        assert isinstance(key, str) and key, f"bad key: {key!r}"
        assert entries, f"TOPIC_REPO_MAP[{key!r}] is empty"
        for repo, path in entries:
            assert isinstance(repo, str) and "/" in repo, f"bad repo in {key!r}: {repo!r}"
            assert isinstance(path, str) and path, f"bad path in {key!r}: {path!r}"


TESTS = [
    test_client_construction,
    test_parse_code_match,
    test_parse_file_contents_text,
    test_parse_file_contents_binary,
    test_parse_file_contents_errors,
    test_get_file_contents_input_validation,
    test_match_topic_known,
    test_match_topic_specificity_ipv6_beats_socket,
    test_match_topic_unknown_returns_empty,
    test_match_topic_empty_input,
    test_match_topic_iterates_in_insertion_order,
    test_find_topic_sources_uses_curated_map,
    test_find_topic_sources_unknown_topic_uses_search,
    test_topic_repo_map_is_nonempty_and_well_formed,
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
    print("\nall smoke tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
