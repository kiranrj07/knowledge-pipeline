"""Smoke tests for the Parallel.ai MCP client + SDK client, including SSE parsing.

Run with:
    /home/janak/ai/knowledge-pipeline/.venv/bin/python tests/smoke_parallel_search.py

The SSE parser and SDK tests are pure-unit (no network, no parallel-web SDK
needed). The live MCP test is opt-in via the PARALLEL_LIVE=1 env var since
it requires the public MCP endpoint.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge_pipeline.discovery.parallel_search import (  # noqa: E402
    ParallelMCPClient,
    ParallelMCPError,
    ParallelSDKClient,
    _parse_sse_jsonrpc,
    create_parallel_client,
)


# ---- SSE parser (pure unit) ----------------------------------------------


def test_sse_well_formed_single_event() -> None:
    body = (
        'event: message\n'
        'data: {"jsonrpc":"2.0","id":1,"result":{"results":[]}}\n'
        '\n'
    )
    parsed = _parse_sse_jsonrpc(body)
    assert parsed == {"jsonrpc": "2.0", "id": 1, "result": {"results": []}}


def test_sse_compact_single_line() -> None:
    body = 'data: {"jsonrpc":"2.0","id":1,"result":{"results":[]}}'
    parsed = _parse_sse_jsonrpc(body)
    assert parsed["result"] == {"results": []}


def test_sse_multiple_events_returns_last() -> None:
    body = (
        'event: message\n'
        'data: {"jsonrpc":"2.0","id":1,"result":{"order":"first"}}\n'
        '\n'
        'event: message\n'
        'data: {"jsonrpc":"2.0","id":1,"result":{"order":"last"}}\n'
        '\n'
    )
    parsed = _parse_sse_jsonrpc(body)
    assert parsed["result"] == {"order": "last"}


def test_sse_crlf_normalized() -> None:
    body = (
        'event: message\r\n'
        'data: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}\r\n'
        '\r\n'
    )
    parsed = _parse_sse_jsonrpc(body)
    assert parsed["result"] == {"ok": True}


def test_sse_skips_malformed_data_lines() -> None:
    body = (
        'data: not-json-at-all\n'
        '\n'
        'data: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n'
        '\n'
    )
    parsed = _parse_sse_jsonrpc(body)
    assert parsed["result"] == {"ok": True}


def test_sse_bare_json_without_data_prefix() -> None:
    body = '{"jsonrpc":"2.0","id":1,"result":{"ok":true}}'
    parsed = _parse_sse_jsonrpc(body)
    assert parsed["result"] == {"ok": True}


def test_sse_empty_body_raises() -> None:
    try:
        _parse_sse_jsonrpc("")
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("expected ValueError on empty SSE body")


def test_sse_no_decodable_data_raises() -> None:
    body = "event: message\ndata: not json\n\n"
    try:
        _parse_sse_jsonrpc(body)
    except ValueError as exc:
        assert "no `data:`" in str(exc)
    else:
        raise AssertionError("expected ValueError when no data lines decode")


# ---- Client construction / input validation ------------------------------


def test_client_construction() -> None:
    c = ParallelMCPClient()
    assert c._endpoint == "https://search.parallel.ai/mcp"
    assert len(c.session_id) == 32
    c2 = ParallelMCPClient(endpoint="https://example.com/mcp/")
    assert c2._endpoint == "https://example.com/mcp"
    c3 = ParallelMCPClient(session_id="custom_session")
    assert c3.session_id == "custom_session"


def test_search_validates_queries() -> None:
    c = ParallelMCPClient()
    try:
        c.search(objective="x", search_queries=[])
    except ValueError as exc:
        assert "search_queries" in str(exc)
    else:
        raise AssertionError("expected ValueError on empty search_queries")


def test_fetch_validates_urls() -> None:
    c = ParallelMCPClient()
    try:
        c.fetch(urls=[])
    except ValueError as exc:
        assert "urls" in str(exc)
    else:
        raise AssertionError("expected ValueError on empty urls")

    try:
        c.fetch(urls=["a"] * 21)
    except ValueError as exc:
        assert "20" in str(exc)
    else:
        raise AssertionError("expected ValueError on >20 urls")


# ---- SDK client + factory (pure unit, uses fake sdk via _sdk_factory) ---


class _FakeSearchResult:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeExtractResult:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeSDK:
    """Stand-in for the parallel-web SDK; records call args."""

    def __init__(self, *, search_result=None, extract_result=None) -> None:
        self._search_result = search_result
        self._extract_result = extract_result
        self.search_calls: list[dict] = []
        self.extract_calls: list[dict] = []

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return self._search_result

    def extract(self, **kwargs):
        self.extract_calls.append(kwargs)
        return self._extract_result


def test_sdk_client_uses_factory_and_parses_search() -> None:
    fake_search = _FakeSearchResult(
        search_id="sdk_search_1",
        session_id="sdk_session_1",
        results=[
            _FakeSearchResult(url="https://example.com/a", title="A",
                              publish_date=None, excerpts=["ex1"]),
            _FakeSearchResult(url="https://example.com/b", title="B",
                              publish_date=None, excerpts=["ex2", "ex3"]),
        ],
    )
    factory = _FakeSDK(search_result=fake_search)
    client = ParallelSDKClient(api_key="test_key", _sdk_factory=lambda key: factory)

    response = client.search(objective="find TCP internals", search_queries=["tcp", "kernel"])

    assert len(factory.search_calls) == 1
    assert factory.search_calls[0]["objective"] == "find TCP internals"
    assert factory.search_calls[0]["search_queries"] == ["tcp", "kernel"]
    assert response.search_id == "sdk_search_1"
    assert response.session_id == "sdk_session_1"
    assert len(response.results) == 2
    assert response.results[0].url == "https://example.com/a"
    assert response.results[1].excerpts == ["ex2", "ex3"]


def test_sdk_client_parses_extract_results_and_errors() -> None:
    fake_extract = _FakeExtractResult(
        extract_id="sdk_extract_1",
        session_id="sdk_session_1",
        results=[
            _FakeExtractResult(url="https://example.com/a", title="A",
                               publish_date=None, excerpts=["ex1"],
                               full_content=None),
        ],
        errors=[
            _FakeExtractResult(url="https://example.com/b",
                               error_type="fetch_failed",
                               http_status_code=404, content="Not Found"),
        ],
    )
    factory = _FakeSDK(extract_result=fake_extract)
    client = ParallelSDKClient(api_key="test_key", _sdk_factory=lambda key: factory)

    response = client.fetch(urls=["https://example.com/a", "https://example.com/b"],
                            objective="extract TCP docs")

    assert len(factory.extract_calls) == 1
    assert factory.extract_calls[0]["urls"] == [
        "https://example.com/a", "https://example.com/b"
    ]
    assert factory.extract_calls[0]["objective"] == "extract TCP docs"
    assert response.extract_id == "sdk_extract_1"
    assert len(response.results) == 1
    assert response.results[0].url == "https://example.com/a"
    assert len(response.errors) == 1
    assert response.errors[0].http_status_code == 404
    assert response.errors[0].error_type == "fetch_failed"


def test_sdk_client_validates_inputs() -> None:
    try:
        ParallelSDKClient(api_key="")
    except ValueError as e:
        assert "api_key" in str(e)
    else:
        raise AssertionError("expected ValueError on empty api_key")

    client = ParallelSDKClient(api_key="key", _sdk_factory=lambda k: _FakeSDK())
    try:
        client.search(objective="x", search_queries=[])
    except ValueError as e:
        assert "search_queries" in str(e)
    else:
        raise AssertionError("expected ValueError on empty search_queries")

    try:
        client.fetch(urls=[])
    except ValueError as e:
        assert "urls" in str(e)
    else:
        raise AssertionError("expected ValueError on empty urls")

    try:
        client.fetch(urls=["a"] * 21)
    except ValueError as e:
        assert "20" in str(e)
    else:
        raise AssertionError("expected ValueError on >20 urls")


def test_create_parallel_client_prefers_sdk_with_key() -> None:
    client = create_parallel_client(parallel_api_key="my_key")
    assert isinstance(client, ParallelSDKClient)


def test_create_parallel_client_falls_back_to_mcp() -> None:
    client = create_parallel_client(parallel_mcp_url="https://example.com/mcp")
    assert isinstance(client, ParallelMCPClient)
    assert client._endpoint == "https://example.com/mcp"


def test_create_parallel_client_default_mcp_endpoint() -> None:
    client = create_parallel_client()
    assert isinstance(client, ParallelMCPClient)
    assert client._endpoint == "https://search.parallel.ai/mcp"


# ---- Live MCP test (opt-in) ----------------------------------------------


def test_live_mcp_returns_results() -> None:
    """Opt-in live test against the public MCP endpoint.

    Skipped unless PARALLEL_LIVE=1 is set in the environment.
    """
    if os.environ.get("PARALLEL_LIVE") != "1":
        return
    c = ParallelMCPClient(timeout_seconds=60.0)
    response = c.search(
        objective=(
            "Find authoritative sources explaining how Kubernetes networking "
            "internals work, including CNI plugins and pod-to-pod packet flow."
        ),
        search_queries=["Kubernetes CNI plugin", "pod networking internals"],
    )
    assert isinstance(response.results, list)
    # Don't assert len > 0 — Parallel.ai free tier may rate-limit us — just
    # make sure the call parses cleanly and returns a well-formed response.
    assert response.session_id


TESTS = [
    test_sse_well_formed_single_event,
    test_sse_compact_single_line,
    test_sse_multiple_events_returns_last,
    test_sse_crlf_normalized,
    test_sse_skips_malformed_data_lines,
    test_sse_bare_json_without_data_prefix,
    test_sse_empty_body_raises,
    test_sse_no_decodable_data_raises,
    test_client_construction,
    test_search_validates_queries,
    test_fetch_validates_urls,
    test_sdk_client_uses_factory_and_parses_search,
    test_sdk_client_parses_extract_results_and_errors,
    test_sdk_client_validates_inputs,
    test_create_parallel_client_prefers_sdk_with_key,
    test_create_parallel_client_falls_back_to_mcp,
    test_create_parallel_client_default_mcp_endpoint,
    test_live_mcp_returns_results,
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
