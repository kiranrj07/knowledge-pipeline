"""Smoke tests for the SearXNG client and the create_web_discovery_client
factory routing.

Run with:
    /home/janak/ai/knowledge-pipeline/.venv/bin/python tests/smoke_searxng.py

All tests are pure-unit (mocked HTTP). No Docker, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge_pipeline.discovery.parallel_search import (  # noqa: E402
    ParallelMCPClient,
    ParallelSDKClient,
    create_parallel_client,
    create_web_discovery_client,
)
from knowledge_pipeline.discovery.searxng import (  # noqa: E402
    SearXNGClient,
    SearXNGError,
)


# ---- Test fixtures -------------------------------------------------------


def _mock_response(payload: dict) -> MagicMock:
    """Build a mock requests.Response with .json() and .raise_for_status()."""
    mock = MagicMock()
    mock.json.return_value = payload
    mock.raise_for_status = MagicMock()
    return mock


def _sample_payload() -> dict:
    return {
        "query": "TCP SYN backlog",
        "number_of_results": 42,
        "results": [
            {
                "url": "https://example.com/a",
                "title": "Stack Overflow: SYN backlog",
                "content": "tcp_max_syn_backlog controls the queue length",
                "publishedDate": None,
                "engine": "duckduckgo",
            },
            {
                "url": "https://example.com/b",
                "title": "Linux Journal: TCP listen() backlog",
                "content": "the listen() backlog is bounded by somaxconn",
                "publishedDate": "2024-06-12",
                "engine": "brave",
            },
            {
                "url": "https://example.com/c",
                "title": "No content here",
                "content": "",
                "publishedDate": None,
                "engine": "startpage",
            },
        ],
        "suggestions": [
            {"suggestion": "tcp backlog"},
            {"suggestion": "syn cookies"},
        ],
    }


# ---- SearXNGClient construction / validation ----------------------------


def test_client_construction() -> None:
    c = SearXNGClient(base_url="http://127.0.0.1:8888")
    assert c.base_url == "http://127.0.0.1:8888"
    c2 = SearXNGClient(base_url="http://example.com:9999/")
    assert c2.base_url == "http://example.com:9999"
    assert c2.supports_fetch is False

    try:
        SearXNGClient(base_url="")
    except ValueError as exc:
        assert "base_url" in str(exc)
    else:
        raise AssertionError("expected ValueError on empty base_url")


def test_search_validates_inputs() -> None:
    c = SearXNGClient(base_url="http://test")
    try:
        c.search(query="")
    except ValueError as exc:
        assert "query" in str(exc)
    else:
        raise AssertionError("expected ValueError on empty query")

    try:
        c.search(query="x", max_results=0)
    except ValueError as exc:
        assert "max_results" in str(exc)
    else:
        raise AssertionError("expected ValueError on max_results=0")

    try:
        c.search(query="x", max_results=101)
    except ValueError as exc:
        assert "max_results" in str(exc)
    else:
        raise AssertionError("expected ValueError on max_results>100")

    try:
        c.search(query="x", pageno=0)
    except ValueError as exc:
        assert "pageno" in str(exc)
    else:
        raise AssertionError("expected ValueError on pageno<1")


def test_fetch_raises() -> None:
    c = SearXNGClient(base_url="http://test")
    try:
        c.fetch(urls=["https://x.com"])
    except SearXNGError as exc:
        assert "does not provide" in str(exc)
    else:
        raise AssertionError("expected SearXNGError on fetch")


# ---- SearXNGClient.search() parsing (mocked HTTP) ----------------------


def test_search_parses_results_and_suggestions() -> None:
    client = SearXNGClient(base_url="http://test")
    mock_resp = _mock_response(_sample_payload())

    with patch("requests.get", return_value=mock_resp) as mock_get:
        response = client.search(query="TCP SYN backlog", max_results=10)

    # Verify the HTTP call was made correctly
    mock_get.assert_called_once()
    call_args, call_kwargs = mock_get.call_args
    assert call_args[0] == "http://test/search"
    assert call_kwargs["params"]["q"] == "TCP SYN backlog"
    assert call_kwargs["params"]["format"] == "json"
    assert call_kwargs["params"]["pageno"] == 1

    # Verify response parsing
    assert response.search_id == "42"  # from number_of_results
    assert len(response.results) == 3
    # Empty-content result is preserved but with empty excerpts (caller filters).
    assert response.results[0].url == "https://example.com/a"
    assert response.results[0].title == "Stack Overflow: SYN backlog"
    assert response.results[0].excerpts == ["tcp_max_syn_backlog controls the queue length"]
    assert response.results[0].publish_date is None
    assert response.results[1].publish_date == "2024-06-12"
    assert response.results[2].excerpts == []  # empty content
    assert response.warnings == ["tcp backlog", "syn cookies"]


def test_search_respects_max_results() -> None:
    client = SearXNGClient(base_url="http://test")
    payload = {
        "number_of_results": 100,
        "results": [
            {"url": f"https://e.com/{i}", "title": f"T{i}", "content": f"c{i}"}
            for i in range(20)
        ],
    }
    mock_resp = _mock_response(payload)

    with patch("requests.get", return_value=mock_resp):
        response = client.search(query="x", max_results=5)

    assert len(response.results) == 5


def test_search_passes_language_and_categories() -> None:
    client = SearXNGClient(base_url="http://test")
    mock_resp = _mock_response({"number_of_results": 0, "results": []})

    with patch("requests.get", return_value=mock_resp) as mock_get:
        client.search(query="kernel debugging", language="en", categories="science,it")

    params = mock_get.call_args.kwargs["params"]
    assert params["q"] == "kernel debugging"
    assert params["language"] == "en"
    assert params["categories"] == "science,it"
    assert params["format"] == "json"


def test_search_omits_language_and_categories_when_none() -> None:
    client = SearXNGClient(base_url="http://test")
    mock_resp = _mock_response({"number_of_results": 0, "results": []})

    with patch("requests.get", return_value=mock_resp) as mock_get:
        client.search(query="x", language=None, categories=None)

    params = mock_get.call_args.kwargs["params"]
    assert "language" not in params
    assert "categories" not in params


def test_search_http_error_wrapped() -> None:
    import requests as req

    client = SearXNGClient(base_url="http://test")
    with patch("requests.get", side_effect=req.ConnectionError("refused")):
        try:
            client.search(query="x")
        except SearXNGError as exc:
            assert "HTTP error" in str(exc)
        else:
            raise AssertionError("expected SearXNGError on connection error")


def test_search_non_json_response_wrapped() -> None:
    client = SearXNGClient(base_url="http://test")
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.side_effect = ValueError("not json")
    with patch("requests.get", return_value=mock_resp):
        try:
            client.search(query="x")
        except SearXNGError as exc:
            assert "non-JSON" in str(exc)
        else:
            raise AssertionError("expected SearXNGError on non-JSON response")


# ---- create_web_discovery_client factory routing -------------------------


def test_factory_picks_searxng_when_url_set() -> None:
    client = create_web_discovery_client(searxng_url="http://my-searxng:8888")
    assert isinstance(client, SearXNGClient)
    assert client.base_url == "http://my-searxng:8888"


def test_factory_falls_back_to_parallel_sdk_when_no_searxng() -> None:
    client = create_web_discovery_client(parallel_api_key="my_key")
    assert isinstance(client, ParallelSDKClient)


def test_factory_falls_back_to_parallel_mcp_when_neither_set() -> None:
    client = create_web_discovery_client()
    assert isinstance(client, ParallelMCPClient)
    assert client._endpoint == "https://search.parallel.ai/mcp"


def test_factory_falls_back_to_mcp_with_custom_url() -> None:
    client = create_web_discovery_client(parallel_mcp_url="https://alt-mcp.example.com/mcp")
    assert isinstance(client, ParallelMCPClient)
    assert client._endpoint == "https://alt-mcp.example.com/mcp"


def test_factory_searxng_takes_priority_over_parallel() -> None:
    """When both SEARXNG_URL and PARALLEL_API_KEY are set, SearXNG wins."""
    client = create_web_discovery_client(
        searxng_url="http://searxng:8888",
        parallel_api_key="key",
    )
    assert isinstance(client, SearXNGClient)


def test_create_parallel_client_alias_still_works() -> None:
    """The back-compat alias must still create clients."""
    c1 = create_parallel_client(parallel_api_key="k")
    assert isinstance(c1, ParallelSDKClient)
    c2 = create_parallel_client(parallel_mcp_url="https://alt.example.com/mcp")
    assert isinstance(c2, ParallelMCPClient)
    assert c2._endpoint == "https://alt.example.com/mcp"


def test_search_accepts_parallel_style_signature() -> None:
    """Orchestrator calls search(objective=..., search_queries=[...]); must work."""
    client = SearXNGClient(base_url="http://test")
    mock_resp = _mock_response({
        "number_of_results": 3,
        "results": [
            {"url": "https://e.com/a", "title": "A", "content": "snippet a"},
        ],
    })

    with patch("requests.get", return_value=mock_resp) as mock_get:
        response = client.search(
            objective="find TCP internals",
            search_queries=["TCP SYN backlog", "TCP three-way handshake"],
            max_results=5,
        )

    params = mock_get.call_args.kwargs["params"]
    # The joined search_queries form the actual SearXNG query string.
    assert "TCP SYN backlog" in params["q"]
    assert "TCP three-way handshake" in params["q"]
    assert response.results[0].url == "https://e.com/a"


def test_search_requires_query_or_search_queries() -> None:
    client = SearXNGClient(base_url="http://test")
    try:
        client.search()  # neither kwarg provided
    except ValueError as exc:
        assert "query" in str(exc) or "search_queries" in str(exc)
    else:
        raise AssertionError("expected ValueError when neither query nor search_queries given")


TESTS = [
    test_client_construction,
    test_search_validates_inputs,
    test_fetch_raises,
    test_search_parses_results_and_suggestions,
    test_search_respects_max_results,
    test_search_passes_language_and_categories,
    test_search_omits_language_and_categories_when_none,
    test_search_http_error_wrapped,
    test_search_non_json_response_wrapped,
    test_search_accepts_parallel_style_signature,
    test_search_requires_query_or_search_queries,
    test_factory_picks_searxng_when_url_set,
    test_factory_falls_back_to_parallel_sdk_when_no_searxng,
    test_factory_falls_back_to_parallel_mcp_when_neither_set,
    test_factory_falls_back_to_mcp_with_custom_url,
    test_factory_searxng_takes_priority_over_parallel,
    test_create_parallel_client_alias_still_works,
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
