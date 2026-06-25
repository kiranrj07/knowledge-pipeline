"""Parallel.ai MCP client for source discovery and content extraction.

Uses the no-auth MCP endpoint at https://search.parallel.ai/mcp to call two tools:
- web_search: natural-language objective + keyword queries -> ranked URLs + excerpts
- web_fetch:  explicit URLs -> clean markdown excerpts (or full content on request)

Note: web_fetch covers the same surface as Crawl4AI for our purposes (handle
JS-rendered pages and PDFs, return LLM-optimized markdown). Using one provider
for both discovery and extraction removes a dependency and keeps the rate limit
accountable to one vendor. Crawl4AI is still useful for pages behind auth or
when Parallel's excerpt mode is too lossy.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

import requests


# ---- Result types ---------------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    url: str
    title: str | None = None
    publish_date: str | None = None
    excerpts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FetchResult:
    url: str
    title: str | None = None
    publish_date: str | None = None
    excerpts: list[str] = field(default_factory=list)
    full_content: str | None = None


@dataclass(frozen=True)
class FetchError:
    url: str
    error_type: str
    http_status_code: int | None = None
    content: str | None = None


@dataclass(frozen=True)
class SearchResponse:
    search_id: str
    session_id: str
    results: list[SearchResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FetchResponse:
    extract_id: str
    session_id: str
    results: list[FetchResult] = field(default_factory=list)
    errors: list[FetchError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---- Errors ---------------------------------------------------------------


class ParallelMCPError(RuntimeError):
    """Raised when the Parallel.ai MCP endpoint returns a transport or JSON-RPC error."""


# ---- Client ---------------------------------------------------------------


class ParallelMCPClient:
    """Thin client for the Parallel.ai MCP HTTP endpoint.

    A stable `session_id` is generated per instance (UUID hex) and reused across
    every tool call so Parallel can rate-limit per session instead of per call.
    No authentication is required for the public endpoint.
    """

    def __init__(
        self,
        endpoint: str = "https://search.parallel.ai/mcp",
        timeout_seconds: float = 60.0,
        session_id: str | None = None,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout_seconds
        self._session_id = session_id or uuid.uuid4().hex
        self._request_id = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    # ---- Transport --------------------------------------------------------

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        return self._parse_response(self._post(payload))

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = requests.post(
                self._endpoint,
                json=payload,
                timeout=self._timeout,
                headers={
                    "Content-Type": "application/json",
                    # MCP servers commonly stream tool-call responses as SSE.
                    # Accepting both lets the client handle plain JSON and SSE
                    # transparently — we parse whichever comes back.
                    "Accept": "application/json, text/event-stream",
                },
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ParallelMCPError(f"HTTP error calling {self._endpoint}: {exc}") from exc
        content_type = (response.headers.get("Content-Type") or "").lower()
        body = response.text
        if "text/event-stream" in content_type or body.lstrip().startswith(("event:", "data:")):
            try:
                return _parse_sse_jsonrpc(body)
            except (ValueError, json.JSONDecodeError) as exc:
                raise ParallelMCPError(
                    f"Could not parse SSE response from {self._endpoint}: {exc}. "
                    f"First 200 chars: {body[:200]!r}"
                ) from exc
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise ParallelMCPError(
                f"Non-JSON response from {self._endpoint}: {exc}. "
                f"First 200 chars: {body[:200]!r}"
            ) from exc

    @staticmethod
    def _parse_response(body: dict[str, Any]) -> dict[str, Any]:
        if "error" in body and body["error"]:
            error = body["error"]
            message = error.get("message", "unknown error") if isinstance(error, dict) else str(error)
            raise ParallelMCPError(f"Parallel.ai MCP error: {message}")
        if "result" not in body:
            raise ParallelMCPError(f"Parallel.ai MCP returned no result field: {body}")
        return body["result"]

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._call("tools/call", {"name": name, "arguments": arguments})

    # ---- Public API -------------------------------------------------------

    def search(
        self,
        *,
        objective: str,
        search_queries: list[str],
        model_name: str | None = None,
    ) -> SearchResponse:
        """Search the live web for authoritative sources on the given objective.

        Args:
            objective: natural-language research goal (full sentences OK).
            search_queries: 2-3 diverse 3-6 word keyword queries.
            model_name: optional LLM slug for Parallel's product analytics;
                does not affect search behavior.

        Returns:
            SearchResponse with ranked results and Parallel's session_id.

        Raises:
            ValueError: if search_queries is empty.
            ParallelMCPError: on transport or JSON-RPC errors.
        """
        if not search_queries:
            raise ValueError("search_queries must contain at least one query")

        arguments: dict[str, Any] = {
            "objective": objective,
            "search_queries": search_queries,
            "session_id": self._session_id,
        }
        if model_name is not None:
            arguments["model_name"] = model_name

        result = self._call_tool("web_search", arguments)
        return _parse_search_response(result)

    def fetch(
        self,
        *,
        urls: list[str],
        objective: str | None = None,
        search_queries: list[str] | None = None,
        full_content: bool = False,
        model_name: str | None = None,
    ) -> FetchResponse:
        """Fetch one or more URLs and return LLM-optimized excerpts (or full content).

        Args:
            urls: 1-20 URLs to extract.
            objective: optional natural-language description of what to focus on
                (max 200 chars per the MCP schema). Strongly recommended when
                fetching many URLs to keep excerpts relevant.
            search_queries: optional queries from the prior web_search call;
                used together with objective to focus excerpts.
            full_content: when True, returns the entire page as markdown. Off
                by default because responses can be tens of thousands of tokens.
            model_name: optional LLM slug for Parallel's product analytics.

        Raises:
            ValueError: if urls is empty or exceeds the 20-URL limit.
            ParallelMCPError: on transport or JSON-RPC errors.
        """
        if not urls:
            raise ValueError("urls must contain at least one URL")
        if len(urls) > 20:
            raise ValueError("urls may not exceed 20 per call (Parallel.ai limit)")

        arguments: dict[str, Any] = {
            "urls": urls,
            "session_id": self._session_id,
            "full_content": full_content,
        }
        if objective is not None:
            arguments["objective"] = objective
        if search_queries is not None:
            arguments["search_queries"] = search_queries
        if model_name is not None:
            arguments["model_name"] = model_name

        result = self._call_tool("web_fetch", arguments)
        return _parse_fetch_response(result)


# ---- Response parsing -----------------------------------------------------


def _parse_sse_jsonrpc(body: str) -> dict[str, Any]:
    """Extract a JSON-RPC body from an MCP Server-Sent Events stream.

    SSE events look like:
        event: message
        data: {"jsonrpc":"2.0","id":1,"result":{...}}

        (blank line separates events)

    Some servers send the response as a single long line with embedded newlines;
    others split into multiple events. We tolerate both: parse every `data:`
    line, then return the last successfully-decoded JSON-RPC object. That
    matches the convention of "the terminal event carries the result".
    """
    if not body:
        raise ValueError("empty SSE body")
    # Normalize CRLF and split on event boundaries (blank line) OR on data: prefixes
    # so we handle both well-formed and compact streams.
    chunks: list[str] = []
    for piece in body.replace("\r\n", "\n").split("\n\n"):
        piece = piece.strip()
        if piece:
            chunks.append(piece)
    if not chunks:
        # Fallback: split every line and treat each `data:` line as its own chunk.
        chunks = [line.strip() for line in body.splitlines() if line.strip()]

    parsed: list[dict[str, Any]] = []
    for chunk in chunks:
        data_line: str | None = None
        for line in chunk.splitlines():
            stripped = line.strip()
            if stripped.startswith("data:"):
                data_line = stripped[len("data:"):].strip()
                break
        if data_line is None:
            # Some servers send the JSON on a bare line with no `data:` prefix.
            if chunk.startswith("{") and chunk.endswith("}"):
                data_line = chunk
        if data_line is None:
            continue
        try:
            parsed.append(json.loads(data_line))
        except json.JSONDecodeError:
            continue
    if not parsed:
        raise ValueError("no `data:` lines in SSE stream could be decoded as JSON")
    return parsed[-1]


def _parse_search_response(result: dict[str, Any]) -> SearchResponse:
    parsed_results: list[SearchResult] = []
    for item in result.get("results") or []:
        if isinstance(item, dict) and "url" in item:
            parsed_results.append(
                SearchResult(
                    url=item["url"],
                    title=item.get("title"),
                    publish_date=item.get("publish_date"),
                    excerpts=list(item.get("excerpts") or []),
                )
            )
    return SearchResponse(
        search_id=str(result.get("search_id", "")),
        session_id=str(result.get("session_id", "")),
        results=parsed_results,
        warnings=_collect_warnings(result.get("warnings")),
    )


def _parse_fetch_response(result: dict[str, Any]) -> FetchResponse:
    parsed_results: list[FetchResult] = []
    for item in result.get("results") or []:
        if isinstance(item, dict) and "url" in item:
            parsed_results.append(
                FetchResult(
                    url=item["url"],
                    title=item.get("title"),
                    publish_date=item.get("publish_date"),
                    excerpts=list(item.get("excerpts") or []),
                    full_content=item.get("full_content"),
                )
            )
    parsed_errors: list[FetchError] = []
    for item in result.get("errors") or []:
        if isinstance(item, dict) and "url" in item:
            parsed_errors.append(
                FetchError(
                    url=item["url"],
                    error_type=str(item.get("error_type", "unknown")),
                    http_status_code=item.get("http_status_code"),
                    content=item.get("content"),
                )
            )
    return FetchResponse(
        extract_id=str(result.get("extract_id", "")),
        session_id=str(result.get("session_id", "")),
        results=parsed_results,
        errors=parsed_errors,
        warnings=_collect_warnings(result.get("warnings")),
    )


def _collect_warnings(raw: Any) -> list[str]:
    collected: list[str] = []
    for warning in raw or []:
        if isinstance(warning, dict):
            message = warning.get("message", "")
            if message:
                collected.append(str(message))
        elif warning:
            collected.append(str(warning))
    return collected

# ---- SDK client (optional, requires PARALLEL_API_KEY) -------------------
#
# The SDK gives a richer free tier than the no-auth MCP endpoint, with
# stronger relevance for technical queries. We import it lazily so the MCP
# path keeps working when the SDK isn't installed.


class ParallelSDKClient:
    """Client backed by the parallel-web Python SDK. Requires an API key.

    Drop-in alternative to ParallelMCPClient for the discovery and extraction
    stages. Same method signatures, same return types. Pick this when
    PARALLEL_API_KEY is set; the orchestrator does that automatically via
    `create_parallel_client`.
    """

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 60.0,
        _sdk_factory: Any = None,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        self._api_key = api_key
        self._timeout = timeout_seconds
        self._session_id = uuid.uuid4().hex
        # _sdk_factory is a test seam: optional callable(api_key) -> sdk-instance.
        # When None, we lazy-import the real `parallel` module.
        self._sdk_factory = _sdk_factory

    @property
    def session_id(self) -> str:
        return self._session_id

    def _get_sdk(self) -> Any:
        if self._sdk_factory is not None:
            return self._sdk_factory(self._api_key)
        try:
            from parallel import Parallel
        except ImportError as exc:
            raise RuntimeError(
                "parallel-web SDK is not installed. Run "
                "`pip install parallel-web` to use ParallelSDKClient."
            ) from exc
        return Parallel(api_key=self._api_key)

    def search(
        self,
        *,
        objective: str,
        search_queries: list[str],
        model_name: str | None = None,
    ) -> SearchResponse:
        if not search_queries:
            raise ValueError("search_queries must contain at least one query")
        sdk = self._get_sdk()
        result = sdk.search(objective=objective, search_queries=search_queries)
        results = [
            SearchResult(
                url=str(getattr(r, "url", "")),
                title=getattr(r, "title", None),
                publish_date=getattr(r, "publish_date", None),
                excerpts=list(getattr(r, "excerpts", []) or []),
            )
            for r in (getattr(result, "results", None) or [])
        ]
        return SearchResponse(
            search_id=str(getattr(result, "search_id", "")),
            session_id=str(getattr(result, "session_id", self._session_id)),
            results=results,
        )

    def fetch(
        self,
        *,
        urls: list[str],
        objective: str | None = None,
        search_queries: list[str] | None = None,
        full_content: bool = False,
        model_name: str | None = None,
    ) -> FetchResponse:
        if not urls:
            raise ValueError("urls must contain at least one URL")
        if len(urls) > 20:
            raise ValueError("urls may not exceed 20 per call (Parallel.ai limit)")
        sdk = self._get_sdk()
        result = sdk.extract(
            urls=urls,
            objective=objective or "extract relevant content",
        )
        results = [
            FetchResult(
                url=str(getattr(r, "url", "")),
                title=getattr(r, "title", None),
                publish_date=getattr(r, "publish_date", None),
                excerpts=list(getattr(r, "excerpts", []) or []),
                full_content=getattr(r, "full_content", None),
            )
            for r in (getattr(result, "results", None) or [])
        ]
        errors = [
            FetchError(
                url=str(getattr(e, "url", "")),
                error_type=str(getattr(e, "error_type", "unknown")),
                http_status_code=getattr(e, "http_status_code", None),
                content=getattr(e, "content", None),
            )
            for e in (getattr(result, "errors", None) or [])
        ]
        return FetchResponse(
            extract_id=str(getattr(result, "extract_id", "")),
            session_id=str(getattr(result, "session_id", self._session_id)),
            results=results,
            errors=errors,
        )


def create_web_discovery_client(
    *,
    searxng_url: str | None = None,
    parallel_api_key: str | None = None,
    parallel_mcp_url: str | None = None,
) -> "SearXNGClient | ParallelMCPClient | ParallelSDKClient":
    """Return the best available web discovery client.

    Priority:
        1. SearXNG (self-hosted, no API key) — preferred when SEARXNG_URL is set.
        2. Parallel.ai SDK (needs PARALLEL_API_KEY) — paid tier with better relevance.
        3. Parallel.ai MCP (no-auth fallback) — last resort, often returns 0 results
           for deep technical queries.
    """
    if searxng_url:
        # Lazy import: SearXNG module is independent of the parallel-web SDK.
        from knowledge_pipeline.discovery.searxng import SearXNGClient
        return SearXNGClient(base_url=searxng_url)
    if parallel_api_key:
        return ParallelSDKClient(api_key=parallel_api_key)
    return ParallelMCPClient(
        endpoint=parallel_mcp_url or "https://search.parallel.ai/mcp"
    )


# Back-compat alias: existing callers using the old name keep working.
create_parallel_client = create_web_discovery_client
