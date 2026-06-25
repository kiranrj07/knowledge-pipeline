"""SearXNG client for self-hosted, key-free web discovery.

SearXNG aggregates results from many engines (DuckDuckGo, Bing, Brave,
Startpage, ...) and returns them via JSON. We use it as the default
discovery backend because it requires no API key and runs entirely on the
user's own Docker host.

Compared to Parallel.ai:
- Pro: no API key, no rate limits, no vendor lock-in.
- Con: SearXNG returns snippets in the `content` field of search results;
  there is no separate "fetch full page" endpoint. So our orchestrator
  treats the search-response `content` as the excerpt and skips the
  fetch step (which raises NotImplementedError below).

The SearXNG image we use (searxng/searxng:latest, 2026.6.x+) ships with
granian and binds to container port 8080. Map host:8888 -> container:8080
when running the container.
"""
from __future__ import annotations

from typing import Any

import requests

# Reuse the shared discovery result types. They live in parallel_search.py for
# historical reasons; renaming to a shared module is a future cleanup.
from knowledge_pipeline.discovery.parallel_search import (
    FetchResponse,
    SearchResponse,
    SearchResult,
)


class SearXNGError(RuntimeError):
    """Raised on any SearXNG failure (HTTP, parse, or unsupported operation)."""


class SearXNGClient:
    """Client for a self-hosted SearXNG instance with JSON output enabled.

    The container must be started with the SearXNG settings.yml that has
    `search.formats: [html, json]` (the default config has JSON disabled).
    """

    #: SearXNG does not provide a separate full-content fetch endpoint;
    #: the search response already carries per-result snippets.
    supports_fetch: bool = False

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8888",
        timeout_seconds: float = 30.0,
    ) -> None:
        if not base_url:
            raise ValueError("base_url must not be empty")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    @property
    def base_url(self) -> str:
        return self._base_url

    # ---- Public API -------------------------------------------------------

    def search(
        self,
        *,
        query: str | None = None,
        objective: str | None = None,
        search_queries: list[str] | None = None,
        max_results: int = 10,
        language: str | None = "en",
        categories: str | None = None,
        pageno: int = 1,
    ) -> SearchResponse:
        """Search SearXNG and return ranked results with snippets.

        Accepts either of two call signatures:
        - `query="..."` — direct SearXNG-style single string.
        - `objective="...", search_queries=[...]` — Parallel.ai-compatible
          shape used by the orchestrator. We join the search_queries with a
          space to form the actual SearXNG query string (SearXNG does
          keyword search, so the richer "objective" context has no equivalent).

        Args:
            query: search query string (preferred direct form).
            objective: ignored at query level; carried for Parallel-API parity.
            search_queries: list of keyword queries (Parallel-API parity).
            max_results: cap on results returned (1..100; SearXNG's hard limit).
            language: optional ISO-639-1 code (e.g. "en", "de"). None = no filter.
            categories: optional comma-separated category list (e.g. "general",
                "news", "science"). None = all categories.
            pageno: 1-based page number.

        Returns:
            SearchResponse with up to max_results SearchResult entries. Each
            result's `excerpts` contains one entry: the SearXNG `content`
            snippet for that hit.

        Raises:
            ValueError: on empty query or invalid max_results.
            SearXNGError: on HTTP or parse failures.
        """
        # Accept either signature. When called via the Parallel-compatible
        # shape, derive the actual SearXNG query from search_queries.
        if query is None:
            if not search_queries:
                raise ValueError("either query or search_queries is required")
            # Join multiple queries so SearXNG hits any of them. SearXNG does
            # AND/OR matching on the full query string; joining with a space
            # keeps it simple and works well in practice.
            query = " ".join(q for q in search_queries if q)
        if not query or not query.strip():
            raise ValueError("query must not be empty")
        if max_results < 1 or max_results > 100:
            raise ValueError("max_results must be between 1 and 100")
        if pageno < 1:
            raise ValueError("pageno must be >= 1")

        params: dict[str, Any] = {
            "q": query,
            "format": "json",
            "pageno": pageno,
        }
        if language:
            params["language"] = language
        if categories:
            params["categories"] = categories

        url = f"{self._base_url}/search"
        try:
            response = requests.get(url, params=params, timeout=self._timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise SearXNGError(f"SearXNG HTTP error on {url}: {exc}") from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise SearXNGError(f"SearXNG returned non-JSON: {exc}") from exc

        return _parse_search_response(payload, max_results=max_results)

    def fetch(
        self,
        *,
        urls: list[str],
        objective: str | None = None,
        **_: Any,
    ) -> FetchResponse:
        """SearXNG has no separate fetch endpoint; raise clearly.

        The orchestrator should detect `client.supports_fetch is False` and
        use the search-response snippets directly instead of calling this.
        """
        raise SearXNGError(
            "SearXNG does not provide a fetch endpoint. Use search() "
            "results (which already include per-result snippets) directly, "
            "or set supports_fetch=True on a custom subclass."
        )


# ---- Response parsing -----------------------------------------------------


def _parse_search_response(payload: dict[str, Any], *, max_results: int) -> SearchResponse:
    raw_results = payload.get("results") or []
    results: list[SearchResult] = []
    for item in raw_results[:max_results]:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not url:
            continue
        title = item.get("title")
        publish_date = item.get("publishedDate") or None
        content = item.get("content") or ""
        excerpts = [content.strip()] if content.strip() else []
        results.append(
            SearchResult(
                url=str(url),
                title=str(title) if title else None,
                publish_date=str(publish_date) if publish_date else None,
                excerpts=excerpts,
            )
        )
    suggestions_raw = payload.get("suggestions") or []
    suggestions = [
        str(s.get("suggestion", ""))
        for s in suggestions_raw
        if isinstance(s, dict) and s.get("suggestion")
    ]
    return SearchResponse(
        search_id=str(payload.get("number_of_results", "")),
        session_id="",
        results=results,
        warnings=suggestions,
    )
