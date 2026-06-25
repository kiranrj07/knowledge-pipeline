"""Embedding client backed by Ollama's /api/embeddings endpoint.

Default model is `nomic-embed-text` (768-dim, ~274 MB, fast on CPU). Override
via `EMBEDDING_MODEL` in .env or by passing a different model to the client.

Why Ollama for embeddings instead of a hosted API:
- Already running locally (we use it for the LLMs).
- No per-token cost; safe to embed the whole corpus on every research run.
- Same JSON-over-HTTP interface, no extra daemon.

Note: Ollama's /api/embeddings takes a single `prompt` per call. For batches
we issue sequential calls; Qdrant can ingest thousands of points/sec so the
embedding step is the bottleneck, not the upsert.
"""
from __future__ import annotations

from typing import Any

import requests


# Known embedding dimensions for common Ollama embedding models. Used by the
# Qdrant store to size the collection correctly without an extra config knob.
KNOWN_EMBEDDING_DIMS: dict[str, int] = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
}


class EmbeddingClient:
    """Thin client for Ollama's /api/embeddings endpoint."""

    def __init__(
        self,
        *,
        model: str = "nomic-embed-text",
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 60.0,
    ) -> None:
        if not model:
            raise ValueError("model must not be empty")
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int | None:
        """Embedding dimension for the configured model, if known.

        Returns None for unknown models so callers can either fail loudly or
        discover the dim by embedding a probe and inspecting the response.
        """
        return KNOWN_EMBEDDING_DIMS.get(self._model)

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            response = requests.post(
                url,
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=self._timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama embedding HTTP error on {url}: {exc}") from exc
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError(f"Ollama embedding returned non-JSON: {exc}") from exc

    def embed(self, text: str) -> list[float]:
        """Embed a single string. Returns the embedding vector."""
        if not text:
            raise ValueError("text must not be empty")
        body = self._post("/api/embeddings", {"model": self._model, "prompt": text})
        embedding = body.get("embedding")
        if not isinstance(embedding, list):
            raise RuntimeError(f"Ollama embedding response missing 'embedding' list: {body}")
        return [float(x) for x in embedding]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of strings sequentially. Order matches input.

        Empty input returns an empty list. Each input must be non-empty.
        """
        if any(not t for t in texts):
            raise ValueError("texts must all be non-empty")
        return [self.embed(t) for t in texts]

    def probe_dimension(self) -> int:
        """Embed a one-token probe and return the dimension. Useful when the
        configured model isn't in KNOWN_EMBEDDING_DIMS.
        """
        return len(self.embed("dimension probe"))
