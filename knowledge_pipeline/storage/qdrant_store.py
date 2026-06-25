"""Local-mode Qdrant wrapper for the research pipeline.

Uses `QdrantClient(path=...)` so Qdrant runs in-process from a local directory.
No Docker daemon, no separate service to start. Storage lives under
`QDRANT_PATH` (default `./qdrant_storage`).

Trade-offs vs server-mode:
- Pro: zero ops; safe for solo development and CI.
- Con: single-process; can't be shared across multiple workers.
- Con: not all Qdrant features (e.g. distributed, sharding) are available.

For MVP-1 this is exactly right. Move to server-mode only when you have
concurrent writers or want to share the store with the video-production side.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


# ---- Result types ---------------------------------------------------------


@dataclass(frozen=True)
class DocumentChunk:
    """A single chunk to be stored in Qdrant."""

    text: str
    source_url: str | None = None
    source_path: str | None = None
    source_type: str = "unknown"  # e.g. "rfc", "official_doc", "kernel_source", "engineering_blog"
    title: str | None = None
    topic: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by similarity search, with its score."""

    chunk_id: str
    text: str
    score: float
    source_url: str | None
    source_path: str | None
    source_type: str
    title: str | None
    topic: str | None
    extra: dict[str, Any]


# ---- Errors ---------------------------------------------------------------


class QdrantStoreError(RuntimeError):
    """Raised on any Qdrant store failure."""


# ---- Store ----------------------------------------------------------------


class QdrantStore:
    """Local-mode Qdrant wrapper with a single named collection per topic.

    The store owns one collection per topic. Each chunk is a point with a
    payload carrying source metadata so downstream filters work without a
    second index.
    """

    def __init__(
        self,
        *,
        storage_path: str,
        collection_name: str,
        vector_dim: int,
    ) -> None:
        if not storage_path:
            raise ValueError("storage_path must not be empty")
        if not collection_name:
            raise ValueError("collection_name must not be empty")
        if vector_dim <= 0:
            raise ValueError("vector_dim must be > 0")
        self._storage_path = storage_path
        self._collection_name = collection_name
        self._vector_dim = vector_dim
        self._client = QdrantClient(path=storage_path)

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def close(self) -> None:
        """Release the underlying Qdrant client."""
        self._client.close()

    # ---- Collection management --------------------------------------------

    def ensure_collection(self) -> None:
        """Create the collection if it doesn't exist. Idempotent."""
        try:
            collections = self._client.get_collections().collections
        except Exception as exc:  # noqa: BLE001
            raise QdrantStoreError(f"Failed to list Qdrant collections: {exc}") from exc
        if any(c.name == self._collection_name for c in collections):
            return
        try:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self._vector_dim,
                    distance=qmodels.Distance.COSINE,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            raise QdrantStoreError(
                f"Failed to create collection {self._collection_name!r}: {exc}"
            ) from exc

    def delete_collection(self) -> None:
        """Drop the collection. Used by tests and rebuild-from-scratch flows."""
        try:
            self._client.delete_collection(collection_name=self._collection_name)
        except Exception as exc:  # noqa: BLE001
            raise QdrantStoreError(
                f"Failed to delete collection {self._collection_name!r}: {exc}"
            ) from exc

    def count(self) -> int:
        """Return the number of points currently in the collection."""
        try:
            info = self._client.get_collection(collection_name=self._collection_name)
        except Exception as exc:  # noqa: BLE001
            raise QdrantStoreError(
                f"Failed to read collection {self._collection_name!r}: {exc}"
            ) from exc
        return int(getattr(info, "points_count", 0) or 0)

    # ---- Ingest -----------------------------------------------------------

    def upsert(
        self,
        vectors: list[list[float]],
        chunks: list[DocumentChunk],
    ) -> list[str]:
        """Upsert vectors + payloads into the collection. Returns the assigned IDs.

        Args:
            vectors: one embedding per chunk, same length as chunks.
            chunks: the corresponding DocumentChunk payloads.

        Raises:
            ValueError: if vectors and chunks have different lengths.
            QdrantStoreError: on Qdrant transport failure.
        """
        if len(vectors) != len(chunks):
            raise ValueError(
                f"vectors and chunks length mismatch: {len(vectors)} vs {len(chunks)}"
            )
        if not vectors:
            return []

        ids = [str(uuid.uuid4()) for _ in chunks]
        points = [
            qmodels.PointStruct(
                id=point_id,
                vector=vector,
                payload=_chunk_payload(chunk),
            )
            for point_id, vector, chunk in zip(ids, vectors, chunks)
        ]
        try:
            self._client.upsert(
                collection_name=self._collection_name,
                points=points,
                wait=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise QdrantStoreError(
                f"Failed to upsert {len(points)} points into {self._collection_name!r}: {exc}"
            ) from exc
        return ids

    # ---- Retrieval --------------------------------------------------------

    def search(
        self,
        *,
        query_vector: list[float],
        limit: int = 5,
        source_type: str | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """Similarity search with optional source-type filter.

        Args:
            query_vector: embedding of the query.
            limit: max results to return.
            source_type: optional filter, e.g. "kernel_source" or "rfc".
            score_threshold: optional minimum similarity score (cosine).

        Returns:
            Ordered list of RetrievedChunk (highest score first).
        """
        if not query_vector:
            raise ValueError("query_vector must not be empty")
        if limit <= 0:
            raise ValueError("limit must be > 0")

        query_filter = None
        if source_type is not None:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="source_type",
                        match=qmodels.MatchValue(value=source_type),
                    )
                ]
            )

        try:
            response = self._client.query_points(
                collection_name=self._collection_name,
                query=query_vector,
                limit=limit,
                query_filter=query_filter,
                score_threshold=score_threshold,
                with_payload=True,
            )
            hits = response.points
        except Exception as exc:  # noqa: BLE001
            raise QdrantStoreError(
                f"Qdrant search failed on {self._collection_name!r}: {exc}"
            ) from exc

        return [_hit_to_chunk(hit) for hit in hits]


# ---- Helpers --------------------------------------------------------------


def _chunk_payload(chunk: DocumentChunk) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": chunk.text,
        "source_type": chunk.source_type,
    }
    if chunk.source_url is not None:
        payload["source_url"] = chunk.source_url
    if chunk.source_path is not None:
        payload["source_path"] = chunk.source_path
    if chunk.title is not None:
        payload["title"] = chunk.title
    if chunk.topic is not None:
        payload["topic"] = chunk.topic
    for key, value in chunk.extra.items():
        payload.setdefault(key, value)
    return payload


def _hit_to_chunk(hit: Any) -> RetrievedChunk:
    payload = getattr(hit, "payload", None) or {}
    if not isinstance(payload, dict):
        payload = {}
    return RetrievedChunk(
        chunk_id=str(getattr(hit, "id", "")),
        text=str(payload.get("text", "")),
        score=float(getattr(hit, "score", 0.0) or 0.0),
        source_url=payload.get("source_url"),
        source_path=payload.get("source_path"),
        source_type=str(payload.get("source_type", "unknown")),
        title=payload.get("title"),
        topic=payload.get("topic"),
        extra={k: v for k, v in payload.items()
               if k not in {"text", "source_url", "source_path", "source_type", "title", "topic"}},
    )
