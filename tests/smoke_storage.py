"""Smoke tests for the storage module (chunker + embeddings + qdrant_store).

Run directly with the venv python (no pytest needed):

    /home/janak/ai/knowledge-pipeline/.venv/bin/python tests/smoke_storage.py

The qdrant_store tests use a tempdir and do NOT touch the configured
QDRANT_PATH, so it's safe to run alongside a real collection.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge_pipeline.storage.chunker import split_by_chars, split_paragraphs  # noqa: E402
from knowledge_pipeline.storage.embeddings import (  # noqa: E402
    KNOWN_EMBEDDING_DIMS,
    EmbeddingClient,
)
from knowledge_pipeline.storage.qdrant_store import (  # noqa: E402
    DocumentChunk,
    QdrantStore,
    QdrantStoreError,
)


# ---- chunker --------------------------------------------------------------


def test_chunker_empty() -> None:
    assert split_paragraphs("", 100, 10) == []
    assert split_by_chars("", 100, 10) == []
    assert split_paragraphs("   \n\n  ", 100, 10) == []


def test_chunker_single_short_paragraph() -> None:
    text = "Hello world."
    chunks = split_paragraphs(text, 100, 10)
    assert chunks == ["Hello world."]


def test_chunker_packs_small_paragraphs() -> None:
    text = "Para one.\n\nPara two.\n\nPara three."
    chunks = split_paragraphs(text, 1000, 10)
    # All three paragraphs should fit in one chunk since they're tiny.
    assert len(chunks) == 1
    assert "Para one" in chunks[0] and "Para three" in chunks[0]


def test_chunker_splits_long_paragraphs() -> None:
    para = "x" * 200
    chunks = split_paragraphs(para, 50, 5)
    assert len(chunks) > 1
    assert all(len(c) <= 50 for c in chunks)


def test_chunker_splits_oversized_paragraph_by_chars() -> None:
    para = "abcdefghij" * 50  # 500 chars
    chunks = split_by_chars(para, 100, 10)
    assert len(chunks) == 6  # 100 + 90 + 90 + 90 + 90 + 40 (last partial)
    assert all(len(c) <= 100 for c in chunks)
    # Overlap: chunk[1] should start with the last 10 chars of chunk[0].
    assert chunks[1].startswith(chunks[0][-10:])


def test_chunker_input_validation() -> None:
    for bad_chunk_size in (0, -1):
        try:
            split_paragraphs("x", bad_chunk_size, 0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError on chunk_size={bad_chunk_size}")

    for bad_overlap in (-1, 100, 200):
        try:
            split_paragraphs("x", 100, bad_overlap)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError on overlap={bad_overlap}")


# ---- embeddings -----------------------------------------------------------


def test_embeddings_known_dims() -> None:
    assert KNOWN_EMBEDDING_DIMS["nomic-embed-text"] == 768
    assert KNOWN_EMBEDDING_DIMS["mxbai-embed-large"] == 1024
    assert KNOWN_EMBEDDING_DIMS["all-minilm"] == 384


def test_embeddings_dimension_property() -> None:
    c = EmbeddingClient(model="nomic-embed-text")
    assert c.dimension == 768
    assert c.model == "nomic-embed-text"
    c2 = EmbeddingClient(model="unknown-model-xyz")
    assert c2.dimension is None


def test_embeddings_construction() -> None:
    c = EmbeddingClient(model="x", base_url="http://example.com/")
    assert c._base_url == "http://example.com"
    try:
        EmbeddingClient(model="")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on empty model")


def test_embeddings_input_validation() -> None:
    c = EmbeddingClient(model="nomic-embed-text")
    try:
        c.embed("")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on empty text")

    try:
        c.embed_batch(["x", ""])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on empty text in batch")


# ---- qdrant_store ---------------------------------------------------------


def _make_tmp_store(name: str = "test_collection") -> tuple[QdrantStore, str]:
    """Create a QdrantStore in a fresh tempdir. Returns (store, tmpdir)."""
    tmpdir = tempfile.mkdtemp(prefix="kp_qdrant_test_")
    store = QdrantStore(storage_path=tmpdir, collection_name=name, vector_dim=4)
    return store, tmpdir


def _vec(seed: int, dim: int = 4) -> list[float]:
    """Deterministic fake vector: one-hot at position (seed % dim), scaled."""
    pos = seed % dim
    return [1.0 if i == pos else 0.0 for i in range(dim)]


def test_qdrant_construction_validation() -> None:
    try:
        QdrantStore(storage_path="", collection_name="c", vector_dim=4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on empty storage_path")

    try:
        QdrantStore(storage_path="/tmp", collection_name="", vector_dim=4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on empty collection_name")

    try:
        QdrantStore(storage_path="/tmp", collection_name="c", vector_dim=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on non-positive vector_dim")


def test_qdrant_ensure_collection_idempotent() -> None:
    store, tmpdir = _make_tmp_store()
    try:
        store.ensure_collection()
        # Second call must not raise.
        store.ensure_collection()
        assert store.count() == 0
    finally:
        store.delete_collection()
        store.close()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_qdrant_upsert_length_mismatch() -> None:
    store, tmpdir = _make_tmp_store()
    try:
        store.ensure_collection()
        try:
            store.upsert(vectors=[_vec(0)], chunks=[])
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError on length mismatch")
    finally:
        store.delete_collection()
        store.close()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_qdrant_upsert_and_search_roundtrip() -> None:
    store, tmpdir = _make_tmp_store()
    try:
        store.ensure_collection()
        chunks = [
            DocumentChunk(text="TCP SYN backlog fills when accept queue overflows.",
                          source_url="https://example.com/a", source_type="rfc",
                          title="RFC 793", topic="tcp"),
            DocumentChunk(text="BGP route reflectors reduce full-mesh requirements.",
                          source_url="https://example.com/b", source_type="official_doc",
                          title="BGP Guide", topic="bgp"),
            DocumentChunk(text="Cilium uses eBPF for high-performance networking.",
                          source_url="https://example.com/c", source_type="engineering_blog",
                          title="Cilium Internals", topic="ebpf"),
        ]
        ids = store.upsert(vectors=[_vec(i) for i in range(3)], chunks=chunks)
        assert len(ids) == 3
        assert store.count() == 3

        # Query vector identical to chunk[0]'s vector -> top hit should be chunk[0].
        results = store.search(query_vector=_vec(0), limit=2)
        assert len(results) >= 1
        assert results[0].text.startswith("TCP SYN backlog")
        assert results[0].source_type == "rfc"
        assert results[0].source_url == "https://example.com/a"
        assert results[0].topic == "tcp"

        # Filter by source_type.
        results = store.search(query_vector=_vec(2), limit=5, source_type="engineering_blog")
        assert all(r.source_type == "engineering_blog" for r in results)

        # Empty upsert is a no-op (returns []).
        empty_ids = store.upsert(vectors=[], chunks=[])
        assert empty_ids == []
        assert store.count() == 3
    finally:
        store.delete_collection()
        store.close()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_qdrant_search_input_validation() -> None:
    store, tmpdir = _make_tmp_store()
    try:
        store.ensure_collection()
        try:
            store.search(query_vector=[], limit=1)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError on empty query_vector")

        try:
            store.search(query_vector=[1.0, 0.0, 0.0, 0.0], limit=0)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError on non-positive limit")
    finally:
        store.delete_collection()
        store.close()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


TESTS = [
    test_chunker_empty,
    test_chunker_single_short_paragraph,
    test_chunker_packs_small_paragraphs,
    test_chunker_splits_long_paragraphs,
    test_chunker_splits_oversized_paragraph_by_chars,
    test_chunker_input_validation,
    test_embeddings_known_dims,
    test_embeddings_dimension_property,
    test_embeddings_construction,
    test_embeddings_input_validation,
    test_qdrant_construction_validation,
    test_qdrant_ensure_collection_idempotent,
    test_qdrant_upsert_length_mismatch,
    test_qdrant_upsert_and_search_roundtrip,
    test_qdrant_search_input_validation,
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
