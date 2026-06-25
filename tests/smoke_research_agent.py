"""Smoke tests for the research agent (run directly; no pytest).

Run with:
    /home/janak/ai/knowledge-pipeline/.venv/bin/python tests/smoke_research_agent.py

Uses FakeEmbeddingClient and FakeOllamaGenerator to avoid hitting external
services. Qdrant runs in local mode against a fresh tmpdir per test.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge_pipeline.research.agent import (  # noqa: E402
    OllamaGenerator,
    ResearchAgent,
    SourceDocument,
    _format_sources,
)
from knowledge_pipeline.storage.qdrant_store import QdrantStore  # noqa: E402


# ---- Test doubles ---------------------------------------------------------


class FakeEmbeddingClient:
    """Deterministic embedding double: maps text to a stable vector based on length."""

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim
        self.embed_calls: list[str] = []
        self.batch_calls: list[list[str]] = []

    @property
    def model(self) -> str:
        return "fake-embed"

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        if not text:
            raise ValueError("text must not be empty")
        self.embed_calls.append(text)
        # Deterministic one-hot keyed off text length.
        pos = len(text) % self._dim
        return [1.0 if i == pos else 0.0 for i in range(self._dim)]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if any(not t for t in texts):
            raise ValueError("texts must all be non-empty")
        self.batch_calls.append(list(texts))
        return [self.embed(t) for t in texts]


class FakeOllamaGenerator(OllamaGenerator):
    """OllamaGenerator that returns a canned response instead of calling Ollama."""

    def __init__(self, canned: str) -> None:
        super().__init__(model="fake-llm")
        self._canned = canned
        self.generate_calls: list[dict] = []

    def generate(self, *, prompt: str, system: str | None = None, temperature: float = 0.2, think: bool = False) -> str:
        self.generate_calls.append({"prompt": prompt, "system": system, "temperature": temperature})
        return self._canned


# ---- Fixtures -------------------------------------------------------------


def _make_store(name: str = "test_research") -> tuple[QdrantStore, str]:
    tmpdir = tempfile.mkdtemp(prefix="kp_research_")
    return QdrantStore(storage_path=tmpdir, collection_name=name, vector_dim=4), tmpdir


def _make_agent(llm_response: str = "## fake brief") -> tuple[ResearchAgent, QdrantStore, FakeOllamaGenerator, FakeEmbeddingClient, str]:
    store, tmpdir = _make_store()
    embeddings = FakeEmbeddingClient(dim=4)
    llm = FakeOllamaGenerator(canned=llm_response)
    agent = ResearchAgent(
        llm=llm,
        embedding_client=embeddings,
        qdrant_store=store,
        chunk_size=1500,
        chunk_overlap=200,
        max_context_chunks=5,
    )
    return agent, store, llm, embeddings, tmpdir


# ---- OllamaGenerator ------------------------------------------------------


def test_ollama_generator_construction() -> None:
    g = OllamaGenerator(model="qwen3:14b")
    assert g.model == "qwen3:14b"
    assert g._base_url == "http://127.0.0.1:11434"
    g2 = OllamaGenerator(model="x", base_url="http://example.com/")
    assert g2._base_url == "http://example.com"
    try:
        OllamaGenerator(model="")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on empty model")


def test_ollama_generator_input_validation() -> None:
    g = OllamaGenerator(model="x")
    try:
        g.generate(prompt="")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on empty prompt")


# ---- ResearchAgent construction ------------------------------------------


def test_research_agent_construction_validation() -> None:
    store, tmpdir = _make_store()
    embeddings = FakeEmbeddingClient()
    llm = FakeOllamaGenerator(canned="x")
    import shutil
    try:
        try:
            ResearchAgent(llm=llm, embedding_client=embeddings, qdrant_store=store, chunk_size=0, chunk_overlap=0, max_context_chunks=1)
        except ValueError as e:
            assert "chunk_size" in str(e)
        else:
            raise AssertionError("expected ValueError on chunk_size=0")

        try:
            ResearchAgent(llm=llm, embedding_client=embeddings, qdrant_store=store, chunk_size=100, chunk_overlap=100, max_context_chunks=1)
        except ValueError as e:
            assert "chunk_overlap" in str(e)
        else:
            raise AssertionError("expected ValueError on overlap >= chunk_size")

        try:
            ResearchAgent(llm=llm, embedding_client=embeddings, qdrant_store=store, chunk_size=100, chunk_overlap=10, max_context_chunks=0)
        except ValueError as e:
            assert "max_context_chunks" in str(e)
        else:
            raise AssertionError("expected ValueError on max_context_chunks=0")
    finally:
        store.delete_collection()
        store.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---- index_sources --------------------------------------------------------


def test_index_sources_empty_input() -> None:
    agent, store, llm, embeddings, tmpdir = _make_agent()
    import shutil
    try:
        summary = agent.index_sources("topic", [])
        assert summary.documents == 0
        assert summary.chunks == 0
        assert summary.chunk_ids == []
    finally:
        store.delete_collection()
        store.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_index_sources_chunks_and_upserts() -> None:
    agent, store, llm, embeddings, tmpdir = _make_agent()
    import shutil
    try:
        docs = [
            SourceDocument(
                text="TCP SYN backlog fills when the accept queue is exceeded.",
                source_url="https://example.com/a", source_type="rfc",
                title="RFC 793",
            ),
            SourceDocument(
                text="BGP route reflectors reduce the iBGP full-mesh requirement.",
                source_url="https://example.com/b", source_type="official_doc",
                title="BGP Guide",
            ),
        ]
        summary = agent.index_sources("TCP SYN backlog", docs)
        assert summary.topic == "TCP SYN backlog"
        assert summary.documents == 2
        assert summary.chunks == 2
        assert len(summary.chunk_ids) == 2
        assert store.count() == 2
        # The agent made one batch embed call.
        assert len(embeddings.batch_calls) == 1
        assert len(embeddings.batch_calls[0]) == 2
    finally:
        store.delete_collection()
        store.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_index_sources_validates_topic() -> None:
    agent, store, llm, embeddings, tmpdir = _make_agent()
    import shutil
    try:
        try:
            agent.index_sources("", [SourceDocument(text="x")])
        except ValueError as e:
            assert "topic" in str(e)
        else:
            raise AssertionError("expected ValueError on empty topic")
    finally:
        store.delete_collection()
        store.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---- build_synthesis_prompt ----------------------------------------------


def test_build_synthesis_prompt_structure() -> None:
    agent, store, llm, embeddings, tmpdir = _make_agent()
    import shutil
    try:
        agent.index_sources(
            "TCP SYN backlog",
            [
                SourceDocument(
                    text="The SYN backlog tracks half-open connections waiting to complete the three-way handshake.",
                    source_url="https://example.com/tcp", source_type="rfc",
                    title="RFC 793",
                ),
                SourceDocument(
                    text="net/ipv4/tcp_input.c: tcp_rcv_state_process handles incoming SYN segments.",
                    source_url="https://github.com/torvalds/linux/blob/master/net/ipv4/tcp_input.c",
                    source_type="kernel_source", title="tcp_input.c",
                ),
            ],
        )
        prompt = agent.build_synthesis_prompt("TCP SYN backlog")
        # Topic appears in the prompt.
        assert "TCP SYN backlog" in prompt
        # All 12 section headers from the template appear.
        for header in (
            "## 1. Executive Summary",
            "## 2. Core Concept",
            "## 3. Architecture",
            "## 4. Internal Workflow",
            "## 5. Packet / Data Flow",
            "## 6. Engineering Decisions",
            "## 7. Source-Code and Documentation References",
            "## 8. Troubleshooting Methodology",
            "## 9. Performance / Security Considerations",
            "## 10. Common Misconceptions",
            "## 11. YouTube Content Gap",
            "## 12. Recommended Video Angle",
            "## Retrieved Sources",
        ):
            assert header in prompt, f"missing section header: {header}"
        # At least one retrieved source is embedded.
        assert "RFC 793" in prompt or "tcp_input.c" in prompt
        # The query was embedded exactly once.
        assert embeddings.embed_calls.count("TCP SYN backlog") == 1
    finally:
        store.delete_collection()
        store.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_build_synthesis_prompt_empty_corpus() -> None:
    agent, store, llm, embeddings, tmpdir = _make_agent()
    import shutil
    try:
        prompt = agent.build_synthesis_prompt("some topic")
        assert "(no sources retrieved)" in prompt
    finally:
        store.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---- synthesize_brief ----------------------------------------------------


def test_synthesize_brief_returns_llm_output() -> None:
    canned = "## 1. Executive Summary\n\nA brief.\n\n## 12. Recommended Video Angle\n\nDo the packet walk."
    agent, store, llm, embeddings, tmpdir = _make_agent(llm_response=canned)
    import shutil
    try:
        agent.index_sources(
            "TCP SYN backlog",
            [SourceDocument(text="SYN backlog holds half-open connections.", source_type="rfc")],
        )
        brief = agent.synthesize_brief("TCP SYN backlog")
        assert brief == canned
        # The agent invoked the LLM exactly once with a non-empty prompt and system.
        assert len(llm.generate_calls) == 1
        call = llm.generate_calls[0]
        assert call["prompt"]
        assert call["system"]
        assert call["temperature"] == 0.2
    finally:
        store.delete_collection()
        store.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---- _format_sources ------------------------------------------------------


def test_format_sources_empty() -> None:
    assert _format_sources([]) == "(no sources retrieved)"


def test_format_sources_includes_metadata() -> None:
    from knowledge_pipeline.storage.qdrant_store import RetrievedChunk

    chunks = [
        RetrievedChunk(
            chunk_id="c1", text="TCP SYN backlog explanation.", score=0.9,
            source_url="https://example.com/a", source_path=None,
            source_type="rfc", title="RFC 793", topic="tcp", extra={},
        )
    ]
    text = _format_sources(chunks)
    assert "[1]" in text
    assert "TCP SYN backlog explanation." in text
    assert "RFC 793" in text
    assert "https://example.com/a" in text
    assert "type=rfc" in text


TESTS = [
    test_ollama_generator_construction,
    test_ollama_generator_input_validation,
    test_research_agent_construction_validation,
    test_index_sources_empty_input,
    test_index_sources_chunks_and_upserts,
    test_index_sources_validates_topic,
    test_build_synthesis_prompt_structure,
    test_build_synthesis_prompt_empty_corpus,
    test_synthesize_brief_returns_llm_output,
    test_format_sources_empty,
    test_format_sources_includes_metadata,
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
