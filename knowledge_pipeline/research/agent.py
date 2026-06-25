"""Research agent: turns source documents into a structured research_brief.md.

Flow:
1. `index_sources(topic, documents)` -> chunk + embed + upsert into Qdrant.
2. `synthesize_brief(topic)` -> retrieve top-k chunks relevant to the topic,
   build a structured prompt, call the local LLM (Qwen3:14b via Ollama),
   return the markdown brief.

The brief follows the master_knowledge_document.md structure from the channel
plan: executive summary, core concept, architecture, internal workflow,
packet/data flow, engineering decisions, source refs, troubleshooting,
perf/security, misconceptions, YouTube gap, recommended video angle. This is
the load-bearing artifact the video-production pipeline consumes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import requests

from knowledge_pipeline.storage.chunker import split_paragraphs
from knowledge_pipeline.storage.embeddings import EmbeddingClient
from knowledge_pipeline.storage.qdrant_store import DocumentChunk, QdrantStore


# ---- Source + result types ------------------------------------------------


@dataclass(frozen=True)
class SourceDocument:
    """User-facing source document handed to the agent for indexing."""

    text: str
    source_url: str | None = None
    source_path: str | None = None
    source_type: str = "unknown"
    title: str | None = None


@dataclass(frozen=True)
class IndexedSummary:
    """Summary of what was indexed."""

    topic: str
    documents: int
    chunks: int
    chunk_ids: list[str] = field(default_factory=list)


# ---- Errors ---------------------------------------------------------------


class ResearchAgentError(RuntimeError):
    """Raised when the research agent cannot complete synthesis."""


# ---- LLM client (Ollama /api/generate) ------------------------------------


class OllamaGenerator:
    """Minimal Ollama /api/generate client. Used to call Qwen for synthesis."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 600.0,
    ) -> None:
        if not model:
            raise ValueError("model must not be empty")
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self,
        *,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.2,
        think: bool = False,
    ) -> str:
        """Run a single /api/generate call and return the response text."""
        if not prompt:
            raise ValueError("prompt must not be empty")
        body: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "think": think,
            "options": {"temperature": temperature},
        }
        if system:
            body["system"] = system
        url = f"{self._base_url}/api/generate"
        try:
            response = requests.post(url, json=body, timeout=self._timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ResearchAgentError(f"Ollama HTTP error on {url}: {exc}") from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise ResearchAgentError(f"Ollama returned non-JSON: {exc}") from exc
        text = payload.get("response", "")
        if not isinstance(text, str):
            raise ResearchAgentError(f"Ollama response missing 'response' string: {payload}")
        return text.strip()


# ---- Research agent -------------------------------------------------------


# Default system + user prompt for brief synthesis. Both are intentionally
# opinionated about the channel's USP (internals, source-grounded, video-ready).
DEFAULT_SYSTEM_PROMPT = (
    "You are a senior systems engineer writing a research brief for an "
    "internals-focused YouTube channel. The brief feeds into a video script "
    "and will be reviewed by a stricter AI before that. Be specific, cite "
    "exact source paths, and avoid generic definitions. If a section cannot "
    "be substantiated from the retrieved sources, say 'insufficient evidence' "
    "rather than inventing."
)

DEFAULT_USER_PROMPT_TEMPLATE = (
    "Topic: {topic}\n\n"
    "Below are excerpts retrieved from authoritative sources (RFCs, official "
    "docs, kernel/library source). Produce a research_brief.md with EXACTLY "
    "these twelve sections, in this order. Each section must be substantive.\n\n"
    "## 1. Executive Summary\n"
    "Two or three sentences: what the topic is and why it matters in practice.\n\n"
    "## 2. Core Concept\n"
    "The fundamental idea, in one paragraph. Use precise terminology.\n\n"
    "## 3. Architecture\n"
    "Components and their roles. Name specific subsystems, files, RFC sections.\n\n"
    "## 4. Internal Workflow\n"
    "Step-by-step internal process. Numbered steps with concrete state transitions.\n\n"
    "## 5. Packet / Data Flow\n"
    "Literal walkthrough: source -> each hop -> destination. Include header "
    "fields, function names, or API calls.\n\n"
    "## 6. Engineering Decisions\n"
    "Why is it designed this way? What tradeoffs were made?\n\n"
    "## 7. Source-Code and Documentation References\n"
    "Bullet list of specific files (path), RFC sections (RFC NNNN §X.Y), or "
    "vendor doc anchors the viewer can read.\n\n"
    "## 8. Troubleshooting Methodology\n"
    "Concrete diagnostic steps. Include commands (ss, ip, tcpdump, kubectl, "
    "iproute2, etc.) where relevant.\n\n"
    "## 9. Performance / Security Considerations\n"
    "Latency, throughput, resource costs. Auth, encryption, attack surface.\n\n"
    "## 10. Common Misconceptions\n"
    "Bulleted list of what people typically get wrong about this topic.\n\n"
    "## 11. YouTube Content Gap\n"
    "What do existing top YouTube videos on this topic miss or oversimplify?\n\n"
    "## 12. Recommended Video Angle\n"
    "A specific, novel angle that would make the resulting video more "
    "insightful than existing coverage.\n\n"
    "Cite sources inline like [RFC 793 §3.9] or [torvalds/linux/net/ipv4/tcp_input.c].\n\n"
    "## Retrieved Sources\n\n{sources}\n"
)


class ResearchAgent:
    """Indexes source documents and synthesizes a research brief on demand."""

    def __init__(
        self,
        *,
        llm: OllamaGenerator,
        embedding_client: EmbeddingClient,
        qdrant_store: QdrantStore,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        max_context_chunks: int = 20,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        user_prompt_template: str = DEFAULT_USER_PROMPT_TEMPLATE,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        if max_context_chunks <= 0:
            raise ValueError("max_context_chunks must be > 0")

        self._llm = llm
        self._embeddings = embedding_client
        self._store = qdrant_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_context_chunks = max_context_chunks
        self._system_prompt = system_prompt
        self._user_prompt_template = user_prompt_template

    # ---- Indexing ---------------------------------------------------------

    def index_sources(self, topic: str, documents: list[SourceDocument]) -> IndexedSummary:
        """Chunk, embed, and upsert the given documents into the Qdrant store.

        Returns a summary including the list of chunk IDs assigned by Qdrant.
        """
        if not topic.strip():
            raise ValueError("topic must not be empty")
        if not documents:
            return IndexedSummary(topic=topic, documents=0, chunks=0, chunk_ids=[])

        self._store.ensure_collection()

        chunk_docs: list[DocumentChunk] = []
        chunk_texts: list[str] = []
        for document in documents:
            pieces = split_paragraphs(document.text, self._chunk_size, self._chunk_overlap)
            for piece in pieces:
                chunk_docs.append(
                    DocumentChunk(
                        text=piece,
                        source_url=document.source_url,
                        source_path=document.source_path,
                        source_type=document.source_type,
                        title=document.title,
                        topic=topic,
                    )
                )
                chunk_texts.append(piece)

        # Embed in one batch call (Ollama /api/embeddings is single-text but
        # the client wraps the loop). For very large corpora this is the
        # bottleneck; consider parallelizing later.
        vectors = self._embeddings.embed_batch(chunk_texts)
        chunk_ids = self._store.upsert(vectors=vectors, chunks=chunk_docs)
        return IndexedSummary(
            topic=topic,
            documents=len(documents),
            chunks=len(chunk_docs),
            chunk_ids=chunk_ids,
        )

    # ---- Synthesis --------------------------------------------------------

    def build_synthesis_prompt(self, topic: str) -> str:
        """Build the user prompt that will be sent to the LLM.

        Useful for tests and for debugging retrieval quality independently of
        the LLM call. Embeds `topic`, retrieves top-k chunks from Qdrant, and
        formats them into the user prompt template.
        """
        retrieved = self._retrieve_context(topic)
        sources_text = _format_sources(retrieved)
        return self._user_prompt_template.format(topic=topic, sources=sources_text)

    def synthesize_brief(self, topic: str) -> str:
        """Retrieve relevant chunks and ask the LLM to produce a research brief.

        Returns the markdown body (no front-matter).
        """
        prompt = self.build_synthesis_prompt(topic)
        return self._llm.generate(
            prompt=prompt,
            system=self._system_prompt,
            temperature=0.2,
            think=False,
        )

    # ---- Internals --------------------------------------------------------

    def _retrieve_context(self, topic: str) -> list[Any]:
        # Ensure the collection exists so callers that bypass index_sources
        # (e.g. an empty corpus) get a clean "(no sources retrieved)" instead
        # of a Qdrant "Collection not found" error.
        self._store.ensure_collection()
        query_vector = self._embeddings.embed(topic)
        return self._store.search(
            query_vector=query_vector,
            limit=self._max_context_chunks,
        )


def _format_sources(chunks: list[Any]) -> str:
    """Format retrieved chunks into the prompt's "Retrieved Sources" block."""
    if not chunks:
        return "(no sources retrieved)"
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        location_parts: list[str] = []
        if getattr(chunk, "title", None):
            location_parts.append(f"title={chunk.title}")
        if getattr(chunk, "source_url", None):
            location_parts.append(f"url={chunk.source_url}")
        if getattr(chunk, "source_path", None):
            location_parts.append(f"path={chunk.source_path}")
        location_parts.append(f"type={getattr(chunk, 'source_type', 'unknown')}")
        location = "; ".join(location_parts)
        text = (getattr(chunk, "text", "") or "").strip()
        blocks.append(f"[{index}] ({location})\n{text}")
    return "\n\n".join(blocks)
