"""End-to-end MVP-1 research pipeline.

Flow:
    topic
      -> Parallel.ai MCP web_search        (ranked authoritative URLs)
      -> Parallel.ai MCP web_fetch         (clean excerpts per URL)
      -> GitHub code search                (relevant kernel/source files)
      -> Qdrant (local mode)               (chunked + embedded store)
      -> Qwen via Ollama                   (research_brief.md)
      -> OpenRouter escalation reviewer    (quality_score.json, optional)

Each stage can be overridden via constructor injection so tests can run the
full pipeline without touching the network.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from knowledge_pipeline.config import PipelineConfig
from knowledge_pipeline.discovery.parallel_search import (
    ParallelMCPClient,
    SearchResponse,
    create_web_discovery_client,
)
from knowledge_pipeline.discovery.youtube import (
    fetch_transcripts,
    format_for_prompt as format_youtube_for_prompt,
)
from knowledge_pipeline.research.agent import (
    OllamaGenerator,
    ResearchAgent,
    SourceDocument,
)
from knowledge_pipeline.research.openrouter_client import (
    OpenRouterClient,
    QualityScore,
)
from knowledge_pipeline.source_code.github_client import (
    FileContents,
    GitHubClient,
)
from knowledge_pipeline.storage.embeddings import EmbeddingClient
from knowledge_pipeline.storage.qdrant_store import QdrantStore


# ---- Result types ---------------------------------------------------------


@dataclass(frozen=True)
class ResearchRunResult:
    topic: str
    output_path: Path
    collection_name: str
    discovered_urls: list[str] = field(default_factory=list)
    github_files: list[dict[str, str]] = field(default_factory=list)
    indexed_chunks: int = 0
    source_documents: int = 0
    quality_score: QualityScore | None = None
    quality_score_path: Path | None = None
    manifest_path: Path | None = None


# ---- Errors ---------------------------------------------------------------


class PipelineError(RuntimeError):
    """Raised when the research pipeline cannot complete."""


# ---- Helpers --------------------------------------------------------------


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "topic"


def _discover(
    client: ParallelMCPClient, *, topic: str, max_urls: int
) -> SearchResponse:
    """Call Parallel.ai web_search and return the ranked response."""
    return client.search(
        objective=(
            f"Find authoritative technical sources explaining the internals "
            f"of: {topic}. Prefer RFCs, official documentation, and source-code "
            f"references over tutorial blogs."
        ),
        search_queries=[
            topic,
            f"{topic} internals",
            f"{topic} implementation",
        ][: max(2, min(3, max_urls))],
    )


def _extract(
    client: ParallelMCPClient,
    *,
    urls: list[str],
    topic: str,
) -> list[SourceDocument]:
    """Call Parallel.ai web_fetch for the discovered URLs and convert to SourceDocument."""
    if not urls:
        return []
    response = client.fetch(
        urls=urls,
        objective=f"Extract content directly relevant to: {topic}",
    )
    sources: list[SourceDocument] = []
    for result in response.results:
        text = "\n\n".join(result.excerpts).strip()
        if not text:
            continue
        sources.append(
            SourceDocument(
                text=text,
                source_url=result.url,
                source_type="web_doc",
                title=result.title,
            )
        )
    return sources


def _discover_and_extract(
    client: Any, *, topic: str, max_urls: int
) -> tuple[list[str], list[SourceDocument]]:
    """Run discovery + extraction in one step, adapting to client capabilities.

    Returns (urls, sources).
    - For clients with `supports_fetch = True` (Parallel.ai): runs search
      then fetch, building SourceDocuments from the extracted excerpts.
    - For clients with `supports_fetch = False` (SearXNG): reuses the
      snippets already present in the search response, skipping fetch.

    Raises:
        PipelineError: if no usable sources are collected downstream.
    """
    search_response = _discover(client, topic=topic, max_urls=max_urls)
    discovered_urls = [r.url for r in search_response.results[:max_urls]]

    if getattr(client, "supports_fetch", True):
        sources = _extract(client, urls=discovered_urls, topic=topic)
    else:
        # SearXNG-style: search response already carries snippets as `excerpts`.
        sources = []
        for r in search_response.results[:max_urls]:
            text = "\n\n".join(r.excerpts).strip()
            if not text:
                continue
            sources.append(
                SourceDocument(
                    text=text,
                    source_url=r.url,
                    source_type="web_doc",
                    title=r.title,
                )
            )
    return discovered_urls, sources


def _ground_with_github(
    client: GitHubClient, *, topic: str, max_files: int
) -> list[SourceDocument]:
    """Search GitHub code and convert hits to SourceDocument."""
    if max_files <= 0:
        return []
    files = client.find_topic_sources(topic, max_files=max_files)
    sources: list[SourceDocument] = []
    for file in files:
        text = file.content.strip()
        if not text:
            continue
        sources.append(
            SourceDocument(
                text=text,
                source_url=file.html_url,
                source_path=file.path,
                source_type="source_code",
                title=f"{file.repo_full_name}/{file.path}",
            )
        )
    return sources


# ---- Main entry point -----------------------------------------------------


def run_research_pipeline(
    *,
    topic: str,
    output_path: Path,
    config: PipelineConfig,
    parallel_client: ParallelMCPClient | None = None,
    github_client: GitHubClient | None = None,
    openrouter_client: OpenRouterClient | None = None,
    embeddings: EmbeddingClient | None = None,
    qdrant_store: QdrantStore | None = None,
    llm: OllamaGenerator | None = None,
    research_agent: ResearchAgent | None = None,
    youtube_context: str | None = None,
    enable_review: bool = True,
    max_urls: int | None = None,
    max_github_files: int | None = None,
    collection_name: str | None = None,
    manifest_path: Path | None = None,
) -> ResearchRunResult:
    """Run the end-to-end MVP-1 research pipeline.

    Args:
        topic: the research topic (free-form string).
        output_path: where to write research_brief.md.
        config: validated PipelineConfig.
        parallel_client / github_client / openrouter_client / embeddings /
            qdrant_store / llm / research_agent: optional injected clients.
            When None, real ones are constructed from `config`.
        enable_review: whether to call the OpenRouter reviewer at the end.
        max_urls: override for the discovered-URL cap.
        max_github_files: override for the GitHub file cap.
        collection_name: override for the Qdrant collection name (default: topic slug).
        manifest_path: optional path for a JSON manifest of what was collected.

    Returns:
        ResearchRunResult with counts, paths, and optional QualityScore.
    """
    if not topic.strip():
        raise PipelineError("topic must not be empty")
    if output_path is None:
        raise PipelineError("output_path must not be None")

    max_urls = max_urls if max_urls is not None else config.max_discovery_urls
    max_github_files = (
        max_github_files if max_github_files is not None else config.max_github_files
    )
    collection = collection_name or _slugify(topic)

    discovery = parallel_client or create_web_discovery_client(
        searxng_url=config.searxng_url,
        parallel_api_key=config.parallel_api_key,
        parallel_mcp_url=config.parallel_mcp_url,
    )
    github = github_client or GitHubClient(token=config.github_token)

    # Stage 1 + 2: discovery + extraction (adapts to client capabilities).
    # SearXNG returns snippets in the search response, so no separate fetch
    # step; Parallel.ai uses search -> fetch. _discover_and_extract handles both.
    discovered_urls, web_sources = _discover_and_extract(
        discovery, topic=topic, max_urls=max_urls
    )

    # Stage 2.5: YouTube transcript discovery (the main USP).
    # SearXNG's web results often include YouTube URLs; filter + fetch transcripts.
    yt_transcripts, yt_errors = fetch_transcripts(discovered_urls, max_videos=5)
    youtube_context = (
        format_youtube_for_prompt(yt_transcripts) if yt_transcripts else ""
    )
    youtube_meta = [
        {"url": t.url, "video_id": t.video_id, "summary_chars": len(t.summary)}
        for t in yt_transcripts
    ]

    # Stage 3: source-code grounding
    code_sources = _ground_with_github(github, topic=topic, max_files=max_github_files)
    github_meta = [
        {"repo": s.source_url or "", "path": s.source_path or "", "title": s.title or ""}
        for s in code_sources
    ]

    all_sources = web_sources + code_sources
    if not all_sources:
        raise PipelineError(
            "No source documents collected: discovery returned no URLs and "
            "GitHub search returned no files. Check your topic and API access."
        )

    # Stage 4: index into Qdrant
    emb = embeddings or EmbeddingClient(
        model=config.embedding_model, base_url=config.ollama_base_url
    )
    dim = emb.dimension
    if dim is None:
        # Fall back to probing with a one-token sample. Costs one extra embed call.
        dim = emb.probe_dimension()
    store = qdrant_store or QdrantStore(
        storage_path=str(config.qdrant_path),
        collection_name=collection,
        vector_dim=dim,
    )

    generator = llm or OllamaGenerator(
        model=config.research_llm, base_url=config.ollama_base_url
    )
    agent = research_agent or ResearchAgent(
        llm=generator,
        embedding_client=emb,
        qdrant_store=store,
        chunk_size=config.chunk_size_chars,
        chunk_overlap=config.chunk_overlap_chars,
    )

    summary = agent.index_sources(topic, all_sources)

    # Stage 5: synthesize brief
    brief_markdown = agent.synthesize_brief(topic)

    # Persist brief
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(brief_markdown.rstrip() + "\n", encoding="utf-8")

    quality_score: QualityScore | None = None
    quality_path: Path | None = None
    if enable_review:
        try:
            reviewer = openrouter_client or OpenRouterClient(
                api_key=config.openrouter_api_key
            )
            quality_score = reviewer.review_quality(
                brief_text=brief_markdown, topic=topic, model=config.review_llm
            )
            quality_path = output_path.with_suffix(output_path.suffix + ".quality.json")
            quality_path.write_text(
                json.dumps(
                    {
                        "technical_accuracy": quality_score.technical_accuracy,
                        "depth": quality_score.depth,
                        "uniqueness": quality_score.uniqueness,
                        "troubleshooting_value": quality_score.troubleshooting_value,
                        "source_grounding": quality_score.source_grounding,
                        "composite": quality_score.composite,
                        "ready_for_script": quality_score.ready_for_script,
                        "rationale": quality_score.rationale,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            # Review is best-effort; never fail the run because the reviewer flaked.
            quality_path = output_path.with_suffix(output_path.suffix + ".quality.json")
            quality_path.write_text(
                json.dumps(
                    {
                        "review_skipped": True,
                        "error": str(exc),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    manifest_written: Path | None = None
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                _build_manifest(
                    topic=topic,
                    collection=collection,
                    discovered_urls=discovered_urls,
                    github_files=github_meta,
                    summary=summary,
                    output_path=output_path,
                    quality_path=quality_path,
                    quality_score=quality_score,
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest_written = manifest_path

    return ResearchRunResult(
        topic=topic,
        output_path=output_path,
        collection_name=collection,
        discovered_urls=discovered_urls,
        github_files=github_meta,
        indexed_chunks=summary.chunks,
        source_documents=len(all_sources),
        quality_score=quality_score,
        quality_score_path=quality_path,
        manifest_path=manifest_written,
    )


def _build_manifest(
    *,
    topic: str,
    collection: str,
    discovered_urls: list[str],
    github_files: list[dict[str, str]],
    summary: Any,
    output_path: Path,
    quality_path: Path | None,
    quality_score: QualityScore | None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "topic": topic,
        "collection": collection,
        "discovered_urls": discovered_urls,
        "github_files": github_files,
        "indexed_chunks": summary.chunks,
        "source_documents": summary.documents,
        "output": str(output_path),
    }
    if quality_path is not None:
        manifest["quality_score_path"] = str(quality_path)
    if quality_score is not None:
        manifest["quality_score"] = {
            "technical_accuracy": quality_score.technical_accuracy,
            "depth": quality_score.depth,
            "uniqueness": quality_score.uniqueness,
            "troubleshooting_value": quality_score.troubleshooting_value,
            "source_grounding": quality_score.source_grounding,
            "composite": quality_score.composite,
            "ready_for_script": quality_score.ready_for_script,
        }
    return manifest
