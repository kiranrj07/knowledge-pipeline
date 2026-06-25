"""Smoke tests for the end-to-end MVP-1 pipeline + CLI.

Run with:
    /home/janak/ai/knowledge-pipeline/.venv/bin/python tests/smoke_pipeline.py

Uses fake clients/services so no network, Ollama, Qdrant persistence, or
OpenRouter call is made. Tests the orchestration logic in isolation.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from knowledge_pipeline.config import PipelineConfig  # noqa: E402
from knowledge_pipeline.discovery.parallel_search import (  # noqa: E402
    FetchResponse,
    FetchResult,
    SearchResponse,
    SearchResult,
)
from knowledge_pipeline.orchestration.cli import parse_args, run_research_command  # noqa: E402
from knowledge_pipeline.orchestration.pipeline import (  # noqa: E402
    PipelineError,
    ResearchRunResult,
    _discover,
    _extract,
    _ground_with_github,
    _slugify,
    run_research_pipeline,
)
from knowledge_pipeline.research.agent import OllamaGenerator  # noqa: E402
from knowledge_pipeline.research.openrouter_client import QualityScore  # noqa: E402
from knowledge_pipeline.source_code.github_client import FileContents  # noqa: E402
from knowledge_pipeline.storage.qdrant_store import QdrantStore  # noqa: E402
from tests.smoke_research_agent import (  # noqa: E402
    FakeEmbeddingClient,
    FakeOllamaGenerator,
)


# ---- Test doubles for orchestration ---------------------------------------


class FakeParallelClient:
    """Records calls and returns canned search/fetch responses."""

    def __init__(
        self,
        *,
        search_response: SearchResponse,
        fetch_response: FetchResponse,
    ) -> None:
        self._search_response = search_response
        self._fetch_response = fetch_response
        self.search_calls: list[dict] = []
        self.fetch_calls: list[dict] = []

    def search(self, *, objective: str, search_queries: list[str], model_name: str | None = None) -> SearchResponse:
        self.search_calls.append({"objective": objective, "search_queries": list(search_queries)})
        return self._search_response

    def fetch(self, *, urls: list[str], objective: str | None = None, **kwargs) -> FetchResponse:
        self.fetch_calls.append({"urls": list(urls), "objective": objective})
        return self._fetch_response


class FakeGitHubClient:
    """Records find_topic_sources calls and returns canned FileContents."""

    def __init__(self, files: list[FileContents]) -> None:
        self._files = files
        self.find_calls: list[dict] = []

    def find_topic_sources(self, topic: str, *, max_files: int = 8, language: str | None = "c") -> list[FileContents]:
        self.find_calls.append({"topic": topic, "max_files": max_files, "language": language})
        return self._files[:max_files]


class FakeOpenRouterClient:
    """Records review_quality calls and returns a canned QualityScore."""

    def __init__(self, score: QualityScore) -> None:
        self._score = score
        self.review_calls: list[dict] = []

    def review_quality(self, *, brief_text: str, topic: str, model: str) -> QualityScore:
        self.review_calls.append({"topic": topic, "model": model, "length": len(brief_text)})
        return self._score


# ---- Fixtures -------------------------------------------------------------


def _config(tmpdir: str) -> PipelineConfig:
    """Build a PipelineConfig for tests. The dataclass is frozen but accepts
    arbitrary values from tests; we construct via __init__-style kwargs."""
    return PipelineConfig(
        searxng_url="",  # tests inject a fake discovery_client directly
        parallel_mcp_url="https://example.invalid/mcp",
        parallel_api_key="",
        github_token="ghp_test_token",
        openrouter_api_key="sk-or-v1-test",
        qdrant_path=Path(tmpdir).expanduser().resolve(),
        embedding_model="nomic-embed-text",
        ollama_base_url="http://127.0.0.1:11434",
        research_llm="qwen3:14b",
        review_llm="anthropic/claude-sonnet-4-20250514",
        max_discovery_urls=5,
        max_github_files=3,
        chunk_size_chars=1500,
        chunk_overlap_chars=200,
        project_root=Path(tmpdir),
    )


def _tmp_config() -> tuple[PipelineConfig, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory(prefix="kp_pipeline_test_")
    return _config(tmp.name), tmp


# ---- _slugify ------------------------------------------------------------


def test_slugify() -> None:
    assert _slugify("TCP SYN backlog") == "tcp-syn-backlog"
    assert _slugify("AWS Transit Gateway / Appliance Mode") == "aws-transit-gateway-appliance-mode"
    assert _slugify("") == "topic"
    assert _slugify("___") == "topic"


# ---- _discover / _extract / _ground_with_github ---------------------------


def test_discover() -> None:
    canned = SearchResponse(
        search_id="search_x", session_id="session_x",
        results=[SearchResult(url="https://example.com/a", title="A")],
    )
    client = FakeParallelClient(search_response=canned, fetch_response=FetchResponse(extract_id="e", session_id="s"))
    response = _discover(client, topic="TCP SYN backlog", max_urls=3)
    assert response is canned
    assert len(client.search_calls) == 1
    assert "TCP SYN backlog" in client.search_calls[0]["objective"]
    assert len(client.search_calls[0]["search_queries"]) >= 2


def test_extract_empty() -> None:
    client = FakeParallelClient(
        search_response=SearchResponse(search_id="s", session_id="s"),
        fetch_response=FetchResponse(extract_id="e", session_id="s"),
    )
    sources = _extract(client, urls=[], topic="x")
    assert sources == []
    assert client.fetch_calls == []  # no call when urls is empty


def test_extract_populates_sources() -> None:
    canned = FetchResponse(
        extract_id="e1", session_id="s1",
        results=[
            FetchResult(url="https://example.com/a", title="Doc A", excerpts=["Excerpt A1", "Excerpt A2"]),
            FetchResult(url="https://example.com/b", title="Doc B", excerpts=[]),  # skipped (empty text)
            FetchResult(url="https://example.com/c", title="Doc C", excerpts=["Excerpt C1"]),
        ],
    )
    client = FakeParallelClient(
        search_response=SearchResponse(search_id="s", session_id="s"),
        fetch_response=canned,
    )
    sources = _extract(client, urls=["https://example.com/a", "https://example.com/b"], topic="x")
    assert len(sources) == 2  # b skipped
    assert sources[0].source_type == "web_doc"
    assert sources[0].source_url == "https://example.com/a"
    assert "Excerpt A1" in sources[0].text and "Excerpt A2" in sources[0].text
    assert sources[1].title == "Doc C"


def test_ground_with_github_zero_cap() -> None:
    client = FakeGitHubClient(files=[])
    sources = _ground_with_github(client, topic="x", max_files=0)
    assert sources == []
    assert client.find_calls == []  # no call when cap is zero


def test_ground_with_github_skips_empty_content() -> None:
    files = [
        FileContents(repo_full_name="torvalds/linux", path="net/ipv4/tcp_input.c",
                     sha="abc", size=100, content="static int tcp_rcv_state(...) {}",
                     html_url="https://github.com/torvalds/linux/blob/master/net/ipv4/tcp_input.c"),
        FileContents(repo_full_name="torvalds/linux", path="net/ipv4/empty.c",
                     sha="def", size=0, content="   \n  "),  # whitespace-only -> skipped
    ]
    client = FakeGitHubClient(files=files)
    sources = _ground_with_github(client, topic="TCP SYN backlog", max_files=5)
    assert len(sources) == 1
    assert sources[0].source_type == "source_code"
    assert sources[0].source_path == "net/ipv4/tcp_input.c"
    assert sources[0].source_url is not None
    assert "torvalds/linux/net/ipv4/tcp_input.c" == sources[0].title


# ---- run_research_pipeline (end-to-end with mocks) -----------------------


def test_pipeline_end_to_end_with_mocks() -> None:
    config, tmp = _tmp_config()
    try:
        # Build canned external responses.
        canned_search = SearchResponse(
            search_id="s1", session_id="sess",
            results=[
                SearchResult(url="https://example.com/tcp-a", title="TCP RFC"),
                SearchResult(url="https://example.com/tcp-b", title="Linux Kernel Notes"),
            ],
        )
        canned_fetch = FetchResponse(
            extract_id="e1", session_id="sess",
            results=[
                FetchResult(url="https://example.com/tcp-a", title="TCP RFC",
                            excerpts=["The SYN backlog holds half-open connections."]),
            ],
        )
        canned_gh = [
            FileContents(repo_full_name="torvalds/linux", path="net/ipv4/tcp_input.c",
                         sha="abc", size=200,
                         content="static void tcp_rcv_state_process(struct sock *sk) { ... }"),
        ]
        canned_score = QualityScore(
            technical_accuracy=8, depth=9, uniqueness=7,
            troubleshooting_value=6, source_grounding=9,
            ready_for_script=False,
            rationale="Strong internals coverage; weak on troubleshooting commands.",
        )

        parallel = FakeParallelClient(search_response=canned_search, fetch_response=canned_fetch)
        github = FakeGitHubClient(files=canned_gh)
        openrouter = FakeOpenRouterClient(score=canned_score)
        embeddings = FakeEmbeddingClient(dim=4)
        llm = FakeOllamaGenerator(canned="## 1. Executive Summary\nA brief on TCP SYN backlog.")

        # Real Qdrant store in a tmpdir (so we exercise the real query_points API).
        qdrant = QdrantStore(
            storage_path=str(config.qdrant_path), collection_name="tcp-syn-backlog", vector_dim=4,
        )

        out_path = Path(tmp.name) / "brief.md"
        manifest_path = Path(tmp.name) / "manifest.json"

        result = run_research_pipeline(
            topic="TCP SYN backlog",
            output_path=out_path,
            config=config,
            parallel_client=parallel,
            github_client=github,
            openrouter_client=openrouter,
            embeddings=embeddings,
            qdrant_store=qdrant,
            llm=llm,
            enable_review=True,
            manifest_path=manifest_path,
        )

        # Return shape
        assert isinstance(result, ResearchRunResult)
        assert result.topic == "TCP SYN backlog"
        assert result.output_path == out_path
        assert result.collection_name == "tcp-syn-backlog"
        assert result.discovered_urls == ["https://example.com/tcp-a", "https://example.com/tcp-b"]
        assert len(result.github_files) == 1
        assert result.source_documents == 2  # 1 web + 1 code
        assert result.indexed_chunks >= 1
        assert result.quality_score is canned_score
        assert result.quality_score_path is not None
        assert result.manifest_path == manifest_path

        # Files written
        assert out_path.exists()
        assert "Executive Summary" in out_path.read_text()
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["topic"] == "TCP SYN backlog"
        assert manifest["collection"] == "tcp-syn-backlog"
        assert manifest["discovered_urls"] == result.discovered_urls
        assert manifest["indexed_chunks"] == result.indexed_chunks
        assert manifest["quality_score"]["composite"] == canned_score.composite

        quality_path = out_path.with_suffix(out_path.suffix + ".quality.json")
        assert quality_path.exists()
        quality = json.loads(quality_path.read_text())
        assert quality["composite"] == canned_score.composite
        assert quality["ready_for_script"] is False

        # External services were called.
        assert len(parallel.search_calls) == 1
        assert parallel.fetch_calls[0]["urls"] == result.discovered_urls
        assert len(github.find_calls) == 1
        assert github.find_calls[0]["topic"] == "TCP SYN backlog"
        assert len(openrouter.review_calls) == 1
        assert llm.generate_calls  # called at least once for synthesis
    finally:
        tmp.cleanup()


def test_pipeline_no_sources_raises() -> None:
    config, tmp = _tmp_config()
    try:
        parallel = FakeParallelClient(
            search_response=SearchResponse(search_id="s", session_id="s", results=[]),
            fetch_response=FetchResponse(extract_id="e", session_id="s"),
        )
        github = FakeGitHubClient(files=[])
        embeddings = FakeEmbeddingClient(dim=4)
        llm = FakeOllamaGenerator(canned="unused")
        qdrant = QdrantStore(
            storage_path=str(config.qdrant_path), collection_name="x", vector_dim=4,
        )

        out_path = Path(tmp.name) / "brief.md"
        try:
            run_research_pipeline(
                topic="Empty topic",
                output_path=out_path,
                config=config,
                parallel_client=parallel,
                github_client=github,
                embeddings=embeddings,
                qdrant_store=qdrant,
                llm=llm,
                enable_review=False,
            )
        except PipelineError as e:
            assert "No source documents collected" in str(e)
        else:
            raise AssertionError("expected PipelineError when no sources found")
    finally:
        tmp.cleanup()


def test_pipeline_empty_topic_raises() -> None:
    config, tmp = _tmp_config()
    try:
        try:
            run_research_pipeline(
                topic="",
                output_path=Path(tmp.name) / "brief.md",
                config=config,
            )
        except PipelineError as e:
            assert "topic" in str(e)
        else:
            raise AssertionError("expected PipelineError on empty topic")
    finally:
        tmp.cleanup()


# ---- CLI parsing ---------------------------------------------------------


def test_cli_required_args() -> None:
    try:
        parse_args([])
    except SystemExit:
        pass
    else:
        raise AssertionError("expected SystemExit on missing subcommand")


def test_cli_research_parses() -> None:
    args = parse_args(["research", "--topic", "TCP", "--output", "/tmp/brief.md"])
    assert args.command == "research"
    assert args.topic == "TCP"
    assert str(args.output) == "/tmp/brief.md"
    assert args.review is True  # default enabled
    assert args.max_urls is None
    assert args.max_github_files is None


def test_cli_research_overrides() -> None:
    args = parse_args([
        "research", "--topic", "TCP", "--output", "/tmp/brief.md",
        "--max-urls", "8", "--max-github-files", "4",
        "--collection", "tcp-coll", "--no-review", "--quiet",
        "--manifest", "/tmp/manifest.json",
    ])
    assert args.max_urls == 8
    assert args.max_github_files == 4
    assert args.collection == "tcp-coll"
    assert args.review is False
    assert args.quiet is True
    assert str(args.manifest) == "/tmp/manifest.json"


TESTS = [
    test_slugify,
    test_discover,
    test_extract_empty,
    test_extract_populates_sources,
    test_ground_with_github_zero_cap,
    test_ground_with_github_skips_empty_content,
    test_pipeline_end_to_end_with_mocks,
    test_pipeline_no_sources_raises,
    test_pipeline_empty_topic_raises,
    test_cli_required_args,
    test_cli_research_parses,
    test_cli_research_overrides,
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
