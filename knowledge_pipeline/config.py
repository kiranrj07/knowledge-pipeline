"""Environment + configuration loader for the knowledge pipeline.

Loads `.env` from the project root and validates required credentials up front
so failures happen at startup rather than mid-run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root. We deliberately do NOT override real env vars
# so that secrets injected by the shell (e.g. systemd, docker) take precedence.
load_dotenv(_PROJECT_ROOT / ".env", override=False)


_REQUIRED_ENV_VARS: tuple[str, ...] = ("GITHUB_TOKEN", "OPENROUTER_API_KEY")


def _missing_env_vars() -> list[str]:
    return [name for name in _REQUIRED_ENV_VARS if not os.environ.get(name)]


def _optional(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


@dataclass(frozen=True)
class PipelineConfig:
    """Validated runtime configuration for the knowledge pipeline.

    Construct via `load_config()` — the dataclass itself does no validation
    so it can be created from tests with arbitrary values.
    """

    # Web discovery (SearXNG preferred; Parallel.ai as fallback tiers)
    searxng_url: str
    parallel_mcp_url: str
    parallel_api_key: str
    # Source-code grounding
    github_token: str
    # Escalation review
    openrouter_api_key: str
    # Local services
    qdrant_path: Path
    embedding_model: str
    ollama_base_url: str
    # LLM choices
    research_llm: str
    review_llm: str
    # Tunables
    max_discovery_urls: int
    max_github_files: int
    chunk_size_chars: int
    chunk_overlap_chars: int
    project_root: Path


def load_config() -> PipelineConfig:
    """Load configuration from the environment, validating required vars.

    Raises:
        RuntimeError: if any required env var (GITHUB_TOKEN, OPENROUTER_API_KEY)
            is missing. The error message names every missing var so the user
            can fix them in one pass.
    """
    missing = _missing_env_vars()
    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill them in."
        )

    return PipelineConfig(
        searxng_url=_optional("SEARXNG_URL", "http://127.0.0.1:8888"),
        parallel_mcp_url=_optional("PARALLEL_MCP_URL", "https://search.parallel.ai/mcp"),
        parallel_api_key=_optional("PARALLEL_API_KEY", ""),
        github_token=os.environ["GITHUB_TOKEN"],
        openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
        qdrant_path=Path(_optional("QDRANT_PATH", str(_PROJECT_ROOT / "qdrant_storage"))).expanduser().resolve(),
        embedding_model=_optional("EMBEDDING_MODEL", "nomic-embed-text"),
        ollama_base_url=_optional("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        research_llm=_optional("RESEARCH_LLM", "qwen3:14b"),
        review_llm=_optional("REVIEW_LLM", "openrouter/anthropic/claude-sonnet-4-20250514"),
        max_discovery_urls=int(_optional("MAX_DISCOVERY_URLS", "12")),
        max_github_files=int(_optional("MAX_GITHUB_FILES", "8")),
        chunk_size_chars=int(_optional("CHUNK_SIZE_CHARS", "1500")),
        chunk_overlap_chars=int(_optional("CHUNK_OVERLAP_CHARS", "200")),
        project_root=_PROJECT_ROOT,
    )
