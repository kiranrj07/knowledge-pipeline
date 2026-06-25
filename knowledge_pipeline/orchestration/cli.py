"""CLI entrypoint for the knowledge-pipeline MVP-1.

Usage:
    python -m knowledge_pipeline.orchestration.cli research \
        --topic "TCP SYN backlog" \
        --output runs/tcp-syn-backlog/research_brief.md

Environment:
    Loads .env from the project root. Required: GITHUB_TOKEN, OPENROUTER_API_KEY.
    PARALLEL_MCP_URL defaults to https://search.parallel.ai/mcp (no key needed).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from knowledge_pipeline.config import load_config
from knowledge_pipeline.orchestration.pipeline import (
    PipelineError,
    ResearchRunResult,
    run_research_pipeline,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="knowledge-pipeline",
        description="Research pipeline for the internals-focused YouTube channel.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    research = sub.add_parser("research", help="Run the end-to-end research pipeline for a topic.")
    research.add_argument("--topic", required=True, help="Research topic (free-form string).")
    research.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write research_brief.md (any path; parent dirs created).",
    )
    research.add_argument(
        "--max-urls",
        type=int,
        default=None,
        help="Cap on discovered URLs (defaults to MAX_DISCOVERY_URLS from .env).",
    )
    research.add_argument(
        "--max-github-files",
        type=int,
        default=None,
        help="Cap on GitHub source files (defaults to MAX_GITHUB_FILES from .env).",
    )
    research.add_argument(
        "--collection",
        default=None,
        help="Qdrant collection name (default: topic slug).",
    )
    research.add_argument(
        "--review",
        dest="review",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run OpenRouter quality review at the end (default: enabled).",
    )
    research.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to write a JSON manifest of what was collected.",
    )
    research.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )

    return parser.parse_args(argv)


def _print(msg: str, *, quiet: bool) -> None:
    if not quiet:
        print(msg, file=sys.stderr)


def run_research_command(args: argparse.Namespace) -> ResearchRunResult:
    config = load_config()
    result = run_research_pipeline(
        topic=args.topic,
        output_path=args.output,
        config=config,
        enable_review=args.review,
        max_urls=args.max_urls,
        max_github_files=args.max_github_files,
        collection_name=args.collection,
        manifest_path=args.manifest,
    )

    _print(
        f"[discovery] {len(result.discovered_urls)} URLs",
        quiet=args.quiet,
    )
    _print(
        f"[github]    {len(result.github_files)} source files",
        quiet=args.quiet,
    )
    _print(
        f"[index]     {result.indexed_chunks} chunks across {result.source_documents} sources",
        quiet=args.quiet,
    )
    _print(
        f"[output]    {result.output_path}",
        quiet=args.quiet,
    )
    if result.quality_score is not None:
        _print(
            f"[review]    composite={result.quality_score.composite:.2f} "
            f"ready_for_script={result.quality_score.ready_for_script}",
            quiet=args.quiet,
        )
        if result.quality_score_path is not None:
            _print(f"[review]    {result.quality_score_path}", quiet=args.quiet)
    if result.manifest_path is not None:
        _print(f"[manifest]  {result.manifest_path}", quiet=args.quiet)

    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "research":
            run_research_command(args)
            return 0
        print(f"unknown command: {args.command}", file=sys.stderr)
        return 2
    except PipelineError as exc:
        print(f"pipeline error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
