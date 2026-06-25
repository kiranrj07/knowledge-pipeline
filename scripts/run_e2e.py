#!/usr/bin/env python3
"""End-to-end orchestrator: knowledge-pipeline (research) -> whisper (video).

Usage:
    .venv/bin/python scripts/run_e2e.py --topic "AWS VPC"

What it does:
    1. Runs knowledge-pipeline CLI to produce research_brief.md + manifest.json
       (SearXNG web discovery + GitHub source-code + YouTube transcripts + Qwen3 synthesis)
    2. Copies the brief into whisper's output dir so it's available as input context
    3. Runs whisper's run_pipeline.py (Graphviz + Qwen3/Gemma3 + Piper TTS + Whisper STT
       + FFmpeg assembly + optional Moondream review)
    4. Both pipelines share the same --output-dir so all artifacts land in one place

Why this exists:
    The two projects were built independently. knowledge-pipeline produces a
    13-section research_brief.md with the YouTube Coverage Delta USP section.
    whisper's run_pipeline.py does its own internal research synthesis. This
    orchestrator runs them back-to-back so a single --topic command produces
    both the research brief AND the final video.

Future work (not yet wired):
    - whisper should accept --input research_brief.md and skip its internal
      research when present, replacing it with the higher-quality knowledge-pipeline
      output. Currently the brief is just copied into the output dir as context.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def slugify(value: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in value).strip("-")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="knowledge-pipeline -> whisper end-to-end orchestrator"
    )
    parser.add_argument("--topic", required=True, help="Research topic / video topic")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Shared output directory. Defaults to runs/<topic-slug>.",
    )
    parser.add_argument(
        "--kp-root",
        type=Path,
        default=Path("/home/janak/ai/knowledge-pipeline"),
        help="Path to the knowledge-pipeline checkout.",
    )
    parser.add_argument(
        "--wp-root",
        type=Path,
        default=Path("/home/janak/ai/whisper"),
        help="Path to the whisper checkout.",
    )
    parser.add_argument(
        "--max-urls", type=int, default=4, help="SearXNG discovery cap"
    )
    parser.add_argument(
        "--max-github-files", type=int, default=4, help="GitHub source-code cap"
    )
    parser.add_argument(
        "--review-loops", type=int, default=0,
        help="Moondream diagram review loops in whisper (0 = skip)",
    )
    parser.add_argument(
        "--kp-venv",
        type=Path,
        default=None,
        help="Override knowledge-pipeline venv (defaults to <kp-root>/.venv)",
    )
    parser.add_argument(
        "--wp-venv",
        type=Path,
        default=None,
        help="Override whisper venv (defaults to <wp-root>/.venv)",
    )
    parser.add_argument(
        "--skip-knowledge", action="store_true",
        help="Skip knowledge-pipeline step (use existing brief in output-dir)",
    )
    args = parser.parse_args()

    out_dir = args.output_dir or (Path("runs") / slugify(args.topic))
    out_dir.mkdir(parents=True, exist_ok=True)

    kp_venv = args.kp_venv or (args.kp_root / ".venv")
    wp_venv = args.wp_venv or (args.wp_root / ".venv")
    kp_python = kp_venv / "bin" / "python"
    wp_python = wp_venv / "bin" / "python"

    for p, label in [(kp_python, "knowledge-pipeline venv"), (wp_python, "whisper venv")]:
        if not p.exists():
            print(f"ERROR: {label} python not found at {p}", file=sys.stderr)
            print(f"  Hint: create it with `python3.12 -m venv {p.parent}`", file=sys.stderr)
            return 1

    print(f"=== Output directory: {out_dir} ===\n")

    # ---- Step 1: knowledge-pipeline research synthesis --------------------
    brief_path = out_dir / "research_brief.md"
    manifest_path = out_dir / "manifest.json"

    if not args.skip_knowledge and not brief_path.exists():
        print(f"=== Step 1/2: knowledge-pipeline research synthesis ===")
        cmd = [
            str(kp_python),
            "-m", "knowledge_pipeline.orchestration.cli", "research",
            "--topic", args.topic,
            "--output", str(brief_path),
            "--manifest", str(manifest_path),
            "--max-urls", str(args.max_urls),
            "--max-github-files", str(args.max_github_files),
            "--no-review",  # skip quality review here; whisper does its own
        ]
        print(f"  $ {' '.join(cmd[:8])} ...")
        result = subprocess.run(cmd, cwd=str(args.kp_root))
        if result.returncode != 0:
            print(f"\nERROR: knowledge-pipeline failed (exit {result.returncode})", file=sys.stderr)
            return result.returncode
        print(f"  -> brief: {brief_path}\n")
    else:
        print(f"Step 1/2: knowledge-pipeline skipped (using existing {brief_path})\n")

    # ---- Step 1.5: copy brief into whisper output dir as input context -------
    if brief_path.exists():
        wp_brief = out_dir / "input_research_brief.md"
        shutil.copy2(brief_path, wp_brief)
        print(f"  Copied brief -> {wp_brief} (whisper context)\n")

    # ---- Step 2: whisper video production ----------------------------------
    print(f"=== Step 2/2: whisper video production ===")
    cmd = [
        str(wp_python),
        str(args.wp_root / "run_pipeline.py"),
        "--topic", args.topic,
        "--output-dir", str(out_dir),
        "--review-loops", str(args.review_loops),
    ]
    print(f"  $ {' '.join(cmd[:6])} ...")
    result = subprocess.run(cmd, cwd=str(args.wp_root))
    if result.returncode != 0:
        print(f"\nERROR: whisper pipeline failed (exit {result.returncode})", file=sys.stderr)
        return result.returncode

    print(f"\n=== Done. Artifacts in: {out_dir} ===")
    print(f"  research_brief.md        — knowledge-pipeline output")
    print(f"  manifest.json            — discovery + sources audit")
    print(f"  scenes/ + final_video.mp4 — whisper output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
