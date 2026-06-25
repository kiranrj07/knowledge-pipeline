# knowledge-pipeline

Research and knowledge-layer for the **internals-focused** YouTube automation
channel. Built on top of the existing local video pipeline at
`/home/janak/ai/whisper/`. This project handles stages 1–8 of the overall
architecture (discovery, source collection, knowledge fusion); the existing
pipeline at `whisper/` handles stages 9–15 (diagrams, narration, mux,
visual review).

## What this does

```
topic
  ↓
SearXNG  ──→ ranked authoritative URLs (+ YouTube URLs filtered out)
  ↓
SearXNG web_fetch  ──→ clean Markdown excerpts
  ↓
GitHub TOPIC_REPO_MAP  ──→ curated kernel / library source files
  ↓
YouTube transcripts (NEW)  ──→ fetched via youtube-transcript-api
  ↓
Qdrant (local mode)  ──→ chunked + embedded store
  ↓
Qwen3:14b (Ollama)  ──→ research_brief.md (13 sections)
  ↓
OpenRouter (escalation review)  ──→ quality_score.json
```

The point: **primary research must come from RFCs, official docs, and source
code.** YouTube transcripts only exist to find gaps, never as a source of
truth. Section 13 ("YouTube Coverage Delta") of the brief is the unique
deliverable that makes this pipeline stand out from existing YouTube videos.

## Setup

```bash
cd /home/janak/ai/knowledge-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

cp .env.example .env
# Edit .env: fill in GITHUB_TOKEN and OPENROUTER_API_KEY

# Start SearXNG via Docker (JSON format enabled)
docker pull searxng/searxng
docker run -d --name searxng --restart unless-stopped \
    -p 8888:8080 \
    -v $PWD/searxng-data/settings.yml:/etc/searxng/settings.yml:ro \
    -e SEARXNG_SECRET=changeme \
    searxng/searxng

# Ensure Ollama is running and the required models are pulled
ollama pull qwen3:14b
ollama pull nomic-embed-text

# Install YouTube transcript support
pip install youtube-transcript-api
```

## Run MVP-1 (research synthesis only)

```bash
python -m knowledge_pipeline.orchestration.cli research \
    --topic "TCP SYN backlog" \
    --output runs/tcp-syn-backlog/research_brief.md
```

The output is a `research_brief.md` (13 sections) + `manifest.json` (audit trail)
+ optional `quality_score.json` (OpenRouter review).

## Run the full pipeline (research → video)

The `scripts/run_e2e.py` orchestrator runs knowledge-pipeline first, then
chains into the whisper video pipeline. Both pipelines live in separate
repos; this script bridges them.

```bash
.venv/bin/python scripts/run_e2e.py --topic "AWS VPC"
```

What happens:

1. knowledge-pipeline CLI produces `research_brief.md` + `manifest.json`
2. The brief is copied into the whisper output directory as input context
3. whisper's `run_pipeline.py` produces the final `final_video.mp4` (via
   Graphviz diagrams + Piper TTS + Whisper STT + FFmpeg mux)

Options:

```bash
# Override output directory
.venv/bin/python scripts/run_e2e.py --topic "BGP route reflectors" \
    --output-dir runs/bgp-rr

# Enable Moondream diagram review in whisper
.venv/bin/python scripts/run_e2e.py --topic "eBPF XDP" --review-loops 2

# Skip knowledge-pipeline (use existing brief in output dir)
.venv/bin/python scripts/run_e2e.py --topic "..." --skip-knowledge
```

### Future integration (not yet wired)

whisper's `run_pipeline.py` does its own internal research synthesis. The
ideal future state: whisper accepts `--input research_brief.md` and skips its
internal research when present, using knowledge-pipeline's higher-quality
output instead. Today, the orchestrator just copies the brief into the whisper
output dir as context.

## Run the smoke tests

```bash
.venv/bin/python tests/smoke_parallel_search.py
.venv/bin/python tests/smoke_searxng.py
.venv/bin/python tests/smoke_github_client.py
.venv/bin/python tests/smoke_storage.py
.venv/bin/python tests/smoke_research_agent.py
.venv/bin/python tests/smoke_pipeline.py
.venv/bin/python tests/smoke_youtube.py
```

Total: 87 unit tests covering URL parsing, transcript extraction, prompt
construction, Qdrant indexing, SearXNG routing, end-to-end pipeline.

## Project layout

```
knowledge-pipeline/
├── config.py                       # env + PipelineConfig
├── DOCUMENTATION.md                # full architecture reference
├── USER_MANUAL.md                  # step-by-step user guide
├── scripts/
│   └── run_e2e.py                  # cross-pipeline orchestrator (NEW)
├── knowledge_pipeline/
│   ├── discovery/
│   │   ├── parallel_search.py       # Parallel.ai MCP + SDK clients
│   │   ├── searxng.py               # SearXNGClient + factory
│   │   └── youtube.py               # YouTubeTranscriptClient (NEW)
│   ├── source_code/
│   │   └── github_client.py         # GitHubClient + TOPIC_REPO_MAP
│   ├── storage/
│   │   ├── chunker.py               # paragraph-aware chunking
│   │   ├── embeddings.py            # EmbeddingClient (nomic-embed-text via Ollama)
│   │   └── qdrant_store.py          # QdrantStore (local mode)
│   ├── research/
│   │   ├── agent.py                 # ResearchAgent (Qwen3 via Ollama) + 13th section
│   │   └── openrouter_client.py     # QualityScore reviewer (json_object mode)
│   └── orchestration/
│       ├── pipeline.py              # run_research_pipeline()
│       └── cli.py                   # argparse CLI
└── tests/                          # 87 unit smoke tests
    ├── smoke_parallel_search.py
    ├── smoke_searxng.py
    ├── smoke_github_client.py
    ├── smoke_storage.py
    ├── smoke_research_agent.py
    ├── smoke_pipeline.py
    └── smoke_youtube.py
```

## The YouTube Coverage Delta (Section 13)

The unique deliverable. After indexing web sources + GitHub code, the
research agent writes section 13 of the brief:

> ## 13. YouTube Coverage Delta
> Based on the YouTube Coverage Context above, list the top 3–5 existing
> videos on this topic and what each emphasizes. Then for each, call out
> what it MISSES or gets shallow that this brief covers. End with a 1–2 sentence
> statement of the unique gap this brief fills that no YouTube video currently does.

This makes the brief demonstrably better than existing YouTube content — the
exact differentiator this pipeline exists to create.
