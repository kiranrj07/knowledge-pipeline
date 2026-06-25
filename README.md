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
Parallel.ai MCP  ──→ ranked authoritative URLs
  ↓
Crawl4AI / Extract API  ──→ clean Markdown
  ↓
GitHub MCP  ──→ relevant kernel / library source files
  ↓
Qdrant (local mode)  ──→ chunked + embedded store
  ↓
Qwen3:14b (Ollama)  ──→ research_brief.md
  ↓
OpenRouter (escalation review)  ──→ quality_score.json
```

The point: **primary research must come from RFCs, official docs, and source
code.** YouTube transcripts only exist to find gaps, never as a source of
truth.

## Setup

```bash
cd /home/janak/ai/knowledge-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Edit .env: fill in GITHUB_TOKEN and OPENROUTER_API_KEY

# Ensure Ollama is running and the embedding model is pulled:
ollama pull nomic-embed-text
```

## Run MVP-1

```bash
python -m knowledge_pipeline.orchestration.cli research \
    --topic "TCP SYN backlog" \
    --output runs/tcp-syn-backdrop/research_brief.md
```

The output is a `research_brief.md` that the existing `whisper/` pipeline can
ingest via its `--research-brief` flag (TODO: wire up in whisper).

## Project layout

```
knowledge_pipeline/
├── config.py              # env + PipelineConfig
├── discovery/             # Parallel.ai MCP → URL list
├── source_code/           # GitHub client → source files
├── storage/               # Qdrant local-mode + embeddings
├── research/              # Qwen agent + OpenRouter reviewer
└── orchestration/         # end-to-end CLI
```
