# knowledge-pipeline — Complete Documentation

A local-first research pipeline that turns a **topic** into a structured **12-section research brief** grounded in real source code and web documentation, with a unique **YouTube comparison layer** that makes the output demonstrably better than existing YouTube content.

---

## 1. What it does

Given a free-form topic string, the pipeline:

1. **Discovers** authoritative web sources via **SearXNG** (self-hosted Docker)
2. **Fetches** YouTube video transcripts for the top search hits
3. **Grabs** curated source-code from GitHub via a topic-to-repo map (kernel sources, etc.)
4. **Indexes** all of the above into **Qdrant** (local mode, no Docker daemon)
5. **Synthesizes** a 12-section `research_brief.md` via **Qwen3:14b** through Ollama, using the YouTube transcripts as the 13th "YouTube Coverage Delta" section
6. **Reviews** quality via **OpenRouter** (free tier)

The result is a single `.md` brief that cites RFCs, kernel source files, and YouTube videos — with explicit coverage gaps vs existing YouTube content.

---

## 2. Architecture

```
                       ┌───────────────────────────────┐
                       │  topic (CLI arg)              │
                       └──────────────┬────────────────┘
                                      │
       ┌──────────────────────────────┼──────────────────────────────┐
       │                              │                              │
       ▼                              ▼                              ▼
┌──────────────┐              ┌──────────────┐        ┌─────────────────┐
│  SearXNG     │              │  TOPIC_       │        │  Qwen3:14b      │
│  (Docker)    │              │  REPO_MAP     │        │  via Ollama     │
│  discovery   │              │  → GitHub     │        │  synthesis      │
│              │              │  API          │        │                 │
│  web URLs +  │              │ source-code   │        │  12-section     │
│  excerpts    │              │ files         │        │  research brief │
└──────┬───────┘              └──────┬───────┘        └────────▲────────┘
       │                              │                         │
       │  ┌──────────────────────────┴────────────┐        │
       │  │  YouTubeTranscriptClient (NEW)        │        │
       │  │  youtube-transcript-api              │        │
       │  │  13th "YouTube Coverage Delta" sec   │────────┘
       │  └──────────────────────────┬────────────┘
       │                             │
       ▼                             ▼
┌──────────────────────────────────────────────────┐
│  Qdrant (local mode)                              │
│  chunk + embed + retrieve                          │
│  embedding model: nomic-embed-text via Ollama     │
└──────────────────────────┬───────────────────────┘
                           │
                           ▼
                  research_brief.md + manifest.json
                           │
                           ▼
              ┌─────────────────────────────┐
              │  OpenRouter (json_object)    │
              │  quality review (optional)    │
              └─────────────────────────────┘
```

### Three discovery / grounding sources

| Source | What it provides | When used |
|---|---|---|
| **SearXNG** (Docker, host:8888) | Web URLs + extracted excerpts | Always (primary discovery) |
| **TOPIC_REPO_MAP** | Curated GitHub source files | Topic matches a known repo (TCP→torvalds/linux, BGP→FRRouting, etc.) |
| **YouTube transcripts** | Auto-generated captions for top-N search hits | Always (fetches first N YouTube URLs from SearXNG results) |

### Three LLMs

| LLM | Role | When |
|---|---|---|
| **Qwen3:14b** (Ollama) | Synthesis (the brief itself) | Always |
| **Nomic-embed-text** (Ollama) | Embedding for Qdrant indexing + retrieval | Always |
| **OpenRouter** (free tier: `openrouter/free`) | Quality review (composite score 1-10) | Optional, enabled by default |

---

## 3. Repository layout

```
/home/janak/ai/knowledge-pipeline/
├── DOCUMENTATION.md                     # This file
├── README.md                            # Quick-start guide
├── pyproject.toml                        # Dependencies + optional extras
├── .env / .env.example                  # Environment config
├── searxng-data/settings.yml             # SearXNG config (JSON enabled)
│
├── knowledge_pipeline/
│   ├── __init__.py
│   ├── config.py                         # PipelineConfig dataclass + load_config()
│   │
│   ├── discovery/
│   │   ├── parallel_search.py            # ParallelMCPClient (SSE), ParallelSDKClient
│   │   ├── searxng.py                    # SearXNGClient + factory
│   │   └── youtube.py                    # YouTubeTranscriptClient (transcripts, format_for_prompt)
│   │
│   ├── source_code/
│   │   └── github_client.py              # GitHubClient + TOPIC_REPO_MAP
│   │
│   ├── storage/
│   │   ├── chunker.py                    # split_paragraphs (paragraph-aware chunking)
│   │   ├── embeddings.py                 # EmbeddingClient (nomic-embed-text via Ollama)
│   │   └── qdrant_store.py               # QdrantStore (local mode, in-process)
│   │
│   ├── research/
│   │   ├── agent.py                      # ResearchAgent (Qwen3 via Ollama)
│   │   └── openrouter_client.py          # QualityScore reviewer
│   │
│   └── orchestration/
│       ├── pipeline.py                   # run_research_pipeline() entry point
│       └── cli.py                        # argparse CLI
│
├── tests/
│   ├── conftest.py                       # sys.path setup
│   ├── smoke_parallel_search.py          # 18 tests
│   ├── smoke_searxng.py                  # 17 tests
│   ├── smoke_github_client.py            # 14 tests
│   ├── smoke_storage.py                  # 15 tests
│   ├── smoke_research_agent.py           # 11 tests
│   ├── smoke_pipeline.py                 # 12 tests
│   └── smoke_youtube.py                  # 14 tests (NEW)
│
└── runs/                                # Generated per-run output dirs
    └── <topic-slug>/
        ├── research_brief.md
        ├── research_brief.md.quality.json
        └── manifest.json
```

---

## 4. Installation

### Prerequisites

| Component | Minimum version | Notes |
|---|---|---|
| Docker | 24+ | For SearXNG |
| Ollama | 0.5+ | For Qwen3:14b and nomic-embed-text |
| Python | 3.11+ | Tested with 3.12 |
| OpenRouter key | free tier | For quality review (optional) |

### Setup steps

```bash
cd /home/janak/ai/knowledge-pipeline
python3.12 -m venv .venv
source .venv/bin/activate

pip install -e .
pip install youtube-transcript-api

# Start SearXNG via Docker (once)
docker pull searxng/searxng
docker run -d --name searxng --restart unless-stopped \
    -p 8888:8080 \
    -v $PWD/searxng-data/settings.yml:/etc/searxng/settings.yml:ro \
    -e SEARXNG_SECRET=changeme \
    searxng/searxng

# Pull required Ollama models (once)
ollama pull qwen3:14b
ollama pull nomic-embed-text

# Configure env (copy and edit)
cp .env.example .env
# Edit .env and fill in:
#   GITHUB_TOKEN=ghp_...
#   OPENROUTER_API_KEY=sk-or-v1-...
#   SEARXNG_URL=http://127.0.0.1:8888
```

### Optional dependency extras

```bash
pip install -e ".[parallel]"   # parallel-web SDK (for paid Parallel.ai tier)
```

---

## 5. Configuration (.env)

```ini
# Required
GITHUB_TOKEN=ghp_xxxxxxxxxxxx

# Required (for quality review)
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxx

# Optional with sensible defaults
SEARXNG_URL=http://127.0.0.1:8888
EMBEDDING_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://127.0.0.1:11434
RESEARCH_LLM=qwen3:14b
REVIEW_LLM=openrouter/free

# Discovery / retrieval knobs
MAX_DISCOVERY_URLS=4
MAX_GITHUB_FILES=4
CHUNK_SIZE_CHARS=1500
CHUNK_OVERLAP_CHARS=200

# SearXNG knobs
PARALLEL_API_KEY=
PARALLEL_MCP_URL=https://search.parallel.ai/mcp
```

---

## 6. CLI reference

```bash
source .venv/bin/activate
python -m knowledge_pipeline.orchestration.cli research \
    --topic "TCP SYN backlog" \
    --output runs/tcp-syn-backlog/research_brief.md \
    --max-urls 4 \
    --max-github-files 4
```

### Flags

| Flag | Default | Purpose |
|---|---|---|
| `--topic` | required | Free-form topic string |
| `--output` | required | Path to write `research_brief.md` |
| `--max-urls` | 4 | Cap on SearXNG discovery hits |
| `--max-github-files` | 4 | Cap on TOPIC_REPO_MAP files |
| `--collection` | topic slug | Qdrant collection name override |
| `--review / --no-review` | review | Enable/disable OpenRouter quality review |
| `--manifest` | (none) | Optional path to write manifest.json |
| `--quiet` | (verbose) | Suppress per-stage progress output |

### Output files

| File | Contents |
|---|---|
| `research_brief.md` | The 12-section structured brief |
| `research_brief.md.quality.json` | `{technical_accuracy, depth, …, ready_for_script, rationale}` (when review enabled) |
| `manifest.json` | `{topic, collection, discovered_urls, github_files, youtube_videos, indexed_chunks, source_documents, …}` |

---

## 7. Brief structure

The synthesized `research_brief.md` has exactly **thirteen sections**:

```
## 1. Executive Summary
## 2. Core Concept
## 3. Architecture
## 4. Internal Workflow
## 5. Packet / Data Flow
## 6. Engineering Decisions
## 7. Source-Code and Documentation References
## 8. Troubleshooting Methodology
## 9. Performance / Security Considerations
## 10. Common Misconceptions
## 11. YouTube Content Gap
## 12. Recommended Video Angle
## 13. YouTube Coverage Delta   ← NEW (USP section)
```

Section 13 is populated by the YouTube comparison layer: lists the top 3-5 existing YouTube videos on the topic and what each misses that this brief covers.

---

## 8. Module reference

### `knowledge_pipeline.config`
`PipelineConfig` dataclass + `load_config()`. Reads `.env`, validates required keys (GITHUB_TOKEN, OPENROUTER_API_KEY), fills defaults for everything else.

### `knowledge_pipeline.discovery.searxng`
- **`SearXNGClient`** — wraps the SearXNG JSON API at `base_url/search`
  - `.search(objective, search_queries, max_results, language, categories, ...)` — returns `SearchResponse` with ranked `SearchResult` items. Supports both query-only (`query=`) and Parallel-compatible (`objective + search_queries`) signatures.
  - `supports_fetch` is False (SearXNG returns snippets in search response)
- **`create_web_discovery_client(searxng_url, parallel_api_key, parallel_mcp_url)`** — factory:
  1. SearXNG if `searxng_url` set (preferred)
  2. Parallel.ai SDK if `parallel_api_key` set
  3. Parallel.ai MCP fallback if `parallel_mcp_url` set

### `knowledge_pipeline.discovery.youtube` ← NEW
- **`fetch_transcripts(urls, max_videos=5, summary_chars=500, languages=("en",))`** → `(list[YouTubeTranscript], list[YouTubeFetchError])`
- **`extract_video_id(url)`** — parses YouTube watch / youtu.be / embed / shorts URLs
- **`filter_youtube_urls(urls)`** — keeps only YouTube URLs, deduplicates, preserves order
- **`format_for_prompt(transcripts, max_chars_per_video=400)`** — compact "URL — excerpt" lines for LLM context
- Each `YouTubeTranscript`: `url`, `video_id`, `text` (full), `summary` (truncated for prompts)

### `knowledge_pipeline.source_code.github_client`
- **`GitHubClient(token, ...)`** — wraps GitHub REST API v3
  - `.search_code(query, max_results=10)` — returns `list[CodeMatch]`
  - `.get_file_contents(repo, path, ref=None)` — returns `FileContents`
  - `.find_topic_sources(topic, max_files=8, language="c")` — high-level helper
- **`TOPIC_REPO_MAP`** — curated dict mapping topic keywords to curated `(repo, path)` lists. First-match-wins ordering. Currently covers TCP, IPv6, IPv4, IP, TCP, SYN, socket, epoll, iptables, nftables, conntrack, NAT, BGP, OSPF, ISIS, eBPF, K8s, CNI, Cilium, Envoy.

### `knowledge_pipeline.storage`
- **`chunker.split_paragraphs(text, chunk_size, overlap)`** — paragraph-aware text chunking with overlap
- **`EmbeddingClient(model, base_url)`** — calls Ollama `/api/embeddings` (sync, one at a time)
- **`QdrantStore(path, collection, vector_dim)`** — local-mode Qdrant wrapper (no Docker daemon)
  - `.ensure_collection()`, `.upsert(vectors, chunks)`, `.search(query_vector, limit, source_type=None)`

### `knowledge_pipeline.research.agent`
- **`ResearchAgent(llm, embedding_client, qdrant_store, chunk_size, chunk_overlap, max_context_chunks, youtube_context=None, system_prompt, user_prompt_template)`** — index + synthesize
  - `.index_sources(topic, sources)` → `IndexedSummary`
  - `.build_synthesis_prompt(topic)` → str (for debugging / tests)
  - `.synthesize_brief(topic)` → str (the markdown body)
- `youtube_context` is injected into the synthesis prompt as a "YouTube Coverage Context" block, then surfaced in the prompt's section 13 instructions.

### `knowledge_pipeline.research.openrouter_client`
- **`OpenRouterClient(api_key, base_url, ...)`** — OpenRouter chat client
  - `.chat(model, messages, temperature, response_format, ...)` — OpenAI-compatible
  - `.review_quality(brief_text, topic, model)` → `QualityScore`
  - `QualityScore`: `technical_accuracy, depth, uniqueness, troubleshooting_value, source_grounding, ready_for_script, rationale` + `composite` property
- Uses **`response_format: {"type": "json_object"}`** (not `json_schema`) — free-tier models reject strict json_schema.

### `knowledge_pipeline.orchestration.pipeline`
- **`run_research_pipeline(topic, output_path, config, parallel_client, github_client, openrouter_client, embeddings, qdrant_store, llm, research_agent, enable_review, max_urls, max_github_files, collection_name, manifest_path, youtube_context)`** — full orchestrator. Returns `ResearchRunResult` with discovery, github, youtube counts + paths.
- **`_discover_and_extract(client, topic, max_urls)`** — handles SearXNG (no fetch) vs Parallel (search → fetch)
- **`_ground_with_github(client, topic, max_files)`** — TOPIC_REPO_MAP-aware fetcher
- **`_discover_and_extract_youtube(...)`** ← NEW: extracts YouTube URLs from discovery results, fetches transcripts, returns `youtube_context` string + `youtube_meta` list

### `knowledge_pipeline.orchestration.cli`
Argparse-based CLI. Subcommands: `research`. Exit codes: 0 success, 1 pipeline error, 2 usage error, 130 SIGINT.

---

## 9. Data flow (end-to-end)

1. **CLI parses args** → calls `run_research_pipeline(topic, output_path, ...)`
2. **Config + clients constructed** (SearXNG via factory, GitHubClient, QdrantStore, EmbeddingClient, OllamaGenerator)
3. **YouTube transcripts fetched** ← NEW: `fetch_transcripts(discovered_urls, max_videos=5)`
4. **Web discovery** via `_discover_and_extract(client, topic, max_urls)` → `(discovered_urls, web_sources)`
5. **GitHub grounding** via `_ground_with_github(github, topic, max_files)` → `code_sources`
6. **Qdrant indexing** of all sources
7. **ResearchAgent build_synthesis_prompt** — includes YouTube context block + 13th section instructions
8. **Qwen3:14b synthesis** → markdown body
9. **Write research_brief.md + quality.json + manifest.json**
10. **OpenRouter review** (if enabled) → quality score

---

## 10. Testing

Run all smoke tests individually:

```bash
.venv/bin/python tests/smoke_parallel_search.py
.venv/bin/python tests/smoke_searxng.py
.venv/bin/python tests/smoke_github_client.py
.venv/bin/python tests/smoke_storage.py
.venv/bin/python tests/smoke_research_agent.py
.venv/bin/python tests/smoke_pipeline.py
.venv/bin/python tests/smoke_youtube.py
```

Total: **87 tests, all pure-unit, no network required** (except the end-to-end `smoke_pipeline.py` which uses mocks for everything except Ollama calls).

---

## 11. End-to-end example

```bash
$ python -m knowledge_pipeline.orchestration.cli research \
    --topic "TCP SYN backlog" \
    --output runs/tcp-syn-backlog/research_brief.md \
    --max-urls 4 --max-github-files 4

[discovery] 4 URLs
[github]    4 source files
[index]     371 chunks across 8 sources
[output]    runs/tcp-syn-backlog/research_brief.md
[review]    composite=2.20 ready_for_script=False
[review]    runs/tcp-syn-backlog/research_brief.md.quality.json
[manifest]  runs/tcp-syn-backlog/manifest.json
```

### Output directory after a run

```
runs/tcp-syn-backlog/
├── research_brief.md
├── research_brief.md.quality.json
└── manifest.json
```

---

## 12. Known limitations

| Area | Status | Workaround |
|---|---|---|
| OpenRouter free tier | Some slugs 404 or reject structured output | Use `openrouter/free` (recommended) or `--no-review` |
| TOPIC_REPO_MAP | Static, hardcoded in `github_client.py` | Edit the dict; future: load from YAML |
| YouTube transcripts | Best-effort; some videos have transcripts disabled | Errors collected per-video; doesn't fail the pipeline |
| Qdrant retrieval | Simple top-k by similarity; no re-ranking | Acceptable for ~1k-chunk corpora |
| Brief synthesis quality | Limited by Qwen3:14b (free local model) | Quality review scores it; iterate prompts over time |
| Concurrent runs | Qdrant local-mode is single-process | Use multiple collections or external Qdrant server |

---

## 13. Quick recipes

### Run on a new topic
```bash
python -m knowledge_pipeline.orchestration.cli research \
    --topic "BGP route reflectors" \
    --output runs/bgp-rr/research_brief.md
```

### Run without quality review (faster)
```bash
python -m knowledge_pipeline.orchestration.cli research \
    --topic "..." --output ... --no-review
```

### Add a new topic to TOPIC_REPO_MAP
Edit `knowledge_pipeline/source_code/github_client.py`:

```python
TOPIC_REPO_MAP["envoy"] = [
    ("envoyproxy/envoy", "source/common/network/connection_impl.cc"),
    ("envoyproxy/envoy", "source/common/upstream/cluster_manager_impl.cc"),
]
```

(Insert before more generic keys like "socket" so first-match-wins picks the specific entry.)

### Run SearXNG manually for a query
```bash
curl -s "http://127.0.0.1:8888/search?q=ebpf+XDP&format=json&categories=general" \
    | python -c "import json,sys; print(len(json.load(sys.stdin)['results']))"
```

---

## 14. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Connection refused` on SearXNG | Container not running | `docker ps` then `docker start searxng` |
| `module 'youtube_transcript_api' has no attribute 'YouTubeTranscriptClient'` | Stale build | Re-check `pipeline.py` imports — the class was removed |
| OpenRouter returns 400 "No endpoints found" | Slug not in your free tier | Use `openrouter/free` or `--no-review` |
| Brief mentions generic TCP instead of the topic | qwen3:14b hallucinating | Add `index_sources` debugging; consider `temperature=0.1` |
| Qdrant `Collection not found` | First-run on empty corpus | `ensure_collection()` now handles this automatically |

---

## 15. Version history

| Version | Date | Notes |
|---|---|---|
| MVP-1 (knowledge) | initial | SearXNG + GitHub + Qdrant + Qwen3 + OpenRouter review |
| YouTube layer | follow-up | `youtube-transcript-api`, 13th "YouTube Coverage Delta" section |
