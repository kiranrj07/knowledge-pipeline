# knowledge-pipeline — User Manual

A complete, copy-paste-ready guide to run the program end-to-end and get a research brief on any topic.

---

## TL;DR

```bash
source .venv/bin/activate
python -m knowledge_pipeline.orchestration.cli research \
    --topic "AWS VPC" \
    --output runs/aws-vpc/research_brief.md
```

That's it. Read the rest only if something breaks or you want to customize.

---

## 1. Prerequisites

You need **four** things running locally:

| Service | Default URL | How to verify |
|---|---|---|
| Ollama | http://127.0.0.1:11434 | `curl -s http://127.0.0.1:11434/api/tags \| grep qwen3` |
| SearXNG (Docker) | http://127.0.0.1:8888 | `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8888/` → should be `200` |
| GitHub PAT | — | `echo $GITHUB_TOKEN` should be non-empty |
| OpenRouter key (optional) | — | `echo $OPENROUTER_API_KEY` should be non-empty |

### Required Ollama models

```bash
ollama pull qwen3:14b
ollama pull nomic-embed-text
```

### Required Python packages

The `qwen3:14b` model (~9 GB) and `nomic-embed-text` (~274 MB) must already be present. Run `ollama list` to confirm.

### Required Python packages

```bash
pip install "parallel-web>=1.0.1" "youtube-transcript-api>=0.6.2"
```

---

## 2. One-time setup (already done for you)

If you're running this from the repo as cloned (github.com/kiranrj07/knowledge-pipeline), the following are already in place:

| Item | Where |
|---|---|
| SearXNG container | Running at `127.0.0.1:8888` (via Docker) |
| SearXNG JSON config | `searxng-data/settings.yml` with `json` format enabled |
| Ollama models | `qwen3:14b`, `nomic-embed-text` (via Ollama) |
| Local git repo | Initialized with 1 commit on `main` branch |
| All Python deps | Installed in `.venv/` |

If any of those aren't in place on your machine, follow section 2A below.

### 2A. If starting from scratch

```bash
# 1. Clone and enter the project
git clone https://github.com/kiranrj07/knowledge-pipeline.git
cd knowledge-pipeline
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install Python deps
pip install -e ".[parallel]"
pip install "youtube-transcript-api>=0.6.2"

# 3. Configure env
cp .env.example .env
# Then edit .env and fill in:
#   GITHUB_TOKEN=ghp_xxxxxxxxxxxx
#   OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxx

# 4. Start SearXNG (Docker)
docker pull searxng/searxng
docker run -d --name searxng --restart unless-stopped \
    -p 8888:8080 \
    -v $PWD/searxng-data/settings.yml:/etc/searxng/settings.yml:ro \
    -e SEARXNG_SECRET=changeme \
    searxng/searxng

# 5. Pull Ollama models
ollama pull qwen3:14b
ollama pull nomic-embed-text

# 6. Verify everything is up
curl -s http://127.0.0.1:8888/ -o /dev/null -w "SearXNG: %{http_code}\n"
curl -s http://127.0.0.1:11434/api/tags | python3 -c "import json,sys; print('Ollama:', [m['name'] for m in json.load(sys.stdin)['models']])"
```

---

## 3. Run the program

The CLI is `knowledge_pipeline.orchestration.cli`. Required argument: `--topic`.

### Basic run

```bash
source .venv/bin/activate
python -m knowledge_pipeline.orchestration.cli research \
    --topic "AWS VPC" \
    --output runs/aws-vpc/research_brief.md
```

### What happens during a run

```
[discovery] 4 URLs              ← SearXNG found 4 web hits for "AWS VPC"
[github]    4 source files     ← TOPIC_REPO_MAP / GitHub code search
[index]     371 chunks          ← chunked, embedded, stored in Qdrant
[output]    runs/.../research_brief.md
[review]    composite=2.20        ← OpenRouter quality score (optional)
[manifest]  runs/.../manifest.json
```

Total runtime: ~2-4 minutes depending on model load + Qwen synthesis speed.

### CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--topic` | required | The research topic |
| `--output` | required | Where to write `research_brief.md` |
| `--max-urls` | 4 | Cap on SearXNG discovery hits |
| `--max-github-files` | 4 | Cap on GitHub source-code files |
| `--collection` | topic slug | Qdrant collection name (overrides auto-slug) |
| `--review / --no-review` | `--review` | Enable/disable OpenRouter quality review |
| `--manifest` | (none) | Path to also write `manifest.json` |
| `--quiet` | verbose | Suppress per-stage progress lines |

### Examples

```bash
# Run with explicit manifest output
python -m knowledge_pipeline.orchestration.cli research \
    --topic "AWS VPC" \
    --output runs/aws-vpc/research_brief.md \
    --manifest runs/aws-vpc/manifest.json

# Run with deeper discovery
python -m knowledge_pipeline.orchestration.cli research \
    --topic "BGP route reflectors" \
    --output runs/bgp-rr/research_brief.md \
    --max-urls 8 --max-github-files 6

# Run without quality review (faster)
python -m knowledge_pipeline.orchestration.cli research \
    --topic "AWS VPC" \
    --output runs/aws-vpc/research_brief.md \
    --no-review
```

---

## 3.5. Run the full e2e pipeline (research → video)

The `scripts/run_e2e.py` orchestrator runs knowledge-pipeline first, then chains into the whisper video pipeline. Both pipelines live in separate repos; this script bridges them.

```bash
cd /home/janak/ai/knowledge-pipeline
.venv/bin/python scripts/run_e2e.py --topic "AWS VPC"
```

What happens:

1. knowledge-pipeline CLI produces `research_brief.md` + `manifest.json` (~3 min)
2. The brief is copied into the whisper output directory as input context
3. whisper's `run_pipeline.py` produces the final `final_video.mp4` via Graphviz diagrams + Piper TTS + Whisper STT + FFmpeg mux (~5-15 min depending on topic)

Options:

```bash
# Override output directory
.venv/bin/python scripts/run_e2e.py --topic "BGP route reflectors" \
    --output-dir runs/bgp-rr

# Enable Moondream diagram review in whisper
.venv/bin/python scripts/run_e2e.py --topic "eBPF XDP" --review-loops 2

# Skip knowledge-pipeline (use existing brief in output dir)
.venv/bin/python scripts/run_e2e.py --topic "..." --skip-knowledge

# Override venv or repo paths
.venv/bin/python scripts/run_e2e.py --topic "..." \
    --kp-root /path/to/knowledge-pipeline \
    --wp-root /path/to/whisper
```

Future integration: whisper's `run_pipeline.py` does its own internal research synthesis. The ideal future state is whisper accepting `--input research_brief.md` and skipping its internal research when present. Today the orchestrator just copies the brief into whisper's output dir as context.

---

## 4. Output files

After a successful run, the `--output` directory contains:

```
runs/aws-vpc/
├── research_brief.md              ← the 13-section brief
├── research_brief.md.quality.json  ← OpenRouter scores (if --review)
└── manifest.json                  ← full audit trail (if --manifest)
```

### `research_brief.md` structure

The brief has exactly **13 sections**:

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
## 13. YouTube Coverage Delta    ← THE USP
```

Section 13 is unique to this pipeline: it lists existing top YouTube videos on the topic and what each one misses that this brief covers.

### `manifest.json` structure

```json
{
  "topic": "AWS VPC",
  "collection": "aws-vpc",
  "discovered_urls": ["https://docs.aws.amazon.com/vpc/...", "..."],
  "github_files": [
    {"repo": "...", "path": "...", "title": "..."},
    {"repo": "...", "path": "...", "title": "..."}
  ],
  "youtube_videos": [
    {"url": "https://youtube.com/watch?v=...", "video_id": "...", "summary_chars": 245}
  ],
  "youtube_transcript_errors": [],
  "indexed_chunks": 371,
  "source_documents": 8,
  "output": "runs/aws-vpc/research_brief.md",
  "quality_score_path": "runs/aws-vpc/research_brief.md.quality.json"
}
```

### `quality.json` structure

```json
{
  "technical_accuracy": 3,
  "depth": 2,
  "uniqueness": 3,
  "troubleshooting_value": 2,
  "source_grounding": 3,
  "composite": 2.6,
  "ready_for_script": false,
  "rationale": "..."
}
```

---

## 5. Run a specific test (AWS VPC)

```bash
cd /home/janak/ai/knowledge-pipeline
source .venv/bin/activate

# Quick run (~3-4 min, includes quality review)
python -m knowledge_pipeline.orchestration.cli research \
    --topic "AWS VPC" \
    --output runs/aws-vpc/research_brief.md

# Check results
ls -la runs/aws-vpc/
head -50 runs/aws-vpc/research_brief.md
cat runs/aws-vpc/quality.json
```

Expected: A 200-400 line markdown file with sections covering AWS VPC concepts, source-code references (likely from `aws/amazon-vpc-resource-controller-k8s` or similar), and a section 13 comparing against existing AWS VPC YouTube tutorials.

---

## 6. Topic coverage map

The pipeline has curated source-code for these topics (hits them via TOPIC_REPO_MAP before falling back to generic GitHub search):

- **Networking**: TCP, IPv4, IPv6, IP, socket, epoll, iptables, nftables, conntrack, NAT, SYN
- **Routing**: BGP, OSPF, ISIS
- **Observability**: eBPF, Cilium
- **Orchestration**: kubernetes, k8s, CNI, Envoy

For **AWS VPC** specifically: not in TOPIC_REPO_MAP → falls back to GitHub generic search. SearXNG still finds authoritative AWS docs and YouTube still finds tutorials. The brief will be solid but the source-code grounding will be generic rather than AWS-specific.

If you want AWS VPC-specific source-code grounding, edit `knowledge_pipeline/source_code/github_client.py`:

```python
TOPIC_REPO_MAP["aws_vpc"] = [
    ("aws/amazon-vpc-resource-controller-k8s", "pkg/controllers/vpc/vpc_controller.go"),
    ("aws/amazon-vpc-cni-k8s", "pkg/ipamd/ipamd.go"),
    ("aws/aws-vpc-cni-k8s", "scripts/install-aws-vpc-cni.sh"),
]
```

(Insert before more generic keys like "socket" so first-match-wins picks it.)

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Connection refused: 8888` | SearXNG container not running | `docker ps` then `docker start searxng` |
| `Connection refused: 11434` | Ollama not running | `systemctl --user start ollama` or `ollama serve` |
| `404 from openrouter/free` | Rate-limited / down | Use `--no-review` or try `qwen/qwen-2-7b-instruct:free` |
| `Model 'qwen3:14b' not found` | Model not pulled | `ollama pull qwen3:14b` |
| Brief mentions generic TCP instead of topic | qwen3 hallucinating | Reduce `--max-urls` and `--max-github-files` to focus context |
| `ImportError: cannot import name 'YouTubeTranscriptClient'` | Stale build / wrong file | Re-pull from git or check `knowledge_pipeline/discovery/youtube.py` |
| `git push` fails with "Permission denied" | PAT lacks `repo` scope | Generate new PAT at github.com/settings/tokens with `repo` checked |

---

## 8. Verifying the program is connected end-to-end

Run this smoke check before any real topic:

```bash
# Test SearXNG
curl -s "http://127.0.0.1:8888/search?q=test&format=json" | python3 -c "import json,sys; d=json.load(sys.stdin); print('SearXNG OK:', len(d['results']), 'results')"

# Test Ollama
curl -s http://127.0.0.1:11434/api/tags | python3 -c "import json,sys; d=json.load(sys.stdin); print('Ollama OK:', [m['name'] for m in d['models']])"

# Test GitHub
curl -s -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/user | python3 -c "import json,sys; d=json.load(sys.stdin); print('GitHub OK:', d['login'])"

# Test OpenRouter
curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" https://openrouter.ai/api/v1/auth/key | python3 -c "import json,sys; d=json.load(sys.stdin); print('OpenRouter OK:', 'data' in d)"

# Test Qdrant (Python)
python3 -c "
from qdrant_client import QdrantClient
import tempfile, pathlib
with tempfile.TemporaryDirectory() as td:
    q = QdrantClient(path=td); q.close()
print('Qdrant OK')
"

# Test Python deps
python3 -c "import youtube_transcript_api; import parallel; from qdrant_client import QdrantClient; print('All Python deps OK')"
```

If all six checks pass, the pipeline is fully connected.

---

## 9. Next steps

- For a deep-dive session on a new topic: run with `--max-urls 8 --max-github-files 6 --manifest`
- For batch processing: write a shell loop over a list of topics
- For sharing with an external AI: point them at https://github.com/kiranrj07/knowledge-pipeline — DOCUMENTATION.md is rendered at the repo root
- For tuning the brief quality: edit `knowledge_pipeline/research/agent.py` `DEFAULT_SYSTEM_PROMPT` and `DEFAULT_USER_PROMPT_TEMPLATE`

---

## 10. Quick command card

| Task | Command |
|---|---|
| Activate env | `source .venv/bin/activate` |
| Run on a topic | `python -m knowledge_pipeline.orchestration.cli research --topic "..." --output runs/.../research_brief.md` |
| Run with manifest | add `--manifest runs/.../manifest.json` |
| Skip quality review | add `--no-review` |
| Deeper discovery | add `--max-urls 8 --max-github-files 6` |
| Inspect SearXNG | `curl -s "http://127.0.0.1:8888/search?q=test&format=json" \| jq .` |
| Inspect Ollama | `curl -s http://127.0.0.1:11434/api/tags \| jq -r '.models[].name'` |
| View the brief | `cat runs/<topic>/research_brief.md` |
| View the manifest | `cat runs/<topic>/manifest.json \| jq .` |
| Start SearXNG | `docker start searxng` (or run the `docker run` from section 2A) |

---

## 11. AWS VPC — what to expect

Given the current setup, a run on "AWS VPC" will produce:

- **Discovery (4 URLs)**: docs.aws.amazon.com/vpc, aws.amazon.com/vpc/features, plus 2-3 tutorial blog posts or Stack Overflow pages
- **Source-code (4 files)**: Generic GitHub search hits — NOT AWS-specific unless you add to TOPIC_REPO_MAP (see section 6)
- **YouTube (top 5)**: Likely tutorials from "Tech With Lucy", "Stephane Maarek", etc. — section 13 will list them with what each misses
- **Brief**: 13 sections covering VPC concepts, design choices, troubleshooting, plus a 13th "YouTube Coverage Delta" section showing what's unique vs existing videos

Total runtime: 2-4 minutes.
