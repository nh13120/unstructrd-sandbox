# Unstructrd Sandbox

**An A/B testing harness that measures how well AI shopping assistants understand your brand — and how much better they get when handed structured product data.**

Point it at a brand, and it fires hundreds of realistic shopper questions at Claude, GPT, Gemini, and Perplexity, grades every answer with an LLM judge, and produces an **Agentic Readiness Score** with a self-contained HTML dashboard.

The core experiment is a two-agent comparison:

| | Agent A (baseline) | Agent B (Unstructrd) |
|---|---|---|
| Sees | Brand website via web search | Same website **plus** a structured product feed in context |
| Simulates | Today's AI assistants browsing your site | An assistant reading `unstructrd.yourbrand.com` |

The gap between A and B is what a machine-readable catalog would buy you.

---

## How it works

```
brand name + URL (+ optional ground-truth CSV)
        │
        ▼
 [1] Brand research ─── Gemini + Google Search learns positioning, flagship products, price range
        │
        ▼
 [2] Query matrix ───── ~10 categories × N queries each, brand-aware, seeded for reproducibility
        │
        ▼
 [3] Agent A / Agent B ─ every query × every model, async with per-provider rate limits
        │
        ▼
 [4] LLM judge ──────── scores each response (accuracy, hallucination, specificity, citations)
        │
        ▼
 [5] Scoring ────────── nDCG, bootstrap CIs, per-category and per-model breakdowns,
        │               cross-model consistency, token cost
        ▼
 [6] Report ─────────── HTML dashboard + raw JSON + plain-text transcript
```

### Two modes

- **Discovery mode** (no CSV) — the default. The tool researches the brand itself and judges answers by cross-referencing models against each other and against the live website. Also computes a **Reliability Score** across four dimensions: self-consistency (rephrased probes), cross-model convergence, source grounding, and specificity.
- **Ground-truth mode** (`--ground-truth products.csv`) — you supply the real catalog. Queries are generated from actual products, the judge grades against known-correct answers, and Agent B automatically gets a structured feed built from your CSV.

### Query categories

Each audit generates queries across these categories, grouped into two layers:

**Layer 1 — Discoverability**
`table_stakes` · `occasion_reasoning` · `blind_discovery` · `product_comparison` · `cross_platform_consistency` · `token_efficiency`

**Layer 2 — Brand DNA**
`brand_knowledge` · `adversarial` · `attribute_completeness` · `temporal_freshness`

Ground-truth mode adds `inventory_oos` and `multi_product_comparison`, which need real stock data to grade.

`blind_discovery` is the one category where the agent is *not* told the brand — it has to surface on its own merits from an open-ended question.

### Readiness levels

| Score | Level |
|---|---|
| ≥ 70% | competitive |
| 40–69% | at-risk |
| < 40% | critical |

---

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/nh13120/unstructrd-sandbox.git
cd unstructrd-sandbox
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` with whichever provider keys you have. Models without a key are skipped automatically.

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...        # also used by the judge and brand researcher
PERPLEXITY_API_KEY=pplx-...

MAX_CONCURRENCY=4             # per provider
QUERIES_PER_CATEGORY=10
BOOTSTRAP_ITERATIONS=1000
```

> `GOOGLE_API_KEY` is the one you really want: Gemini is the default judge (`JUDGE_MODEL`) and powers pre-audit brand research.

---

## Usage

### Web UI (recommended)

```bash
python3 interface.py
```

Launches a Gradio interface with a public share link. Remembers your last brand name and URL between runs.

### CLI

```bash
# Quick smoke test — one query per category, all four models
python3 run_audit.py \
  --brand-name "Charles & Keith" \
  --brand-url "https://www.charleskeith.com" \
  --models claude gpt gemini perplexity \
  --queries-per-category 1

# Full discovery audit with reproducible queries
python3 run_audit.py \
  --brand-name "Acme Outdoor" \
  --brand-url "https://acmeoutdoor.com" \
  --queries-per-category 10 \
  --seed 42

# Ground-truth A/B test from your own catalog
python3 run_audit.py \
  --brand-name "Acme Outdoor" \
  --brand-url "https://acmeoutdoor.com" \
  --ground-truth ground_truth/acme.csv

# Test against a live Unstructrd endpoint (hits /health, then pulls /feed)
python3 run_audit.py \
  --brand-name "Acme Outdoor" \
  --brand-url "https://acmeoutdoor.com" \
  --unstructrd-url "https://unstructrd.acmeoutdoor.com"
```

| Flag | Default | Purpose |
|---|---|---|
| `--brand-name` | required | Display name in the report |
| `--brand-url` | required | Site the agents browse |
| `--ground-truth` | — | CSV of real products; omit for discovery mode |
| `--structured-data` | — | JSON feed to hand Agent B (instead of fetching or building one) |
| `--unstructrd-url` | — | Live endpoint to fetch the feed from |
| `--models` | all four | Any of `claude gpt gemini perplexity` |
| `--queries-per-category` | 10 | Matrix size; `≥ 3` needed for reliability probes |
| `--probe-consistency` / `--no-probe-consistency` | on | Self-consistency probes (adds ~36 calls) |
| `--probes-per-category` | 1 | Probe queries per category |
| `--seed` | random | Fix for reproducible query generation |
| `--output-dir` | `output` | Where reports land |

A provider that fails three times in a row is dropped for the rest of the run rather than stalling everything.

### Ground-truth CSV format

Required columns:

```
product_id, product_name, price, currency, material, care_instructions,
return_policy, shipping_standard_days, shipping_express_days, occasion_tags,
warmth_score, waterproof_score, office_appropriate_score, non_existent_colors
```

Stock is read from any `stock_{SIZE}_{COLOR}` columns (e.g. `stock_M_black`). `non_existent_colors` seeds adversarial queries — colors the agent should say *don't* exist.

The code references `ground_truth/template.csv` as a starting point; `.gitignore` excludes `*.csv`, so create that directory locally.

---

## Output

Every run writes three files to `output/`:

| File | What it is |
|---|---|
| `{brand}_audit_{timestamp}.html` | Self-contained dashboard: readiness score, per-category and per-model breakdowns, top failures, layer scores, reliability |
| `{brand}_raw_{timestamp}.json` | Every query, response, judge verdict, and computed score |
| `{brand}_transcript_{timestamp}.txt` | Human-readable log of every prompt and answer |

---

## Project layout

```
run_audit.py            CLI entrypoint
interface.py            Gradio web UI
config.py               Env-driven settings, model IDs, per-provider pricing
src/
  brand_researcher.py   Pre-audit brand intelligence via Gemini + Google Search
  query_generator.py    Category templates, personas, discovery/ground-truth matrices, probes
  data_loader.py        CSV → Product objects; builds Agent B's structured feed
  agents/
    base_agent.py       Retry, timeouts, per-provider semaphores, token + latency capture,
                        provider-specific web search wiring
    agent_a.py          Website-only baseline
    agent_b.py          Website + structured feed in system prompt
  judge.py              LLM-as-a-judge, cross-model comparisons, catalog extraction
  scorer.py             nDCG, bootstrap CIs, precision/recall, consistency, cost
  reliability.py        Four-dimension reliability score for discovery mode
  reporter.py           HTML dashboard, raw JSON, transcript
```

`src/unstructrd_sandbox/` is an earlier package scaffold (thin provider wrappers, Pydantic config) that the current pipeline doesn't import. Safe to ignore or delete.

### Models

Default model IDs live in [config.py](config.py):

| Key | Model | Search backend |
|---|---|---|
| `claude` | `claude-sonnet-4-6` | Anthropic `web_search` tool |
| `gpt` | `gpt-5.2` | OpenAI Responses `web_search_preview` |
| `gemini` | `gemini-3.1-pro-preview` | Google Search grounding |
| `perplexity` | `sonar-reasoning` | Native |

Judge defaults to Gemini; override with `JUDGE_MODEL` in `.env`. Per-provider pricing in `config.py` drives the token-cost estimates in the report.

---

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

Line length is 100; `ruff` enforces `E`, `F`, `I`, `W`.
