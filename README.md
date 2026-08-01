# AI-Powered Suspicious Activity Detection — AML Agent

An agentic system for AML compliance: a natural-language query goes in, the agent parses intent,
**builds a query-specific execution plan** (not a fixed pipeline), calls only the tools that plan needs,
and returns risk-scored, explained, escalation-tagged flags — with the plan itself shown to the user, so
a reviewer can see exactly what the agent decided to do and why.

Built for a 48-hour hackathon, Problem Statement 1 (AI-Powered Suspicious Activity Detection).

---

## Table of contents

1. [Problem statement](#problem-statement)
2. [Domain background](#domain-background)
3. [Solution approach](#solution-approach)
4. [Why this is agentic](#why-this-is-agentic)
5. [Architecture](#architecture)
6. [Repo structure](#repo-structure)
7. [Tech stack](#tech-stack)
8. [Datasets](#datasets)
9. [Setup](#setup)
10. [Usage — example queries](#usage--example-queries)
11. [Results](#results)
12. [Limitations](#limitations)
13. [Team](#team)

---

## Problem statement

Traditional rule-based AML systems generate excessive false positives, overwhelming compliance teams.
Sophisticated laundering techniques — structuring, smurfing, layering — evade naive threshold rules. The
challenge: build an autonomous agent that parses a compliance query, dynamically decides which analysis
tools it needs, detects suspicious patterns (rule-based, ML, or hybrid), scores risk, explains every flag
in plain language, and recommends an escalation action — reducing false positives while staying
explainable enough for a human analyst to trust and act on.

## Domain background

- **The $10,000 threshold.** US banks must file a **Currency Transaction Report (CTR)** for any cash
  transaction ≥ $10,000 (Bank Secrecy Act). **Structuring** (31 U.S.C. § 5324) is the crime of splitting
  transactions to stay under that threshold — independent of whether the underlying funds are illicit.
  This is *why* our structuring rule watches the $9,000–$9,999.99 band specifically, not "large
  transactions" generally.
- **Smurfing** ("fan-out"): distributing funds through many accounts/couriers to obscure the money trail.
- **Layering**: moving funds through a chain of accounts/jurisdictions to sever the audit trail —
  laundering's obfuscation stage.
- **Rapid cash-out**: converting an inbound electronic transfer to physical cash quickly — laundering's
  integration stage.
- **FATF 40 Recommendations** (1, 3, 10) require enhanced due diligence on exactly these patterns.
- **SARs** (Suspicious Activity Reports, FinCEN Form 114) are filed regardless of amount when laundering
  is suspected — our agent drafts one automatically for every `HIGH`-risk flag.

Full regulatory citations, per-rule thresholds, and business justification: **[AML_LOGIC.md](docs/AML_LOGIC.md)**.

## Solution approach

**Hybrid detection = rule-based (explainable, precise) + ML anomaly detection (recall, catches novel
patterns) + a fused risk score.**

- **Rules R1–R7** (structuring, smurfing, layering, rapid cash-out, velocity spike, dormant
  reactivation, and receiver-side structuring)
  — each with a documented regulatory rationale and threshold, emitting rule-specific evidence.
- **ML**: IsolationForest + LocalOutlierFactor over per-customer AML features (rolling sums,
  threshold-proximity ratio, self-deviation z-scores, velocity, pass-through ratios, ...).
- **Fusion** (`docs/CONTRACTS.md` Contract 5): `risk_score = 100 × (0.6 × rule_weight + 0.4 × ml_percentile)`,
  banded into `HIGH → report` / `MEDIUM → review` / `LOW → monitor` / `NONE → no_action`.
- **Explanation**: a deterministic template built from each rule's actual evidence (always accurate,
  always available, LLM-optional). An LLM polish pass rewrites `HIGH`-risk explanations into an
  analyst-facing paragraph — capped to `HIGH` only, both to protect a free-tier rate limit and because
  those are the flags that matter most (they're the ones getting a SAR draft).

## Why this is agentic

The system is graded on **not** being a fixed pipeline. The planner (`backend/agent/planner.py`) builds a
genuinely different tool sequence per query intent — verified by an automated test
(`tests/test_planner.py`, `tests/test_integration.py`) that asserts the plans for these three queries
*differ*:

| Query | Tools invoked | Tools explicitly skipped (and why) |
|---|---|---|
| *"Is customer 4521 suspicious?"* | `load_data → filter_data → entity_lookup → feature_engineer → rule_detect → ml_detect → risk_classify` | `eda_profile` (not exploring) |
| *"Show transaction distribution by country"* | `load_data → filter_data → eda_profile` | `feature_engineer`, `rule_detect`, `ml_detect`, `risk_classify` (no detection requested) |
| *"Which customers made 10+ transactions under $10,000?"* | `load_data → filter_data → aggregate_query` | `feature_engineer`, `ml_detect`, `eda_profile` (a deterministic count answers this exactly) |
| *"Analyse this dataset for suspicious activity"* | `load_data → eda_profile → feature_engineer → rule_detect → ml_detect → risk_classify` | — (full sweep) |

> `ml_detect` used to be skipped for the single-entity query too, on the reasoning that one customer is
> too small a sample to rank. That reasoning didn't match the plan it was in: `feature_engineer` there
> runs across the *whole* population, so `ml_detect` receives all 270 customers. Skipping it zeroed the
> ML half of the risk formula, and C-STR02 came back **51.00 MEDIUM** when asked about directly versus
> **89.84 HIGH** in a full sweep. The sample-size guard belongs on data size, not intent name — and it
> already exists twice at runtime (see the re-planning rules below).

The executor also **re-plans mid-run**, not just at planning time:
- `rule_detect` returns 0 hits → appends `ml_detect` to widen the net
- filtered subset < 50 rows → drops a queued `ml_detect` (insufficient sample)
- `filter_data` returns 0 rows → stops early with an explanatory summary, not an empty crash

Every decision — what ran, what was skipped, what got added mid-run, and why — is logged to
`plan.decisions[]` / `plan.tools_considered_but_skipped[]` and shown directly in the UI's execution-plan
trace panel. That panel, not the ML, is the thing this project is actually graded on.

Full intent → tool mapping table: **[docs/CONTRACTS.md](docs/CONTRACTS.md) Contract 4**.

## Architecture

```mermaid
flowchart LR
    Q(["Natural-language query"]) --> P

    subgraph AGENT ["Agent core"]
        direction TB
        P["Parse intent"] --> B["Build a plan for this query"]
        B --> E["Execute · re-plan mid-run"]
        E --> N["Score · explain · draft SAR"]
    end

    E <-->|only the tools the plan needs| T["9 tools<br/>load · filter · features<br/>rules · ML · risk"]
    N --> R(["Risk-scored flags<br/>and the plan that produced them"])
```

**Full technical documentation — architecture, analysis algorithms, and UI design in one place:
[DOCUMENTATION.md](docs/DOCUMENTATION.md).**

Component detail, the full Pydantic contract, and sequence diagrams: **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**.
The frozen interface both halves of this project are built against: **[docs/CONTRACTS.md](docs/CONTRACTS.md)**.
The two-person parallel build plan (for context on how this repo came together):
**[WORKPLAN.md](docs/WORKPLAN.md)**.

## Repo structure

```
soc/
├── backend/
│   ├── main.py                 # FastAPI app — POST /query, GET /health
│   ├── config.py                # env-driven settings (AML_USE_MOCKS, AML_DATASET_PATH, ...)
│   ├── schemas.py                # FROZEN — Pydantic contract (QueryIntent, ExecutionPlan, AgentResponse, ...)
│   ├── agent/
│   │   ├── intent_parser.py     # NL query → QueryIntent (LLM primary, regex fallback)
│   │   ├── planner.py           # QueryIntent → ExecutionPlan (query-specific tool sequence)
│   │   ├── executor.py          # runs the plan, threads ToolContext, conditional re-planning
│   │   ├── narrator.py          # explanation + escalation text per flag
│   │   └── registry.py          # auto-discovers @tool-decorated functions
│   ├── llm/
│   │   └── client.py             # provider-agnostic adapter (Gemini/OpenAI/Groq/Ollama) + regex fallback
│   └── tools/
│       ├── base.py               # FROZEN — ToolContext, ToolResult, @tool decorator
│       ├── data_loader.py        # Kaggle/synthetic CSV → canonical schema
│       ├── filters.py, aggregate.py, entity.py, eda.py
│       ├── features.py           # per-customer AML feature engineering
│       ├── rules.py              # R1–R7 rule-based detectors
│       ├── ml_detect.py           # IsolationForest + LocalOutlierFactor
│       └── risk.py                # rule + ML score fusion → HIGH/MEDIUM/LOW/NONE
├── frontend/
│   ├── app.py                    # Streamlit entry point
│   └── components/
│       ├── plan_trace.py         # execution-plan trace panel
│       ├── flag_cards.py         # flagged-entity cards + SAR draft
│       ├── charts.py              # Plotly visualisations
│       └── theme.py               # shared styling
├── data/
│   ├── sample/aml_sample.csv     # committed synthetic demo dataset (labelled ground truth)
│   ├── generate_synthetic.py     # synthetic dataset generator (fixed seed)
│   ├── build_ibm_cache.py        # optional: real IBM Kaggle dataset → canonical schema
│   └── adapters/                  # per-source schema adapters
├── tests/                          # pytest — planner, executor, rules, ML, no-label-leakage, API, ...
├── docs/                            # all project documentation
│   ├── CONTRACTS.md               # FROZEN — the interface Track A and Track B build against
│   ├── DOCUMENTATION.md            # full technical documentation (architecture + algorithms + UI)
│   ├── ARCHITECTURE.md             # agent design detail and component sequences
│   ├── AML_LOGIC.md                # rule definitions, thresholds, regulatory justification
│   ├── DATA_CARD.md                # dataset sources, schema, preprocessing decisions
│   ├── WORKPLAN.md                 # the two-person parallel build plan
│   └── TRACK_A_*.md, ANTI_HALLUCINATION_*.md, OLLAMA_SETUP_MAC.md
├── run_demo.py                     # starts backend + frontend, opens the browser
├── requirements.txt / requirements-data.txt
└── README.md, CLAUDE.md            # the only markdown kept at repo root
```

## Tech stack

| Layer | Choice |
|---|---|
| Backend | FastAPI + uvicorn, Pydantic v2 |
| Agent core | Python — intent parser, planner, executor, narrator (no agent framework; the plan/execute/re-plan loop is hand-rolled and fully inspectable) |
| LLM | Gemini, OpenAI, Groq, or **Ollama (local, no key/quota)** — one adapter, always with a regex fallback |
| Data / detection | pandas, numpy, scikit-learn (IsolationForest, LOF), networkx (layering chains) |
| Frontend | Streamlit + Plotly |
| Tests | pytest — 190+ tests |

## Datasets

| Dataset | Role | Source | License / citation |
|---|---|---|---|
| **IBM Transactions for AML** (HI-Small) | Primary real-world base | [Kaggle: ealtman2019/ibm-transactions-for-anti-money-laundering-aml](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) | Altman, Baeck, Gerlach — "Realistic Synthetic Financial Transactions for Anti-Money Laundering Models," NeurIPS 2023 Datasets and Benchmarks |
| **Synthetic overlay** (`data/sample/aml_sample.csv`) | Original committed demo dataset — guarantees structuring/smurfing/layering/rapid-cashout patterns are present and labelled, no Kaggle download required to run the demo | `data/generate_synthetic.py`, fixed seed (42) | Ours — full schema, field definitions, and generation logic documented in [DATA_CARD.md](docs/DATA_CARD.md) |
| **Alt-schema synthetic dataset** (`data/sample/aml_sample_alt.csv`, 1,710 transactions / 294 customers) | **Default dataset the live agent actually queries** (`load_data`'s `source` parameter defaults to `"synthetic_alt"` — see `backend/tools/data_loader.py`). Same laundering typologies as the original synthetic set, but generated with a deliberately different raw schema (renamed headers, coded enums, `ACC-`-prefixed account IDs, no `is_cross_border` column) to prove the canonical-schema adapter (`docs/CONTRACTS.md` Contract 0) generalises to a raw format it wasn't hand-fit to, rather than being hardcoded to one CSV's column names | `data/generate_synthetic_alt.py`, fixed seed | Ours |

All three are adapted into one canonical schema (`docs/CONTRACTS.md` Contract 0) before any detection code
touches them — the datasets are fully swappable via `load_data(source=...)` (`'ibm'`, `'ibm_stratified'`,
`'synthetic'`, or `'synthetic_alt'`). Full field-by-field preprocessing decisions, raw dataset statistics,
and every assumption made by each synthetic generator: **[DATA_CARD.md](docs/DATA_CARD.md)**.

## Setup

```bash
git clone <this repo>
cd soc

python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt

cp .env.example .env            # optional — see below

python run_demo.py              # starts backend (FastAPI) + frontend (Streamlit), opens the browser
```

The committed sample dataset (`data/sample/aml_sample.csv`) means the demo runs **with no Kaggle download
and no LLM API key** — `AML_USE_MOCKS=0` (the default in `.env.example`) points at the real detection
pipeline over that sample data; without an LLM key, intent parsing and explanations use the
always-available regex/template fallback, not a degraded mode.

**To use a real LLM** (optional, improves messy/informal query phrasing and polishes HIGH-risk
explanations): set `LLM_PROVIDER` and the matching key in `.env` (`gemini`, `openai`, or `groq`). Free-tier
rate limits are respected by design — LLM calls happen at most once per query for intent parsing (and are
response-cached, so re-running the same query during a demo costs nothing further), and explanation
polishing is capped to `HIGH`-risk flags only (a `full_analysis` query can produce dozens of flags;
polishing all of them would exhaust a free-tier quota on a single request for no benefit, since the
template text is already accurate).

**To use a local LLM instead (no key, no quota at all)**: set `LLM_PROVIDER=ollama` after installing
[Ollama](https://ollama.com/download) and pulling a model (`ollama pull qwen2.5:7b-instruct`). See
[OLLAMA_SETUP_MAC.md](docs/OLLAMA_SETUP_MAC.md) for Mac-specific setup; the same `LLM_PROVIDER=ollama` works
identically on Windows/Linux once Ollama is installed there.

**To download the real IBM Kaggle dataset instead of the synthetic one**, `kaggle`/`kagglehub` credentials
are required — see [DATA_CARD.md](docs/DATA_CARD.md) §1.1.

**Manual start** (equivalent to `run_demo.py`, useful for separate terminals / debugging):
```bash
uvicorn backend.main:app --port 8000       # terminal 1
streamlit run frontend/app.py               # terminal 2 (reads AML_API_URL, defaults to localhost:8000)
```

**Run the test suite:**
```bash
pytest tests/ -v
```

## Usage — example queries

Type a query, or click one of the UI's example buttons. Each of these exercises a different point in the
intent → plan mapping table above:

| Query | Intent | What you'll see |
|---|---|---|
| `Analyse this dataset for suspicious activity` | `full_analysis` | Full EDA + rule + ML sweep; dozens of flags across risk bands |
| `Find structuring patterns in the last 30 days` | `pattern_search` | Only structuring-scoped features/rules run; date filter applied (anchored to the dataset's own date range, not wall-clock "today") |
| `Which customers made 10+ transactions under $10,000?` | `threshold_query` | Direct aggregation, **no ML step at all** — visibly absent from the plan trace |
| `Is customer 4521 suspicious?` | `entity_investigation` | Single-entity scoring; a bare number is resolved against the real dataset's customer IDs (which aren't purely numeric — see [Limitations](#limitations)) |
| `Top 5 highest-risk customers` | `ranking` | Full sweep, truncated to the top 5 by risk score |
| `Show transaction distribution by country` | `eda` | Profiling only — no detection tools run |
| `Why was customer C-STR02 flagged?` | `explain_flag` | Scores just that one entity and returns its explanation directly |

Every response includes: the detected intent + extracted filters/entities, the full execution plan (steps
taken, steps skipped and why, any mid-run re-planning), the flagged entities with risk score/level/escalation/
explanation, and (for `HIGH` risk) a SAR draft.

## Results

Rule thresholds and their regulatory justification are documented per-rule in
[AML_LOGIC.md](docs/AML_LOGIC.md) — e.g. R1 (structuring) requires **3 transactions in a 7-day window** in the
$9,000–$9,999.99 band, which is what separates it from a naive "flag any transaction over $9,000" rule
(the latter would flag every legitimate large transaction; ours requires a *pattern*, corroborated further
by the ML anomaly score before reaching `HIGH`/SAR territory — see Contract 5's fusion formula).

### Quantitative validation

Computed against the original synthetic dataset's ground truth (`data/sample/aml_sample.csv`'s
`label_is_laundering` field — 202 of 2,002 transactions, injected by the generator across the
structuring/smurfing/rapid-cashout/layering cohorts; see [DATA_CARD.md](docs/DATA_CARD.md)). Not validated
against the raw IBM Kaggle dataset — that requires a Kaggle download not run in this environment; the
synthetic set is the labelled ground truth actually available here.

Every number below is **generated, not hand-written** — regenerate with:

```bash
python -m evaluation.run_evaluation
```

> **Note:** `load_data` defaults to the alt-schema synthetic dataset (`source="synthetic_alt"`, see
> [Datasets](#datasets)) for live queries. The evaluation pins `source="synthetic"` explicitly, because
> the labelled ground truth lives in `aml_sample.csv`; scoring flags from one dataset against labels from
> the other silently compares two different customer populations, and the harness now fails loudly rather
> than reporting it.

**Methodology**: our system flags *customers*, not individual transactions, so transaction labels must be
lifted to customer level. There are two defensible ways to do that, and they answer different questions,
so both are reported:

- **Sender-side** — positive if the customer *sent* at least one labelled transaction (51 of 270). This
  matches what most rules look at.
- **Broader** — positive if they sent *or received* one (114 of 270). The extra 63 are receive-only
  participants, e.g. the destination accounts in a structuring scheme.

The naive baseline ([AML_LOGIC.md](docs/AML_LOGIC.md) §6: "flag any transaction with `amount > $9,000`") is
translated the same way, to a fair customer-level comparison.

**Sender-side ground truth**

| | Flagged | Precision | Recall | False-positive rate |
|---|---|---|---|---|
| **Naive baseline** (any txn > $9,000) | 259 / 270 | 0.197 | 1.000 | 0.950 |
| **Our system — any flag** (LOW/MEDIUM/HIGH) | 41 / 270 | 0.561 | 0.451 | 0.082 |
| **Our system — HIGH only** (the SAR-draft tier) | 27 / 270 | 0.778 | 0.412 | 0.027 |

**Broader ground truth (sender or receiver)**

| | Flagged | Precision | Recall | False-positive rate |
|---|---|---|---|---|
| **Naive baseline** (any txn > $9,000) | 259 / 270 | 0.421 | 0.956 | 0.962 |
| **Our system — any flag** (LOW/MEDIUM/HIGH) | 41 / 270 | 0.854 | 0.307 | 0.038 |
| **Our system — HIGH only** (the SAR-draft tier) | 27 / 270 | 0.926 | 0.219 | 0.013 |

The naive rule "catches everything" by flagging 96% of all customers — exactly the
compliance-team-drowning-in-false-positives problem the brief describes. Our system flags **6.3× fewer
customers** (41 vs. 259) at a **12× lower false-positive rate**, and under the broader ground truth
reaches **0.854 precision** — while the naive rule manages 0.421.

**Why one table looks worse than the other.** R7 (receiver-side structuring) flags accounts that *receive*
repeated sub-threshold deposits. Those customers are positives under the broader definition and
**negatives under the sender-side one**, so the same 11 flags read as true positives in one table and
false positives in the other. That is a property of the ground truth, not of the detector: flagging the
beneficiary account of a structuring scheme is correct AML practice, and the sender-side definition simply
cannot credit it. Before R7, sender-side precision was 0.793 and broader recall was 0.219.

**The receiver-side gap is structural, and mostly cannot be closed.** 63 customers appear only as
receivers of labelled transactions. R7 recovers 12 of them with no measured false positives. The other 51
are not reachable by any inbound rule, and we checked rather than assumed: a classic fan-in ("funnel
account") rule has no discriminative power on this data, because the receive-only positives average **7.6
distinct inbound counterparties against a population average of 6.9**, and in any 48-hour window both top
out at 4. There is no separation to threshold on. 26 of the 51 receive exactly one labelled transaction —
indistinguishable from being an ordinary counterparty of a bad actor.

**R5 (velocity) and R6 (dormant reactivation) never fire on this dataset (0 hits each), for two
different measured reasons — neither of them threshold tuning.**

R6 is inapplicable: the dataset spans 89 days, and R6 needs a 60-day dormancy gap followed by a 7-day
burst with at least 3 pre-gap transactions for its z-score. Only 2 of 268 senders have a gap that long
(the largest anywhere is 64.5 days), 1 clears the burst gate, and 0 clear the z-score. Relaxing the
dormancy threshold does not rescue it — a 30-day gap admits 108 of 268 senders, which is ordinary
transaction cadence, not dormancy. The rule is correct; this data has no dormancy typology in it.

R5 was, until this was fixed, unreachable by construction: `velocity_txns_per_hour` was computed as
(max count in any 24h window) ÷ 24 — a daily average wearing an hourly name — so the documented bar of
2.0 txns/hour silently meant *48 transactions inside one 24h window*. The busiest sender in the dataset
has 25 transactions in total, so the observed maximum was 0.542 and no threshold could ever have fired
the rule. The feature now computes a true peak 1-hour rate. R5 still fires on nobody, and that is now a
measured fact rather than an artifact: the corrected rate admits 15 senders at gate 1 (12 of them
labelled positives), and all 15 fail R5's second gate — the highest self-deviation z-score among them is
2.29 against a threshold of 3.0. Dropping that gate would add 2 true positives and 3 false positives, a
losing trade, so it stands.

Correcting the unit changed the ML feature matrix and therefore the published numbers: sender-side
precision fell from 0.590 to 0.561 as two ML-only negatives crossed the 0.95 percentile floor into the
LOW band. Recall, the HIGH tier, and every rule-driven flag are unchanged. The regression is reported
rather than tuned away — the alternative was shipping a feature that contradicts its own name and
[AML_LOGIC.md](docs/AML_LOGIC.md) §3 R5 in order to protect a metric.

## Limitations

- **LLM path is provider-agnostic (Gemini, OpenAI, Groq, or local Ollama) and always has a working
  fallback**; live-verified against real Gemini and Groq keys — correctly classifies messy/slang phrasing
  the regex fallback alone gets wrong (e.g. "who r my 3 sketchiest customers rn" → `ranking`, `top_n=3`).
  LLM providers often return relative-date shorthand ("-30d", "1 month ago") instead of ISO dates for
  phrases like "last 30 days" — `intent_parser._coerce_relative_date()` resolves these against the
  dataset's own reference date rather than failing validation. Free-tier cloud quotas are small enough
  that repeated identical testing can exhaust them in one session — `complete_json()` caches successful
  completions (not failures) to avoid burning quota on repeat queries during rehearsal.
- **Entity-ID matching is numeric-only.** The real dataset's customer IDs follow the generator's own
  scheme (`C-N0001`, `C-STR02`, `C-HUB01`) rather than plain numbers. A query like "customer 2" resolves
  by matching digits against real IDs (picking the first match on ambiguity, which does occur — multiple
  IDs can share a numeric suffix across different prefixes, so it may occasionally resolve to the wrong
  one of several candidates); a query using a real ID directly (e.g. "C-STR02") always works precisely.
  There's no name-based lookup.
- **`explain_flag` re-scores the entity fresh** rather than reusing a cached prior run — simpler and
  always correct, but means it can't explain a flag from a run using different filters than "all data."
- **The ML term is deliberately blind to the query window.** Anomaly percentiles are ranked against the
  full customer population, fixed, rather than against whichever customers survived the analyst's
  filters. That is what makes a risk band mean the same thing in every query — a threshold that decides
  SAR escalation cannot move because someone added `amount_min` to a search. The cost is real: a customer
  who looks unremarkable across the whole dataset but spikes inside a narrow window no longer stands out
  on the ML half of the score. The rules still run on the filtered frame and still catch them. Before
  this was fixed, an `amount_min=5000` filter shifted percentiles by up to 0.73 and moved four customers
  across a risk band; the same filters now produce zero drift, pinned by
  `tests/test_ml.py::TestPercentileReferencePopulation`.
- **Detection is overwhelmingly sender-side, and the remaining receiver-side gap is structural.** R7 is
  the one receiver-keyed rule; every other rule and all 17 features evaluate outbound behaviour. Of the 63
  customers who appear only as receivers of labelled transactions, R7 recovers 12 and 51 remain
  unreachable. That is not a missing-rule problem: we tested the obvious fix and it fails. A classic
  fan-in ("funnel account") rule cannot discriminate here — receive-only positives average 7.6 distinct
  inbound counterparties versus a population average of 6.9, and in any 48-hour window both peak at 4.
  26 of the 51 receive exactly one labelled transaction, which is not distinguishable from being an
  innocent counterparty. Closing the rest would need a signal this dataset does not contain — account
  ownership, KYC linkage, or device/IP overlap.
- Batch analysis over a sample dataset, not live streaming — explicitly in scope per the brief.
- Synthetic data documents its own generation assumptions (seed, thresholds, ring sizes) in
  [DATA_CARD.md](docs/DATA_CARD.md) — real-world deployment would need those revalidated against production
  transaction volumes and patterns.

## Team

- **Track A** (agent core, orchestration, API) — Kapilan Kathirvel
- **Track B** (data, detection, ML, UI) — Vasudevan Kalyan
  

Full division of labour, ownership matrix, and the anti-merge-conflict protocol used to build this in
parallel: **[WORKPLAN.md](docs/WORKPLAN.md)**.
