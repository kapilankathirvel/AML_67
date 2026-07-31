# WORKPLAN — Team Delegation & Zero-Conflict Protocol

**Project:** AI-Powered Suspicious Activity Detection (AML Agent) — 48h hackathon, Problem Statement 1
**Team size:** 2, both driving coding agents, in parallel, on one repo.

> **If you were just sent this file:** read §1 (your track), §2 (the rules that stop us breaking each
> other's code), §3 (paste-ready instruction for your coding agent), then §6 (your hour-by-hour tasks).
> The interface you code against is in [docs/CONTRACTS.md](CONTRACTS.md) — that's the only other file
> you need.

---

## 0. What we're building, in four sentences

A user types a natural-language question ("Find structuring patterns in the last 30 days", "Is customer 4521
suspicious?"). An **agent** parses the intent, **builds a different execution plan per query**, calls only
the tools that query needs, and returns risk-scored flags with plain-English explanations and an escalation
action (monitor / review / report). A Streamlit UI shows the query, the plan the agent chose, *what it
skipped and why*, and the flagged entities. Batch analysis over a sample dataset — no streaming.

The thing that wins this hackathon is **the visible dynamic plan**, not the ML. Judges will test whether
two different queries produce two different plans. Protect that above everything.

---

## 1. Track assignment

| | Track A — Agent Core & API | Track B — Data, Detection & UI |
|---|---|---|
| **Owner** | _________________ | _________________ |
| **Branch** | `feat/agent-core` | `feat/data-ml-ui` |
| **Owns** | intent parsing, planner, executor, narrator, LLM adapter, FastAPI, mocks | dataset + synthetic generator, filters, EDA, features, rules, ML, risk scoring, Streamlit UI |
| **Leans on** | prompt design, orchestration logic | pandas, scikit-learn, UI |

Fill in the names at kickoff. Either person can take either track.

**The single most important property of this split:** Track A can build and fully test the *entire* agent —
parser, planner, executor, narrator, API — against **mock tools** from hour 2, without one line of Track B's
code. And Track B can build and test *every* tool standalone via `pytest`, without the agent existing. Neither
track ever waits for the other. They meet at H24.

---

## 2. The Zero-Merge-Conflict Protocol

Six rules. Rules 1 and 2 do almost all the work.

### Rule 1 — Every file has exactly one owner. You never edit a file you don't own.
Not even a typo. Not even a one-line import. Not even "while I was in there".
Git only produces a conflict when two people change the same file — so **disjoint ownership makes conflicts
arithmetically impossible**, not merely unlikely. See the matrix in §4.

If you need a change inside the other person's file: **message them, they change it, they push.** Cost: two
minutes. Cost of doing it yourself: a conflict in a 4000-line diff at hour 40.

### Rule 2 — Configure your coding agent with the ownership rule.
This is the #1 way the protocol breaks in practice. Coding agents "helpfully" refactor adjacent files,
add imports to `__init__.py`, or tidy `requirements.txt`. See §3 for the exact text to paste.

### Rule 3 — Branch per track; merge to `main` only at the checkpoints.
```
main                 ← kickoff scaffold, then merges at H24, H40, final
├── feat/agent-core  ← Track A, commits freely, pushes often
└── feat/data-ml-ui  ← Track B, commits freely, pushes often
```
Commit and push to **your own** branch as often as you like (commit history is a graded deliverable — commit
every 30–45 min, real messages). Merge into `main` only at H24, H40, and the final. Because ownership is
disjoint, every merge touches non-overlapping files.

### Rule 4 — `requirements.txt` is split in two.
The classic hackathon conflict file: both people add a dep to the last line, guaranteed conflict.

- `requirements.txt` — **A**: fastapi, uvicorn, pydantic, pydantic-settings, python-dotenv, LLM SDK, pytest, httpx
- `requirements-data.txt` — **B**: pandas, numpy, scikit-learn, networkx, streamlit, plotly, requests

One dependency per line. **Never re-sort the file** (a re-sort turns a 1-line diff into a whole-file diff).
Install: `pip install -r requirements.txt -r requirements-data.txt`.

### Rule 5 — `git add` explicit paths. Never `git add -A` or `git add .`
`git add .` sweeps up the other person's half-finished files, your `.env`, and multi-hundred-MB raw datasets.
```bash
git add backend/agent/planner.py tests/test_planner.py    # yes
git add -A                                                 # no
```
Always `git pull --rebase origin <your-branch>` before pushing. **Never force-push a shared branch.**

### Rule 6 — Empty `__init__.py` files stay empty forever.
No re-exports, no `__all__`. Import by full path (`from backend.tools.rules import detect_structuring`).
Re-export files are silent conflict magnets — every new module wants a line in them.

### Two extras worth the discipline
- **`data/sample/aml_sample.csv` is committed exactly once**, with a fixed seed, and only **B** may
  regenerate it. A regenerated CSV is a several-thousand-line diff that conflicts with its own history.
- **No file is co-authored — including the docs.** A owns `README.md` outright. B writes `DATA_CARD.md`,
  `AML_LOGIC.md`, `DEMO_SCRIPT.md`, and A links to them from the README. **Do not both edit the README.**
  Markdown conflicts are as painful as code conflicts and much easier to merge wrong.

---

## 3. Paste this into your coding agent's instructions

Put it in `CLAUDE.md` (A owns that file; A commits it in the kickoff hour so both agents pick it up), and
paste it into your agent at the start of each session too:

```
FILE OWNERSHIP RULE — this repo is being built by two people in parallel.

I am working on Track <A or B>. You may only CREATE or EDIT files owned by Track <A or B>
per the ownership matrix in WORKPLAN.md §4.

You may READ any file in the repo.

If a change appears necessary in a file owned by the other track — including
requirements.txt, __init__.py files, README.md, or backend/schemas.py — STOP and report
what change is needed and why. Do not make the edit. Do not work around it by duplicating
the file. The other person will make the change and push it.

backend/schemas.py, backend/tools/base.py, and docs/CONTRACTS.md are FROZEN interface
files. Treat them as read-only ground truth and conform to them exactly.

Never run `git add -A` or `git add .` — stage explicit paths only.
Never edit or regenerate data/sample/aml_sample.csv.
Leave every __init__.py empty.
```

---

## 4. File Ownership Matrix — this is law

### Track A owns
```
backend/schemas.py                 ← FROZEN contract (Pydantic models)
backend/config.py
backend/main.py                    ← FastAPI app
backend/agent/intent_parser.py
backend/agent/planner.py
backend/agent/executor.py
backend/agent/narrator.py
backend/agent/registry.py          ← auto-discovers tools; nobody else touches it
backend/llm/client.py
backend/tools/base.py              ← FROZEN contract (ToolContext, ToolResult, @tool)
backend/tools/_mocks.py            ← A's mock tools, for building before B's are ready
requirements.txt  .env.example  .gitignore  run_demo.py  CLAUDE.md
README.md  ARCHITECTURE.md  WORKPLAN.md  docs/CONTRACTS.md
tests/test_intent.py  tests/test_planner.py  tests/test_executor.py  tests/test_api.py
```

### Track B owns
```
backend/tools/data_loader.py       ← Kaggle/synthetic → canonical schema
backend/tools/filters.py
backend/tools/eda.py
backend/tools/features.py
backend/tools/rules.py             ← R1–R6
backend/tools/ml_detect.py
backend/tools/aggregate.py
backend/tools/entity.py
backend/tools/risk.py
data/generate_synthetic.py  data/sample/aml_sample.csv  data/adapters/**
frontend/app.py  frontend/components/**
requirements-data.txt
DATA_CARD.md  AML_LOGIC.md  DEMO_SCRIPT.md
tests/test_rules.py  tests/test_features.py  tests/test_ml.py  tests/fixtures/**
notebooks/**
```

Note `backend/tools/` is a **shared directory with per-file ownership** — that's fine, git tracks files not
directories. A owns exactly two files in there (`base.py`, `_mocks.py`); B owns the rest.

**Only three files are read by both and written by one** — `backend/schemas.py`, `backend/tools/base.py`,
`docs/CONTRACTS.md`. All owned by A, all frozen after the kickoff hour. A contract change after the freeze
requires **both people present**: A edits, pushes straight to `main`, tells B, B rebases before continuing.

---

## 5. Hour 0–2 — TOGETHER, one machine, one person typing

**Do not parallelize this.** Everything downstream depends on getting these interfaces right, and two agents
writing the contract simultaneously is exactly the conflict we're designing away.

1. **First 15 minutes:** create the public GitHub repo, push the folder skeleton. Commit history is graded —
   start it immediately, not at hour 30.
2. Write, in this order:
   - `backend/schemas.py` — all Pydantic models (see [docs/CONTRACTS.md](CONTRACTS.md))
   - `backend/tools/base.py` — `ToolContext`, `ToolResult`, the `@tool` decorator
   - `backend/agent/registry.py` — **auto-discovery** via `pkgutil.iter_modules` over `backend/tools/`
   - `backend/tools/_mocks.py` — one mock per tool name, returning plausible fixture data
   - `docs/CONTRACTS.md`, `WORKPLAN.md`, `CLAUDE.md`
   - `requirements.txt` + `requirements-data.txt`, `.gitignore` (ignore `data/raw/`, `data/processed/`,
     `.env`, `__pycache__`, `.venv`)
3. **B** generates and commits `data/sample/aml_sample.csv` — ~2000 rows in the canonical schema, with all
   four patterns already present, fixed seed. Both tracks now have real data to work against.
4. 20 minutes of domain reading (FATF 40 Recommendations; FinCEN CTR/SAR — the $10,000 CTR threshold is
   *why* structuring targets $9,xxx). Notes go straight into `AML_LOGIC.md`.

**Exit criterion — do not split until all three pass:**
- `pytest` collects with no import errors
- `AML_USE_MOCKS=1 uvicorn backend.main:app` starts
- `POST /query {"query": "analyse this dataset"}` returns a valid, fully-populated mocked `AgentResponse`

Then: merge to `main`, both branch off, split.

### Why the registry auto-discovers (the key trick)
A hand-written registry is an import list — B adds a tool, B edits `registry.py`, and that file conflicts
every single time. Instead `registry.py` walks the `backend/tools/` package, imports each module, and
collects every function carrying `@tool`. **B adds a tool by editing a file B already owns; B never opens
`registry.py`.** A writes it once in the kickoff hour and never touches it again either.

---

## 6. Hour-by-hour tasks

### Track A — Agent Core, API, Orchestration

**H2–H8 · Intent parsing**
- `backend/llm/client.py` — one function, provider-agnostic (Gemini or OpenAI free tier), JSON mode,
  short timeout, **returns `None` on any failure** (no key, rate limit, bad JSON, no network). Every caller
  must handle `None`.
- `backend/agent/intent_parser.py` — LLM prompt → `QueryIntent`. Then a **deterministic regex/keyword
  fallback** covering all 7 intents, used when the LLM returns `None`. Must extract: relative dates
  ("last 30 days", "in March", "since January"), entity IDs (`4521`, `C-4521`, `T-8891`), amount thresholds
  ("under $10,000"), counts ("10+ transactions"), countries, transaction types, `top_n` ("top 10").
- `tests/test_intent.py` — ~20 phrasings → expected intent + filters, with the LLM stubbed out.

**H8–H16 · The planner and executor — this is the graded core, give it the most care**
- `planner.py` — `QueryIntent` → `ExecutionPlan`. Every step carries a `reason` string ("entity filter first
  so downstream tools see only customer 4521's 87 transactions"). Populate
  `tools_considered_but_skipped` with a reason for each omission — that list is what proves to a judge the
  agent *decided*, rather than ran a pipeline.
- `executor.py` — run steps in order, thread one `ToolContext` through, merge each `ToolResult`,
  time each step, and **isolate errors**: a failing tool marks its step `error`, appends a warning, and the
  run continues. The API must never 500 because one tool raised.
- **Conditional re-planning** (at least these two, both logged to `plan.decisions[]`):
  - `rule_detect` returned 0 hits → append `ml_detect` to widen the net
  - filtered subset < 50 rows → drop `ml_detect`, note "insufficient sample for anomaly detection"
- `tests/test_planner.py` — includes the **plan-divergence assertions** from §8. Write these early; they
  are the specification of "agentic" for this project.

**H16–H24 · Narrator + API**
- `narrator.py` — a deterministic explanation template per rule R1–R6, built from the evidence dict
  (always accurate, always available). Then an optional LLM pass that rewrites the bundle into an analyst
  paragraph. **Never let the LLM invent numbers — pass it only computed facts.** Map risk band → escalation
  (report / review / monitor / no_action) and draft a short SAR-style summary for HIGH.
- `main.py` — `POST /query`, `GET /health`, `GET /dataset/summary`, `GET /plan/{plan_id}`, CORS open for
  Streamlit, and an in-memory run cache so `explain_flag` queries can reference a previous run.
- **At H24: merge to `main` and send B the OpenAPI URL (`localhost:8000/docs`).** This is B's unblock for
  wiring the real UI.

**H24–H34 · Integration**
- Set `AML_USE_MOCKS=0`, run against B's real tools, fix the mismatches (there will be some — that's what
  this block is for). Report any needed contract change to B before changing it.
- Write `README.md` and `ARCHITECTURE.md`.

**H34–H44 · Hardening**
- All 10 demo queries work end-to-end. Graceful, human answers for empty results ("no transactions matched
  that filter" — never an empty table with no explanation).
- `run_demo.py` starts backend + frontend with one command.
- **Rehearse with the LLM key unset.** Venue wifi fails. The fallback path must be demo-quality.

### Track B — Data, Features, Detection, UI

**H2–H8 · Data foundation**
- Download the Kaggle sets (IBM AML **HI-Small**, sample ~200k rows; PaySim as secondary).
- `data_loader.py` — adapters mapping IBM / PaySim / synthetic → **the canonical schema** in
  [docs/CONTRACTS.md](CONTRACTS.md). This adapter layer is what lets you swap datasets later without
  touching any detection code.
- `generate_synthetic.py` — fixed seed, labelled cohorts: structuring customers, smurfing rings, layering
  chains, rapid-cash-out, plus a large normal population. **Log every parameter you choose** (ring sizes,
  time windows, amount distributions) — it goes into `DATA_CARD.md`, which the brief requires.
- `notebooks/01_data_exploration.ipynb` — evidence of EDA for the judges.

**H8–H14 · Filters + EDA**
- `filters.py` — composable filters on date range, country, txn type, amount range, min txn count, entity ID.
  Each returns the filtered frame plus a note describing what it did (the agent surfaces those notes).
- `eda.py` — profile stats, missingness, amount distribution, txn-type and country breakdowns, volume time
  series, and a **threshold-proximity histogram** (transactions bucketed near $10k — this one chart makes
  structuring visible at a glance and is worth showing the judges). Return **both** a metrics dict **and**
  Plotly figure JSON, per the `ToolResult` contract.

**H14–H22 · Features + rules (the detection core)**
- `features.py` — rolling 1d/7d/30d sum & count per customer; `pct_just_below_threshold` ($9k–$10k share);
  amount z-score against **the customer's own** 90-day baseline (self-deviation beats population deviation
  and produces far fewer false positives); velocity (txns/hour, distinct counterparties/day); rapid-cash-out
  ratio; round-amount ratio; night-hours ratio; new-counterparty ratio; cross-border hits.
  **Parameterise it so only the requested pattern's features compute** — the agent passes which patterns it
  cares about, and skipping the rest is part of the "adaptive execution" story.
- `rules.py` — R1–R6 per [AML_LOGIC.md](AML_LOGIC.md), each returning
  `(entity_id, rule_id, evidence_dict, weight)`. The `evidence_dict` is what the narrator turns into
  English, so make it specific: `{"txn_count": 4, "window_days": 7, "amounts": [9500, 9200, 9800, 9100], "total": 37600}`.
  Use `networkx` for R3 layering chains.
- `tests/test_rules.py` — each rule fires on its own synthetic cohort **and stays silent on the normal
  cohort**. That second half is what lets you claim false-positive control.

**H22–H28 · ML + risk scoring**
- `ml_detect.py` — `StandardScaler` → `IsolationForest` (primary) + `LocalOutlierFactor` (secondary),
  percentile-ranked scores, and the **top-3 contributing features** by deviation-from-median z-score
  (cheap explainability, no SHAP needed).
- `risk.py` — fused score `100 * (0.6 * normalized_rule_weight + 0.4 * ml_percentile)`, bands
  HIGH ≥70 / MEDIUM 40–69 / LOW 15–39 / none <15.
- `aggregate.py` (group-by + threshold queries, e.g. "10+ transactions under $10,000"), `entity.py`
  (single-customer lookup + profile).
- **Merge to `main` at H24** — from this point A is running against your real tools.

**H28–H40 · The Streamlit UI (this is what the judges actually see)**
- Query box + a row of **example-query buttons** (so a judge can click rather than type, and so you can't
  fat-finger the demo).
- **The execution-plan trace panel — the highest-value component in the whole project.** Show: detected
  intent, extracted filters and entities, the ordered tool list with each step's `reason` and timing,
  the tools *skipped* and why, and the `decisions[]` re-planning log. Put this directly under the query box,
  above the results.
- Flag cards: risk badge (colour by band), the explanation paragraph, the evidence table, the escalation
  action, and the SAR draft for HIGH.
- EDA charts from `eda.py`'s Plotly JSON; dataset summary in the sidebar; JSON/CSV export.
- **Talk to A's API over HTTP only** (`requests.post("http://localhost:8000/query", ...)`). Never import
  `backend.agent.*` into the frontend — that import is what would couple the two tracks.

**H40–H44 · Docs + the validation table**
- `DATA_CARD.md`, `AML_LOGIC.md`, `DEMO_SCRIPT.md`.
- Validate detections against IBM's `Is Laundering` label: precision, recall, false-positive rate — **and
  the false-positive count against a naive "amount > $9,000" baseline rule.** The brief is explicitly about
  traditional systems drowning teams in false positives, so a table showing "naive rule: 1,240 alerts,
  38 true → ours: 96 alerts, 31 true" is the strongest single piece of evidence you can put in the README.

### H44–H48 — TOGETHER
Rehearse the demo end-to-end 3 times. Screenshots into the README. Clean-clone test on the other person's
machine. Final commits, tag a release.

---

## 7. Sync protocol

Three 10-minute standups: **H8, H24, H40.** One rule between them: **the frozen contracts are the only thing
you may renegotiate, and only at a standup.**

If you're blocked on the other track, **mock it and keep moving** — never idle:
- **A never waits for B.** `AML_USE_MOCKS=1` until H24.
- **B never waits for A.** Test every tool directly with pytest; the UI can render a saved
  `AgentResponse` JSON fixture until the API is live.

### Git commands you'll actually use
```bash
# start of each session
git checkout feat/<your-branch> && git pull --rebase origin feat/<your-branch>

# during work — explicit paths, every 30-45 min
git add backend/agent/planner.py tests/test_planner.py
git commit -m "planner: skip ml_detect for threshold_query intent"
git push origin feat/<your-branch>

# at H24 / H40 / final — check you only touched your own files FIRST
git diff --stat main...HEAD
git checkout main && git pull && git merge feat/<your-branch> && git push origin main
```
If `git diff --stat main...HEAD` lists a file you don't own, revert that hunk before merging. If a conflict
ever does appear, it means the ownership matrix was violated — find out where, fix the ownership, don't just
resolve the conflict and move on.

---

## 8. Definition of done

Shared checks, run at H44:

1. `pytest tests/ -v` green.
2. **Plan-divergence test — the core agentic claim, and the most likely thing a judge probes.** Automate it:
   - `"Is customer 4521 suspicious?"` → plan contains **no** `eda_profile`
   - `"Which customers made 10+ transactions under $10,000?"` → plan contains **no** `ml_detect`
   - `"Show transaction distribution by country"` → plan contains **no** detection tools at all
   - `"Analyse this dataset for suspicious activity"` → plan contains **both**

   > **Amended.** This originally also required that `"Is customer 4521 suspicious?"` contain no
   > `ml_detect`, on the assumption that a single-entity query is too small a sample to rank. The
   > implementation contradicted that assumption: the same plan runs `feature_engineer` across the
   > whole population precisely so the score is comparable, so `ml_detect` would have received all
   > 270 customers rather than one. Skipping it zeroed the ML term and made every single-entity
   > query return `100 × 0.6 × max_rule_weight` — C-STR02 came back **51.00 MEDIUM** when a full
   > sweep called the same customer **89.84 HIGH**. The genuine sample-size guard is §5's rule
   > (*"filtered subset < 50 rows → drop `ml_detect`"*), which lives in `executor.py` and is
   > unaffected. Plan divergence is still demonstrated by the four cases above.
3. All 10 demo queries return a valid `AgentResponse` with a non-empty explanation **and** an escalation on
   every flag.
4. **LLM-off run works** (key unset) — rehearsed, not just assumed.
5. Detection metrics vs IBM's label + the false-positive comparison table, in the README.
6. **Clean-clone test on the other person's machine:** fresh venv, follow the README verbatim, demo runs on
   the committed sample CSV with **no Kaggle download required**.
7. README cites every dataset with URL and licence, and documents the synthetic generation logic.

## 9. If we fall behind — cut in this order
SHAP → LOF (keep IsolationForest) → R6 dormant reactivation → notebooks → the Kaggle adapters
(synthetic-only still satisfies the brief, as long as `DATA_CARD.md` documents it).

**Never cut:** the execution-plan trace panel, the per-flag explanations, the escalation actions,
`README.md` / `DATA_CARD.md`, or the LLM-off fallback path.
