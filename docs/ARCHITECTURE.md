# ARCHITECTURE.md — Agent Design Detail

Companion to [README.md](../README.md) (the pitch) and [docs/CONTRACTS.md](CONTRACTS.md) (the frozen
interface, written first and unchanged in spirit throughout the build). This file explains *how* the
pieces fit together and *why* they're shaped the way they are.

---

## Component diagram

```mermaid
flowchart TD
    Q(["User query — natural language"])

    subgraph HTTP ["HTTP surface"]
        API["FastAPI · backend/main.py<br/>POST /query · GET /health<br/>GET /dataset/summary · GET /plan/:id"]
    end

    subgraph CORE ["Agent core · backend/agent/"]
        IP["intent_parser<br/>str to QueryIntent"]
        PL["planner<br/>QueryIntent to ExecutionPlan"]
        EX["executor<br/>runs plan · threads ToolContext"]
        NA["narrator<br/>risk rows to flags and SAR drafts"]
    end

    LLM["llm/client.py<br/>Gemini · OpenAI · Groq · Ollama"]
    FB["regex fallback<br/>covers all 7 intents alone"]
    REG["registry<br/>pkgutil auto-discovery of @tool"]
    RP{{"Runtime re-planning<br/>0 rule hits, append ml_detect<br/>under 50 rows, drop ml_detect<br/>0 rows, halt with explanation"}}

    subgraph TOOLS ["Tool layer · backend/tools/"]
        direction LR
        LD["load_data"]
        FD["filter_data"]
        EDA["eda_profile"]
        FE["feature_engineer"]
        RD["rule_detect"]
        MD["ml_detect"]
        AQ["aggregate_query"]
        EL["entity_lookup"]
        RC["risk_classify"]
    end

    RESP["AgentResponse JSON<br/>intent · plan · flags · tables · charts"]
    UI["Streamlit UI<br/>HTTP client, imports nothing from backend"]

    Q --> API
    API --> IP
    IP -.->|primary| LLM
    IP -.->|always available| FB
    IP --> PL
    PL --> EX
    EX <-->|resolve by name| REG
    REG --> TOOLS
    EX --> RP
    RP -.->|mutates remaining steps| EX
    EX --> NA
    NA --> RESP
    RESP --> API
    API -->|HTTP| UI
```

**The one rule that makes the two-person build possible**: dependencies flow **agent → tools, never the
reverse**. No tool imports `backend.agent.*`; no tool imports another tool; the Streamlit frontend imports
nothing from `backend/` at all — HTTP only.

```mermaid
flowchart LR
    FE["Streamlit frontend"] -->|HTTP only| API["FastAPI"]
    API --> AC["Agent core"]
    AC --> TL["Tool layer"]
    TL --> DATA[("Canonical schema<br/>transactions · customers")]

    AC -.->|never| FE
    TL -.->|never imports| AC
    TL -.->|never imports another| TL

    linkStyle 4,5,6 stroke:#8f2d3c,stroke-dasharray:4 4
```

---

## The four agent-core components

### 1. Intent parser (`backend/agent/intent_parser.py`)

`raw_query: str → QueryIntent`. LLM-first (`backend/llm/client.py`, JSON mode, provider-agnostic), with a
**complete regex/keyword fallback** that alone covers all 7 intents — this is the path actually exercised
in every test, since the LLM path has no live-key test yet (see README Limitations).

Key design points:
- **Relative dates** ("last 30 days") anchor to the *dataset's own max transaction date*
  (`_dataset_reference_date()`, cached per process), not wall-clock `date.today()` — the demo dataset is
  dated Jan–Mar 2025, so anchoring to real "today" would silently return zero results for the exact
  "last 30 days" query the problem brief itself uses as an example.
- **Entity IDs** are extracted two ways: a `C-`/`T-` prefixed alphanumeric token (real IDs aren't purely
  numeric — `C-STR02`, `C-N0001`) passes through as-is; a bare number with no prefix gets constructed into
  a plausible `C-#####` ID that `executor.py`'s `_resolve_entities()` later reconciles against the real
  dataset by numeric value.
- **Classification precedence** (first match wins, in this order): `explain_flag` → `entity_investigation`
  → `ranking` → `threshold_query` → `pattern_search` → `eda` → `full_analysis` fallback. E.g. a query
  naming both an entity and a pattern keyword resolves to `entity_investigation`, since a named entity is
  a more specific signal.

### 2. Planner (`backend/agent/planner.py`)

`QueryIntent → ExecutionPlan`. One branch per intent, implementing the mapping table in
[docs/CONTRACTS.md](CONTRACTS.md) Contract 4 exactly. Every `ToolCall` carries a `reason`; every
tool *not* included carries an entry in `tools_considered_but_skipped` with its own reason — this list is
what makes "the agent decided, it didn't just run a pipeline" a checkable claim rather than a marketing
line.

Tool call params are built to match what `backend/tools/*.py` **actually accept** — this mattered in
practice: several params (`filter_data`'s flattened `Filters` fields, `feature_engineer`'s `pattern_types`,
`aggregate_query`'s `group_by`/`agg_func`/`threshold`) were initially guessed differently and had to be
corrected once real tools existed to check against (see the Phase 6/7 history in
[TRACK_A_PROGRESS.md](TRACK_A_PROGRESS.md) if you want the full story).

### 2b. LLM planner (`backend/agent/llm_planner.py`, opt-in)

`build_plan` above is deterministic: the tool sequence inside each branch is a constant. With
`AML_LLM_PLANNER=1`, `plan_query()` puts an LLM planning step in front of it and the model selects the
tools itself. Three modules cooperate:

| Module | Job |
|---|---|
| `tool_schema.py` | Renders the tool catalog from each tool's own `_tool_description`/`_tool_params` — the metadata the `@tool` decorator has always recorded and nothing previously read. One reader, so the prompt cannot drift from what the tools accept. |
| `plan_validator.py` | Twelve rules (V0–V11) over the proposal: registry membership, dependency ordering, no duplicates, `load_data` first, ≤12 steps, declared param names, a stated reason per step. Collects **all** violations, not the first. |
| `llm_planner.py` | Prompt, fallback, and audit trail. |

Two design points worth stating explicitly:

**The deterministic planner is the floor, not the alternative.** Every failure path — LLM unavailable,
unparseable JSON, any validation rejection — returns `build_plan(intent)`. Contract 4 in
[CONTRACTS.md](CONTRACTS.md) is therefore still honoured as the guaranteed behaviour of this system; the
LLM can only do *better* than it for a given query, never something illegal. Contract 4 is unchanged and
was not edited for this feature.

**Legality (V0–V11) and answerability (V12) are different things.** V5–V7 mirror real preconditions in the
tool bodies (`rules.py` reads `ctx.artifacts["features"]`; `risk.py` reads `rule_hits`/`ml_scores`), so a
passing plan cannot fail on a missing artifact. V12 additionally requires the terminal tool each intent
needs to produce its answer.

V12 was not in the original design, which held that anything beyond dependency legality would mean
encoding the deterministic planner's opinions back into the validator. Measurement showed that was too
permissive: with V0–V11 alone a local 3B model reached **60% acceptance while only 7% of plans could
answer the query** — it had learned that shorter plans pass, because a truncated plan satisfies every
ordering rule vacuously. V12 constrains the plan's *output*, not the route to it: a `ranking` query that
cannot return a ranking is broken, not merely suboptimal. Which filters, which detectors, rules vs ML —
all still the model's call.

A legal, answerable but clumsy plan still passes, and should. Judging elegance is not a whitelist's job.

**Repair vs rejection.** Defects with exactly one correct fix and no judgement involved are repaired and
logged, not rejected: a missing or misplaced `load_data`, `filter_data`'s params when left empty,
`entity_lookup`'s `entity_id`. `load_data` moved into this category after measurement — it was 8 of 13
rejections, yet all eight deterministic branches start with it and no query exists where omitting it is
correct, so requiring the model to emit it tested nothing. Anything involving a real choice is never
repaired; a duplicated `load_data` is still rejected, because two of them signals confusion rather than an
omitted preamble.

The audit trail (`planner: source=` / `proposed =` / `rejected —` / `executed =`) goes into
`plan.decisions[]`, which the UI already renders, so this needed no frontend change. `executed` is
appended after the run and therefore includes the executor's own re-planning — that is how
proposed-vs-executed divergence stays visible.

`_tool_params` is only checked when non-empty: `backend/tools/_mocks.py` declares no param schemas, so
strict checking would reject every plan under `AML_USE_MOCKS=1`. An empty declaration means "unvalidated"
and emits a note, keeping the gap auditable rather than silent.

### 3. Executor (`backend/agent/executor.py`)

`(QueryIntent, ExecutionPlan) → AgentResponse`. Threads one `ToolContext` through every step, merging each
`ToolResult`'s `df`/`artifacts`/`tables`/`charts`/`metrics` and timing each step. **Never lets one tool's
failure break the response** — a raised exception or `ok=False` marks that step `"error"`, appends a
warning, and the run continues with whatever data survived.

Three behaviors live here specifically because they need data that only exists *after* a step has run
(the planner can't know these at plan-build time):

- **Conditional re-planning**: 0 rule hits → insert `ml_detect` next; <50 filtered rows → drop a queued
  `ml_detect`; 0 filtered rows → stop early with an explanatory summary.
- **Entity-ID resolution** (`_resolve_entities()`): right after `load_data` populates `ctx.customers`,
  reconciles any unresolved bare-number entity against the real dataset's `customer_id` values **by
  numeric id** (digits-only, integer comparison — not substring, which would false-match short numbers).
  Propagates the resolved ID into `intent.entities` and into any already-built `entity_lookup` step's
  params.
- **Post-`risk_classify` scoping**: `risk_classify` scores the whole population (it has no per-entity or
  top-N parameter), so `entity_investigation`/`explain_flag` filter `risk_rows` down to the requested
  entity, and `ranking` slices to `intent.top_n`, after the fact.

### 4. Narrator (`backend/agent/narrator.py`)

`risk_rows → Flag[]`. A deterministic template layer (`_build_evidence()` + `_explain()`) built directly
from each rule's evidence — **always accurate, always available**, since `rule_detect`'s evidence dicts
are rule-specific free-form dicts (structuring's fields differ from layering's — see
[AML_LOGIC.md](AML_LOGIC.md)), adapted into the frozen `Evidence` schema by pairing `evidence[i]` with
`triggered_rules[i]` positionally (how `risk_classify` builds them).

LLM polish is **capped to `HIGH`-risk flags only** — a `full_analysis` result can carry dozens of flags,
and one LLM call per flag would exhaust a free-tier rate limit on a single query for no real benefit (the
polish silently falls back to the identical template text on any LLM failure anyway). `HIGH` flags are
also the only ones getting a `sar_draft`, so this is where LLM value is actually concentrated.

---

## The frozen contracts (docs/CONTRACTS.md)

Three things, fixed before either half of this project was built, and unchanged since:

1. **`backend/schemas.py`** — every Pydantic model (`QueryIntent`, `ExecutionPlan`, `ToolCall`, `Flag`,
   `AgentResponse`, ...).
2. **`backend/tools/base.py`** — `ToolContext`, `ToolResult`, the `@tool` decorator every tool registers
   with.
3. **`backend/agent/registry.py`** — auto-discovers tools by walking `backend/tools/` with
   `pkgutil.iter_modules`, so adding a tool means adding a file, not editing a hand-maintained import list.

### The registry's one subtlety

`TOOLS` (in `backend/tools/base.py`) is a single global dict; a module's `@tool` decorators only execute
on that module's *first* import. `registry.load_tools(use_mocks=...)` therefore **clears `TOOLS` and
`importlib.reload()`s already-imported modules** on every call — without this, calling it more than once
with different `use_mocks` values in one process (which happens across a test session, not in normal
single-mode server operation) leaves some tool names on stale bindings from whichever mode last imported
them. Found via a full-suite test failure that only reproduced in specific file-import orderings.

---

## Intent → tool sequence (Contract 4, summarized)

| Intent | Sequence | Notably skipped |
|---|---|---|
| `full_analysis` | load → eda → features → rules → ml → risk | — |
| `pattern_search` | load → filter → features(scoped) → rules(scoped) → ml → risk | eda |
| `threshold_query` | load → filter → aggregate | features, ml, eda |
| `entity_investigation` | load → filter → entity_lookup → features → rules → ml → risk *(→ scoped to entity)* | eda |
| `ranking` | load → filter → features → rules → ml → risk *(→ sliced to top_n)* | eda |
| `eda` | load → filter → eda | features, rules, ml, risk |
| `explain_flag` | load → entity_lookup → features → rules → ml → risk *(→ scoped to entity)* | eda |

`explain_flag`'s inclusion of `load_data` is a deliberate deviation from Contract 4's original text
("reuse a cached run") — that mechanism was never actually wired to anything, so the intent always
returned empty. Scoring the entity fresh (same shape as `entity_investigation`, minus `filter_data`, since
"why was X flagged" implies no extra scoping) makes the feature actually answer the question.

Both single-entity intents run `ml_detect` despite asking about one customer, which looks wrong until you
read the rest of their plan: `feature_engineer` runs across the **whole population** so the resulting score
is comparable, and `ml_detect` therefore receives all 270 customers rather than one. They originally
skipped it — `WORKPLAN.md` §8 even pinned that as a definition-of-done item — and the effect was to zero
the ML half of Contract 5's formula. Every single-entity query returned exactly `100 × 0.6 ×
max_rule_weight`: C-STR02 scored **51.00 MEDIUM ("review")** when asked about directly and **89.84 HIGH
("report")** in a full sweep. Asking about a customer directly was the one query that understated their
risk, and it downgraded them out of the SAR-drafting tier.

Genuinely small samples are still handled, by the two guards that fire on data size rather than intent
name: the executor drops `ml_detect` when `filter_data` leaves under 50 rows, and `ml_detect` itself
no-ops below `IF_MIN_SAMPLES`. Plan divergence — the core agentic claim — is unaffected: `eda_profile`
stays out of both, `threshold_query` still skips ML entirely, and `eda` still runs no detection at all.

---

## Sequence: three contrasting queries

One request end to end first, then the three traces that show how much the shape changes per query.

```mermaid
sequenceDiagram
    autonumber
    actor A as Analyst
    participant API as FastAPI
    participant IP as intent_parser
    participant PL as planner
    participant EX as executor
    participant T as Tool layer
    participant NA as narrator

    A->>API: POST /query - find structuring in the last 30 days
    API->>IP: parse_intent(query)
    Note over IP: LLM in JSON mode<br/>regex fallback on failure or timeout
    IP-->>API: QueryIntent - pattern_search, structuring, date window
    API->>PL: build_plan(intent)
    PL-->>API: ExecutionPlan - 6 steps, eda_profile skipped
    API->>EX: run_plan(intent, plan)

    EX->>T: load_data
    T-->>EX: transactions and customers
    Note over EX: resolve bare entity IDs<br/>against real customer_id values
    EX->>T: filter_data - date window
    T-->>EX: narrowed frame
    EX->>T: feature_engineer - structuring only
    T-->>EX: 9 of 17 features
    EX->>T: rule_detect - R1
    T-->>EX: rule hits with evidence

    alt zero rule hits
        EX->>EX: append ml_detect to widen the net
    else filtered set under 50 rows
        EX->>EX: drop the queued ml_detect
    end

    EX->>T: ml_detect
    T-->>EX: percentile-ranked anomaly scores
    EX->>T: risk_classify
    T-->>EX: risk rows sorted by score
    EX->>NA: build_flags(risk_rows)
    NA-->>EX: flags with evidence, explanation, SAR for high risk
    EX-->>API: AgentResponse
    API-->>A: JSON with intent, plan, flags, charts, tables
```

**"Is customer 4521 suspicious?"** — bare number, entity_investigation
```
parse → entities=["C-04521"] (constructed guess, not yet a real ID)
plan  → load_data, filter_data, entity_lookup, feature_engineer, rule_detect, ml_detect,
        risk_classify        (eda_profile skipped — this is not exploration)
exec  → load_data runs → _resolve_entities() matches "04521" by numeric id against real
        customer_ids → resolves to e.g. "C-N0002" if a match exists, else leaves unresolved
      → features/rules/ml all run across the full population, so the score this entity gets
        is the same one a full sweep would give it
      → risk_classify scores everyone → executor filters risk_rows to just this one entity
narrate → 0 or 1 Flag, never a crash either way
```

**"Which customers made 10+ transactions under $10,000?"** — threshold_query
```
parse → filters.min_txn_count=10, filters.amount_max=10000.0
plan  → load_data, filter_data, aggregate_query   (ml_detect never even considered)
exec  → filter_data narrows to sub-$10k txns from senders with ≥10 such txns
      → aggregate_query groups by sender_id, counts, reports row_count
narrate → no Flags at all (by design — this intent skips risk_classify entirely);
          summary states the count directly from aggregate_query's metrics
```

**"Analyse this dataset for suspicious activity"** — full_analysis
```
parse → no filters/entities, full_analysis
plan  → all 6 tools, nothing skipped
exec  → eda_profile runs alongside detection; rule_detect finds hits across R1-R7;
        ml_detect scores every customer; risk_classify fuses both signals
narrate → dozens of Flags across HIGH/MEDIUM/LOW, each with its own evidence-based explanation
```

---

## Where to look for more detail

- **Rule definitions, thresholds, regulatory justification**: [AML_LOGIC.md](AML_LOGIC.md)
- **Dataset schema, sources, synthetic generation logic**: [DATA_CARD.md](DATA_CARD.md)
- **The full frozen interface**: [docs/CONTRACTS.md](CONTRACTS.md)
- **Build history, what was fixed and why, test counts over time**: [TRACK_A_PROGRESS.md](TRACK_A_PROGRESS.md)
- **The two-person parallel-build plan this repo followed**: [WORKPLAN.md](WORKPLAN.md)

---

## Complete system reference

Everything in one view — data sources through adapters, agent-core internals, the artifact handshake
between tools, the frozen contracts, and the frontend. This is a reference diagram, not a teaching one:
the three diagrams above each carry a single idea, while this one is for when you want the whole surface
at once.

```mermaid
flowchart TD

    %% ==================== DATA SOURCES ====================
    subgraph SRC ["Data sources — swappable by design"]
        S1[("IBM AML HI-Small · Kaggle<br/>real-world base")]
        S2[("Synthetic · seed 42<br/>2002 txns · 270 customers")]
        S3[("Synthetic alt schema · seed 42<br/>1710 txns · 294 customers")]
    end

    subgraph ADAPT ["Adapter layer · tools/data_loader.py"]
        AD1["_adapt_ibm"]
        AD2["_adapt_synthetic"]
        AD3["_adapt_synthetic_alt<br/>renamed cols · coded enums"]
    end

    CANON[("Canonical schema · Contract 0<br/>transactions · customers")]

    S1 --> AD1 --> CANON
    S2 --> AD2 --> CANON
    S3 --> AD3 --> CANON

    %% ==================== ENTRY ====================
    USER(["Analyst · natural-language query"])

    subgraph HTTPX ["HTTP surface · backend/main.py"]
        EP1["POST /query"]
        EP2["GET /health"]
        EP3["GET /dataset/summary"]
        EP4["GET /plan/:plan_id"]
    end

    USER --> EP1

    %% ==================== AGENT CORE ====================
    subgraph COREX ["Agent core · backend/agent/"]

        subgraph G1 ["Step 1 · intent_parser.py"]
            IP1["classify intent — 7 types<br/>first match wins"]
            IP2["extract filters · entities · patterns"]
            IP3["anchor relative dates to<br/>dataset max date, not wall clock"]
            IP4["coerce provider shorthand<br/>such as minus 30d"]
        end

        subgraph G2 ["Step 2 · planner.py"]
            PL1["one branch per intent"]
            PL2["attach a reason to every step"]
            PL3["record tools skipped and why"]
        end

        subgraph G3 ["Step 3 · executor.py"]
            EX1["thread one ToolContext · time each step"]
            EX2["conditional re-planning"]
            EX3["resolve bare entity IDs by numeric match"]
            EX4["post-risk scoping — entity or top_n"]
            EX5["failure isolation — step errors, run continues"]
        end

        subgraph G4 ["Step 4 · narrator.py"]
            NA1["adapt rule evidence to frozen shape"]
            NA2["deterministic template explanation"]
            NA3["LLM polish — top 5 HIGH only"]
            NA4["SAR draft — HIGH only"]
        end
    end

    %% ==================== LLM ====================
    subgraph LLMX ["LLM layer · backend/llm/client.py"]
        PV1["Gemini"]
        PV2["OpenAI"]
        PV3["Groq"]
        PV4["Ollama — local, no quota"]
        CACHE["response cache<br/>successes only, not failures"]
    end

    RGX["regex and keyword fallback<br/>covers all 7 intents alone"]

    EP1 --> G1
    G1 -.->|primary, JSON mode| LLMX
    G1 -.->|always available| RGX
    LLMX --- CACHE
    G1 -->|QueryIntent| G2
    G2 -->|ExecutionPlan| G3
    G3 -->|risk rows| G4

    RPX{{"Runtime re-planning triggers<br/>zero rule hits — append ml_detect<br/>under 50 rows — drop ml_detect<br/>zero rows — halt with explanation"}}
    G3 --> RPX
    RPX -.->|mutates remaining steps| G3

    %% ==================== REGISTRY + TOOLS ====================
    REGX["registry.py<br/>pkgutil auto-discovery of @tool<br/>clears and reloads on each call"]
    G3 <-->|resolve by name| REGX

    subgraph TOOLSX ["Tool layer · backend/tools/ — 9 tools, no cross-imports"]

        subgraph TG1 ["Data access"]
            T1["load_data"]
            T2["filter_data"]
            T3["entity_lookup"]
        end

        subgraph TG2 ["Profiling and aggregation"]
            T4["eda_profile<br/>5 Plotly figures"]
            T5["aggregate_query<br/>group-by · threshold · top-N"]
        end

        subgraph TG3 ["Detection"]
            T6["feature_engineer<br/>17 features, scoped per pattern"]
            T7["rule_detect<br/>R1 to R6 · weights 0.60 to 0.85"]
            T8["ml_detect<br/>IsolationForest 0.6 + LOF 0.4"]
        end

        subgraph TG4 ["Scoring"]
            T9["risk_classify<br/>100 x 0.6 rule + 0.4 ml percentile"]
        end
    end

    REGX --> TOOLSX
    CANON --> T1

    %% ---------- artifact handshake between tools ----------
    T1 -->|ctx.df · customers| T2
    T2 -->|narrowed df| T6
    T2 --> T4
    T2 --> T5
    T2 --> T3
    T6 -->|features · feature_list| T7
    T6 -->|features · feature_list| T8
    T7 -->|rule_hits with evidence| T9
    T8 -->|ml_scores · percentiles| T9

    %% ==================== CONTRACTS ====================
    subgraph FROZEN ["Frozen contracts — written before implementation"]
        C1["schemas.py · Contract 1<br/>QueryIntent · ExecutionPlan · Flag · AgentResponse"]
        C2["tools/base.py · Contract 2<br/>ToolContext · ToolResult · @tool"]
        C3["CONTRACTS.md<br/>Contract 0 schema · Contract 4 intent map"]
    end

    C1 -.->|validates| COREX
    C2 -.->|shapes| TOOLSX

    %% ==================== RESPONSE + UI ====================
    RESPX["AgentResponse JSON<br/>query · intent · plan · flags<br/>tables · charts · metrics · warnings"]

    T9 -->|risk_rows| G4
    G4 --> RESPX
    RESPX --> EP1

    subgraph FEX ["Presentation · frontend/ — HTTP client only"]
        APP["app.py<br/>live mode, fixture fallback"]
        PT["plan_trace.py<br/>steps · skips · decisions"]
        FC["flag_cards.py<br/>badge · evidence table · SAR"]
        CH["charts.py<br/>Plotly JSON rendered as-is"]
        TH["theme.py<br/>colour tokens · metric alias resolver"]
        FIX[("fixtures/full_analysis.json<br/>loaded only after a real call fails")]
    end

    EP1 -->|HTTP| APP
    EP2 --> APP
    EP3 --> APP
    APP --> PT
    APP --> FC
    APP --> CH
    TH -.-> PT
    TH -.-> FC
    TH -.-> CH
    FIX -.->|on API failure| APP

    %% ==================== STYLING ====================
    classDef core fill:#d7e8e6,stroke:#0d6e69,stroke-width:1px,color:#10201f
    classDef tool fill:#e6ecef,stroke:#3f5b66,stroke-width:1px,color:#131a1c
    classDef data fill:#efe7d8,stroke:#8a6a1f,stroke-width:1px,color:#241d0c
    classDef frozen fill:#f4dfe2,stroke:#8f2d3c,stroke-width:1px,color:#2b1216
    classDef ui fill:#e3e6f0,stroke:#4a4f7a,stroke-width:1px,color:#15172b

    class IP1,IP2,IP3,IP4,PL1,PL2,PL3,EX1,EX2,EX3,EX4,EX5,NA1,NA2,NA3,NA4,RPX core
    class T1,T2,T3,T4,T5,T6,T7,T8,T9,REGX tool
    class S1,S2,S3,AD1,AD2,AD3,CANON data
    class C1,C2,C3 frozen
    class APP,PT,FC,CH,TH,FIX ui
```
