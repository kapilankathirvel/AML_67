# CONTRACTS — the frozen interface

Both tracks code against this file. It is the whole seam between Track A (agent core + API) and Track B
(data, detection, UI). Written in the kickoff hour, then **frozen**.

> **Frozen means:** `backend/schemas.py`, `backend/tools/base.py`, and this document are read-only ground
> truth for everyone except their owner (Track A). Changing them after the kickoff hour requires **both
> people present** — A edits, pushes straight to `main`, tells B, B rebases before continuing.
> See [../WORKPLAN.md](WORKPLAN.md) §4.

**The one architectural rule that keeps the tracks independent:**
dependencies flow **agent → tools, never the reverse.**
- No tool imports anything from `backend/agent/`.
- No tool imports another tool.
- The frontend imports nothing from `backend/` — it talks HTTP only.

Every tool is a pure function of `(ToolContext, **params) → ToolResult`. That's it.

---

## Contract 0 — The canonical data schema

Every loader adapts its source (IBM AML, PaySim, our synthetic generator) into exactly these columns. All
detection code downstream assumes only this — which is what makes the datasets swappable.

### `transactions`
| Column | Type | Notes |
|---|---|---|
| `txn_id` | str | unique, e.g. `T-000123` |
| `timestamp` | datetime64[ns] | tz-naive, UTC |
| `sender_id` | str | FK → `customers.customer_id` |
| `receiver_id` | str | FK → `customers.customer_id` |
| `amount` | float | positive, in `currency` |
| `currency` | str | ISO 4217, e.g. `USD` |
| `txn_type` | str | `deposit` \| `withdrawal` \| `transfer` \| `wire` \| `cash` |
| `channel` | str | `atm` \| `branch` \| `online` \| `mobile` \| `wire` |
| `sender_country` | str | ISO 3166 alpha-2 |
| `receiver_country` | str | ISO 3166 alpha-2 |
| `is_cross_border` | bool | derived: `sender_country != receiver_country` |
| `label_is_laundering` | bool \| null | ground truth where available (IBM), else null |
| `pattern_label` | str \| null | synthetic only: `structuring` \| `smurfing` \| `layering` \| `rapid_cashout` \| null |

### `customers`
| Column | Type | Notes |
|---|---|---|
| `customer_id` | str | unique, e.g. `C-04521` |
| `name` | str | synthetic |
| `account_open_date` | date | for account-age / dormancy features |
| `customer_type` | str | `individual` \| `business` |
| `country` | str | ISO 3166 alpha-2 |
| `occupation` | str | free text |
| `risk_rating` | str | `low` \| `medium` \| `high` — the bank's own KYC rating |
| `kyc_status` | str | `verified` \| `pending` \| `incomplete` |
| `is_pep` | bool | politically exposed person |
| `expected_monthly_volume` | float | KYC-declared; the baseline for deviation features |

Entity IDs in user queries may arrive bare (`4521`) — the intent parser normalises to `C-04521`.

---

## Contract 1 — `backend/schemas.py` *(owner: Track A · read-only for B)*

```python
from datetime import date
from typing import Any, Literal
from pydantic import BaseModel, Field

Intent = Literal[
    "full_analysis", "pattern_search", "threshold_query",
    "entity_investigation", "ranking", "eda", "explain_flag",
]
PatternType = Literal[
    "structuring", "smurfing", "layering", "rapid_cashout",
    "velocity", "dormant_reactivation", "unknown",
]
RiskLevel  = Literal["high", "medium", "low", "none"]
Escalation = Literal["report", "review", "monitor", "no_action"]


class Filters(BaseModel):
    date_from: date | None = None
    date_to: date | None = None
    countries: list[str] = []
    txn_types: list[str] = []
    amount_min: float | None = None
    amount_max: float | None = None
    min_txn_count: int | None = None      # "10+ transactions"
    customer_segment: str | None = None   # e.g. "business", "pep", "high_risk"


class QueryIntent(BaseModel):
    raw_query: str
    intent: Intent
    filters: Filters = Filters()
    entities: list[str] = []              # normalised customer / txn IDs
    pattern_types: list[PatternType] = []
    top_n: int = 10
    confidence: float = 0.0
    parsed_by: Literal["llm", "rules"]    # honesty about the fallback path


class ToolCall(BaseModel):
    tool: str
    params: dict = {}
    reason: str                           # WHY the planner chose this step
    status: Literal["pending", "ok", "skipped", "error"] = "pending"
    duration_ms: int | None = None
    skip_reason: str | None = None


class ExecutionPlan(BaseModel):
    plan_id: str
    steps: list[ToolCall] = []
    decisions: list[str] = []             # re-planning log, rendered in the UI
    tools_considered_but_skipped: list[str] = []


class Evidence(BaseModel):
    rule_id: str | None = None            # "R1"
    feature: str | None = None            # "pct_just_below_threshold"
    value: float | str
    threshold: float | str | None = None
    note: str = ""


class Flag(BaseModel):
    entity_type: Literal["customer", "transaction"]
    entity_id: str
    risk_score: float = Field(ge=0, le=100)
    risk_level: RiskLevel
    escalation: Escalation
    patterns: list[PatternType] = []
    triggered_rules: list[str] = []       # ["R1", "R4"]
    ml_score: float | None = None         # percentile 0-1
    evidence: list[Evidence] = []
    explanation: str                      # human-readable; NEVER empty
    sar_draft: str | None = None          # HIGH risk only


class AgentResponse(BaseModel):
    query: str
    intent: QueryIntent
    plan: ExecutionPlan
    flags: list[Flag] = []
    tables: dict[str, list[dict]] = {}    # name -> records (UI renders as DataFrame)
    charts: dict[str, dict] = {}          # name -> Plotly figure JSON
    metrics: dict[str, Any] = {}
    summary: str = ""                     # narrative answer to the question asked
    warnings: list[str] = []
```

### HTTP surface *(owner: Track A)*
| Method | Path | Body / Params | Returns |
|---|---|---|---|
| `POST` | `/query` | `{"query": str, "dataset": str \| null}` | `AgentResponse` |
| `GET` | `/health` | — | `{"status": "ok", "llm_available": bool, "mocks": bool}` |
| `GET` | `/dataset/summary` | — | row counts, date range, customer count, column list |
| `GET` | `/plan/{plan_id}` | — | the cached `ExecutionPlan` + flags of a previous run |

Live schema at `http://localhost:8000/docs` once the API is up. **The frontend uses these four endpoints and
nothing else.**

---

## Contract 2 — the tool signature: `backend/tools/base.py` *(owner: Track A · read-only for B)*

```python
from dataclasses import dataclass, field
from typing import Any, Callable
import pandas as pd
from pydantic import BaseModel


@dataclass
class ToolContext:
    df: pd.DataFrame                       # current working transaction set
    customers: pd.DataFrame | None = None
    intent: "QueryIntent" = None
    artifacts: dict[str, Any] = field(default_factory=dict)


class ToolResult(BaseModel):
    ok: bool = True
    df: Any | None = None        # if not None, REPLACES ctx.df for later steps
    artifacts: dict = {}         # MERGED into ctx.artifacts
    tables: dict = {}            # surfaced in AgentResponse.tables
    charts: dict = {}            # Plotly figure JSON -> AgentResponse.charts
    metrics: dict = {}
    notes: list[str] = []        # feed plan.decisions[] / response.warnings[]
    error: str | None = None

    class Config:
        arbitrary_types_allowed = True


TOOLS: dict[str, Callable] = {}

def tool(name: str, params: dict | None = None, description: str = ""):
    """Register a tool. The registry auto-discovers everything decorated with this."""
    def deco(fn):
        fn._tool_name = name
        fn._tool_params = params or {}
        fn._tool_description = description
        TOOLS[name] = fn
        return fn
    return deco
```

### Writing a tool (Track B — this is the only pattern you need)

```python
# backend/tools/rules.py
from backend.tools.base import tool, ToolContext, ToolResult

@tool(
    name="rule_detect",
    params={"patterns": "list[str] — which AML patterns to test"},
    description="Applies rule-based AML detectors R1-R6 to the working set.",
)
def rule_detect(ctx: ToolContext, patterns: list[str] | None = None, **kw) -> ToolResult:
    hits = []   # [{"entity_id", "rule_id", "evidence", "weight"}, ...]
    ...
    return ToolResult(
        artifacts={"rule_hits": hits},
        metrics={"rules_fired": len(hits)},
        notes=[f"R1 structuring matched {n} customers in a 7-day window"],
    )
```

Rules for tool authors:
- **Never mutate `ctx.df` in place.** Return a new frame in `ToolResult.df` if the step narrows the data.
- **Never raise** for an expected condition (empty result, missing column). Return
  `ToolResult(ok=False, error="...")` — the executor turns that into a warning and keeps going.
- `notes` are user-visible. Write them as facts a compliance analyst would want: *"filtered to 1,204 of
  200,000 transactions (2025-06-25 → 2025-07-25)"*.
- Put anything a later tool needs in `artifacts`, under the agreed keys below.

### Agreed `artifacts` keys (the tool-to-tool handshake)
| Key | Written by | Shape |
|---|---|---|
| `transactions_reference` | `load_data` | DataFrame — the transactions as loaded, before `filter_data` narrows `ctx.df`. Read-only for everyone downstream. |
| `features` | `feature_engineer` | DataFrame indexed by `customer_id` |
| `feature_list` | `feature_engineer` | `list[str]` — which features were actually computed |
| `features_reference` | `feature_engineer` | DataFrame, same columns as `features`, computed over `transactions_reference`. The fixed peer group `ml_detect` ranks percentiles against; the same object as `features` when no filter ran. |
| `rule_hits` | `rule_detect` | `list[{entity_id, rule_id, evidence: dict, weight: float}]` |
| `ml_scores` | `ml_detect` | `list[{entity_id, score: float, percentile: float, top_features: list[str]}]` |
| `risk_rows` | `risk_classify` | `list[{entity_id, risk_score, risk_level, escalation, patterns, triggered_rules, evidence}]` |
| `entity_profile` | `entity_lookup` | `dict` — one customer's profile + txn summary |
| `eda` | `eda_profile` | `dict` of stats (charts go in `ToolResult.charts`) |

**The narrator consumes `risk_rows` and turns each into a `Flag`.** As long as `risk_classify` emits that
shape, Track A's narrator works — no other coupling exists between the tracks.

### Tool names the planner may call (fixed list)
`load_data` · `filter_data` · `eda_profile` · `feature_engineer` · `rule_detect` · `ml_detect` ·
`aggregate_query` · `entity_lookup` · `risk_classify`

Track A's `_mocks.py` registers a mock for each of these names under `AML_USE_MOCKS=1`, so the agent runs
end-to-end from hour 2. Adding a name to this list is a **contract change** — standup only.

---

## Contract 3 — the auto-discovering registry: `backend/agent/registry.py` *(owner: Track A)*

```python
import importlib, pkgutil
import backend.tools as tools_pkg
from backend.tools.base import TOOLS

def load_tools(use_mocks: bool = False) -> dict:
    for mod in pkgutil.iter_modules(tools_pkg.__path__):
        if mod.name == "base":
            continue
        if mod.name == "_mocks" and not use_mocks:
            continue
        if mod.name != "_mocks" and use_mocks:
            continue
        importlib.import_module(f"backend.tools.{mod.name}")
    return dict(TOOLS)
```

**Why this matters for the delegation:** a hand-maintained import list would conflict every time Track B
adds a tool. Here, B adds a tool by editing a file B already owns, and the registry picks it up on next
import. Neither person ever edits `registry.py` after the kickoff hour.

---

## Contract 4 — intent → plan mapping *(owner: Track A; B reads it to know which tools get exercised)*

| Intent | Tools invoked, in order | Deliberately skipped |
|---|---|---|
| `full_analysis` | `load_data` → `eda_profile` → `feature_engineer` → `rule_detect` → `ml_detect` → `risk_classify` | — |
| `pattern_search` | `load_data` → `filter_data` → `feature_engineer`(scoped) → `rule_detect` → `ml_detect` → `risk_classify` | `eda_profile` — user asked for a specific pattern, not exploration |
| `threshold_query` | `load_data` → `filter_data` → `aggregate_query` | `feature_engineer`, `ml_detect` — a deterministic count answers this exactly |
| `entity_investigation` | `load_data` → `filter_data`(entity) → `entity_lookup` → `feature_engineer`(scoped) → `rule_detect` → `risk_classify` | `eda_profile`, `ml_detect` — one entity is too small a sample for anomaly detection |
| `ranking` | `load_data` → `filter_data` → `feature_engineer` → `rule_detect` → `ml_detect` → `risk_classify` (sort, `top_n`) | `eda_profile` |
| `eda` | `load_data` → `filter_data` → `eda_profile` | all detection — the user asked to look, not to flag |
| `explain_flag` | cached-run lookup → `narrator` | everything — reuse the prior run |

Unknown / unparseable intent → `full_analysis` on a sample, with an explicit entry in `plan.decisions[]`
saying the intent was unclear and what was assumed.

**Conditional re-planning** (all logged to `plan.decisions[]`):
- `rule_detect` returned 0 hits → append `ml_detect` to widen the net
- filtered subset < 50 rows → drop `ml_detect`, note "insufficient sample for anomaly detection"
- `filter_data` returned 0 rows → stop, return a `summary` explaining which filter emptied the set

This table is the specification of the project's core claim. The plan-divergence test in
[../WORKPLAN.md](WORKPLAN.md) §8 asserts it holds.

---

## Contract 5 — risk bands and escalation *(shared; B implements in `risk.py`, A renders in `narrator.py`)*

```
risk_score = 100 * (0.6 * normalized_rule_weight + 0.4 * ml_percentile)
```
| Score | `risk_level` | `escalation` | Meaning |
|---|---|---|---|
| ≥ 70 | `high` | `report` | File a SAR; narrator emits `sar_draft` |
| 40–69 | `medium` | `review` | Route to a compliance analyst |
| 15–39 | `low` | `monitor` | Enhanced monitoring, no case opened |
| < 15 | `none` | `no_action` | Baseline behaviour |

Justify these bands in `AML_LOGIC.md` — stated business logic scores better with judges than tuned numbers.

---

## Changing a contract

1. Raise it at a standup (H8 / H24 / H40), or message the other person if it genuinely blocks you.
2. **Track A** makes the edit — to `schemas.py`, `tools/base.py`, and this file.
3. A pushes straight to `main` and says so.
4. B runs `git pull --rebase origin main` before continuing.

Never fork the contract locally to unblock yourself. A duplicated `ToolResult` is how a 48-hour project
loses hour 30 to a debugging session neither person can reproduce.
