"""
Executor: runs an ExecutionPlan's steps against the tool registry. Owner: Track A.

Threads one ToolContext through all steps, times each step, isolates tool
failures (a failing tool marks its step "error" and the run continues), and
performs the conditional re-planning specified in docs/CONTRACTS.md Contract 4:
  - rule_detect returns 0 hits -> append ml_detect
  - filtered subset < 50 rows -> drop a queued ml_detect
  - filter_data returns 0 rows -> stop early with an explanatory summary
"""

import re
import time
from typing import Any

import numpy as np

from backend.agent import registry
from backend.agent import replanner
from backend.agent.narrator import build_flags
from backend.config import settings
from backend.schemas import AgentResponse, ExecutionPlan, QueryIntent, ToolCall
from backend.tools.base import ToolContext

_TOOLS_CACHE: dict[str, Any] | None = None


def _get_tools() -> dict[str, Any]:
    global _TOOLS_CACHE
    if _TOOLS_CACHE is None:
        _TOOLS_CACHE = registry.load_tools(use_mocks=settings.aml_use_mocks)
    return _TOOLS_CACHE


def run_plan(intent: QueryIntent, plan: ExecutionPlan) -> AgentResponse:
    tools = _get_tools()
    ctx = ToolContext(df=None, customers=None, intent=intent, artifacts={})
    response = AgentResponse(query=intent.raw_query, intent=intent, plan=plan)

    steps = list(plan.steps)
    i = 0
    replans_used = 0
    failure_replans_used = 0
    while i < len(steps):
        step = steps[i]
        result, failure = _execute_step(step, tools, ctx)

        # The three hardcoded rules below only make sense after a step that
        # produced something, so they stay on the success path. The re-planner
        # does not: a failure is an observation, and the most useful one there
        # is. See the "Failures are observations too" note in replanner.py.
        if failure is not None:
            response.warnings.append(failure)
        else:
            if result.df is not None:
                ctx.df = result.df
            ctx.artifacts.update(result.artifacts)
            response.tables.update(result.tables)
            response.charts.update(result.charts)
            response.metrics.update(result.metrics)
            plan.decisions.extend(result.notes)

            if step.tool == "load_data" and ctx.customers is not None and intent.entities:
                resolved, resolve_notes = _resolve_entities(intent.entities, ctx.customers)
                # always log resolve_notes (even the no-match case) and keep entity_lookup's
                # already-built params in sync — not just when resolved != intent.entities,
                # since an unmatched entity still produced a note worth surfacing
                intent.entities = resolved
                plan.decisions.extend(resolve_notes)
                for later in steps[i + 1:]:
                    if later.tool == "entity_lookup" and "entity_id" in later.params:
                        later.params["entity_id"] = intent.entities[0] if intent.entities else None

            if step.tool == "filter_data" and ctx.df is not None:
                if len(ctx.df) == 0:
                    plan.decisions.append("filter_data returned 0 rows — stopping execution early")
                    response.summary = "No transactions matched the given filters."
                    plan.steps = steps
                    return response
                if len(ctx.df) < 50:
                    remaining = steps[i + 1:]
                    still_has_ml = any(s.tool == "ml_detect" for s in remaining)
                    if still_has_ml:
                        steps[i + 1:] = [s for s in remaining if s.tool != "ml_detect"]
                        plan.decisions.append(
                            "sample too small for anomaly detection (<50 rows) — skipping ml_detect"
                        )

            if step.tool == "rule_detect":
                hits = ctx.artifacts.get("rule_hits", [])
                already_planned = any(s.tool == "ml_detect" for s in steps[i + 1:])
                if not hits and not already_planned:
                    steps.insert(i + 1, ToolCall(tool="ml_detect", reason="no rule hits — widening to ML anomaly detection"))
                    plan.decisions.append("no rule hits — widening the net with ml_detect")

        # Observe -> decide -> act. Runs AFTER the three rules above, which
        # remain the floor: if the model declines or proposes something
        # illegal, behaviour is exactly what it was before this existed.
        # Placed here so the model sees the result of the step that just ran,
        # including any adjustment those rules just made, and — when that step
        # errored — the failure itself.
        # Failures draw on their own reserved allowance — see MAX_FAILURE_REPLANS.
        # Sharing one budget meant the routine steps spent it before any failure
        # could use it, which left the loop blind in practice.
        budget_left = (
            failure_replans_used < replanner.MAX_FAILURE_REPLANS
            if failure is not None
            else replans_used < replanner.MAX_REPLANS
        )
        if settings.aml_llm_replanner and budget_left and i + 1 < len(steps):
            revised = replanner.replan(
                intent=intent,
                plan=plan,
                ctx=ctx,
                executed=steps[: i + 1],
                remaining=steps[i + 1:],
                tools=tools,
                failure=failure,
            )
            if failure is not None:
                failure_replans_used += 1
            else:
                replans_used += 1
            if revised is not None:
                steps[i + 1:] = revised

        i += 1

    risk_rows = ctx.artifacts.get("risk_rows", [])

    if intent.intent in ("entity_investigation", "explain_flag") and intent.entities:
        # filter_data has no per-entity dimension (see planner.py), so risk_classify
        # scores the whole population — narrow to the requested entity/entities here.
        wanted = set(intent.entities)
        risk_rows = [r for r in risk_rows if r.get("entity_id") in wanted]

    if intent.intent == "ranking":
        # risk_classify has no top_n param (see planner.py) — rows arrive pre-sorted
        # descending by risk_score (backend/tools/risk.py), so a plain slice is correct.
        risk_rows = risk_rows[: intent.top_n]

    response.flags = build_flags(risk_rows)
    if not response.summary:
        response.summary = _summarise(intent, response)
    plan.steps = steps

    # tables/charts/metrics are Any-typed (Contract 1) — Pydantic validates but
    # does not coerce their *contents*, so a raw numpy.ndarray (e.g. embedded in
    # eda_profile's Plotly-figure .to_dict() output) passes straight through.
    # FastAPI's JSON response serialization then crashes on it — invisible to
    # every test here, since none of them serialize the response to JSON
    # (run_plan() is called directly, only the real HTTP layer hits this).
    response.tables = _sanitize_for_json(response.tables)
    response.charts = _sanitize_for_json(response.charts)
    response.metrics = _sanitize_for_json(response.metrics)
    return response


def _execute_step(step: ToolCall, tools: dict[str, Any], ctx: ToolContext) -> tuple[Any, str | None]:
    """Run one step. Returns (result, failure_note); exactly one is meaningful.

    Collapses the three ways a step can fail — no such tool, the tool raised,
    the tool returned ok=False — into a single value the caller handles in one
    place. That is what lets the re-planner see failures: each of these was
    previously a `continue` in the main loop, which skipped every decision
    point below it, including the one whose entire job is deciding what to do
    when something goes wrong.

    The failure strings are unchanged from those three branches, because they
    are already surfaced to the user as `response.warnings`.
    """
    fn = tools.get(step.tool)
    if fn is None:
        step.status = "error"
        return None, f"unknown tool '{step.tool}' — skipped"

    t0 = time.perf_counter()
    try:
        result = fn(ctx, **step.params)
    except Exception as exc:  # isolate any tool failure, never let it 500 the API
        step.status = "error"
        step.duration_ms = int((time.perf_counter() - t0) * 1000)
        return None, f"{step.tool} raised {type(exc).__name__}: {exc}"
    step.duration_ms = int((time.perf_counter() - t0) * 1000)

    if not result.ok:
        step.status = "error"
        return None, result.error or f"{step.tool} returned ok=False"

    step.status = "ok"
    return result, None


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _resolve_entities(entities: list[str], customers: Any) -> tuple[list[str], list[str]]:
    """Best-effort resolution of parser-normalised entity IDs against the real
    customer_id values in the loaded dataset.

    intent_parser normalises a bare number in a query (e.g. "4521") into
    "C-04521" (docs/CONTRACTS.md Contract 0's stated convention), but real
    customer IDs follow the data generator's own scheme (e.g. "C-STR02",
    "C-N0001"), so that normalised ID never exact-matches a real record.
    Falls back to leaving an entity unresolved (matches nothing downstream —
    the same graceful "not found" behaviour as before this fix) when no real
    customer shares its numeric id.
    """
    if customers is None or "customer_id" not in getattr(customers, "columns", []):
        return entities, []

    real_ids = customers["customer_id"].astype(str).tolist()
    real_id_set = set(real_ids)
    resolved: list[str] = []
    notes: list[str] = []

    for entity in entities:
        if entity in real_id_set:
            resolved.append(entity)
            continue

        digits = re.sub(r"\D", "", entity)
        entity_num = int(digits) if digits else None
        candidates = []
        if entity_num is not None:
            for rid in real_ids:
                rid_digits = re.sub(r"\D", "", rid)
                if rid_digits and int(rid_digits) == entity_num:
                    candidates.append(rid)

        if candidates:
            match = candidates[0]
            resolved.append(match)
            extra = f" ({len(candidates)} candidates shared numeric id {entity_num}, used first)" if len(candidates) > 1 else ""
            notes.append(f"resolved entity '{entity}' to real customer '{match}' by numeric id{extra}")
        else:
            resolved.append(entity)
            notes.append(f"no real customer found matching '{entity}' — proceeding with original id (no match expected)")

    return resolved, notes


def _summarise(intent: QueryIntent, response: AgentResponse) -> str:
    n = len(response.flags)
    if intent.intent == "entity_investigation":
        entity = intent.entities[0] if intent.entities else "the requested entity"
        if n:
            f = response.flags[0]
            return f"{entity} is flagged {f.risk_level} risk (score {f.risk_score:.0f}) — recommended action: {f.escalation}."
        return f"{entity} shows no flagged risk indicators in the current data."
    if intent.intent == "explain_flag":
        entity = intent.entities[0] if intent.entities else "the requested entity/transaction"
        if n:
            f = response.flags[0]
            return f"{entity} was flagged {f.risk_level} risk (score {f.risk_score:.0f}): {f.explanation}"
        return f"No active flag found for {entity} in the current data."
    if intent.intent == "threshold_query":
        count = response.metrics.get("row_count", n)
        return f"{count} customer(s) matched the specified threshold."
    if intent.intent == "eda":
        txn_count = response.metrics.get("txn_count")
        cust_count = response.metrics.get("customer_count")
        if txn_count is not None and cust_count is not None:
            return f"Profiled {txn_count:,} transactions across {cust_count:,} customers — see the charts and metrics below."
        return "Dataset profile ready — see the charts and metrics below."
    if n:
        return f"{n} entity(ies) flagged for review across the analysed data."
    return "No suspicious activity was flagged for this query."
