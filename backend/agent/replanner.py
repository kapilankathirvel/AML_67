"""
Observe -> decide -> act: let the model revise the rest of the plan mid-run.

This is the piece that was missing for the system to be an agent rather than a
planner. Until now the model chose the whole tool sequence up front, before a
single row of data was loaded, and never saw a byte of what its own plan
produced. `ToolContext.artifacts` has carried the observations since the
project began — it was threaded through every step and read by nothing but
three `if` statements in the executor. This module is the first thing to turn
it into a prompt.

What it may and may not do
--------------------------
It may revise the steps that have not run yet. It may not rewrite history, and
it may not touch the risk formula, thresholds or bands — those are the parts a
regulator has to be able to check, and they stay in code.

Every revision is validated by the same `plan_validator.validate_proposal` the
one-shot planner uses. That is deliberate reuse rather than a parallel rule
set: assembling `executed_prefix + proposed_suffix` into one step list and
validating it whole means V3 (no duplicates, so `load_data` cannot be re-run),
V4 (`load_data` first), V5-V7 (dependencies), V12 (the terminal tool still
present) and V14 (capabilities the model may not reach) all apply to a
mid-flight revision for free, with no second implementation to drift.

Failures are observations too
-----------------------------
The first version of this module only ever ran after a step that succeeded:
the executor's three error paths each `continue`d straight past the call. That
made the loop blind in exactly the situation it is most useful for — a step
just died, and everything queued behind it depends on output that now does not
exist. A failed `rule_detect` is the clearest case: `risk_classify` is still
queued with nothing to classify, the hardcoded "no rule hits -> add ml_detect"
rule cannot fire because it lives on the success path, and `ml_detect` is still
a legal, useful revision. The failure note is now passed in and leads the
digest.

Relationship to the executor's three hardcoded rules
----------------------------------------------------
They stay, underneath this, exactly as `planner.build_plan` sits underneath
`llm_planner.plan_query`. If the model declines to revise or proposes something
illegal, the deterministic behaviour is unchanged. The floor is never removed;
this only ever adds a path above it.

Bounds that are not optional
----------------------------
- MAX_REPLANS caps how many times the model may intervene per request. Without
  it, a model that keeps appending one more tool never terminates.
- The observation digest must contain the changing counters. backend/llm/client
  caches on the exact (prompt, schema_hint) pair, so a digest that looked the
  same on iteration 2 would replay iteration 1's answer verbatim — a silent
  loop making the same decision forever while appearing to think.
"""

from __future__ import annotations

from typing import Any, Callable

from backend.agent.plan_validator import validate_proposal
from backend.agent.tool_schema import render_catalog
from backend.llm.client import complete_json
from backend.schemas import ExecutionPlan, QueryIntent, ToolCall
from backend.tools.base import ToolContext

# Two is enough to demonstrate the loop and bounded enough to stay inside the
# frontend's request timeout: each re-plan is a full LLM round trip, and live
# queries already reach ~47s against a 60s ceiling.
MAX_REPLANS = 2

# Failures get their own reserved allowance rather than sharing the one above.
# Measured while adding failure observation: on a five-step plan the routine
# budget is spent on the first two (successful) steps, so a failure at step
# four found nothing left and the loop stayed blind in practice even though it
# could now see. The cap exists to bound latency and guarantee termination, and
# one extra round trip on a path that is rare by definition costs neither:
# worst case is MAX_REPLANS + MAX_FAILURE_REPLANS calls, still a constant.
MAX_FAILURE_REPLANS = 1

_SCHEMA_HINT = (
    'Return JSON: {"revise": true|false, "steps": [{"tool": "<name>", "params": {}, '
    '"reason": "<why this tool, given what was just observed>"}]}. '
    'Set revise=false and omit steps to keep the current plan — that is the '
    'right answer unless the observation genuinely changes what is needed.'
)


def observe(
    ctx: ToolContext,
    executed: list[str],
    remaining: list[str],
    failure: str | None = None,
) -> str:
    """A compact factual digest of what has actually happened so far.

    Deliberately counts and names only. Feeding rows or entity IDs to the model
    would invite it to reason about individual customers, which is the rules'
    job and is where a hallucinated number would do real damage.

    `failure` is the note from a step that just errored. It leads the digest
    because it changes what the rest of the observation means: a zero in
    `rule hits` after a successful `rule_detect` is a finding, whereas the same
    zero after a failed one is just an artifact that was never written.
    """
    lines = [
        f"steps already run: {' -> '.join(executed) or '(none)'}",
        f"steps still queued: {' -> '.join(remaining) or '(none)'}",
    ]
    if failure:
        lines.insert(0, f"THE STEP THAT JUST RAN FAILED: {failure}")

    df = ctx.df
    lines.append(f"working rows: {0 if df is None else len(df)}")

    rule_hits = ctx.artifacts.get("rule_hits")
    if rule_hits is not None:
        rules = sorted({str(h.get("rule_id")) for h in rule_hits})
        lines.append(
            f"rule hits: {len(rule_hits)} across rules {rules or '[]'}, "
            f"{len({str(h.get('entity_id')) for h in rule_hits})} distinct entities"
        )

    ml_scores = ctx.artifacts.get("ml_scores")
    if ml_scores is not None:
        above = sum(1 for m in ml_scores if float(m.get("percentile", 0.0)) >= 0.95)
        lines.append(f"ml scores: {len(ml_scores)} entities, {above} at or above the 95th percentile")

    risk_rows = ctx.artifacts.get("risk_rows")
    if risk_rows is not None:
        levels: dict[str, int] = {}
        for r in risk_rows:
            levels[str(r.get("risk_level"))] = levels.get(str(r.get("risk_level")), 0) + 1
        lines.append(f"risk rows: {len(risk_rows)} scored, by level {levels}")

    features = ctx.artifacts.get("features")
    if features is not None:
        lines.append(f"features computed: {getattr(features, 'shape', ('?', '?'))[1]} columns")

    return "\n".join(lines)


def _build_prompt(
    intent: QueryIntent,
    digest: str,
    remaining: list[ToolCall],
    tools: dict[str, Callable],
    failure: str | None = None,
) -> str:
    queued = " -> ".join(s.tool for s in remaining) or "(nothing)"
    # The default advice ("keeping the plan is usually correct") is calibrated
    # for the happy path, where the queued steps still make sense. After a
    # failure it points the wrong way, so it is replaced rather than softened.
    guidance = (
        "Decide whether the remaining steps should change in light of the "
        "observation above. Keeping the plan is usually correct — revise only "
        "when the observation shows the queued steps will not answer the "
        "query. "
        if not failure
        else (
            "The step that just ran FAILED, so any queued step that needed its "
            "output will fail too. Revising is more likely to be right here "
            "than it usually is: prefer a route to the same answer that does "
            "not depend on what failed. Keep the plan only if the queued steps "
            "genuinely do not need the failed step's output. Note that a step "
            "which has already run cannot be run again, including the one that "
            "failed. "
        )
    )
    return (
        "You are part-way through executing a plan for an AML compliance query, "
        "and you can now see what the steps so far actually produced.\n\n"
        f"QUERY: {intent.raw_query}\n"
        f"PARSED INTENT: {intent.intent}\n\n"
        "WHAT HAS HAPPENED:\n"
        f"{digest}\n\n"
        f"STEPS STILL QUEUED: {queued}\n\n"
        "AVAILABLE TOOLS:\n"
        f"{render_catalog(tools)}\n\n"
        f"{guidance}"
        "If you revise, return the FULL replacement list for the "
        "remaining steps (not the steps that already ran).\n"
        "The same rules apply as when the plan was built: feature_engineer "
        "before rule_detect and ml_detect, risk_classify after a detector, no "
        "tool twice, and do not re-run a step that has already run."
    )


def replan(
    intent: QueryIntent,
    plan: ExecutionPlan,
    ctx: ToolContext,
    executed: list[ToolCall],
    remaining: list[ToolCall],
    tools: dict[str, Callable],
    failure: str | None = None,
) -> list[ToolCall] | None:
    """Ask the model whether to revise the queued steps. None = keep them.

    Returns the validated replacement for `remaining`, or None in every other
    case — model declined, LLM unavailable, unparseable, or the revision failed
    validation. Every outcome is written to `plan.decisions` with a
    `replanner:` prefix, so declining and failing are distinguishable in the
    audit trail rather than both looking like silence.

    `failure` is set when the step that just ran errored. That is the case this
    loop is most useful for and the one it could not originally see, because
    the executor's error paths skipped past it — see the note in executor.py.
    """
    digest = observe(
        ctx,
        [s.tool for s in executed],
        [s.tool for s in remaining],
        failure=failure,
    )
    if failure:
        plan.decisions.append(f"replanner: observing a failed step — {failure}")

    raw = complete_json(
        _build_prompt(intent, digest, remaining, tools, failure=failure), _SCHEMA_HINT
    )
    if raw is None:
        plan.decisions.append("replanner: no usable response — keeping the current plan")
        return None

    if not isinstance(raw, dict) or not raw.get("revise"):
        plan.decisions.append("replanner: model reviewed the observation and kept the plan")
        return None

    # Validate prefix + proposal as one whole plan, so every existing rule
    # applies to the revision. The prefix is what already ran, so a duplicate
    # of it fails V3 rather than silently re-running load_data.
    proposed = raw.get("steps")
    if not isinstance(proposed, list):
        plan.decisions.append("replanner: revision had no usable 'steps' — keeping the current plan")
        return None

    prefix = [{"tool": s.tool, "params": dict(s.params), "reason": s.reason or "already executed"}
              for s in executed]
    result = validate_proposal({"steps": prefix + proposed}, intent, tools)

    if not result.ok:
        plan.decisions.append("replanner: revision rejected — keeping the current plan")
        for r in result.rejections:
            plan.decisions.append(f"replanner: rejected — {r}")
        return None

    suffix = result.steps[len(prefix):]
    if not suffix:
        plan.decisions.append("replanner: revision left nothing to run — keeping the current plan")
        return None

    plan.decisions.append(
        "replanner: revised the remaining plan after observing results — "
        f"{' -> '.join(s.tool for s in remaining) or '(nothing)'} "
        f"becomes {' -> '.join(s.tool for s in suffix)}"
    )
    for note in result.notes:
        plan.decisions.append(f"replanner: {note}")
    return suffix


def digest_for_tests(
    ctx: ToolContext,
    executed: list[str],
    remaining: list[str],
    failure: str | None = None,
) -> str:
    """Public alias so tests can assert on the observation without importing a
    private name. The digest's contents are load-bearing: they are what makes
    each iteration's prompt distinct and therefore cache-missing."""
    return observe(ctx, executed, remaining, failure=failure)
