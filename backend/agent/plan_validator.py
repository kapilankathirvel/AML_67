"""
Whitelist validation for an LLM-proposed execution plan.

This is the component that makes LLM tool-selection safe to run in a compliance
setting. The model proposes; this module decides whether the proposal is legal;
backend/agent/llm_planner.py falls back to the deterministic planner if it is
not. Nothing an LLM returns reaches the executor unchecked.

What it validates and what it deliberately does not
---------------------------------------------------
Two kinds of rule, and the boundary between them was drawn by measurement
rather than by taste.

V0-V11 enforce *safety and dependency legality*. V5-V7 mirror real
preconditions in the tool bodies — backend/tools/rules.py reads
ctx.artifacts["features"], backend/tools/risk.py reads rule_hits/ml_scores — so
a plan that passes cannot fail on a missing artifact.

V12 and V13 enforce *answerability*: the plan must contain the tool that
produces the response its intent promises (V12), and any parameter drawn from a
closed set must actually be in it (V13). Both were added after a plan passed
every other rule and still returned nothing.

V14 enforces *authority*: some capabilities are not the model's to invoke at
any value. That is a stronger claim than validating a value, and it is the one
that belongs in a compliance system — a plan may choose how to analyse, but it
may not choose which dataset the product runs on or write to disk. This was not in the original design, which held
that anything beyond dependency legality would mean encoding the deterministic
planner's opinions back into the validator. Measuring showed that reasoning was
too permissive. With V0-V11 alone a local model hit 60% acceptance while only 1
plan in 15 was useful: it had learned that shorter plans pass, and a truncated
plan satisfies every ordering rule vacuously. "Who are my riskiest customers?"
produced load_data -> filter_data -> feature_engineer — legal, and it returns
nothing.

The distinction that makes V12 legitimate rather than a taste rule: it
constrains the plan's OUTPUT, not the route to it. A ranking query that cannot
return a ranking is broken, not suboptimal. Everything upstream stays the
model's call — which filters, rules or ML or both, whether to profile first.

What still passes and should: a legal, answerable, but clumsy plan (profiling
the dataset before a single-entity lookup, say). The trace records what was
chosen and why; judging elegance is not a whitelist's job.

All violations are collected rather than short-circuiting on the first, so the
audit trail states everything wrong with a rejected proposal.

Repair vs rejection
-------------------
Some defects have exactly one correct fix and no judgement in applying it:
a missing load_data (every plan needs it), filter_data's params when the model
left them empty (they come from the parsed query), entity_lookup's entity_id.
Those are repaired and logged rather than rejected — see
_ensure_load_data_first and _normalise. Anything involving a real choice —
which detectors run, which patterns to test — is never repaired, because
silently rewriting those would make "the LLM chose this plan" untrue.

The V10 exemption is load-bearing
---------------------------------
Parameter names are only checked when the tool actually declares a parameter
schema. backend/tools/_mocks.py declares `params` on none of its nine tools, so
under AML_USE_MOCKS=1 (the default, and what most of the test suite runs)
every declared set is empty. Strict checking would reject every non-trivial plan
in exactly the configuration used for testing. An empty set therefore means
"cannot be validated" and emits a note, so the gap is itself auditable rather
than silent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, get_args

from backend.agent.tool_schema import declared_params
from backend.schemas import PatternType, QueryIntent, ToolCall

# A generous ceiling: the longest deterministic plan is 8 steps and no tool may
# repeat, so 12 cannot constrain a legitimate plan. It exists to bound a
# degenerate model response, not to express a policy about plan length.
MAX_STEPS = 12

# Tools whose preconditions are satisfied by another tool having already run.
# Kept as data so a new dependency is one line, not a new branch.
_REQUIRES_BEFORE: dict[str, tuple[str, ...]] = {
    "rule_detect": ("feature_engineer",),
    "ml_detect": ("feature_engineer",),
}

# risk_classify needs at least one of these, not all of them.
_RISK_CLASSIFY_ANY_OF = ("rule_detect", "ml_detect")

# V12 — the terminal output each intent promises in its response.
#
# This is the one rule here that is about the plan DOING ITS JOB rather than
# about dependency legality, and it was added after measuring: with V0-V11
# alone, a local model reached 60% acceptance while only 1 in 15 plans was
# actually useful. It had learned to emit SHORT plans, and a truncated plan
# satisfies every ordering rule trivially — "who are my riskiest customers?"
# came back as load_data -> filter_data -> feature_engineer, which is legal,
# computes features, detects nothing, and returns zero flags.
#
# The line this draws is deliberately narrow: a plan must be able to produce
# the response shape its intent promises. That is not a matter of taste, which
# is why it belongs here — a ranking query that can return no ranking is
# broken, not merely suboptimal. Everything upstream of the terminal tool is
# still the model's choice: which filters, whether to use rules or ML or both,
# whether to profile first. This constrains the output, not the route to it.
# V13 — parameters whose VALUES come from a closed set, not just whose names
# have to exist.
#
# Added after a live run against a local model. The plan below was ACCEPTED —
# every tool real, every dependency satisfied, the terminal tool present:
#
#   proposed = load_data -> filter_data -> feature_engineer
#              -> rule_detect -> risk_classify
#   feature_engineer: 294 customers x 0 features for pattern_types=[risk]
#   rule_detect: 0 total hits, rules evaluated=[]
#   risk_classify: no rule hits and no ML anomalies - clean dataset
#
# "risk" is not a PatternType. feature_engineer computed nothing, rule_detect
# ran nothing, and "who are my riskiest customers?" returned an empty answer
# with no warning. V10 checks that `pattern_types` is a declared parameter NAME
# on that tool; nothing checked that its contents were legal. That is the same
# failure V12 exists to prevent — a plan that cannot answer — arriving through
# a different door.
#
# Scope is deliberately narrow: only parameters whose legal values are a fixed
# literal in the frozen schema. Free-form values (amounts, dates, entity ids)
# are the tools' business, and guessing at their validity here would duplicate
# logic that already lives in the tools and drift from it.
_PATTERN_VALUES: frozenset[str] = frozenset(get_args(PatternType))

_ENUM_PARAMS: dict[str, frozenset[str]] = {
    # feature_engineer and ml_detect spell it pattern_types; rule_detect's
    # frozen contract spells it patterns, with pattern_types as an alias.
    "pattern_types": _PATTERN_VALUES,
    "patterns": _PATTERN_VALUES,
}

# V14 — capabilities that are not the model's to invoke.
#
# V13 says "a parameter whose legal values are knowable must be checked, not
# merely named". V14 is the stronger neighbouring claim: some parameters must
# not be reachable from an LLM plan at all, whatever value is offered.
#
# load_data is the case that forced it. It declares `source` and
# `force_rebuild` (backend/tools/data_loader.py), V10 checks only that those
# names exist, and _normalise never touches load_data — so a proposal of
# {"source": "ibm", "force_rebuild": true} was fully legal. That would switch
# the product onto a different dataset mid-request AND trigger a parquet cache
# rebuild, which is a filesystem write, on a model's say-so. In a system whose
# whole claim is that the validator makes LLM tool-selection safe, a
# model-triggered disk write is the wrong side of the line.
#
# 'ibm'/'ibm_stratified' are additionally blocked for a plain correctness
# reason: they read data/raw/, which is empty in this repo, so a plan choosing
# them fails at runtime anyway. Rejecting turns a confusing mid-run error into
# a clear, logged refusal.
_ALLOWED_SOURCES: frozenset[str] = frozenset({"synthetic", "synthetic_alt"})

# Params an LLM plan may not set at any value. Each maps to why, so the
# rejection message explains rather than just refuses.
_FORBIDDEN_PARAMS: dict[tuple[str, str], str] = {
    ("load_data", "force_rebuild"):
        "it rewrites the on-disk parquet cache — a model must not trigger a filesystem write",
    ("load_data", "nrows"):
        "it is an 'ibm'-source testing knob and has no meaning for the sources a plan may use",
    ("load_data", "target_size"):
        "it only affects 'ibm_stratified', which a plan may not select",
    ("load_data", "max_pos_customers"):
        "it only affects 'ibm_stratified', which a plan may not select",
    ("load_data", "seed"):
        "it only affects 'ibm_stratified' sampling, which a plan may not select",
}

_REQUIRED_TERMINAL: dict[str, str] = {
    "full_analysis": "risk_classify",
    "pattern_search": "risk_classify",
    "entity_investigation": "risk_classify",
    "ranking": "risk_classify",
    "explain_flag": "risk_classify",
    "eda": "eda_profile",
    "threshold_query": "aggregate_query",
}


@dataclass
class ValidationResult:
    ok: bool
    steps: list[ToolCall] = field(default_factory=list)
    rejections: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _coerce_steps(raw: Any, rejections: list[str]) -> list[dict] | None:
    """V0 — shape check. Returns None if the payload is unusable."""
    if not isinstance(raw, dict):
        rejections.append(f"malformed proposal: expected an object, got {type(raw).__name__}")
        return None

    steps = raw.get("steps")
    if not isinstance(steps, list):
        rejections.append(f"malformed proposal: 'steps' must be a list, got {type(steps).__name__}")
        return None
    if not steps:
        rejections.append("malformed proposal: 'steps' is empty")
        return None

    clean: list[dict] = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            rejections.append(f"malformed proposal: step {i} is {type(step).__name__}, expected an object")
            return None
        tool = step.get("tool")
        if not isinstance(tool, str) or not tool.strip():
            rejections.append(f"malformed proposal: step {i} has no usable 'tool' name")
            return None
        params = step.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            rejections.append(f"malformed proposal: step {i} ('{tool}') has non-object 'params'")
            return None
        reason = step.get("reason", "")
        clean.append({"tool": tool.strip(), "params": params, "reason": str(reason or "").strip()})
    return clean


def _ensure_load_data_first(steps: list[dict]) -> tuple[list[dict], str]:
    """Put load_data at the front, prepending it if the model left it out.

    Repair rather than rejection, and the reasoning is worth stating because it
    is the one place this module bends.

    load_data is not a planning decision. All eight branches of the
    deterministic planner start with it, every legal plan requires it, and no
    query exists for which omitting it is correct — so it carries zero
    information about how the model chose to answer the question. There is
    exactly one right fix and no judgement in applying it, which puts it in the
    same class as injecting filter_data's params from the parsed query.

    Measured cost of treating it as a rejection instead: against a local
    qwen2.5:3b-instruct it was 8 of 13 rejections — the model kept opening with
    filter_data — so more than half the failures were ceremony rather than bad
    planning.

    The line this does NOT cross: nothing here adds or removes a *detector*, or
    decides which patterns to test. Those are the choices being delegated, and
    repairing them would make "the LLM chose the plan" untrue. A duplicated
    load_data is still a rejection (V3), because two of them means the model
    misunderstood the plan rather than merely omitted a preamble.
    """
    names = [s["tool"] for s in steps]
    if names.count("load_data") > 1:
        return steps, ""  # V3 will reject; repairing would hide the confusion
    if names and names[0] == "load_data":
        return steps, ""

    if "load_data" in names:
        idx = names.index("load_data")
        moved = steps.pop(idx)
        return [moved] + steps, (
            f"moved load_data from position {idx + 1} to the front "
            "(every plan must start by loading the data)"
        )

    return [{
        "tool": "load_data",
        "params": {},
        "reason": "load the working dataset (required first step of every plan)",
    }] + steps, "prepended load_data (omitted by the planner; every plan requires it)"


def validate_proposal(
    raw: Any,
    intent: QueryIntent,
    tools: dict[str, Callable],
) -> ValidationResult:
    """Validate an LLM plan proposal against the registered tools.

    `tools` must be the SAME registry snapshot the executor will dispatch
    against — see llm_planner.plan_query, which passes executor._get_tools().
    Validating against real tools while mocks execute would let a plan through
    that the executor cannot actually run.
    """
    rejections: list[str] = []
    notes: list[str] = []

    steps = _coerce_steps(raw, rejections)
    if steps is None:
        return ValidationResult(ok=False, rejections=rejections)

    steps, load_note = _ensure_load_data_first(steps)
    if load_note:
        notes.append(load_note)

    names = [s["tool"] for s in steps]
    declared = declared_params(tools)

    # V1 — length
    if len(steps) > MAX_STEPS:
        rejections.append(f"proposed {len(steps)} steps, max is {MAX_STEPS}")

    # V2 — every tool exists
    for name in names:
        if name not in tools:
            rejections.append(f"unknown tool '{name}' not in registry")

    # V3 — no duplicates
    for name in sorted(set(names)):
        if names.count(name) > 1:
            rejections.append(f"tool '{name}' proposed more than once")

    # V4 — load_data first. Normally already true: _ensure_load_data_first
    # repairs it above. This remains as a backstop for the one case that is not
    # repaired (a duplicated load_data, which V3 also flags).
    if names[0] != "load_data":
        rejections.append(f"load_data must be the first step, got '{names[0]}'")

    # V5/V6 — ordering dependencies
    for tool_name, prerequisites in _REQUIRES_BEFORE.items():
        if tool_name not in names:
            continue
        idx = names.index(tool_name)
        for prereq in prerequisites:
            if prereq not in names or names.index(prereq) > idx:
                rejections.append(f"{tool_name} requires {prereq} before it")

    # V7 — risk_classify needs something to fuse
    if "risk_classify" in names:
        idx = names.index("risk_classify")
        if not any(d in names and names.index(d) < idx for d in _RISK_CLASSIFY_ANY_OF):
            rejections.append("risk_classify requires rule_detect or ml_detect before it")

    # V8 — filter_data cannot precede the data
    if "filter_data" in names and names.index("filter_data") == 0:
        rejections.append("filter_data must follow load_data")

    # V9 — entity_lookup needs an entity to look up
    if "entity_lookup" in names and not intent.entities:
        rejections.append("entity_lookup proposed but no entity was extracted from the query")

    # V10 — parameter names, only where a schema was declared
    for step in steps:
        allowed = declared.get(step["tool"])
        if allowed is None:
            continue  # unknown tool; already rejected by V2
        if not allowed:
            if step["params"]:
                notes.append(
                    f"params for '{step['tool']}' not validated — tool declares no param schema"
                )
            continue
        for key in sorted(step["params"]):
            if key not in allowed:
                rejections.append(f"{step['tool']}: undeclared param '{key}'")

    # V11 — every step must justify itself; the reason is user-facing audit text
    for step in steps:
        if not step["reason"]:
            rejections.append(f"{step['tool']}: missing reason")

    # V12 — the plan must be able to answer the question that was asked
    required = _REQUIRED_TERMINAL.get(intent.intent)
    if required and required not in names:
        rejections.append(
            f"intent '{intent.intent}' needs {required} to produce a result, "
            f"but the plan ends at '{names[-1]}'"
        )

    # V13 — closed-set parameter VALUES, not just names
    for step in steps:
        for key, allowed_values in _ENUM_PARAMS.items():
            if key not in step["params"]:
                continue
            value = step["params"][key]
            values = value if isinstance(value, list) else [value]
            for item in values:
                if item not in allowed_values:
                    rejections.append(
                        f"{step['tool']}: '{item}' is not a valid {key} — "
                        f"expected one of {sorted(allowed_values)}"
                    )

    # V14 — capabilities a plan may not reach at all
    for step in steps:
        for key in sorted(step["params"]):
            why = _FORBIDDEN_PARAMS.get((step["tool"], key))
            if why is not None:
                rejections.append(f"{step['tool']}: may not set '{key}' — {why}")

        if step["tool"] == "load_data" and "source" in step["params"]:
            source = step["params"]["source"]
            if source not in _ALLOWED_SOURCES:
                rejections.append(
                    f"load_data: source '{source}' is not available to a plan — "
                    f"expected one of {sorted(_ALLOWED_SOURCES)}"
                )

    if rejections:
        return ValidationResult(ok=False, rejections=rejections, notes=notes)

    calls, norm_notes = _normalise(steps, intent)
    return ValidationResult(ok=True, steps=calls, rejections=[], notes=notes + norm_notes)


def _normalise(steps: list[dict], intent: QueryIntent) -> tuple[list[ToolCall], list[str]]:
    """Repair a valid plan's params from the parsed intent.

    This runs only after every rule has passed, and can never reject: it fills
    in what the model left out rather than judging it. Two things must be
    injected or the plan silently does the wrong thing:

      filter_data — without the parsed Filters the query's own date/amount/
      country constraints are dropped and the plan analyses everything.

      entity_lookup — the `entity_id` KEY must exist even when the value is
      None, because backend/agent/executor.py only re-syncs the resolved real
      customer ID into a later step `if "entity_id" in later.params`.
    """
    # Imported here rather than at module scope: planner imports nothing from
    # this module, and keeping the dependency one-way avoids a cycle if the
    # planner ever wants to validate its own output.
    from backend.agent.planner import _filter_kwargs

    notes: list[str] = []
    calls: list[ToolCall] = []

    for step in steps:
        params = dict(step["params"])

        if step["tool"] == "filter_data":
            injected = []
            for key, value in _filter_kwargs(intent.filters).items():
                if value is None:
                    continue
                if key not in params:
                    params[key] = value
                    injected.append(key)
            if injected:
                notes.append(
                    "injected filter_data params from the parsed query: " + ", ".join(sorted(injected))
                )

        if step["tool"] == "entity_lookup" and "entity_id" not in params:
            params["entity_id"] = intent.entities[0] if intent.entities else None
            notes.append("injected entity_lookup entity_id from the parsed query")

        if step["tool"] == "aggregate_query":
            # backend/agent/planner.py always passes group_by/agg_func/threshold
            # for a threshold_query; nothing was passing them into an LLM plan.
            # `threshold` is the one that silently produces a WRONG ANSWER
            # rather than an obviously broken one: aggregate_query's signature
            # defaults it to None, so "which customers made 10+ transactions?"
            # ran the aggregation with no threshold at all and returned every
            # sender. Same failure shape as pattern_types=["risk"] — a legal
            # plan that quietly answers a different question.
            injected = []
            for key, value in (
                ("group_by", ["sender_id"]),
                ("agg_func", "count"),
                ("threshold", intent.filters.min_txn_count),
            ):
                if value is None:
                    continue
                if key not in params:
                    params[key] = value
                    injected.append(key)
            if injected:
                notes.append(
                    "injected aggregate_query params from the parsed query: "
                    + ", ".join(sorted(injected))
                )

        calls.append(ToolCall(tool=step["tool"], params=params, reason=step["reason"]))

    return calls, notes
