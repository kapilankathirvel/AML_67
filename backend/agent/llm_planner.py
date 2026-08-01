"""
LLM-driven planning with a deterministic floor.

The model is shown the real tool catalog (built from each tool's own @tool
declaration — see backend/agent/tool_schema.py) and proposes a sequence of tool
calls for the specific query. backend/agent/plan_validator.py then decides
whether that proposal is legal. If the LLM is unavailable, returns nothing
usable, or proposes something illegal, this falls back to
backend/agent/planner.py::build_plan and records why.

Relationship to docs/CONTRACTS.md Contract 4
--------------------------------------------
Contract 4 specifies a fixed intent -> tool table. That contract is NOT
weakened here and the file is not edited: build_plan still implements it
exactly, and build_plan is what the fallback returns, so the contract remains
the guaranteed floor of this system's behaviour. What changes is that the
planner may now do better than the floor for a specific query — and when it
does, the divergence is written into plan.decisions rather than being silent.
An auditor can read what was proposed, whether it was accepted, why it was
rejected if it was, and what actually ran including the executor's own mid-run
re-planning.

Why build_plan is not modified
------------------------------
There is no tests/conftest.py in this repo. Every test file stubs the LLM by
monkeypatching the importing module's own `complete_json` symbol, and a .env
with live API keys sits at the repo root. Putting an LLM call inside build_plan
would make roughly 26 existing call sites — across test_planner.py,
test_executor.py, test_integration.py and evaluation/run_evaluation.py — start
issuing real network requests. Keeping the LLM path in a separate entry point
means every one of those callers is unaffected by construction rather than by
remembering to stub something.

The `aml_llm_planner` setting defaults to False for the same reason: the
default configuration of this repo behaves exactly as it did before this module
existed.
"""

from __future__ import annotations

import uuid
from typing import Any, Callable

from backend.agent.plan_validator import _REQUIRED_TERMINAL, validate_proposal
from backend.agent.planner import build_plan
from backend.agent.tool_schema import render_catalog
from backend.config import settings
from backend.llm.client import complete_json
from backend.schemas import ExecutionPlan, QueryIntent

# The two hardest rules are repeated here rather than left to the prompt body.
# Measured reason: against a local qwen2.5:3b-instruct, 10 of 15 proposals
# omitted load_data as the first step and 10 of 15 left `reason` empty, while 0
# of 15 produced unparseable JSON. The model reliably honours the *schema* and
# unreliably honours prose further up the prompt, so the constraints that were
# being dropped belong in the schema line.
_SCHEMA_HINT = (
    'Return JSON: {"steps": [{"tool": "<tool name>", "params": {}, '
    '"reason": "<one sentence on why this tool is needed for THIS query>"}]}. '
    'The FIRST step must always be {"tool": "load_data", ...}. '
    'Every step must include a non-empty "reason". '
    'Never emit a step without both "tool" and "reason".'
)

# Stated in the prompt AND enforced in plan_validator. The prompt raises the
# acceptance rate; the validator is what actually holds. Neither replaces the
# other — a model that ignores the prompt is exactly the case the validator
# exists for.
_DEPENDENCY_RULES = """\
- load_data MUST be the first step, always, and appear exactly once
- every step MUST carry a non-empty "reason"
- feature_engineer must come before rule_detect and before ml_detect
- risk_classify must come after rule_detect or ml_detect
- filter_data must come after load_data
- entity_lookup only if the query names a specific customer
- no tool may appear more than once; at most 12 steps
- only use parameter names listed under that tool above"""

# A worked example. Small models pattern-match a concrete example far more
# reliably than they follow a list of constraints — and it demonstrates the
# two rules that were being dropped (load_data first, reason on every step)
# rather than only asserting them.
_EXAMPLE = """\
EXAMPLE — for the query "which customers are structuring under $10,000?":
{"steps": [
  {"tool": "load_data", "params": {}, "reason": "the query needs the transaction set"},
  {"tool": "filter_data", "params": {"amount_max": 10000}, "reason": "restrict to the amount band asked about"},
  {"tool": "feature_engineer", "params": {"pattern_types": ["structuring"]}, "reason": "structuring features are needed by the rules"},
  {"tool": "rule_detect", "params": {"patterns": ["structuring"]}, "reason": "apply the structuring detectors"},
  {"tool": "risk_classify", "params": {}, "reason": "turn the rule hits into ranked risk scores"}
]}"""


def _describe_intent(intent: QueryIntent) -> str:
    """Compact one-line summary of the parsed intent; omits empty fields."""
    bits = [f"intent={intent.intent}"]
    if intent.entities:
        bits.append(f"entities={intent.entities}")
    if intent.pattern_types:
        bits.append(f"patterns={intent.pattern_types}")
    active = {
        k: v for k, v in intent.filters.model_dump().items()
        if v not in (None, [], "")
    }
    if active:
        bits.append(f"filters={active}")
    if intent.top_n:
        bits.append(f"top_n={intent.top_n}")
    return "  ".join(bits)


def _build_prompt(intent: QueryIntent, tools: dict[str, Callable]) -> str:
    # State THIS query's required terminal tool explicitly rather than leaving
    # the model to infer it from a general rule. Measured: a general "detection
    # queries must end with risk_classify" line did not stop the model
    # truncating plans, because it has to first decide its query is a detection
    # query. Naming the tool removes that inference step.
    required = _REQUIRED_TERMINAL.get(intent.intent)
    requirement = (
        f"\nThis query's intent is '{intent.intent}', so the plan MUST include "
        f"{required} — without it the plan returns nothing and is rejected.\n"
        if required else ""
    )
    return (
        "You are planning tool calls for an AML compliance query.\n\n"
        f"QUERY: {intent.raw_query}\n"
        f"PARSED INTENT: {_describe_intent(intent)}\n"
        f"{requirement}\n"
        "AVAILABLE TOOLS:\n"
        f"{render_catalog(tools)}\n\n"
        "DEPENDENCY RULES YOU MUST OBEY:\n"
        f"{_DEPENDENCY_RULES}\n\n"
        f"{_EXAMPLE}\n\n"
        "Choose only the tools this specific query needs, but never at the cost "
        "of the required tool above. Do not include a tool whose output would "
        "not be used to answer the query."
    )


def propose_plan(intent: QueryIntent, tools: dict[str, Callable]) -> dict[str, Any] | None:
    """Ask the LLM for a plan. Returns None on any failure (see llm/client.py)."""
    return complete_json(_build_prompt(intent, tools), _SCHEMA_HINT)


def _arrow(names: list[str]) -> str:
    return " -> ".join(names) if names else "(empty)"


def _proposed_names(raw: Any) -> list[str]:
    """Tool names from an UNVALIDATED payload, for the audit line.

    This runs before validate_proposal has vetted anything, so it must survive
    every shape a model can emit — a bare string for `steps`, a list of
    strings, a non-dict at the top level. Getting this wrong crashes the exact
    code path whose job is to report that the payload was malformed.
    """
    if not isinstance(raw, dict):
        return []
    steps = raw.get("steps")
    if not isinstance(steps, list):
        return []
    names: list[str] = []
    for step in steps:
        if isinstance(step, dict):
            names.append(str(step.get("tool")))
        else:
            names.append(repr(step))
    return names


def _fallback(intent: QueryIntent, audit: list[str]) -> ExecutionPlan:
    """Deterministic plan, with the audit trail of why we are using it.

    Audit lines are PREPENDED so build_plan's own decisions (its low-confidence
    note, its unrecognised-intent note) survive intact underneath.
    """
    plan = build_plan(intent)
    plan.decisions[:0] = audit
    return plan


def plan_query(intent: QueryIntent) -> ExecutionPlan:
    """Build an execution plan, using the LLM when enabled and legal.

    This is the entry point backend/main.py calls. build_plan remains the
    entry point for tests and the evaluation harness.
    """
    if not settings.aml_llm_planner:
        return build_plan(intent)

    # Reuse the executor's registry snapshot rather than calling
    # registry.load_tools() again: that clears the global TOOLS dict and
    # reloads the tool modules, which would swap out the very function objects
    # the executor's _TOOLS_CACHE is holding. Validate against exactly what
    # will execute.
    from backend.agent import executor as executor_mod

    tools = executor_mod._get_tools()

    raw = propose_plan(intent, tools)
    if raw is None:
        return _fallback(intent, [
            "planner: source=deterministic (LLM unavailable or returned no usable JSON)",
        ])

    result = validate_proposal(raw, intent, tools)

    if not result.ok:
        proposed = _proposed_names(raw)
        audit = [
            "planner: source=deterministic (LLM plan rejected)",
            f"planner: proposed = {_arrow(proposed)}",
        ]
        audit += [f"planner: rejected — {r}" for r in result.rejections]
        audit.append(f"planner: fell back to the deterministic plan for intent '{intent.intent}'")
        return _fallback(intent, audit)

    selected = [s.tool for s in result.steps]
    decisions = [
        "planner: source=llm",
        f"planner: proposed = {_arrow(selected)}",
        f"planner: validated OK against {len(tools)} registered tools",
    ]
    decisions += [f"planner: {note}" for note in result.notes]

    return ExecutionPlan(
        plan_id=uuid.uuid4().hex[:12],
        steps=result.steps,
        decisions=decisions,
        tools_considered_but_skipped=[
            f"{name}: not selected by the LLM planner"
            for name in sorted(set(tools) - set(selected))
        ],
    )


def record_executed_plan(plan: ExecutionPlan) -> None:
    """Append what actually ran, after the executor has finished.

    Called from main.py AFTER run_plan so it captures the executor's own
    re-planning — the ml_detect insertion when rules find nothing, the
    ml_detect drop on a small sample. That is what makes proposed-vs-executed
    divergence visible instead of the plan appearing to have been followed
    exactly. The executor itself is deliberately left unmodified.
    """
    executed = [
        f"{s.tool}({s.status})" if s.status != "ok" else s.tool
        for s in plan.steps
    ]
    plan.decisions.append(f"planner: executed = {_arrow(executed)}")
