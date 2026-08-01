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

from backend.agent.plan_validator import validate_proposal
from backend.agent.planner import build_plan
from backend.agent.tool_schema import render_catalog
from backend.config import settings
from backend.llm.client import complete_json
from backend.schemas import ExecutionPlan, QueryIntent

_SCHEMA_HINT = (
    'Return JSON: {"steps": [{"tool": "<tool name>", "params": {}, '
    '"reason": "<one sentence on why this tool is needed for THIS query>"}]}'
)

# Stated in the prompt AND enforced in plan_validator. The prompt raises the
# acceptance rate; the validator is what actually holds. Neither replaces the
# other — a model that ignores the prompt is exactly the case the validator
# exists for.
_DEPENDENCY_RULES = """\
- load_data must be the first step, and appear exactly once
- feature_engineer must come before rule_detect and before ml_detect
- risk_classify must come after rule_detect or ml_detect
- filter_data must come after load_data
- entity_lookup only if the query names a specific customer
- no tool may appear more than once; at most 12 steps
- only use parameter names listed under that tool above"""


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
    return (
        "You are planning tool calls for an AML compliance query.\n\n"
        f"QUERY: {intent.raw_query}\n"
        f"PARSED INTENT: {_describe_intent(intent)}\n\n"
        "AVAILABLE TOOLS:\n"
        f"{render_catalog(tools)}\n\n"
        "DEPENDENCY RULES YOU MUST OBEY:\n"
        f"{_DEPENDENCY_RULES}\n\n"
        "Choose only the tools this specific query needs. Do not include a tool "
        "whose output would not be used to answer it."
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
