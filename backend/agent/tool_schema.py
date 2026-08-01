"""
Tool metadata -> LLM-readable catalog.

The @tool decorator in backend/tools/base.py has always recorded `_tool_name`,
`_tool_params` and `_tool_description` on every registered tool, and until this
module existed nothing in the repo read any of them. Every tool carries a
hand-written description and per-parameter docs (see backend/tools/filters.py,
rules.py, ml_detect.py) that were assembled into nothing and sent to no model.

This module is the single reader. It exists so the LLM planner describes tools
from the tools' own declarations rather than from a second, hand-maintained copy
that would drift the first time a tool changed. Adding a tool therefore extends
the planner's vocabulary automatically, exactly the way the registry's pkgutil
walk already extends the executor's.

Determinism note: every function here sorts by tool name. The rendered catalog
goes into the planning prompt, and backend/llm/client.py caches on the exact
(prompt, schema_hint) pair — an unstable dict ordering would produce a different
prompt string on each process and silently defeat that cache.
"""

from __future__ import annotations

from typing import Any, Callable


def tool_schema(tools: dict[str, Callable]) -> list[dict[str, Any]]:
    """Describe each registered tool as {name, description, params}.

    `params` maps parameter name -> its documentation string, straight from the
    tool's own @tool(params={...}) declaration. An empty dict means the tool
    declares no parameter schema (which is different from "takes no parameters"
    — see declared_params).
    """
    out: list[dict[str, Any]] = []
    for name in sorted(tools):
        fn = tools[name]
        out.append({
            "name": getattr(fn, "_tool_name", name),
            "description": getattr(fn, "_tool_description", "") or "",
            "params": dict(getattr(fn, "_tool_params", {}) or {}),
        })
    return out


def declared_params(tools: dict[str, Callable]) -> dict[str, set[str]]:
    """Map tool name -> the set of parameter names it declares.

    An EMPTY set is meaningful and must not be read as "accepts nothing": it
    means the tool published no parameter schema, so its parameters cannot be
    validated. backend/tools/_mocks.py declares none at all, and the real
    eda_profile and risk_classify genuinely take no parameters. The validator
    distinguishes the two cases by treating empty as "unvalidated" rather than
    as "reject everything" — see backend/agent/plan_validator.py.
    """
    return {
        name: set((getattr(fn, "_tool_params", {}) or {}).keys())
        for name, fn in sorted(tools.items())
    }


def render_catalog(tools: dict[str, Callable]) -> str:
    """Render the tool catalog as plain text for the planning prompt.

    Plain text rather than JSON: it is materially shorter for the same content,
    which matters because this block dominates the prompt and every provider
    here is on a free tier.
    """
    lines: list[str] = []
    for entry in tool_schema(tools):
        lines.append(f"- {entry['name']}: {entry['description'].strip()}")
        for param in sorted(entry["params"]):
            doc = str(entry["params"][param]).strip()
            lines.append(f"    params.{param}: {doc}")
        if not entry["params"]:
            lines.append("    (no parameters)")
    return "\n".join(lines)
