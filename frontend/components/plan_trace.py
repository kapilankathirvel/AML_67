"""
frontend/components/plan_trace.py

Renders the execution-plan trace panel.
WORKPLAN.md: "the highest-value component in the whole project."
Placed directly under the query box, above results.

Renders from AgentResponse fields:
  - intent  : QueryIntent
  - plan    : ExecutionPlan
    - steps : list[ToolCall]
    - decisions[]
    - tools_considered_but_skipped[]

Owner: Track B. No backend.agent.* imports.
"""

from __future__ import annotations

import streamlit as st

from frontend.components.theme import PLAN_STEP_COLOR, PLAN_STEP_TEXT_ON, TEXT_MUTED

_INTENT_LABEL: dict[str, str] = {
    "full_analysis":       "🔍 Full Analysis",
    "pattern_search":      "🎯 Pattern Search",
    "threshold_query":     "📊 Threshold Query",
    "entity_investigation":"🧑 Entity Investigation",
    "ranking":             "🏆 Ranking",
    "eda":                 "📈 Exploratory Analysis",
    "explain_flag":        "💡 Explain Flag",
}


def render_plan_trace(response: dict) -> None:
    """Render the full execution-plan trace panel from an AgentResponse dict.

    Wrapped in a collapsed expander so results (flags) are visible first.
    Judges can open the trace to inspect the agentic decision-making on demand.
    """
    intent_obj = response.get("intent", {})
    plan_obj   = response.get("plan", {})

    intent_str = intent_obj.get("intent", "unknown")
    parsed_by  = intent_obj.get("parsed_by", "?")
    label = (
        f"Execution Plan Trace — {_INTENT_LABEL.get(intent_str, intent_str)} "
        f"(parsed by {parsed_by})"
    )

    with st.expander(label, expanded=False, icon="🗺️"):
        # ------------------------------------------------------------------
        # Intent summary row
        # ------------------------------------------------------------------
        confidence  = intent_obj.get("confidence", 0.0)
        entities    = intent_obj.get("entities", [])
        patterns    = intent_obj.get("pattern_types", [])
        filters     = intent_obj.get("filters", {})

        col_a, col_b, col_c = st.columns([2, 1, 1])
        with col_a:
            st.markdown(
                f"**Detected intent:** {_INTENT_LABEL.get(intent_str, intent_str)}"
            )
        with col_b:
            st.markdown(f"**Parsed by:** `{parsed_by}`")
        with col_c:
            st.markdown(f"**Confidence:** `{confidence:.0%}`")

        # Entities + patterns + active filters
        detail_parts: list[str] = []
        if entities:
            detail_parts.append(f"**Entities:** {', '.join(f'`{e}`' for e in entities)}")
        if patterns:
            detail_parts.append(f"**Patterns:** {', '.join(f'`{p}`' for p in patterns)}")

        active_filters = {k: v for k, v in filters.items() if v not in (None, [], "")}
        if active_filters:
            filt_str = " · ".join(f"`{k}={v}`" for k, v in active_filters.items())
            detail_parts.append(f"**Filters:** {filt_str}")

        if detail_parts:
            st.markdown("  \n".join(detail_parts))
        else:
            st.markdown("*No entity, pattern, or filter constraints extracted.*")

        # ------------------------------------------------------------------
        # Tool steps timeline
        # ------------------------------------------------------------------
        steps = plan_obj.get("steps", [])
        if steps:
            st.subheader("Tool Steps", divider="grey")
            for i, step in enumerate(steps, 1):
                status   = step.get("status", "pending")
                colour   = PLAN_STEP_COLOR.get(status, "#64748b")
                text_col = PLAN_STEP_TEXT_ON.get(status, "#ffffff")
                tool     = step.get("tool", "unknown")
                reason   = step.get("reason", "")
                duration = step.get("duration_ms")
                dur_str  = f"`{duration} ms`" if duration is not None else "`—`"

                badge = f'<span style="background:{colour};color:{text_col};border-radius:4px;padding:1px 7px;font-size:12px;font-weight:600;">{status.upper()}</span>'
                st.markdown(
                    f"**{i}. `{tool}`** {badge} &nbsp;&nbsp;{dur_str}",
                    unsafe_allow_html=True,
                )
                st.markdown(f"<span style='color:{TEXT_MUTED};font-size:13px;margin-left:16px;'>↳ {reason}</span>", unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # Skipped tools
        # ------------------------------------------------------------------
        skipped = plan_obj.get("tools_considered_but_skipped", [])
        if skipped:
            st.subheader("Tools Considered but Skipped", divider="grey")
            for s in skipped:
                st.markdown(f"- <span style='color:{TEXT_MUTED};'>{s}</span>", unsafe_allow_html=True)

        # ------------------------------------------------------------------
        # Re-planning decisions log
        # ------------------------------------------------------------------
        decisions = plan_obj.get("decisions", [])
        if decisions:
            # Two different things share this list and read badly mixed together.
            #
            # The planner's own audit trail (source / proposed / rejected /
            # executed) is a contiguous record of one decision, and it is
            # emitted in two bursts — the first lines before execution, the
            # `executed` line after — so in source order it arrives split by
            # every tool's runtime note. Grouping by prefix reassembles it.
            #
            # It also renders as one monospace block rather than one st.info
            # box per line: a six-box stack buries the tool notes underneath,
            # and this content is a log, so it should look like one.
            planner_lines = [d for d in decisions if d.startswith("planner:")]
            runtime_lines = [d for d in decisions if not d.startswith("planner:")]

            if planner_lines:
                st.subheader("Planner Audit Trail", divider="grey")
                rejected = any(d.startswith("planner: rejected") for d in planner_lines)
                source_llm = any("source=llm" in d for d in planner_lines)
                if source_llm:
                    st.success("Plan chosen by the LLM and accepted by the validator.")
                elif rejected:
                    st.warning(
                        "The LLM proposed a plan, the validator rejected it, and the "
                        "deterministic plan ran instead. Both are recorded below."
                    )
                st.code(
                    "\n".join(d.removeprefix("planner: ") for d in planner_lines),
                    language="text",
                )

            if runtime_lines:
                st.subheader("Re-planning Decisions", divider="grey")
                for d in runtime_lines:
                    st.info(d)
