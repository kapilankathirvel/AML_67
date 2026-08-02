"""Tests for the observe -> decide -> act loop.

The property that matters most is the one asserted first: with the flag off,
nothing changes at all. Everything else in this file is about the model being
unable to do damage when the flag is on — it may revise what has not run yet,
and nothing else.
"""

import pytest

import backend.agent.executor as executor_mod
import backend.agent.replanner as replanner_mod
from backend.agent.executor import _get_tools, run_plan
from backend.agent.planner import build_plan
from backend.agent.replanner import MAX_FAILURE_REPLANS, MAX_REPLANS, observe
from backend.config import settings
from backend.schemas import QueryIntent, ToolCall
from backend.tools.base import ToolContext, ToolResult


@pytest.fixture(autouse=True)
def force_mocks(monkeypatch):
    monkeypatch.setattr(settings, "aml_use_mocks", True)
    executor_mod._TOOLS_CACHE = None
    yield
    executor_mod._TOOLS_CACHE = None


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    monkeypatch.setattr(replanner_mod, "complete_json", lambda *a, **kw: None)
    monkeypatch.setattr("backend.agent.narrator.complete_json", lambda *a, **kw: None)


def _intent(name="full_analysis"):
    return QueryIntent(raw_query="analyse this dataset", intent=name,
                       parsed_by="rules", confidence=0.9)


def _run(intent=None):
    intent = intent or _intent()
    plan = build_plan(intent)
    response = run_plan(intent, plan)
    return plan, response


def _decisions(plan):
    return "\n".join(plan.decisions)


def _stub(monkeypatch, payload):
    monkeypatch.setattr(replanner_mod, "complete_json", lambda *a, **kw: payload)


# ---------------------------------------------------------------------------
# Flag off — the default configuration must be untouched
# ---------------------------------------------------------------------------


def test_flag_off_produces_no_replanner_lines(monkeypatch):
    monkeypatch.setattr(settings, "aml_llm_replanner", False)
    plan, _ = _run()
    assert "replanner:" not in _decisions(plan)


def test_flag_off_never_calls_the_llm(monkeypatch):
    def explode(*a, **kw):
        raise AssertionError("replanner called the LLM with the flag off")

    monkeypatch.setattr(settings, "aml_llm_replanner", False)
    monkeypatch.setattr(replanner_mod, "complete_json", explode)
    _run()  # must not raise


def test_flag_off_leaves_the_step_sequence_identical(monkeypatch):
    monkeypatch.setattr(settings, "aml_llm_replanner", False)
    intent = _intent()
    baseline = [s.tool for s in build_plan(intent).steps]
    plan, _ = _run(intent)
    assert [s.tool for s in plan.steps] == baseline


# ---------------------------------------------------------------------------
# The observation digest
# ---------------------------------------------------------------------------


def test_digest_reports_what_actually_happened():
    ctx = ToolContext(df=None)
    ctx.artifacts["rule_hits"] = [
        {"entity_id": "C-1", "rule_id": "R1"},
        {"entity_id": "C-1", "rule_id": "R3"},
        {"entity_id": "C-2", "rule_id": "R1"},
    ]
    ctx.artifacts["risk_rows"] = [{"risk_level": "high"}, {"risk_level": "low"}]

    digest = observe(ctx, ["load_data", "rule_detect"], ["risk_classify"])

    assert "load_data -> rule_detect" in digest
    assert "risk_classify" in digest
    assert "3 across rules ['R1', 'R3']" in digest
    assert "2 distinct entities" in digest
    assert "'high': 1" in digest


def test_digest_changes_as_the_run_progresses():
    """Load-bearing: llm/client.py caches on the exact prompt. If two
    iterations produced the same digest, iteration 2 would replay iteration 1's
    answer and the loop would make the same decision forever."""
    ctx = ToolContext(df=None)
    first = observe(ctx, ["load_data"], ["rule_detect"])
    ctx.artifacts["rule_hits"] = [{"entity_id": "C-1", "rule_id": "R1"}]
    second = observe(ctx, ["load_data", "rule_detect"], [])
    assert first != second


def test_digest_reports_absent_artifacts_by_omission():
    """Nothing computed yet must not be reported as zero of something."""
    digest = observe(ToolContext(df=None), [], ["load_data"])
    assert "rule hits" not in digest
    assert "risk rows" not in digest


# ---------------------------------------------------------------------------
# Flag on — declining, revising, and being refused
# ---------------------------------------------------------------------------


def test_model_declining_is_recorded_distinctly(monkeypatch):
    """Declining and failing must not both look like silence in the trace."""
    monkeypatch.setattr(settings, "aml_llm_replanner", True)
    _stub(monkeypatch, {"revise": False})
    plan, _ = _run()

    assert "replanner: model reviewed the observation and kept the plan" in _decisions(plan)


def test_llm_unavailable_is_recorded_and_harmless(monkeypatch):
    monkeypatch.setattr(settings, "aml_llm_replanner", True)
    _stub(monkeypatch, None)
    intent = _intent()
    baseline = [s.tool for s in build_plan(intent).steps]
    plan, _ = _run(intent)

    assert "replanner: no usable response" in _decisions(plan)
    assert [s.tool for s in plan.steps] == baseline


def test_a_valid_revision_replaces_the_queued_steps(monkeypatch):
    monkeypatch.setattr(settings, "aml_llm_replanner", True)
    _stub(monkeypatch, {"revise": True, "steps": [
        {"tool": "feature_engineer", "params": {}, "reason": "features first"},
        {"tool": "rule_detect", "params": {}, "reason": "detect"},
        {"tool": "risk_classify", "params": {}, "reason": "score"},
    ]})
    plan, response = _run()

    tools = [s.tool for s in plan.steps]
    assert "replanner: revised the remaining plan" in _decisions(plan)
    assert tools[0] == "load_data", "history must not be rewritten"
    assert "ml_detect" not in tools, "the revision dropped ml_detect and that must stick"
    assert not response.warnings


def test_a_revision_that_reruns_load_data_is_rejected(monkeypatch):
    """The prefix is validated with the proposal, so V3 (no duplicates) stops
    the model re-running a step that has already executed."""
    monkeypatch.setattr(settings, "aml_llm_replanner", True)
    _stub(monkeypatch, {"revise": True, "steps": [
        {"tool": "load_data", "params": {}, "reason": "load again"},
        {"tool": "feature_engineer", "params": {}, "reason": "features"},
        {"tool": "rule_detect", "params": {}, "reason": "detect"},
        {"tool": "risk_classify", "params": {}, "reason": "score"},
    ]})
    plan, _ = _run()

    text = _decisions(plan)
    assert "replanner: revision rejected" in text
    assert "proposed more than once" in text


def test_a_revision_that_breaks_a_dependency_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "aml_llm_replanner", True)
    _stub(monkeypatch, {"revise": True, "steps": [
        {"tool": "risk_classify", "params": {}, "reason": "score with nothing to fuse"},
    ]})
    plan, _ = _run()
    assert "replanner: revision rejected" in _decisions(plan)


def test_a_revision_may_not_reach_a_forbidden_capability(monkeypatch):
    """V14 applies to a mid-flight revision exactly as it does up front."""
    monkeypatch.setattr(settings, "aml_llm_replanner", True)
    _stub(monkeypatch, {"revise": True, "steps": [
        {"tool": "feature_engineer", "params": {}, "reason": "features"},
        {"tool": "rule_detect", "params": {"patterns": ["not_a_pattern"]}, "reason": "detect"},
        {"tool": "risk_classify", "params": {}, "reason": "score"},
    ]})
    plan, _ = _run()

    text = _decisions(plan)
    assert "replanner: revision rejected" in text
    assert "is not a valid patterns" in text


def test_malformed_revision_is_harmless(monkeypatch):
    monkeypatch.setattr(settings, "aml_llm_replanner", True)
    _stub(monkeypatch, {"revise": True, "steps": "feature_engineer"})
    intent = _intent()
    baseline = [s.tool for s in build_plan(intent).steps]
    plan, _ = _run(intent)

    assert "replanner:" in _decisions(plan)
    assert [s.tool for s in plan.steps] == baseline


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


def test_replanning_is_capped(monkeypatch):
    """Without a cap, a model that keeps appending one more tool never ends."""
    monkeypatch.setattr(settings, "aml_llm_replanner", True)
    calls = {"n": 0}

    def counting(*a, **kw):
        calls["n"] += 1
        return {"revise": False}

    monkeypatch.setattr(replanner_mod, "complete_json", counting)
    _run()

    assert calls["n"] <= MAX_REPLANS, f"expected at most {MAX_REPLANS} re-plans, got {calls['n']}"


# ---------------------------------------------------------------------------
# Failures are observations too
#
# The loop originally ran only after a step that succeeded: each of the
# executor's three error paths `continue`d straight past it. That made it blind
# in the one situation it exists for.
# ---------------------------------------------------------------------------


def _boom(*a, **kw):
    raise RuntimeError("synthetic tool failure")


def _capture_prompts(monkeypatch, payload=None):
    """Collect every prompt, not just the first. The loop is consulted after
    each step, so the failure prompt is not the earliest one."""
    seen: list[str] = []

    def spy(prompt, schema_hint=""):
        seen.append(prompt)
        return payload

    monkeypatch.setattr(replanner_mod, "complete_json", spy)
    return seen


def test_digest_leads_with_the_failure():
    digest = observe(ToolContext(df=None), ["load_data"], ["risk_classify"],
                     failure="rule_detect raised RuntimeError: boom")
    assert digest.splitlines()[0].startswith("THE STEP THAT JUST RAN FAILED:")
    assert "rule_detect raised RuntimeError: boom" in digest


def test_digest_has_no_failure_line_on_the_happy_path():
    digest = observe(ToolContext(df=None), ["load_data"], ["risk_classify"])
    assert "FAILED" not in digest


def test_a_raising_tool_still_reaches_the_replanner(monkeypatch):
    """The regression this whole change is about."""
    monkeypatch.setattr(settings, "aml_llm_replanner", True)
    seen = _capture_prompts(monkeypatch, {"revise": False})
    monkeypatch.setitem(_get_tools(), "rule_detect", _boom)

    plan, response = _run()

    assert any("synthetic tool failure" in p for p in seen), \
        "the re-planner was never told the step had failed"
    assert any("synthetic tool failure" in w for w in response.warnings)


def test_the_failure_is_written_to_the_audit_trail(monkeypatch):
    monkeypatch.setattr(settings, "aml_llm_replanner", True)
    _stub(monkeypatch, {"revise": False})
    monkeypatch.setitem(_get_tools(), "rule_detect", _boom)

    plan, _ = _run()
    assert "replanner: observing a failed step — rule_detect raised RuntimeError" in _decisions(plan)


def test_a_tool_returning_not_ok_also_reaches_the_replanner(monkeypatch):
    monkeypatch.setattr(settings, "aml_llm_replanner", True)
    seen = _capture_prompts(monkeypatch, {"revise": False})
    monkeypatch.setitem(
        _get_tools(), "rule_detect",
        lambda ctx, **kw: ToolResult(ok=False, error="rule_detect could not run"),
    )

    _run()
    assert any("rule_detect could not run" in p for p in seen)


def test_an_unknown_tool_also_reaches_the_replanner(monkeypatch):
    monkeypatch.setattr(settings, "aml_llm_replanner", True)
    seen = _capture_prompts(monkeypatch, {"revise": False})

    intent = _intent()
    plan = build_plan(intent)
    plan.steps.insert(1, ToolCall(tool="no_such_tool", reason="deliberately bogus"))
    run_plan(intent, plan)

    assert any("unknown tool 'no_such_tool'" in p for p in seen)


def test_the_model_can_route_around_a_failed_detector(monkeypatch):
    """The concrete win. rule_detect dies, so risk_classify is queued with
    nothing to classify, and the hardcoded 'no rule hits -> add ml_detect' rule
    cannot help because it lives on the success path. ml_detect is still a legal
    and useful revision, and only the loop can reach it."""
    monkeypatch.setattr(settings, "aml_llm_replanner", True)
    revision = {"revise": True, "steps": [
        {"tool": "ml_detect", "params": {}, "reason": "rule_detect failed — fall back to anomaly detection"},
        {"tool": "risk_classify", "params": {}, "reason": "score what ml_detect found"},
    ]}
    # Revise only when told something failed; keep the plan otherwise. That is
    # the behaviour being tested, and it also keeps the earlier happy-path
    # consultations from reshaping the plan before rule_detect can fail.
    monkeypatch.setattr(
        replanner_mod, "complete_json",
        lambda prompt, schema_hint="": revision if "JUST RAN FAILED" in prompt else {"revise": False},
    )

    intent = _intent()
    plan = build_plan(intent)
    plan.steps = [s for s in plan.steps if s.tool != "ml_detect"]
    monkeypatch.setitem(_get_tools(), "rule_detect", _boom)
    run_plan(intent, plan)

    assert "replanner: revised the remaining plan" in _decisions(plan)
    assert "ml_detect" in [s.tool for s in plan.steps]


def test_the_failure_allowance_survives_a_spent_routine_budget(monkeypatch):
    """rule_detect is the fourth step, so MAX_REPLANS is already gone by the
    time it fails. Without a reserved allowance the loop would see the failure
    and have no budget left to act on it — which is how it behaved when the
    two budgets were shared."""
    monkeypatch.setattr(settings, "aml_llm_replanner", True)
    seen = _capture_prompts(monkeypatch, {"revise": False})
    monkeypatch.setitem(_get_tools(), "rule_detect", _boom)

    _run()

    routine = [p for p in seen if "JUST RAN FAILED" not in p]
    failed = [p for p in seen if "JUST RAN FAILED" in p]
    assert len(routine) == MAX_REPLANS, "the routine budget should have been spent first"
    assert failed, "the failure got no consultation once the routine budget ran out"


def test_failure_replans_are_capped_too(monkeypatch):
    """The reserved allowance is an allowance, not an exemption."""
    monkeypatch.setattr(settings, "aml_llm_replanner", True)
    seen = _capture_prompts(monkeypatch, {"revise": False})
    for tool in ("feature_engineer", "rule_detect", "ml_detect"):
        monkeypatch.setitem(_get_tools(), tool, _boom)

    _run()

    failed = [p for p in seen if "JUST RAN FAILED" in p]
    assert len(failed) <= MAX_FAILURE_REPLANS, \
        f"expected at most {MAX_FAILURE_REPLANS} failure re-plans, got {len(failed)}"


def test_a_failure_does_not_change_behaviour_with_the_flag_off(monkeypatch):
    """The failure paths must stay exactly as they were by default."""
    monkeypatch.setattr(settings, "aml_llm_replanner", False)
    monkeypatch.setattr(replanner_mod, "complete_json",
                        lambda *a, **kw: pytest.fail("consulted the LLM with the flag off"))
    monkeypatch.setitem(_get_tools(), "rule_detect", _boom)

    intent = _intent()
    baseline = [s.tool for s in build_plan(intent).steps]
    plan, response = _run(intent)

    assert "replanner:" not in _decisions(plan)
    assert [s.tool for s in plan.steps] == baseline
    assert any("rule_detect raised RuntimeError" in w for w in response.warnings)


def test_a_failed_step_is_still_marked_error(monkeypatch):
    """Refactoring the three error branches into one helper must not lose the
    per-step status the frontend's trace renders."""
    monkeypatch.setattr(settings, "aml_llm_replanner", False)
    monkeypatch.setitem(_get_tools(), "rule_detect", _boom)
    plan, _ = _run()

    failed = [s for s in plan.steps if s.tool == "rule_detect"]
    assert failed and all(s.status == "error" for s in failed)


def test_the_run_still_completes_after_a_failure(monkeypatch):
    """A failing step must not abort the request — unchanged from before."""
    monkeypatch.setattr(settings, "aml_llm_replanner", False)
    monkeypatch.setitem(_get_tools(), "rule_detect", _boom)
    _, response = _run()
    assert response.summary


def test_the_hardcoded_rules_still_fire_underneath(monkeypatch):
    """The executor's own re-planning is the floor. With the model declining,
    the 0-rule-hits rule must still insert ml_detect."""
    monkeypatch.setattr(settings, "aml_llm_replanner", True)
    _stub(monkeypatch, {"revise": False})

    intent = QueryIntent(raw_query="find structuring", intent="pattern_search",
                         parsed_by="rules", confidence=0.9)
    plan = build_plan(intent)
    plan.steps = [s for s in plan.steps if s.tool != "ml_detect"]
    monkeypatch.setitem(
        _get_tools(), "rule_detect",
        lambda ctx, **kw: ToolResult(ok=True, artifacts={"rule_hits": []}, metrics={"rules_fired": 0}),
    )
    run_plan(intent, plan)

    assert "no rule hits — widening the net with ml_detect" in _decisions(plan)
    assert "ml_detect" in [s.tool for s in plan.steps]
