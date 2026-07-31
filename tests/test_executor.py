import pytest

import backend.agent.executor as executor_mod
from backend.agent.executor import _get_tools, run_plan
from backend.agent.planner import build_plan
from backend.config import settings
from backend.schemas import QueryIntent


@pytest.fixture(autouse=True)
def force_mocks(monkeypatch):
    """These tests assert against mock fixture data (C-04521, etc.) and must
    not depend on ambient settings.aml_use_mocks (e.g. a real .env with
    AML_USE_MOCKS=0 present for a local demo run) — force mock mode explicitly."""
    monkeypatch.setattr(settings, "aml_use_mocks", True)
    executor_mod._TOOLS_CACHE = None
    yield
    executor_mod._TOOLS_CACHE = None


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """The mock risk_classify fixture (C-04521) is HIGH risk, which triggers
    narrator._explain()'s LLM polish path. Without this, every test in this
    file silently made a REAL network call to whatever provider/key was live
    in .env — discovered after burning through a day's free-tier quota partly
    from repeated full-suite test runs, not just manual query testing."""
    monkeypatch.setattr("backend.agent.narrator.complete_json", lambda *a, **kw: None)


def _full_analysis_intent() -> QueryIntent:
    return QueryIntent(raw_query="analyse this dataset", intent="full_analysis", parsed_by="rules", confidence=0.9)


def test_full_analysis_end_to_end_with_mocks():
    intent = _full_analysis_intent()
    plan = build_plan(intent)
    response = run_plan(intent, plan)

    assert response.flags, "expected at least one flag from mock risk_classify"
    assert response.flags[0].entity_id == "C-04521"
    assert response.flags[0].explanation
    assert response.flags[0].escalation == "report"
    assert all(s.status in ("ok", "error") for s in plan.steps)
    assert response.summary


def test_isolates_tool_errors(monkeypatch):
    tools = _get_tools()

    def boom(ctx, **kw):
        raise RuntimeError("simulated tool failure")

    monkeypatch.setitem(tools, "eda_profile", boom)

    intent = _full_analysis_intent()
    plan = build_plan(intent)
    response = run_plan(intent, plan)

    assert any("eda_profile" in w for w in response.warnings)
    assert response.flags, "pipeline should complete past the failing step"


def test_entity_investigation_returns_scoped_flag():
    intent = QueryIntent(
        raw_query="Is customer 4521 suspicious?",
        intent="entity_investigation",
        entities=["C-04521"],
        parsed_by="rules",
        confidence=0.9,
    )
    plan = build_plan(intent)
    # run_plan mutates plan.steps, so the planner's own output has to be captured
    # first. ml_detect is planned here now (see WORKPLAN.md §8's amendment) but this
    # test runs on the 5-row mock dataset, where the executor's <50-row guard then
    # legitimately drops it again — which is the guard doing its job.
    planned = [s.tool for s in plan.steps]
    assert "ml_detect" in planned
    assert "eda_profile" not in planned

    response = run_plan(intent, plan)

    assert "eda_profile" not in [s.tool for s in plan.steps]
    assert response.flags
