import pytest
from fastapi.testclient import TestClient

import backend.agent.executor as executor_mod
from backend.config import settings
from backend.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def force_mocks(monkeypatch):
    """These tests assert against the mock fixtures (C-04521, 5 rows, etc.) and
    must not depend on whatever settings.aml_use_mocks happens to be set to in
    the ambient environment (e.g. a real .env with AML_USE_MOCKS=0 for a local
    demo run) — force it explicitly, same as test_integration.py does for the
    opposite (real-tools) direction."""
    monkeypatch.setattr(settings, "aml_use_mocks", True)
    executor_mod._TOOLS_CACHE = None
    yield
    executor_mod._TOOLS_CACHE = None


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """The mock risk_classify fixture (C-04521) is HIGH risk, triggering
    narrator._explain()'s LLM polish path via the real /query endpoint —
    without this, every test here silently made a real network call."""
    monkeypatch.setattr("backend.agent.narrator.complete_json", lambda *a, **kw: None)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["mocks"] is True


def test_query_entity_investigation_excludes_eda():
    """The API returns the plan *after* execution, and these tests run against the
    5-row mock dataset, so the executor's <50-row guard strips ml_detect before the
    response is built. That makes this endpoint the wrong place to pin whether the
    planner includes ml_detect — tests/test_planner.py does that on the plan itself.
    What this test still pins is that eda_profile never enters a single-entity plan.
    """
    r = client.post("/query", json={"query": "Is customer 4521 suspicious?"})
    assert r.status_code == 200
    tools = [s["tool"] for s in r.json()["plan"]["steps"]]
    assert "eda_profile" not in tools
    assert "risk_classify" in tools


def test_query_threshold_excludes_ml():
    r = client.post("/query", json={"query": "Which customers made 10+ transactions under $10,000?"})
    assert r.status_code == 200
    tools = [s["tool"] for s in r.json()["plan"]["steps"]]
    assert "ml_detect" not in tools


def test_query_full_analysis_includes_both():
    r = client.post("/query", json={"query": "Analyse this dataset for suspicious activity"})
    assert r.status_code == 200
    tools = [s["tool"] for s in r.json()["plan"]["steps"]]
    assert "eda_profile" in tools
    assert "ml_detect" in tools


def test_query_response_has_explained_flags():
    r = client.post("/query", json={"query": "Analyse this dataset for suspicious activity"})
    body = r.json()
    assert body["flags"]
    for flag in body["flags"]:
        assert flag["explanation"]
        assert flag["escalation"] in ("report", "review", "monitor", "no_action")


def test_dataset_summary():
    r = client.get("/dataset/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["row_count"] > 0


def test_plan_lookup_after_query():
    r = client.post("/query", json={"query": "Analyse this dataset for suspicious activity"})
    plan_id = r.json()["plan"]["plan_id"]
    r2 = client.get(f"/plan/{plan_id}")
    assert r2.status_code == 200


def test_plan_lookup_unknown_id():
    r = client.get("/plan/does-not-exist")
    assert r.status_code == 404
