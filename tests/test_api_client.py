"""Tests for the frontend's transport adapter and the pinned data source.

Two things are protected here.

The first is that the adapter's two transports are interchangeable. The whole
argument for the in-process path is that it runs the SAME code as the HTTP
path, so a demo deployed on one process behaves like a local run on two. That
claim is only true while the in-process branch delegates to backend.main rather
than reimplementing the pipeline, and nothing about a reimplementation would
look wrong until it silently answered differently.

The second is the data-source pin. load_data's signature defaults to
'synthetic_alt', which is a DIFFERENT population from the labelled 'synthetic'
set every published metric describes -- 1,710 txns / 294 customers against
2,002 / 270, with no overlapping customer IDs. A deployment left on the default
would answer questions about one dataset while README.md reports the other.

Nothing here starts a server or loads a real dataset; the backend is stubbed.
"""

import os

import pytest

from backend.config import settings
from backend.schemas import ExecutionPlan, ToolCall
from frontend import api_client


def _clear_backend_cache() -> None:
    """Drop the memoised backend import, if it is still the memoised one.

    Several tests below monkeypatch _backend with a plain function, and
    monkeypatch's own undo does not necessarily run before this fixture's
    teardown -- so the attribute may not have cache_clear at that moment.
    """
    clear = getattr(api_client._backend, "cache_clear", None)
    if clear is not None:
        clear()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Each test decides its own transport, and the import cache is per-process."""
    monkeypatch.delenv("AML_API_URL", raising=False)
    _clear_backend_cache()
    yield
    _clear_backend_cache()


def _plan(*tools: str) -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="test",
        steps=[ToolCall(tool=t, params={}, reason="test") for t in tools],
    )


# ---------------------------------------------------------------------------
# Which transport gets chosen
# ---------------------------------------------------------------------------


def test_no_api_url_means_in_process():
    assert api_client.api_base_url() is None
    assert api_client.in_process() is True


def test_api_url_means_http(monkeypatch):
    monkeypatch.setenv("AML_API_URL", "http://localhost:8000")
    assert api_client.api_base_url() == "http://localhost:8000"
    assert api_client.in_process() is False


def test_blank_api_url_is_treated_as_unset(monkeypatch):
    """A deployment host that defines the variable with an empty value is
    saying 'no API', not 'an API at the empty string'. Streamlit Cloud does
    exactly this when a secret is declared but left blank."""
    monkeypatch.setenv("AML_API_URL", "   ")
    assert api_client.api_base_url() is None
    assert api_client.in_process() is True


def test_transport_is_read_at_call_time_not_import_time(monkeypatch):
    """Streamlit re-runs the script per interaction but does not re-import
    modules, so a value captured at import would survive a secrets change and
    keep pointing at a backend that is no longer configured."""
    assert api_client.in_process() is True
    monkeypatch.setenv("AML_API_URL", "http://elsewhere:9000")
    assert api_client.in_process() is False


# ---------------------------------------------------------------------------
# The in-process path delegates rather than reimplements
# ---------------------------------------------------------------------------


def test_in_process_query_calls_backend_main(monkeypatch):
    """The claim the whole design rests on.

    If this ever stops holding -- if the adapter grows its own
    parse -> plan -> execute -- the deployed demo becomes a second
    implementation that can drift from the real one.
    """
    seen = {}

    class _FakeResponse:
        def model_dump(self, mode=None):
            return {"flags": [], "dumped_with_mode": mode}

    def _fake_query(request):
        seen["query"] = request.query
        seen["dataset"] = request.dataset
        return _FakeResponse()

    from backend.main import QueryRequest

    monkeypatch.setattr(
        api_client, "_backend", lambda: (QueryRequest, None, None, _fake_query)
    )

    out = api_client.post_query("who is riskiest?", dataset="synthetic")

    assert seen == {"query": "who is riskiest?", "dataset": "synthetic"}
    assert out == {"flags": [], "dumped_with_mode": "json"}


def test_in_process_response_is_a_plain_dict(monkeypatch):
    """The UI components call .get() on the payload, so handing back a Pydantic
    model would work right up until the first component that does."""
    class _FakeResponse:
        def model_dump(self, mode=None):
            return {"flags": [{"entity_id": "C-1"}]}

    from backend.main import QueryRequest

    monkeypatch.setattr(
        api_client, "_backend",
        lambda: (QueryRequest, None, None, lambda request: _FakeResponse()),
    )

    out = api_client.post_query("anything")
    assert isinstance(out, dict)
    assert out["flags"][0]["entity_id"] == "C-1"


def test_in_process_health_is_labelled_as_such(monkeypatch):
    monkeypatch.setattr(
        api_client, "_backend",
        lambda: (None, None, lambda: {"status": "ok", "mocks": False}, None),
    )
    assert api_client.check_health()["mode"] == "in-process"


def test_a_broken_backend_returns_none_rather_than_raising(monkeypatch):
    """Whoever opened the link gets the fixture banner, not a stack trace."""
    def _explode():
        raise ImportError("no pandas here")

    monkeypatch.setattr(api_client, "_backend", _explode)

    assert api_client.check_health() is None
    assert api_client.get_dataset_summary() is None
    assert api_client.post_query("anything") is None


# ---------------------------------------------------------------------------
# HTTP mode is unchanged, and does not silently fall back
# ---------------------------------------------------------------------------


def test_http_mode_does_not_fall_back_to_in_process(monkeypatch):
    """A configured API that is down is an outage, and must look like one.

    Quietly running the backend in-process instead would turn a broken
    deployment into a working-looking demo, which is the kind of helpfulness
    that costs you the next four hours.
    """
    monkeypatch.setenv("AML_API_URL", "http://localhost:8000")

    def _refuse(*args, **kwargs):
        raise AssertionError("in-process backend must not be imported in HTTP mode")

    monkeypatch.setattr(api_client, "_backend", _refuse)
    monkeypatch.setattr(
        api_client.requests, "get",
        lambda *a, **k: (_ for _ in ()).throw(OSError("connection refused")),
    )
    monkeypatch.setattr(
        api_client.requests, "post",
        lambda *a, **k: (_ for _ in ()).throw(OSError("connection refused")),
    )

    assert api_client.check_health() is None
    assert api_client.post_query("anything") is None


def test_http_health_is_labelled_as_http(monkeypatch):
    monkeypatch.setenv("AML_API_URL", "http://localhost:8000")

    class _R:
        def raise_for_status(self): pass
        def json(self): return {"status": "ok"}

    monkeypatch.setattr(api_client.requests, "get", lambda *a, **k: _R())
    assert api_client.check_health()["mode"] == "http"


def test_http_query_omits_dataset_when_not_given(monkeypatch):
    """The API treats a missing dataset as 'use the configured default'.
    Sending dataset=None explicitly would be a different request."""
    monkeypatch.setenv("AML_API_URL", "http://localhost:8000")
    sent = {}

    class _R:
        def raise_for_status(self): pass
        def json(self): return {}

    def _post(url, json=None, timeout=None):
        sent.update(json)
        return _R()

    monkeypatch.setattr(api_client.requests, "post", _post)
    api_client.post_query("anything")
    assert sent == {"query": "anything"}


# ---------------------------------------------------------------------------
# The data-source pin
# ---------------------------------------------------------------------------


def test_pin_applies_the_configured_source():
    from backend.main import _pin_data_source

    plan = _plan("load_data", "feature_engineer", "rule_detect")
    used = _pin_data_source(plan, None)

    load_step = plan.steps[0]
    assert used == settings.aml_data_source
    assert load_step.params["source"] == settings.aml_data_source


def test_an_explicit_dataset_wins_over_configuration():
    from backend.main import _pin_data_source

    plan = _plan("load_data")
    assert _pin_data_source(plan, "synthetic") == "synthetic"
    assert plan.steps[0].params["source"] == "synthetic"


def test_pin_leaves_every_other_step_alone():
    """It pins which dataset is loaded, not how anything is analysed."""
    from backend.main import _pin_data_source

    plan = _plan("load_data", "feature_engineer", "ml_detect", "risk_classify")
    _pin_data_source(plan, "synthetic")

    assert all(step.params == {} for step in plan.steps[1:])


def test_pin_preserves_existing_params():
    """Overriding the source must not discard params the planner set."""
    from backend.main import _pin_data_source

    plan = ExecutionPlan(
        plan_id="t",
        steps=[ToolCall(tool="load_data", params={"nrows": 500}, reason="test")],
    )
    _pin_data_source(plan, "synthetic")

    assert plan.steps[0].params == {"nrows": 500, "source": "synthetic"}


def test_dataset_summary_uses_the_same_source_as_query(monkeypatch):
    """The sidebar and the results must describe one dataset.

    /dataset/summary called load_data bare, so it took the signature default
    while /query used the configured source. With aml_data_source set, the
    sidebar reported synthetic_alt's 1,710 txns / 294 customers next to results
    computed over synthetic's 2,002 / 270 — caught by running the adapter end
    to end rather than by any test that existed at the time.
    """
    import backend.main as main_mod

    seen = {}

    def _fake_load_data(ctx, **params):
        seen.update(params)

        class _R:
            ok = True
            df = None
            artifacts: dict = {}

        return _R()

    monkeypatch.setattr(
        main_mod.registry, "load_tools", lambda use_mocks: {"load_data": _fake_load_data}
    )
    monkeypatch.setattr(settings, "aml_data_source", "synthetic")

    main_mod.dataset_summary()
    assert seen["source"] == "synthetic"


def test_default_source_matches_load_datas_own_default():
    """The setting exists to make the choice explicit for a deployment, not to
    change what anyone gets today. If these two ever diverge, every existing
    local run silently switches dataset."""
    import inspect

    from backend.tools.data_loader import load_data

    signature_default = inspect.signature(load_data).parameters["source"].default
    assert settings.aml_data_source == signature_default == "synthetic_alt"
