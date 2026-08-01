"""Tests for LLM-driven planning and its whitelist.

The central property under test is that the LLM can never make this system do
something illegal or make it stop working: every failure mode falls back to the
deterministic planner, and the reason is recorded in plan.decisions.

Note the two autouse fixtures. `no_llm` stubs the module-level symbol
backend.agent.llm_planner.complete_json rather than the client function,
matching how test_executor.py and test_integration.py stub the narrator — the
planner imports the name at module scope, so patching the client would not
intercept it. Without this, the tests that enable the planner flag would make
real network calls against whatever key is live in .env.
"""

import pytest

import backend.agent.executor as executor_mod
import backend.agent.llm_planner as llm_planner_mod
from backend.agent.executor import _get_tools, run_plan
from backend.agent.llm_planner import plan_query, record_executed_plan
from backend.agent.plan_validator import MAX_STEPS, validate_proposal
from backend.agent.planner import build_plan
from backend.agent.tool_schema import declared_params, render_catalog, tool_schema
from backend.config import settings
from backend.schemas import Filters, QueryIntent

ALL_INTENTS = [
    "full_analysis", "pattern_search", "threshold_query",
    "entity_investigation", "ranking", "eda", "explain_flag",
]


@pytest.fixture(autouse=True)
def force_mocks(monkeypatch):
    monkeypatch.setattr(settings, "aml_use_mocks", True)
    executor_mod._TOOLS_CACHE = None
    yield
    executor_mod._TOOLS_CACHE = None


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """Default: the LLM is unavailable. Individual tests override with a stub."""
    monkeypatch.setattr(llm_planner_mod, "complete_json", lambda *a, **kw: None)
    monkeypatch.setattr("backend.agent.narrator.complete_json", lambda *a, **kw: None)


def _intent(intent="pattern_search", **kw) -> QueryIntent:
    return QueryIntent(
        raw_query=kw.pop("raw_query", "find structuring in the last 30 days"),
        intent=intent,
        parsed_by="rules",
        confidence=kw.pop("confidence", 0.9),
        **kw,
    )


def _tools():
    return _get_tools()


def _stub_llm(monkeypatch, payload):
    monkeypatch.setattr(llm_planner_mod, "complete_json", lambda *a, **kw: payload)


def _enable(monkeypatch):
    monkeypatch.setattr(settings, "aml_llm_planner", True)


def _decisions(plan) -> str:
    return "\n".join(plan.decisions)


# ---------------------------------------------------------------------------
# tool_schema — proves the catalog is built from the tools' own metadata
# ---------------------------------------------------------------------------


def test_catalog_is_built_from_real_tool_metadata():
    """Not a hand-written copy: these strings live in the tool modules.

    If someone re-describes a tool in a second place, this test keeps passing
    while the catalog goes stale — which is precisely why the catalog reads
    _tool_description instead.
    """
    executor_mod._TOOLS_CACHE = None
    settings.aml_use_mocks = False
    try:
        catalog = render_catalog(_get_tools())
    finally:
        settings.aml_use_mocks = True
        executor_mod._TOOLS_CACHE = None

    assert "IsolationForest" in catalog, "ml_detect's real description should be in the catalog"
    assert "ISO-3166" in catalog, "filter_data's real param docs should be in the catalog"


def test_tool_schema_covers_every_registered_tool_and_is_sorted():
    tools = _tools()
    entries = tool_schema(tools)
    names = [e["name"] for e in entries]

    assert set(names) == set(tools)
    assert names == sorted(names), "unsorted output would destabilise the prompt cache key"


def test_declared_params_distinguishes_empty_from_missing():
    tools = _tools()
    declared = declared_params(tools)
    assert set(declared) == set(tools)
    # Mock tools declare no params at all — the case V10 must not treat as
    # "rejects everything".
    assert all(isinstance(v, set) for v in declared.values())


# ---------------------------------------------------------------------------
# Flag off — the default configuration must be unchanged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("intent_name", ALL_INTENTS)
def test_flag_off_matches_build_plan_exactly(intent_name):
    intent = _intent(intent_name, entities=["C-04521"])
    assert [s.tool for s in plan_query(intent).steps] == [s.tool for s in build_plan(intent).steps]


def test_flag_off_never_calls_the_llm(monkeypatch):
    """The strongest guarantee in this file: with the flag off, no network path
    is reachable at all — not merely stubbed."""
    def explode(*a, **kw):
        raise AssertionError("plan_query called the LLM with aml_llm_planner off")

    monkeypatch.setattr(settings, "aml_llm_planner", False)
    monkeypatch.setattr(llm_planner_mod, "complete_json", explode)
    plan_query(_intent())  # must not raise


# ---------------------------------------------------------------------------
# Fallback paths
# ---------------------------------------------------------------------------


def test_llm_unavailable_falls_back_and_says_so(monkeypatch):
    _enable(monkeypatch)
    intent = _intent()
    plan = plan_query(intent)

    assert [s.tool for s in plan.steps] == [s.tool for s in build_plan(intent).steps]
    assert "source=deterministic" in _decisions(plan)
    assert "LLM unavailable" in _decisions(plan)


@pytest.mark.parametrize("payload,expected", [
    ({"steps": [{"tool": "nope", "params": {}, "reason": "r"}]},
     "unknown tool 'nope'"),
    ({"steps": [{"tool": "load_data", "params": {}, "reason": "r"},
                {"tool": "rule_detect", "params": {}, "reason": "r"},
                {"tool": "feature_engineer", "params": {}, "reason": "r"}]},
     "rule_detect requires feature_engineer before it"),
    ({"steps": [{"tool": "load_data", "params": {}, "reason": "r"},
                {"tool": "risk_classify", "params": {}, "reason": "r"}]},
     "risk_classify requires rule_detect or ml_detect before it"),
    # A plan that cannot answer its intent. Note load_data is NOT the failure
    # here any more — it gets prepended (see the repair tests below) — so what
    # remains is that a pattern_search plan produces no risk scores.
    ({"steps": [{"tool": "eda_profile", "params": {}, "reason": "r"}]},
     "needs risk_classify to produce a result"),
    ({"steps": [{"tool": "load_data", "params": {}, "reason": "r"},
                {"tool": "eda_profile", "params": {}, "reason": "r"},
                {"tool": "eda_profile", "params": {}, "reason": "r"}]},
     "proposed more than once"),
    ({"steps": [{"tool": "load_data", "params": {}, "reason": ""}]},
     "missing reason"),
])
def test_each_rejection_rule_falls_back_and_is_logged(monkeypatch, payload, expected):
    _enable(monkeypatch)
    _stub_llm(monkeypatch, payload)
    intent = _intent()
    plan = plan_query(intent)

    assert [s.tool for s in plan.steps] == [s.tool for s in build_plan(intent).steps]
    assert "source=deterministic (LLM plan rejected)" in _decisions(plan)
    assert expected in _decisions(plan), f"expected rejection {expected!r} in:\n{_decisions(plan)}"


def test_too_many_steps_rejected(monkeypatch):
    _enable(monkeypatch)
    steps = [{"tool": "load_data", "params": {}, "reason": "r"}]
    steps += [{"tool": f"t{i}", "params": {}, "reason": "r"} for i in range(MAX_STEPS + 2)]
    _stub_llm(monkeypatch, {"steps": steps})

    plan = plan_query(_intent())
    assert f"max is {MAX_STEPS}" in _decisions(plan)


def test_entity_lookup_without_an_entity_is_rejected(monkeypatch):
    _enable(monkeypatch)
    _stub_llm(monkeypatch, {"steps": [
        {"tool": "load_data", "params": {}, "reason": "r"},
        {"tool": "entity_lookup", "params": {}, "reason": "r"},
    ]})
    plan = plan_query(_intent(entities=[]))
    assert "no entity was extracted" in _decisions(plan)


@pytest.mark.parametrize("payload", [
    {}, {"steps": "load_data"}, {"steps": []}, {"steps": [{"tool": 3, "reason": "r"}]},
    {"steps": [{"tool": "load_data", "params": "nope", "reason": "r"}]},
    {"steps": ["load_data"]}, None,
])
def test_malformed_payloads_are_rejected_without_raising(monkeypatch, payload):
    _enable(monkeypatch)
    _stub_llm(monkeypatch, payload)
    intent = _intent()
    plan = plan_query(intent)  # must not raise
    assert [s.tool for s in plan.steps] == [s.tool for s in build_plan(intent).steps]


def test_all_violations_are_reported_not_just_the_first(monkeypatch):
    """The audit trail should state everything wrong with a proposal."""
    _enable(monkeypatch)
    _stub_llm(monkeypatch, {"steps": [
        {"tool": "risk_classify", "params": {}, "reason": ""},
        {"tool": "ghost", "params": {}, "reason": "r"},
    ]})
    text = _decisions(plan_query(_intent()))
    assert "unknown tool 'ghost'" in text
    assert "missing reason" in text
    assert "risk_classify requires rule_detect or ml_detect before it" in text


# ---------------------------------------------------------------------------
# Accept path
# ---------------------------------------------------------------------------


def _valid_payload():
    return {"steps": [
        {"tool": "load_data", "params": {}, "reason": "need the working set"},
        {"tool": "filter_data", "params": {}, "reason": "narrow to the window asked about"},
        {"tool": "feature_engineer", "params": {}, "reason": "structuring features"},
        {"tool": "rule_detect", "params": {}, "reason": "apply the structuring rules"},
        {"tool": "risk_classify", "params": {}, "reason": "score what the rules found"},
    ]}


def test_valid_proposal_is_used_verbatim(monkeypatch):
    _enable(monkeypatch)
    _stub_llm(monkeypatch, _valid_payload())
    plan = plan_query(_intent())

    assert [s.tool for s in plan.steps] == [
        "load_data", "filter_data", "feature_engineer", "rule_detect", "risk_classify",
    ]
    assert "source=llm" in _decisions(plan)
    assert "validated OK" in _decisions(plan)
    # ml_detect and eda_profile were not chosen — that must be recorded.
    skipped = " ".join(plan.tools_considered_but_skipped)
    assert "ml_detect" in skipped and "eda_profile" in skipped


def test_plan_differs_from_the_deterministic_one(monkeypatch):
    """Proves the LLM plan is actually taking effect, not coincidentally equal."""
    _enable(monkeypatch)
    _stub_llm(monkeypatch, _valid_payload())
    intent = _intent()
    assert [s.tool for s in plan_query(intent).steps] != [s.tool for s in build_plan(intent).steps]


def test_filter_params_are_injected_from_the_parsed_intent(monkeypatch):
    """Without this the LLM's empty params would silently discard the query's
    own filters and the plan would analyse the whole dataset."""
    _enable(monkeypatch)
    _stub_llm(monkeypatch, _valid_payload())
    intent = _intent(filters=Filters(amount_max=10000.0, countries=["US"]))
    plan = plan_query(intent)

    params = next(s.params for s in plan.steps if s.tool == "filter_data")
    assert params["amount_max"] == 10000.0
    assert params["countries"] == ["US"]
    assert "injected filter_data params" in _decisions(plan)


def test_llm_supplied_filter_params_are_not_overwritten(monkeypatch):
    _enable(monkeypatch)
    payload = _valid_payload()
    for step in payload["steps"]:
        if step["tool"] == "filter_data":
            step["params"] = {"amount_max": 5000.0}
    _stub_llm(monkeypatch, payload)

    plan = plan_query(_intent(filters=Filters(amount_max=10000.0)))
    params = next(s.params for s in plan.steps if s.tool == "filter_data")
    assert params["amount_max"] == 5000.0, "injection must not clobber an explicit choice"


def test_entity_id_key_is_always_present(monkeypatch):
    """executor.py only re-syncs the resolved customer id `if "entity_id" in
    later.params` — the key must exist even when the value is None."""
    _enable(monkeypatch)
    _stub_llm(monkeypatch, {"steps": [
        {"tool": "load_data", "params": {}, "reason": "r"},
        {"tool": "entity_lookup", "params": {}, "reason": "r"},
        {"tool": "feature_engineer", "params": {}, "reason": "r"},
        {"tool": "rule_detect", "params": {}, "reason": "r"},
        {"tool": "risk_classify", "params": {}, "reason": "r"},
    ]})
    plan = plan_query(_intent("entity_investigation", entities=["C-04521"]))
    params = next(s.params for s in plan.steps if s.tool == "entity_lookup")
    assert "entity_id" in params
    assert params["entity_id"] == "C-04521"


# ---------------------------------------------------------------------------
# load_data repair — omitted or misplaced load_data is fixed, not rejected
# ---------------------------------------------------------------------------


def test_missing_load_data_is_prepended_not_rejected(monkeypatch):
    """load_data carries no planning information — every legal plan starts with
    it — so requiring the model to emit it only adds a failure mode. Measured:
    it was 8 of 13 rejections against a local 3B model."""
    _enable(monkeypatch)
    _stub_llm(monkeypatch, {"steps": [
        {"tool": "feature_engineer", "params": {}, "reason": "features"},
        {"tool": "rule_detect", "params": {}, "reason": "detect"},
        {"tool": "risk_classify", "params": {}, "reason": "score"},
    ]})
    plan = plan_query(_intent())

    assert "source=llm" in _decisions(plan), "should be accepted, not fall back"
    assert [s.tool for s in plan.steps][0] == "load_data"
    assert "prepended load_data" in _decisions(plan), "the repair must be logged"


def test_misplaced_load_data_is_moved_to_the_front(monkeypatch):
    _enable(monkeypatch)
    _stub_llm(monkeypatch, {"steps": [
        {"tool": "feature_engineer", "params": {}, "reason": "features"},
        {"tool": "load_data", "params": {}, "reason": "load"},
        {"tool": "rule_detect", "params": {}, "reason": "detect"},
        {"tool": "risk_classify", "params": {}, "reason": "score"},
    ]})
    plan = plan_query(_intent())

    assert [s.tool for s in plan.steps][0] == "load_data"
    assert "moved load_data" in _decisions(plan)


def test_duplicate_load_data_is_still_rejected(monkeypatch):
    """Two of them means the model misunderstood the plan, not that it omitted
    a preamble — repairing that would hide real confusion."""
    _enable(monkeypatch)
    _stub_llm(monkeypatch, {"steps": [
        {"tool": "load_data", "params": {}, "reason": "load"},
        {"tool": "load_data", "params": {}, "reason": "load again"},
        {"tool": "feature_engineer", "params": {}, "reason": "features"},
        {"tool": "rule_detect", "params": {}, "reason": "detect"},
        {"tool": "risk_classify", "params": {}, "reason": "score"},
    ]})
    plan = plan_query(_intent())
    assert "source=deterministic" in _decisions(plan)
    assert "proposed more than once" in _decisions(plan)


def test_repair_never_adds_a_detector(monkeypatch):
    """The line the repair must not cross: it fixes preconditions, never the
    choices being delegated. A plan with no detectors must still be rejected
    rather than quietly completed."""
    _enable(monkeypatch)
    _stub_llm(monkeypatch, {"steps": [
        {"tool": "filter_data", "params": {}, "reason": "narrow"},
    ]})
    plan = plan_query(_intent())

    assert "source=deterministic" in _decisions(plan)
    assert "needs risk_classify to produce a result" in _decisions(plan)


# ---------------------------------------------------------------------------
# V13 — closed-set parameter VALUES
# ---------------------------------------------------------------------------


def _plan_with_pattern(value):
    return {"steps": [
        {"tool": "load_data", "params": {}, "reason": "load"},
        {"tool": "feature_engineer", "params": {"pattern_types": value}, "reason": "features"},
        {"tool": "rule_detect", "params": {}, "reason": "detect"},
        {"tool": "risk_classify", "params": {}, "reason": "score"},
    ]}


def test_invalid_pattern_type_value_is_rejected(monkeypatch):
    """The bug this rule was written for, reproduced.

    Observed live: the model proposed pattern_types=["risk"] on a plan that was
    otherwise entirely legal — real tools, dependencies satisfied, terminal tool
    present — and it was ACCEPTED. "risk" is not a PatternType, so
    feature_engineer computed 0 features, rule_detect evaluated 0 rules, and
    "who are my riskiest customers?" returned an empty answer with no warning.
    """
    _enable(monkeypatch)
    _stub_llm(monkeypatch, _plan_with_pattern(["risk"]))
    plan = plan_query(_intent("ranking"))

    assert "source=deterministic" in _decisions(plan)
    assert "'risk' is not a valid pattern_types" in _decisions(plan)


def test_valid_pattern_type_value_is_accepted(monkeypatch):
    _enable(monkeypatch)
    _stub_llm(monkeypatch, _plan_with_pattern(["structuring"]))
    plan = plan_query(_intent("ranking"))
    assert "source=llm" in _decisions(plan)


def test_pattern_value_checked_on_rule_detect_alias_too(monkeypatch):
    """rule_detect's frozen contract spells it `patterns`; both names carry the
    same closed set and both must be checked."""
    _enable(monkeypatch)
    _stub_llm(monkeypatch, {"steps": [
        {"tool": "load_data", "params": {}, "reason": "load"},
        {"tool": "feature_engineer", "params": {}, "reason": "features"},
        {"tool": "rule_detect", "params": {"patterns": ["nonsense"]}, "reason": "detect"},
        {"tool": "risk_classify", "params": {}, "reason": "score"},
    ]})
    plan = plan_query(_intent("ranking"))
    assert "'nonsense' is not a valid patterns" in _decisions(plan)


def test_bare_string_pattern_value_is_checked_not_iterated_as_chars(monkeypatch):
    """A model may send a string where a list is expected. It must be validated
    as one value, not exploded into characters."""
    _enable(monkeypatch)
    _stub_llm(monkeypatch, _plan_with_pattern("risk"))
    plan = plan_query(_intent("ranking"))

    text = _decisions(plan)
    assert "'risk' is not a valid pattern_types" in text
    assert "'r' is not a valid" not in text


# ---------------------------------------------------------------------------
# V14 — capabilities a plan may not reach, and aggregate_query's threshold
# ---------------------------------------------------------------------------


def _threshold_plan(agg_params=None):
    return {"steps": [
        {"tool": "load_data", "params": {}, "reason": "load"},
        {"tool": "filter_data", "params": {}, "reason": "narrow"},
        {"tool": "aggregate_query", "params": agg_params or {}, "reason": "count per sender"},
    ]}


def test_aggregate_query_threshold_is_injected(monkeypatch):
    """The silent-wrong-answer bug: aggregate_query defaults threshold to None,
    the deterministic planner always sets it from the parsed query, and nothing
    was passing it into an LLM plan. "Which customers made 10+ transactions?"
    ran with no threshold and returned every sender."""
    _enable(monkeypatch)
    _stub_llm(monkeypatch, _threshold_plan())
    intent = _intent("threshold_query", filters=Filters(min_txn_count=10))
    plan = plan_query(intent)

    assert "source=llm" in _decisions(plan)
    params = next(s.params for s in plan.steps if s.tool == "aggregate_query")
    assert params["threshold"] == 10
    assert params["group_by"] == ["sender_id"]
    assert params["agg_func"] == "count"
    assert "injected aggregate_query params" in _decisions(plan)


def test_model_supplied_aggregate_params_are_not_clobbered(monkeypatch):
    _enable(monkeypatch)
    _stub_llm(monkeypatch, _threshold_plan({"threshold": 3, "agg_func": "sum"}))
    plan = plan_query(_intent("threshold_query", filters=Filters(min_txn_count=10)))

    params = next(s.params for s in plan.steps if s.tool == "aggregate_query")
    assert params["threshold"] == 3, "an explicit model choice must win over injection"
    assert params["agg_func"] == "sum"


def test_no_threshold_in_query_means_none_injected(monkeypatch):
    """Nothing to inject is not an error — the tool's own default applies."""
    _enable(monkeypatch)
    _stub_llm(monkeypatch, _threshold_plan())
    plan = plan_query(_intent("threshold_query", filters=Filters()))

    params = next(s.params for s in plan.steps if s.tool == "aggregate_query")
    assert "threshold" not in params
    assert params["group_by"] == ["sender_id"]


@pytest.mark.parametrize("params,expected", [
    ({"source": "ibm"}, "source 'ibm' is not available to a plan"),
    ({"source": "ibm_stratified"}, "source 'ibm_stratified' is not available to a plan"),
    ({"force_rebuild": True}, "may not set 'force_rebuild'"),
    ({"nrows": 100}, "may not set 'nrows'"),
    ({"seed": 7}, "may not set 'seed'"),
    ({"target_size": 1000}, "may not set 'target_size'"),
    ({"max_pos_customers": 5}, "may not set 'max_pos_customers'"),
])
def test_load_data_capabilities_a_plan_may_not_reach(monkeypatch, params, expected):
    """A model-triggered dataset switch or parquet rebuild is a filesystem
    write on the model's say-so. V10 checked only that these names exist."""
    _enable(monkeypatch)
    _stub_llm(monkeypatch, {"steps": [
        {"tool": "load_data", "params": params, "reason": "load"},
        {"tool": "feature_engineer", "params": {}, "reason": "features"},
        {"tool": "rule_detect", "params": {}, "reason": "detect"},
        {"tool": "risk_classify", "params": {}, "reason": "score"},
    ]})
    plan = plan_query(_intent("ranking"))

    assert "source=deterministic" in _decisions(plan)
    assert expected in _decisions(plan)


@pytest.mark.parametrize("source", ["synthetic", "synthetic_alt"])
def test_permitted_sources_still_accepted(monkeypatch, source):
    _enable(monkeypatch)
    _stub_llm(monkeypatch, {"steps": [
        {"tool": "load_data", "params": {"source": source}, "reason": "load"},
        {"tool": "feature_engineer", "params": {}, "reason": "features"},
        {"tool": "rule_detect", "params": {}, "reason": "detect"},
        {"tool": "risk_classify", "params": {}, "reason": "score"},
    ]})
    plan = plan_query(_intent("ranking"))

    assert "source=llm" in _decisions(plan)
    assert next(s.params for s in plan.steps if s.tool == "load_data")["source"] == source


def test_forbidden_param_survives_the_load_data_repair(monkeypatch):
    """The repair moves load_data to the front; V14 must still see its params."""
    _enable(monkeypatch)
    _stub_llm(monkeypatch, {"steps": [
        {"tool": "feature_engineer", "params": {}, "reason": "features"},
        {"tool": "load_data", "params": {"force_rebuild": True}, "reason": "load"},
        {"tool": "rule_detect", "params": {}, "reason": "detect"},
        {"tool": "risk_classify", "params": {}, "reason": "score"},
    ]})
    plan = plan_query(_intent("ranking"))
    assert "may not set 'force_rebuild'" in _decisions(plan)


# ---------------------------------------------------------------------------
# V12 — the plan must be able to answer the question
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("intent_name,terminal", [
    ("full_analysis", "risk_classify"),
    ("pattern_search", "risk_classify"),
    ("ranking", "risk_classify"),
    ("eda", "eda_profile"),
    ("threshold_query", "aggregate_query"),
])
def test_truncated_plans_are_rejected_per_intent(monkeypatch, intent_name, terminal):
    """Before V12 a truncated plan passed every ordering rule vacuously: a
    local model reached 60% acceptance with only 7% of plans able to answer."""
    _enable(monkeypatch)
    _stub_llm(monkeypatch, {"steps": [
        {"tool": "load_data", "params": {}, "reason": "load"},
        {"tool": "filter_data", "params": {}, "reason": "narrow"},
    ]})
    plan = plan_query(_intent(intent_name, entities=["C-04521"]))
    assert f"needs {terminal} to produce a result" in _decisions(plan)


# ---------------------------------------------------------------------------
# Executed-plan recording
# ---------------------------------------------------------------------------


def test_executed_line_records_what_actually_ran(monkeypatch):
    _enable(monkeypatch)
    _stub_llm(monkeypatch, _valid_payload())
    intent = _intent()
    plan = plan_query(intent)
    run_plan(intent, plan)
    record_executed_plan(plan)

    assert _decisions(plan).rstrip().splitlines()[-1].startswith("planner: executed = ")


def test_executed_line_captures_executor_replanning(monkeypatch):
    """The executor appends ml_detect when the rules find nothing. That is a
    plan change made after planning, and the trace must show it."""
    _enable(monkeypatch)
    _stub_llm(monkeypatch, _valid_payload())

    intent = _intent()
    plan = plan_query(intent)
    proposed = [s.tool for s in plan.steps]
    assert "ml_detect" not in proposed

    monkeypatch.setitem(
        _get_tools(), "rule_detect",
        lambda ctx, **kw: __import__(
            "backend.tools.base", fromlist=["ToolResult"]
        ).ToolResult(ok=True, artifacts={"rule_hits": []}, metrics={"rules_fired": 0}),
    )
    run_plan(intent, plan)
    record_executed_plan(plan)

    executed = _decisions(plan).splitlines()[-1]
    assert "ml_detect" in executed, f"executor's insertion should appear in: {executed}"


# ---------------------------------------------------------------------------
# Validator used directly
# ---------------------------------------------------------------------------


def test_validator_returns_all_rejections_and_no_steps():
    result = validate_proposal({"steps": [{"tool": "zzz", "params": {}, "reason": "r"}]},
                               _intent(), _tools())
    assert result.ok is False
    assert result.steps == []
    assert result.rejections
