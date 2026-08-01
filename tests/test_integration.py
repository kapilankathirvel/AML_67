"""
Phase 6 — integration tests against Track B's real tools (AML_USE_MOCKS=0)
and the real sample dataset, not the mock fixtures.

executor._TOOLS_CACHE is a module-level cache keyed on whatever
settings.aml_use_mocks was at first call — earlier tests in the suite call it
with mocks on, so it must be reset (both before and after) whenever a test
here flips the setting, or the real tools never actually get imported.
"""

import backend.agent.executor as executor_mod
from backend.agent.executor import run_plan
from backend.agent.planner import build_plan
from backend.config import settings
from backend.schemas import Filters, QueryIntent

import pytest

# A real customer ID present in data/sample/aml_sample.csv's structuring cohort
# (mock fixtures use C-04521, which does not exist in the real dataset — real
# IDs follow Track B's synthetic generator's own scheme, e.g. C-STR02, C-N0001).
REAL_STRUCTURING_ENTITY = "C-STR02"


@pytest.fixture
def real_tools(monkeypatch):
    monkeypatch.setattr(settings, "aml_use_mocks", False)
    executor_mod._TOOLS_CACHE = None
    yield
    executor_mod._TOOLS_CACHE = None


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    """full_analysis-style tests here run against real data and can produce
    dozens of HIGH-risk flags, each of which triggers narrator._explain()'s
    LLM polish path — without this, every run of this file silently made
    dozens of REAL network calls to whatever provider/key was live in .env.
    None of these tests assert on LLM-polished wording (only counts, tool
    sequences, entity scoping, risk scores), so forcing template-only mode
    costs zero coverage while eliminating what was likely the single largest
    contributor to today's free-tier quota exhaustion."""
    monkeypatch.setattr("backend.agent.narrator.complete_json", lambda *a, **kw: None)


def test_full_analysis_against_real_tools(real_tools):
    intent = QueryIntent(raw_query="Analyse this dataset for suspicious activity",
                          intent="full_analysis", parsed_by="rules", confidence=0.9)
    plan = build_plan(intent)
    response = run_plan(intent, plan)

    assert all(s.status == "ok" for s in plan.steps), [(s.tool, s.status) for s in plan.steps]
    assert not response.warnings
    assert response.flags, "expected flags against the real structuring/smurfing/layering/cashout cohorts"
    assert all(f.explanation for f in response.flags)
    assert all(f.escalation in ("report", "review", "monitor", "no_action") for f in response.flags)


def test_pattern_search_scopes_features_and_rules(real_tools):
    intent = QueryIntent(raw_query="Find structuring patterns in the last 30 days",
                          intent="pattern_search", pattern_types=["structuring"],
                          parsed_by="rules", confidence=0.9)
    plan = build_plan(intent)
    response = run_plan(intent, plan)

    assert all(s.status == "ok" for s in plan.steps)
    # Only the structuring rules should have been evaluated, not the full set.
    # That is R1 (sender side) and R7 (receiver side) — R7 was added as the
    # mirror of the same typology, so scoping to "structuring" legitimately
    # runs both ends of the transaction. The point of this assertion is that
    # smurfing/layering/cashout/velocity/dormancy rules stayed out.
    assert response.metrics.get("rules_evaluated") == ["R1", "R7"]
    assert response.flags


def test_entity_investigation_agrees_with_the_full_sweep(real_tools):
    """Asking about one customer must not change that customer's risk score.

    entity_investigation used to skip ml_detect on the reasoning that one entity is
    too small a sample — but the same plan runs feature_engineer across the whole
    population, so ml_detect would have received all 270 customers. Skipping it
    zeroed the ML half of Contract 5's formula, and every single-entity query came
    back at 100 * 0.6 * max_rule_weight: C-STR02 scored 51.00 MEDIUM ('review') here
    while full_analysis called the same customer 89.84 HIGH ('report'). The direct
    question about a customer was the one query that understated their risk, and it
    downgraded them out of the SAR-drafting tier.
    """
    full_intent = QueryIntent(raw_query="Analyse this dataset for suspicious activity",
                              intent="full_analysis", parsed_by="rules", confidence=0.9)
    full_plan = build_plan(full_intent)
    full_response = run_plan(full_intent, full_plan)
    by_entity = {f.entity_id: f for f in full_response.flags}
    assert REAL_STRUCTURING_ENTITY in by_entity, "fixture entity should be flagged by a full sweep"
    expected = by_entity[REAL_STRUCTURING_ENTITY]

    solo_intent = QueryIntent(raw_query=f"Is customer {REAL_STRUCTURING_ENTITY} suspicious?",
                              intent="entity_investigation", entities=[REAL_STRUCTURING_ENTITY],
                              parsed_by="rules", confidence=0.9)
    solo_plan = build_plan(solo_intent)
    solo_response = run_plan(solo_intent, solo_plan)

    assert "ml_detect" in [s.tool for s in solo_plan.steps]
    assert len(solo_response.flags) == 1
    actual = solo_response.flags[0]

    assert actual.risk_score == expected.risk_score, (
        f"{REAL_STRUCTURING_ENTITY} scored {actual.risk_score} when asked about directly "
        f"but {expected.risk_score} in a full sweep — the score must not depend on the query"
    )
    assert actual.risk_level == expected.risk_level
    assert actual.escalation == expected.escalation


def test_small_sample_still_drops_ml_detect(real_tools):
    """The sample-size guard removed from the planner was redundant, not load-bearing.

    executor.py drops ml_detect at runtime when filter_data leaves under 50 rows, and
    that is the guard WORKPLAN.md §5 actually specifies — it fires on the size of the
    data rather than on the name of the intent.
    """
    # $20,000 leaves 31 of aml_sample.csv's 2,002 transactions — under the 50-row
    # floor. A structuring-band filter would not do: amount_min=9900 still leaves 680
    # rows, because the $9,000-9,999 band is exactly where the planted structuring is.
    intent = QueryIntent(raw_query="Find structuring in transactions over $20,000",
                          intent="pattern_search", pattern_types=["structuring"],
                          filters=Filters(amount_min=20000.0),
                          parsed_by="rules", confidence=0.9)
    plan = build_plan(intent)
    # load_data defaults to synthetic_alt, a much larger dataset where $20,000 leaves
    # far more than 50 rows. Pin the source or this test silently stops testing the
    # guard it is named after.
    for step in plan.steps:
        if step.tool == "load_data":
            step.params = {**step.params, "source": "synthetic"}
    assert "ml_detect" in [s.tool for s in plan.steps], "planner should have planned ml_detect"
    response = run_plan(intent, plan)

    assert all(s.status == "ok" for s in plan.steps), [(s.tool, s.status) for s in plan.steps]
    assert any("too small" in d for d in plan.decisions), (
        f"expected the <50-row guard to fire; decisions were {plan.decisions}"
    )
    # Deliberately NOT asserting ml_detect is absent from the final plan. At $20,000
    # the filtered frame holds 31 rows, the guard drops ml_detect — and then the
    # *other* re-planning rule fires, because no structuring rule hits 31 high-value
    # transactions, and re-adds ml_detect to widen the net. Both rules behaving
    # correctly in sequence produces a plan that still contains ml_detect, so the
    # decision log is what records that the guard ran, not the step list.
    assert any("widening the net" in d for d in plan.decisions)
    assert response.summary


def test_threshold_query_against_real_tools(real_tools):
    intent = QueryIntent(raw_query="Which customers made 10+ transactions under $10,000?",
                          intent="threshold_query",
                          filters=Filters(min_txn_count=10, amount_max=10000.0),
                          parsed_by="rules", confidence=0.9)
    plan = build_plan(intent)
    response = run_plan(intent, plan)

    assert all(s.status == "ok" for s in plan.steps)
    assert "ml_detect" not in [s.tool for s in plan.steps]
    assert response.metrics.get("row_count", 0) > 0
    assert str(response.metrics["row_count"]) in response.summary


def test_entity_investigation_scopes_to_one_entity(real_tools):
    intent = QueryIntent(raw_query=f"Is customer {REAL_STRUCTURING_ENTITY} suspicious?",
                          intent="entity_investigation", entities=[REAL_STRUCTURING_ENTITY],
                          parsed_by="rules", confidence=0.9)
    plan = build_plan(intent)
    response = run_plan(intent, plan)

    assert all(s.status == "ok" for s in plan.steps)
    assert len(response.flags) == 1
    assert response.flags[0].entity_id == REAL_STRUCTURING_ENTITY
    assert response.flags[0].explanation


def test_entity_resolution_maps_bare_number_to_real_customer(real_tools):
    """intent_parser normalises 'customer 2' -> 'C-00002', which doesn't exist
    in the real dataset (real IDs are e.g. C-N0002). The executor should
    resolve it by numeric id to a real customer_id, not just fail to match."""
    intent = QueryIntent(raw_query="Is customer 2 suspicious?", intent="entity_investigation",
                          entities=["C-00002"], parsed_by="rules", confidence=0.9)
    plan = build_plan(intent)
    response = run_plan(intent, plan)

    assert all(s.status == "ok" for s in plan.steps)
    assert intent.entities != ["C-00002"], "entity should have been resolved to a real customer_id"
    assert intent.entities[0].startswith("C-") and intent.entities[0] != "C-00002"
    assert any("resolved entity" in d for d in plan.decisions)
    entity_lookup_step = next(s for s in plan.steps if s.tool == "entity_lookup")
    assert entity_lookup_step.params["entity_id"] == intent.entities[0]


def test_entity_resolution_leaves_out_of_range_id_unresolved(real_tools):
    """A number with no real counterpart (out of the ~270-customer range)
    should degrade gracefully — no crash, no flags, not silently matched to
    the wrong customer."""
    intent = QueryIntent(raw_query="Is customer 4521 suspicious?", intent="entity_investigation",
                          entities=["C-04521"], parsed_by="rules", confidence=0.9)
    plan = build_plan(intent)
    response = run_plan(intent, plan)

    assert all(s.status == "ok" for s in plan.steps)
    assert intent.entities == ["C-04521"]
    assert response.flags == []
    assert any("no real customer found" in d for d in plan.decisions)


def test_entity_resolution_passes_through_already_real_id(real_tools):
    intent = QueryIntent(raw_query=f"Is customer {REAL_STRUCTURING_ENTITY} suspicious?",
                          intent="entity_investigation", entities=[REAL_STRUCTURING_ENTITY],
                          parsed_by="rules", confidence=0.9)
    plan = build_plan(intent)
    run_plan(intent, plan)

    assert intent.entities == [REAL_STRUCTURING_ENTITY]
    assert not any("resolved entity" in d for d in plan.decisions)


def test_entity_investigation_unknown_id_returns_no_flags_not_a_crash(real_tools):
    intent = QueryIntent(raw_query="Is customer 4521 suspicious?", intent="entity_investigation",
                          entities=["C-04521"], parsed_by="rules", confidence=0.9)
    plan = build_plan(intent)
    response = run_plan(intent, plan)

    assert all(s.status == "ok" for s in plan.steps)
    assert response.flags == []
    assert response.summary


def test_ranking_truncates_to_top_n(real_tools):
    intent = QueryIntent(raw_query="Top 5 highest-risk customers", intent="ranking", top_n=5,
                          parsed_by="rules", confidence=0.9)
    plan = build_plan(intent)
    response = run_plan(intent, plan)

    assert all(s.status == "ok" for s in plan.steps)
    assert len(response.flags) == 5
    scores = [f.risk_score for f in response.flags]
    assert scores == sorted(scores, reverse=True)


def test_eda_intent_runs_no_detection_against_real_tools(real_tools):
    intent = QueryIntent(raw_query="Show transaction distribution by country", intent="eda",
                          parsed_by="rules", confidence=0.9)
    plan = build_plan(intent)
    response = run_plan(intent, plan)

    assert all(s.status == "ok" for s in plan.steps)
    assert response.flags == []
    assert response.metrics.get("txn_type_counts") or response.metrics.get("channel_counts")


def test_explain_flag_actually_scores_the_entity(real_tools):
    """Phase 7 fix: explain_flag previously never loaded data (per Contract 4's
    'reuse cached run' design, which was never wired to anything) and always
    returned empty. It now loads data and scores just the requested entity."""
    intent = QueryIntent(raw_query=f"Why was customer {REAL_STRUCTURING_ENTITY} flagged?",
                          intent="explain_flag", entities=[REAL_STRUCTURING_ENTITY],
                          parsed_by="rules", confidence=0.9)
    plan = build_plan(intent)
    response = run_plan(intent, plan)

    assert all(s.status == "ok" for s in plan.steps)
    assert "load_data" in [s.tool for s in plan.steps]
    assert "eda_profile" not in [s.tool for s in plan.steps]
    # ml_detect now runs here — see WORKPLAN.md §8's amendment. Without it the
    # explanation quoted a score the ML term had been zeroed out of.
    assert "ml_detect" in [s.tool for s in plan.steps]
    assert len(response.flags) == 1
    assert response.flags[0].entity_id == REAL_STRUCTURING_ENTITY
    assert response.flags[0].explanation in response.summary


def test_false_positive_reduction_vs_naive_baseline(real_tools):
    """README.md's Results section quantifies this against the synthetic
    dataset's label_is_laundering ground truth: our system flags far fewer
    customers than a naive 'any txn > $9,000' rule, at a far lower
    false-positive rate. This test protects that headline claim without
    pinning exact percentages (which would be brittle to minor threshold
    tuning) — if this ever fails, the false-positive-reduction story in the
    README is no longer true and needs re-validating, not just re-asserting.
    """
    from evaluation.harness import evaluate, load_ground_truth, naive_baseline

    intent = QueryIntent(raw_query="Analyse this dataset for suspicious activity",
                          intent="full_analysis", parsed_by="rules", confidence=0.9)
    plan = build_plan(intent)
    # load_data defaults to source="synthetic_alt", a different dataset from the
    # labelled one this test scores against. Pin it, or the flags and the labels
    # describe different populations and the comparison is meaningless.
    for step in plan.steps:
        if step.tool == "load_data":
            step.params = {**step.params, "source": "synthetic"}
    response = run_plan(intent, plan)
    our_flagged = {f.entity_id for f in response.flags}

    df, gt = load_ground_truth()
    naive_flagged = naive_baseline(df, gt.all_customers)

    ours = evaluate(our_flagged, gt.sender_only, gt.all_customers)
    naive = evaluate(naive_flagged, gt.sender_only, gt.all_customers)

    assert ours.flagged < naive.flagged / 5, (
        f"our system flagged {ours.flagged}, naive flagged {naive.flagged} — "
        "expected at least a 5x reduction in flagged customers"
    )
    assert ours.false_positive_rate < naive.false_positive_rate / 5, (
        f"our FPR {ours.false_positive_rate:.3f} vs naive FPR "
        f"{naive.false_positive_rate:.3f} — expected at least a 5x FPR reduction"
    )
    # sanity: we shouldn't be achieving low FPR by simply not flagging anyone
    assert ours.true_positives > 0, "our system caught zero true positives"


def test_full_analysis_response_is_actually_json_serializable(real_tools):
    """Regression test for a real bug found live: eda_profile's Plotly figures
    (via .to_dict()) embed raw numpy.ndarray values in trace data (x/y/text/
    marker.color). AgentResponse.charts is Any-typed, so Pydantic validates
    these fine at construction time — but FastAPI's JSON response
    serialization crashed with PydanticSerializationError once this actually
    reached the HTTP layer. Every other test calls run_plan() directly and
    inspects the object, which never exercises serialization — this test
    exists specifically to catch that gap. A 500 on the demo's flagship query
    is the worst possible thing to have work in tests and fail live."""
    intent = QueryIntent(raw_query="Analyse this dataset for suspicious activity",
                          intent="full_analysis", parsed_by="rules", confidence=0.9)
    plan = build_plan(intent)
    response = run_plan(intent, plan)

    assert response.charts, "expected eda_profile to have produced charts to test against"
    # this is the exact call FastAPI makes when returning the response over HTTP
    response.model_dump_json()
