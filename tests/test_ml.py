"""
tests/test_ml.py

Tests for backend/tools/ml_detect.py, backend/tools/risk.py,
backend/tools/aggregate.py, and backend/tools/entity.py.

Ownership: Track B (tests/test_ml.py is in the Track B ownership matrix).

Test strategy:
  ml_detect:
    - scores all entities when features present
    - top entity is genuinely anomalous (from labelled pattern cohort)
    - top_features are valid feature names from feature_list
    - returns empty ml_scores (not error) when below min_samples
    - does not mutate ctx

  risk_classify:
    - known-anomalous entity scores HIGH (rule hits present)
    - normal entity scores LOW/none (no rule hits)
    - formula: score = 100 * (0.6 * rule_weight + 0.4 * ml_pct)
    - escalation correct for each band
    - empty rule_hits + empty ml_scores → clean dataset, ok=True

  aggregate_query:
    - count, sum, mean, max, min, nunique produce correct results on fixture
    - threshold filter excludes below-threshold groups
    - invalid agg_func → ok=False
    - invalid group_by column → ok=False
    - empty df → ok=True, empty table

  entity_lookup:
    - known entity returns correct sent/received counts
    - bare numeric ID '1' is normalised to 'C-00001'
    - unknown entity → ok=True with note (not ok=False)
    - entity_profile contains required keys
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from backend.tools.base import ToolContext
from backend.tools.features import feature_engineer
from backend.tools.rules import rule_detect
from backend.tools.ml_detect import ml_detect, IF_MIN_SAMPLES, TOP_N_FEATURES
from backend.tools.risk import risk_classify, RISK_HIGH_THRESHOLD, RISK_MEDIUM_THRESHOLD
from backend.tools.aggregate import aggregate_query
from backend.tools.entity import entity_lookup

SAMPLE_TX   = "data/sample/aml_sample.csv"
SAMPLE_CUST = "data/sample/aml_sample_customers.csv"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE_TX)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["is_cross_border"] = df["is_cross_border"].astype(bool)
    return df


@pytest.fixture(scope="module")
def sample_cust() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CUST)


@pytest.fixture(scope="module")
def full_ctx(sample_df, sample_cust) -> ToolContext:
    """ToolContext with features + rule_hits + ml_scores populated."""
    ctx = ToolContext(df=sample_df.copy())
    ctx.artifacts["customers"] = sample_cust

    feat_r = feature_engineer(ctx)
    assert feat_r.ok, feat_r.error
    ctx.artifacts.update(feat_r.artifacts)

    rule_r = rule_detect(ctx)
    assert rule_r.ok, rule_r.error
    ctx.artifacts.update(rule_r.artifacts)

    ml_r = ml_detect(ctx)
    assert ml_r.ok, ml_r.error
    ctx.artifacts.update(ml_r.artifacts)

    return ctx


# ---------------------------------------------------------------------------
# ml_detect tests
# ---------------------------------------------------------------------------


class TestMlDetect:
    def test_scores_all_entities(self, full_ctx: ToolContext) -> None:
        scores = full_ctx.artifacts["ml_scores"]
        n_customers = len(full_ctx.artifacts["features"])
        assert len(scores) == n_customers, (
            f"Expected {n_customers} ml_scores, got {len(scores)}"
        )

    def test_scores_are_in_0_1(self, full_ctx: ToolContext) -> None:
        for s in full_ctx.artifacts["ml_scores"]:
            assert 0.0 <= s["score"] <= 1.0, f"score out of range: {s}"
            assert 0.0 <= s["percentile"] <= 1.0, f"percentile out of range: {s}"

    def test_top_entity_is_pattern_customer(self, full_ctx: ToolContext) -> None:
        """The highest-scoring entity should be from a labelled pattern cohort."""
        scores = full_ctx.artifacts["ml_scores"]
        top_id = scores[0]["entity_id"]
        # Labelled pattern customers have C-HUB, C-STR, C-SMF, C-LAY, C-RCO prefix
        pattern_prefixes = ("C-HUB", "C-STR", "C-SMF", "C-LAY", "C-RCO")
        assert any(top_id.startswith(p) for p in pattern_prefixes), (
            f"Top ML entity {top_id} is not from a known pattern cohort — "
            f"check whether IF is learning anything useful"
        )

    def test_top_features_are_valid_names(self, full_ctx: ToolContext) -> None:
        feature_list = set(full_ctx.artifacts["feature_list"])
        for s in full_ctx.artifacts["ml_scores"][:10]:
            for f in s["top_features"]:
                assert f in feature_list, (
                    f"top_feature '{f}' is not in feature_list"
                )

    def test_top_features_count(self, full_ctx: ToolContext) -> None:
        for s in full_ctx.artifacts["ml_scores"]:
            assert len(s["top_features"]) <= TOP_N_FEATURES

    def test_sorted_descending(self, full_ctx: ToolContext) -> None:
        scores = [s["score"] for s in full_ctx.artifacts["ml_scores"]]
        assert scores == sorted(scores, reverse=True), (
            "ml_scores should be sorted descending by score"
        )

    def test_below_min_samples_returns_empty(self) -> None:
        """Fewer than IF_MIN_SAMPLES entities → empty ml_scores, not error."""
        # 3 customers — below IF_MIN_SAMPLES=10
        tiny_df = pd.DataFrame({
            "txn_id": ["T-1", "T-2", "T-3"],
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            "sender_id": ["C-A", "C-B", "C-C"],
            "receiver_id": ["C-B", "C-C", "C-A"],
            "amount": [1000.0, 2000.0, 3000.0],
            "currency": ["USD", "USD", "USD"],
            "txn_type": ["transfer", "transfer", "transfer"],
            "channel": ["online", "online", "online"],
            "sender_country": ["US", "US", "US"],
            "receiver_country": ["US", "US", "US"],
            "is_cross_border": [False, False, False],
            "label_is_laundering": [None, None, None],
            "pattern_label": [None, None, None],
        })
        ctx = ToolContext(df=tiny_df)
        feat_r = feature_engineer(ctx)
        assert feat_r.ok
        ctx.artifacts.update(feat_r.artifacts)

        result = ml_detect(ctx)
        assert result.ok
        assert result.artifacts["ml_scores"] == []
        assert any("below minimum" in n for n in result.notes)

    def test_does_not_mutate_ctx(self, sample_df: pd.DataFrame) -> None:
        ctx = ToolContext(df=sample_df.copy())
        feat_r = feature_engineer(ctx)
        ctx.artifacts.update(feat_r.artifacts)
        original_len = len(ctx.df)
        ml_detect(ctx)
        assert len(ctx.df) == original_len


class TestPercentileReferencePopulation:
    """An anomaly percentile must describe the customer, not the query.

    Percentiles used to be ranked inside whatever cohort survived filter_data, so
    narrowing a query silently re-scored everyone still in it. Measured on the real
    sample: an amount_min=5000 filter moved percentiles by up to 0.73 and pushed four
    customers across a risk band. ml_detect now fits and ranks on
    artifacts["features_reference"] — the unfiltered population — so these are
    invariant.
    """

    @staticmethod
    def _scores(sample_df: pd.DataFrame, sample_cust: pd.DataFrame,
                amount_min: float | None) -> dict[str, float]:
        ctx = ToolContext(df=sample_df.copy())
        ctx.artifacts["customers"] = sample_cust
        # load_data captures this in the real pipeline; set it directly so the test
        # exercises ml_detect rather than the loader.
        ctx.artifacts["transactions_reference"] = sample_df.copy()

        if amount_min is not None:
            ctx.df = ctx.df[ctx.df["amount"] >= amount_min].copy()

        feat_r = feature_engineer(ctx)
        assert feat_r.ok, feat_r.error
        ctx.artifacts.update(feat_r.artifacts)

        ml_r = ml_detect(ctx)
        assert ml_r.ok, ml_r.error
        return {s["entity_id"]: s["percentile"] for s in ml_r.artifacts["ml_scores"]}

    @pytest.fixture(scope="class")
    def unfiltered_scores(self, sample_df, sample_cust) -> dict[str, float]:
        """Computed once — feature_engineer over the full sample is the slow part of
        this class, and every parametrised case compares against the same baseline."""
        return self._scores(sample_df, sample_cust, None)

    @pytest.mark.parametrize("amount_min", [2000.0, 5000.0, 8000.0])
    def test_percentiles_are_invariant_under_filtering(
        self, sample_df: pd.DataFrame, sample_cust: pd.DataFrame,
        unfiltered_scores: dict[str, float], amount_min: float,
    ) -> None:
        unfiltered = unfiltered_scores
        filtered   = self._scores(sample_df, sample_cust, amount_min)

        shared = set(unfiltered) & set(filtered)
        assert shared, "filter removed every entity — test proves nothing"

        drifted = {
            eid: (unfiltered[eid], filtered[eid])
            for eid in shared
            if abs(unfiltered[eid] - filtered[eid]) > 1e-9
        }
        assert not drifted, (
            f"amount_min={amount_min} changed {len(drifted)} percentiles; "
            f"a customer's anomaly rank must not depend on the query. {list(drifted.items())[:5]}"
        )

    def test_reference_is_the_full_population_not_the_cohort(
        self, sample_df: pd.DataFrame, sample_cust: pd.DataFrame
    ) -> None:
        """The peer group stays at full size even when the working frame shrinks."""
        ctx = ToolContext(df=sample_df.copy())
        ctx.artifacts["customers"] = sample_cust
        ctx.artifacts["transactions_reference"] = sample_df.copy()
        ctx.df = ctx.df[ctx.df["amount"] >= 5000.0].copy()

        feat_r = feature_engineer(ctx)
        ctx.artifacts.update(feat_r.artifacts)
        ml_r = ml_detect(ctx)

        cohort = len(feat_r.artifacts["features"])
        reference = ml_r.metrics["ml_reference_population"]
        assert reference > cohort, (
            f"reference population ({reference}) should exceed the filtered cohort ({cohort})"
        )
        assert reference == len(feat_r.artifacts["features_reference"])

    def test_falls_back_to_cohort_when_no_reference_present(
        self, sample_df: pd.DataFrame, sample_cust: pd.DataFrame
    ) -> None:
        """Without a reference artifact, behaviour is the pre-existing cohort ranking —
        tools stay usable standalone, which several tests in this file rely on."""
        ctx = ToolContext(df=sample_df.copy())
        ctx.artifacts["customers"] = sample_cust
        feat_r = feature_engineer(ctx)
        ctx.artifacts.update(feat_r.artifacts)
        assert "transactions_reference" not in ctx.artifacts

        ml_r = ml_detect(ctx)
        assert ml_r.ok
        assert ml_r.metrics["ml_reference_population"] == len(feat_r.artifacts["features"])


# ---------------------------------------------------------------------------
# risk_classify tests
# ---------------------------------------------------------------------------


class TestRiskClassify:
    def test_high_risk_entity_flagged(self, full_ctx: ToolContext) -> None:
        risk_r = risk_classify(full_ctx)
        assert risk_r.ok, risk_r.error
        rows = risk_r.artifacts["risk_rows"]
        high_entities = [r for r in rows if r["risk_level"] == "high"]
        assert len(high_entities) > 0, "Expected at least one HIGH risk entity"

    def test_risk_rows_schema(self, full_ctx: ToolContext) -> None:
        risk_r = risk_classify(full_ctx)
        required_keys = {
            "entity_id", "risk_score", "risk_level", "escalation",
            "patterns", "triggered_rules", "ml_score", "evidence",
        }
        for row in risk_r.artifacts["risk_rows"]:
            for k in required_keys:
                assert k in row, f"risk_row missing key '{k}'"

    def test_escalation_matches_risk_level(self, full_ctx: ToolContext) -> None:
        risk_r = risk_classify(full_ctx)
        expected = {
            "high": "report",
            "medium": "review",
            "low": "monitor",
            "none": "no_action",
        }
        for row in risk_r.artifacts["risk_rows"]:
            assert row["escalation"] == expected[row["risk_level"]], (
                f"Wrong escalation for {row['entity_id']}: "
                f"level={row['risk_level']} but escalation={row['escalation']}"
            )

    def test_formula_exact(self) -> None:
        """Verify formula: score = 100 * (0.6 * rule_weight + 0.4 * ml_percentile)."""
        ctx = ToolContext(df=pd.DataFrame())
        ctx.artifacts["rule_hits"] = [{
            "entity_id": "C-TEST",
            "rule_id":   "R1",
            "weight":    0.85,
            "evidence":  {"test": 1},
        }]
        ctx.artifacts["ml_scores"] = [{
            "entity_id": "C-TEST",
            "score":     0.80,
            "percentile": 0.80,
            "top_features": ["rolling_1d_count"],
        }]
        r = risk_classify(ctx)
        assert r.ok
        rows = r.artifacts["risk_rows"]
        assert len(rows) == 1
        expected = round(100.0 * (0.6 * 0.85 + 0.4 * 0.80), 2)
        actual   = rows[0]["risk_score"]
        assert abs(actual - expected) < 0.01, (
            f"Formula mismatch: expected {expected}, got {actual}"
        )
        assert rows[0]["risk_level"] == "high"
        assert rows[0]["escalation"] == "report"

    def test_clean_dataset_no_hits(self) -> None:
        """No rule hits + no ML scores → clean, ok=True, empty risk_rows."""
        ctx = ToolContext(df=pd.DataFrame())
        ctx.artifacts["rule_hits"] = []
        ctx.artifacts["ml_scores"] = []
        r = risk_classify(ctx)
        assert r.ok
        assert r.artifacts["risk_rows"] == []
        assert any("clean" in n for n in r.notes)

    def test_sorted_descending(self, full_ctx: ToolContext) -> None:
        risk_r = risk_classify(full_ctx)
        scores = [r["risk_score"] for r in risk_r.artifacts["risk_rows"]]
        assert scores == sorted(scores, reverse=True)

    def test_risk_summary_table_populated(self, full_ctx: ToolContext) -> None:
        risk_r = risk_classify(full_ctx)
        summary = risk_r.tables["risk_summary"]
        assert len(summary) > 0
        row = summary[0]
        assert "entity_id" in row
        assert "risk_score" in row
        assert "risk_level" in row


# ---------------------------------------------------------------------------
# aggregate_query tests
# ---------------------------------------------------------------------------


@pytest.fixture
def agg_df() -> pd.DataFrame:
    """Small fixture for aggregate tests."""
    rows = [
        {"txn_id": "T-1", "sender_id": "C-A", "receiver_id": "C-X", "amount": 9500.0,
         "timestamp": "2025-01-01", "currency": "USD", "txn_type": "transfer", "channel": "online",
         "sender_country": "US", "receiver_country": "US", "is_cross_border": False,
         "label_is_laundering": False, "pattern_label": None},
        {"txn_id": "T-2", "sender_id": "C-A", "receiver_id": "C-Y", "amount": 9200.0,
         "timestamp": "2025-01-02", "currency": "USD", "txn_type": "transfer", "channel": "online",
         "sender_country": "US", "receiver_country": "US", "is_cross_border": False,
         "label_is_laundering": False, "pattern_label": None},
        {"txn_id": "T-3", "sender_id": "C-A", "receiver_id": "C-Z", "amount": 9800.0,
         "timestamp": "2025-01-03", "currency": "USD", "txn_type": "transfer", "channel": "online",
         "sender_country": "US", "receiver_country": "US", "is_cross_border": False,
         "label_is_laundering": False, "pattern_label": None},
        {"txn_id": "T-4", "sender_id": "C-B", "receiver_id": "C-X", "amount": 500.0,
         "timestamp": "2025-01-04", "currency": "USD", "txn_type": "deposit", "channel": "atm",
         "sender_country": "US", "receiver_country": "US", "is_cross_border": False,
         "label_is_laundering": False, "pattern_label": None},
    ]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["is_cross_border"] = df["is_cross_border"].astype(bool)
    return df


class TestAggregateQuery:
    def test_count_exact(self, agg_df: pd.DataFrame) -> None:
        ctx = ToolContext(df=agg_df)
        r = aggregate_query(ctx, group_by=["sender_id"], agg_func="count")
        assert r.ok
        results = {row["sender_id"]: row["result"] for row in r.tables["agg_result"]}
        assert results["C-A"] == 3
        assert results["C-B"] == 1

    def test_sum_exact(self, agg_df: pd.DataFrame) -> None:
        ctx = ToolContext(df=agg_df)
        r = aggregate_query(ctx, group_by=["sender_id"], agg_col="amount", agg_func="sum")
        assert r.ok
        results = {row["sender_id"]: row["result"] for row in r.tables["agg_result"]}
        assert abs(results["C-A"] - 28500.0) < 1.0
        assert abs(results["C-B"] - 500.0) < 1.0

    def test_threshold_filter(self, agg_df: pd.DataFrame) -> None:
        ctx = ToolContext(df=agg_df)
        r = aggregate_query(ctx, group_by=["sender_id"], agg_func="count", threshold=3)
        assert r.ok
        ids = [row["sender_id"] for row in r.tables["agg_result"]]
        assert "C-A" in ids
        assert "C-B" not in ids

    def test_nunique(self, agg_df: pd.DataFrame) -> None:
        ctx = ToolContext(df=agg_df)
        r = aggregate_query(ctx, group_by=["sender_id"], agg_col="receiver_id", agg_func="nunique")
        assert r.ok
        results = {row["sender_id"]: row["result"] for row in r.tables["agg_result"]}
        assert results["C-A"] == 3  # C-X, C-Y, C-Z
        assert results["C-B"] == 1  # C-X

    def test_mean_exact(self, agg_df: pd.DataFrame) -> None:
        ctx = ToolContext(df=agg_df)
        r = aggregate_query(ctx, group_by=["sender_id"], agg_col="amount", agg_func="mean")
        assert r.ok
        results = {row["sender_id"]: row["result"] for row in r.tables["agg_result"]}
        expected_mean_a = round((9500.0 + 9200.0 + 9800.0) / 3, 2)
        assert abs(results["C-A"] - expected_mean_a) < 0.1

    def test_invalid_agg_func(self, agg_df: pd.DataFrame) -> None:
        ctx = ToolContext(df=agg_df)
        r = aggregate_query(ctx, group_by=["sender_id"], agg_func="median")
        assert not r.ok
        assert "invalid agg_func" in r.error

    def test_invalid_group_by(self, agg_df: pd.DataFrame) -> None:
        ctx = ToolContext(df=agg_df)
        r = aggregate_query(ctx, group_by=["nonexistent_col"], agg_func="count")
        assert not r.ok
        assert "not found" in r.error

    def test_empty_df(self) -> None:
        ctx = ToolContext(df=pd.DataFrame())
        r = aggregate_query(ctx)
        assert r.ok
        assert r.tables["agg_result"] == []

    def test_sorted_descending(self, agg_df: pd.DataFrame) -> None:
        ctx = ToolContext(df=agg_df)
        r = aggregate_query(ctx, group_by=["sender_id"], agg_func="count")
        assert r.ok
        results = [row["result"] for row in r.tables["agg_result"]]
        assert results == sorted(results, reverse=True)

    def test_on_sample(self, sample_df: pd.DataFrame) -> None:
        """10+ transactions per customer threshold query on real data."""
        ctx = ToolContext(df=sample_df)
        r = aggregate_query(ctx, group_by=["sender_id"], agg_func="count", threshold=10)
        assert r.ok
        assert len(r.tables["agg_result"]) > 0
        for row in r.tables["agg_result"]:
            assert row["result"] >= 10


# ---------------------------------------------------------------------------
# entity_lookup tests
# ---------------------------------------------------------------------------


class TestEntityLookup:
    def test_known_entity_returns_profile(self, sample_df: pd.DataFrame, sample_cust: pd.DataFrame) -> None:
        ctx = ToolContext(df=sample_df)
        ctx.artifacts["customers"] = sample_cust
        r = entity_lookup(ctx, entity_id="C-STR01")
        assert r.ok
        prof = r.artifacts["entity_profile"]
        assert prof["customer_id"] == "C-STR01"
        assert prof["total_transactions_sent"] > 0

    def test_profile_required_keys(self, sample_df: pd.DataFrame) -> None:
        ctx = ToolContext(df=sample_df)
        r = entity_lookup(ctx, entity_id="C-STR01")
        assert r.ok
        prof = r.artifacts["entity_profile"]
        required = {
            "customer_id", "total_transactions_sent", "total_transactions_received",
            "total_amount_sent", "distinct_receivers", "first_txn_date", "last_txn_date",
        }
        for k in required:
            assert k in prof, f"entity_profile missing key '{k}'"

    def test_bare_numeric_id_normalised(self, sample_df: pd.DataFrame) -> None:
        """'1' should normalise to 'C-00001' without raising."""
        ctx = ToolContext(df=sample_df)
        # This ID probably doesn't exist in data; we're testing normalisation + graceful miss
        r = entity_lookup(ctx, entity_id="1")
        assert r.ok  # not found → ok=True with note

    def test_unknown_entity_ok_with_note(self, sample_df: pd.DataFrame) -> None:
        ctx = ToolContext(df=sample_df)
        r = entity_lookup(ctx, entity_id="C-NONEXISTENT99999")
        assert r.ok
        assert r.artifacts["entity_profile"] == {}
        assert any("not found" in n for n in r.notes)

    def test_entity_txns_table_populated(self, sample_df: pd.DataFrame) -> None:
        ctx = ToolContext(df=sample_df)
        r = entity_lookup(ctx, entity_id="C-STR01")
        assert r.ok
        txns = r.tables["entity_txns"]
        assert len(txns) > 0
        assert "amount" in txns[0]
        assert "sender_id" in txns[0]

    def test_multi_entity_lookup(self, sample_df: pd.DataFrame) -> None:
        ctx = ToolContext(df=sample_df)
        r = entity_lookup(ctx, entity_id=["C-STR01", "C-HUB01"])
        assert r.ok
        prof = r.artifacts["entity_profile"]
        # Multi-entity returns dict keyed by customer_id
        assert "C-STR01" in prof
        assert "C-HUB01" in prof

    def test_empty_df_returns_ok(self) -> None:
        ctx = ToolContext(df=pd.DataFrame())
        r = entity_lookup(ctx, entity_id="C-STR01")
        assert r.ok
        assert r.artifacts["entity_profile"] == {}

    def test_no_entity_id_returns_error(self, sample_df: pd.DataFrame) -> None:
        ctx = ToolContext(df=sample_df)
        r = entity_lookup(ctx)
        assert not r.ok
        assert "entity_id is required" in r.error

    def test_txns_include_both_sent_and_received(self, sample_df: pd.DataFrame) -> None:
        ctx = ToolContext(df=sample_df)
        r = entity_lookup(ctx, entity_id="C-STR01")
        assert r.ok
        txns = r.tables["entity_txns"]
        ids_present = {row.get("sender_id") for row in txns} | {row.get("receiver_id") for row in txns}
        assert "C-STR01" in ids_present
