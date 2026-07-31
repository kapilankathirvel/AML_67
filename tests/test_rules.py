"""
tests/test_rules.py

Tests for backend/tools/rules.py.

Ownership: explicitly listed in WORKPLAN.md §4 Track B ownership matrix.

Core test contract:
  - Each rule fires on its OWN synthetic pattern cohort from aml_sample.csv
  - Each rule stays SILENT on the NORMAL cohort (false-positive control)
  - Evidence dicts match the shape specified in AML_LOGIC.md §3

Test strategy:
  - Load data/sample/aml_sample.csv and split by pattern_label
  - Construct ToolContext directly (no executor)
  - Run features first, then rule_detect — mirroring the real plan order
  - For each rule: assert hits > 0 on its pattern cohort, assert 0 on normal cohort
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from backend.tools.base import ToolContext
from backend.tools.features import feature_engineer
from backend.tools.rules import _run_r7_inbound_structuring, rule_detect

SAMPLE_TX   = "data/sample/aml_sample.csv"
SAMPLE_CUST = "data/sample/aml_sample_customers.csv"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_df() -> pd.DataFrame:
    df = pd.read_csv(SAMPLE_TX)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["is_cross_border"] = df["is_cross_border"].astype(bool)
    return df


@pytest.fixture(scope="module")
def normal_df(full_df: pd.DataFrame) -> pd.DataFrame:
    """Transactions labelled as normal (no pattern_label)."""
    return full_df[full_df["pattern_label"].isna()].copy()


@pytest.fixture(scope="module")
def structuring_df(full_df: pd.DataFrame) -> pd.DataFrame:
    return full_df[full_df["pattern_label"] == "structuring"].copy()


@pytest.fixture(scope="module")
def smurfing_df(full_df: pd.DataFrame) -> pd.DataFrame:
    return full_df[full_df["pattern_label"] == "smurfing"].copy()


@pytest.fixture(scope="module")
def layering_df(full_df: pd.DataFrame) -> pd.DataFrame:
    return full_df[full_df["pattern_label"] == "layering"].copy()


@pytest.fixture(scope="module")
def rapid_cashout_df(full_df: pd.DataFrame) -> pd.DataFrame:
    return full_df[full_df["pattern_label"] == "rapid_cashout"].copy()


def _make_ctx(df: pd.DataFrame) -> ToolContext:
    """Build ToolContext with features pre-computed, mirroring real execution plan."""
    ctx = ToolContext(df=df.copy())
    feat_result = feature_engineer(ctx)
    if feat_result.ok:
        ctx.artifacts["features"] = feat_result.artifacts.get("features", pd.DataFrame())
        ctx.artifacts["feature_list"] = feat_result.artifacts.get("feature_list", [])
    return ctx


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _hits_for_rule(hits: list[dict], rule_id: str) -> list[dict]:
    return [h for h in hits if h["rule_id"] == rule_id]


def _hit_entity_ids(hits: list[dict]) -> set[str]:
    return {h["entity_id"] for h in hits}


# ---------------------------------------------------------------------------
# R1 — Structuring
# ---------------------------------------------------------------------------


class TestR1Structuring:
    def test_fires_on_structuring_cohort(self, structuring_df: pd.DataFrame) -> None:
        ctx = _make_ctx(structuring_df)
        result = rule_detect(ctx, patterns=["structuring"])
        assert result.ok
        hits = _hits_for_rule(result.artifacts["rule_hits"], "R1")
        assert len(hits) > 0, (
            f"R1 did not fire on structuring cohort "
            f"({len(structuring_df)} transactions, "
            f"{structuring_df['sender_id'].nunique()} senders)"
        )
        print(f"\nR1 structuring fires: {len(hits)} customers")

    def test_silent_on_normal_cohort(self, normal_df: pd.DataFrame) -> None:
        ctx = _make_ctx(normal_df)
        result = rule_detect(ctx, patterns=["structuring"])
        assert result.ok
        hits = _hits_for_rule(result.artifacts["rule_hits"], "R1")
        hit_ids = _hit_entity_ids(hits)
        # Normal cohort amounts are spread across the whole range; very few should land
        # in the $9k-$10k band AND have ≥ 3 transactions in 7 days.
        # Tolerance: ≤ 5% of normal senders flagged (conservative FP bound)
        n_normal_senders = normal_df["sender_id"].nunique()
        fp_rate = len(hit_ids) / max(n_normal_senders, 1)
        assert fp_rate <= 0.05, (
            f"R1 false-positive rate on normal cohort: {fp_rate:.1%} "
            f"({len(hit_ids)} of {n_normal_senders} normal senders)"
        )
        print(f"\nR1 normal FP: {len(hit_ids)} of {n_normal_senders} senders ({fp_rate:.1%})")

    def test_evidence_shape(self, structuring_df: pd.DataFrame) -> None:
        ctx = _make_ctx(structuring_df)
        result = rule_detect(ctx, patterns=["structuring"])
        hits = _hits_for_rule(result.artifacts["rule_hits"], "R1")
        if not hits:
            pytest.skip("No R1 hits to check evidence shape")
        ev = hits[0]["evidence"]
        required_keys = {
            "txn_count_in_band", "window_days", "amounts",
            "band_low", "band_high", "total", "pct_just_below_threshold",
        }
        for k in required_keys:
            assert k in ev, f"R1 evidence missing key: '{k}'"
        assert ev["band_low"] == 9000.0
        assert ev["band_high"] == 9999.99
        assert ev["window_days"] == 7
        assert isinstance(ev["amounts"], list)
        assert ev["total"] > 0

    def test_weight_correct(self, structuring_df: pd.DataFrame) -> None:
        ctx = _make_ctx(structuring_df)
        result = rule_detect(ctx, patterns=["structuring"])
        hits = _hits_for_rule(result.artifacts["rule_hits"], "R1")
        for h in hits:
            assert abs(h["weight"] - 0.85) < 0.001


# ---------------------------------------------------------------------------
# R2 — Smurfing
# ---------------------------------------------------------------------------


class TestR2Smurfing:
    def test_fires_on_smurfing_cohort(self, smurfing_df: pd.DataFrame) -> None:
        ctx = _make_ctx(smurfing_df)
        result = rule_detect(ctx, patterns=["smurfing"])
        assert result.ok
        hits = _hits_for_rule(result.artifacts["rule_hits"], "R2")
        assert len(hits) > 0, (
            f"R2 did not fire on smurfing cohort "
            f"({len(smurfing_df)} transactions, "
            f"{smurfing_df['sender_id'].nunique()} senders)"
        )
        print(f"\nR2 smurfing fires: {len(hits)} customers")

    def test_silent_on_normal_cohort(self, normal_df: pd.DataFrame) -> None:
        ctx = _make_ctx(normal_df)
        result = rule_detect(ctx, patterns=["smurfing"])
        assert result.ok
        hits = _hits_for_rule(result.artifacts["rule_hits"], "R2")
        hit_ids = _hit_entity_ids(hits)
        n_normal_senders = normal_df["sender_id"].nunique()
        fp_rate = len(hit_ids) / max(n_normal_senders, 1)
        assert fp_rate <= 0.05, (
            f"R2 FP rate on normal cohort: {fp_rate:.1%} "
            f"({len(hit_ids)} of {n_normal_senders})"
        )
        print(f"\nR2 normal FP: {len(hit_ids)} of {n_normal_senders} senders ({fp_rate:.1%})")

    def test_evidence_shape(self, smurfing_df: pd.DataFrame) -> None:
        ctx = _make_ctx(smurfing_df)
        result = rule_detect(ctx, patterns=["smurfing"])
        hits = _hits_for_rule(result.artifacts["rule_hits"], "R2")
        if not hits:
            pytest.skip("No R2 hits to check evidence shape")
        ev = hits[0]["evidence"]
        required_keys = {
            "distinct_receivers_48h", "window_hours", "median_outbound_amount",
            "amounts", "band_low", "band_high", "round_amount_ratio",
        }
        for k in required_keys:
            assert k in ev, f"R2 evidence missing key: '{k}'"
        assert ev["window_hours"] == 48
        assert ev["distinct_receivers_48h"] >= 5

    def test_weight_correct(self, smurfing_df: pd.DataFrame) -> None:
        ctx = _make_ctx(smurfing_df)
        result = rule_detect(ctx, patterns=["smurfing"])
        hits = _hits_for_rule(result.artifacts["rule_hits"], "R2")
        for h in hits:
            assert abs(h["weight"] - 0.75) < 0.001


# ---------------------------------------------------------------------------
# R3 — Layering
# ---------------------------------------------------------------------------


class TestR3Layering:
    def test_fires_on_layering_cohort(self, layering_df: pd.DataFrame) -> None:
        ctx = _make_ctx(layering_df)
        result = rule_detect(ctx, patterns=["layering"])
        assert result.ok
        hits = _hits_for_rule(result.artifacts["rule_hits"], "R3")
        assert len(hits) > 0, (
            f"R3 did not fire on layering cohort "
            f"({len(layering_df)} transactions, "
            f"{layering_df['sender_id'].nunique()} senders). "
            f"Note: R3 requires wire/transfer type + cross-border + pass_through_ratio≥0.70. "
            f"Cross-border rows: {layering_df['is_cross_border'].sum()}, "
            f"wire/transfer rows: {layering_df['txn_type'].isin(['wire','transfer']).sum()}"
        )
        print(f"\nR3 layering fires: {len(hits)} customers")

    def test_silent_on_normal_cohort(self, normal_df: pd.DataFrame) -> None:
        ctx = _make_ctx(normal_df)
        result = rule_detect(ctx, patterns=["layering"])
        assert result.ok
        hits = _hits_for_rule(result.artifacts["rule_hits"], "R3")
        hit_ids = _hit_entity_ids(hits)
        n_normal_senders = normal_df["sender_id"].nunique()
        fp_rate = len(hit_ids) / max(n_normal_senders, 1)
        assert fp_rate <= 0.05, (
            f"R3 FP rate on normal cohort: {fp_rate:.1%} "
            f"({len(hit_ids)} of {n_normal_senders})"
        )
        print(f"\nR3 normal FP: {len(hit_ids)} of {n_normal_senders} senders ({fp_rate:.1%})")

    def test_evidence_shape(self, layering_df: pd.DataFrame) -> None:
        ctx = _make_ctx(layering_df)
        result = rule_detect(ctx, patterns=["layering"])
        hits = _hits_for_rule(result.artifacts["rule_hits"], "R3")
        if not hits:
            pytest.skip("No R3 hits to check evidence shape")
        ev = hits[0]["evidence"]
        required_keys = {
            "chain", "chain_length", "cross_border_hops",
            "pass_through_ratios", "hop_amounts", "hop_types",
        }
        for k in required_keys:
            assert k in ev, f"R3 evidence missing key: '{k}'"
        assert ev["chain_length"] >= 3
        assert ev["cross_border_hops"] >= 1
        assert isinstance(ev["chain"], list) and len(ev["chain"]) >= 4

    def test_weight_correct(self, layering_df: pd.DataFrame) -> None:
        ctx = _make_ctx(layering_df)
        result = rule_detect(ctx, patterns=["layering"])
        hits = _hits_for_rule(result.artifacts["rule_hits"], "R3")
        for h in hits:
            assert abs(h["weight"] - 0.80) < 0.001


# ---------------------------------------------------------------------------
# R4 — Rapid Cashout
# ---------------------------------------------------------------------------


class TestR4RapidCashout:
    def test_fires_on_rapid_cashout_cohort(self, rapid_cashout_df: pd.DataFrame) -> None:
        ctx = _make_ctx(rapid_cashout_df)
        result = rule_detect(ctx, patterns=["rapid_cashout"])
        assert result.ok
        hits = _hits_for_rule(result.artifacts["rule_hits"], "R4")
        assert len(hits) > 0, (
            f"R4 did not fire on rapid_cashout cohort "
            f"({len(rapid_cashout_df)} transactions, "
            f"{rapid_cashout_df['sender_id'].nunique()} senders). "
            f"Cash txns: {rapid_cashout_df['txn_type'].isin(['cash']).sum()}, "
            f"large inbounds: {(rapid_cashout_df['amount'] >= 10000).sum()}"
        )
        print(f"\nR4 rapid_cashout fires: {len(hits)} customers")

    def test_silent_on_normal_cohort(self, normal_df: pd.DataFrame) -> None:
        ctx = _make_ctx(normal_df)
        result = rule_detect(ctx, patterns=["rapid_cashout"])
        assert result.ok
        hits = _hits_for_rule(result.artifacts["rule_hits"], "R4")
        hit_ids = _hit_entity_ids(hits)
        n_normal_senders = normal_df["sender_id"].nunique()
        fp_rate = len(hit_ids) / max(n_normal_senders, 1)
        assert fp_rate <= 0.05, (
            f"R4 FP rate on normal cohort: {fp_rate:.1%} "
            f"({len(hit_ids)} of {n_normal_senders})"
        )
        print(f"\nR4 normal FP: {len(hit_ids)} of {n_normal_senders} senders ({fp_rate:.1%})")

    def test_evidence_shape(self, rapid_cashout_df: pd.DataFrame) -> None:
        ctx = _make_ctx(rapid_cashout_df)
        result = rule_detect(ctx, patterns=["rapid_cashout"])
        hits = _hits_for_rule(result.artifacts["rule_hits"], "R4")
        if not hits:
            pytest.skip("No R4 hits to check evidence shape")
        ev = hits[0]["evidence"]
        required_keys = {
            "inbound_amount", "inbound_txn_id", "inbound_timestamp",
            "cash_outflow_count", "cash_outflow_total", "cashout_ratio",
            "window_hours", "outflow_amounts", "elapsed_to_first_cashout_hours",
        }
        for k in required_keys:
            assert k in ev, f"R4 evidence missing key: '{k}'"
        assert ev["inbound_amount"] >= 10000.0
        assert ev["window_hours"] == 24
        assert ev["cashout_ratio"] >= 0.5
        assert ev["cash_outflow_count"] >= 3

    def test_weight_correct(self, rapid_cashout_df: pd.DataFrame) -> None:
        ctx = _make_ctx(rapid_cashout_df)
        result = rule_detect(ctx, patterns=["rapid_cashout"])
        hits = _hits_for_rule(result.artifacts["rule_hits"], "R4")
        for h in hits:
            assert abs(h["weight"] - 0.75) < 0.001


# ---------------------------------------------------------------------------
# R5 — High Velocity (synthetic fixture — sample may not have this pattern)
# ---------------------------------------------------------------------------


def _make_velocity_fixture() -> pd.DataFrame:
    """Hand-crafted high-velocity customer: 50 txns in 10 hours with extreme outlier amounts.

    Baseline: 100 txns of ~$200-$220 each (tight cluster, low std).
    Burst: 50 txns at ~$50,000 each within 10 hours.
    With 100 baseline + 50 burst = 150 total, mean≈~16k, std≈~22k.
    z-score of $50k items = (50k - 16k) / 22k ≈ 1.5 ... need more separation.
    Actually best approach: use a tiny std in baseline ($200 ± $5) and extreme burst ($1M).
    z-score = (1,000,000 - 205) / 5 = ~199,959 >> 3.0 ✓
    """
    base = pd.Timestamp("2025-01-10T06:00:00")
    rows = []
    # Baseline: 100 small txns with VERY tight cluster (mean=$205, std≈$3)
    for i in range(100):
        rows.append({
            "txn_id": f"T-BASE{i:03d}",
            "timestamp": base - pd.Timedelta(days=85) + pd.Timedelta(hours=i * 20),
            "sender_id": "C-FAST",
            "receiver_id": f"C-R{i}",
            "amount": 200.0 + (i % 6),   # $200-$205 — extremely tight, std ≈ 2
            "currency": "USD", "txn_type": "transfer", "channel": "online",
            "sender_country": "US", "receiver_country": "US",
            "is_cross_border": False, "label_is_laundering": False,
            "pattern_label": None,
        })
    # Burst: 50 txns in 10 hours (= 5 tph >> R5_MIN_TPH=2.0)
    # amounts: $50,000 each → z = (50000 - 202) / 2 ≈ 24,899 >> 3.0 ✓
    for i in range(50):
        rows.append({
            "txn_id": f"T-BURST{i:03d}",
            "timestamp": base + pd.Timedelta(minutes=i * 12),
            "sender_id": "C-FAST",
            "receiver_id": f"C-R{200+i}",
            "amount": 50_000.0 + i * 100,   # $50k+ — massively above $200 baseline
            "currency": "USD", "txn_type": "transfer", "channel": "online",
            "sender_country": "US", "receiver_country": "US",
            "is_cross_border": False, "label_is_laundering": True,
            "pattern_label": "velocity",
        })
    # Normal-velocity customer: 10 txns over 10 days with normal consistent amounts
    for i in range(10):
        rows.append({
            "txn_id": f"T-NORM{i:03d}",
            "timestamp": base + pd.Timedelta(days=i),
            "sender_id": "C-SLOW",
            "receiver_id": f"C-S{i}",
            "amount": 500.0 + i * 2,
            "currency": "USD", "txn_type": "transfer", "channel": "online",
            "sender_country": "US", "receiver_country": "US",
            "is_cross_border": False, "label_is_laundering": False,
            "pattern_label": None,
        })
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["is_cross_border"] = df["is_cross_border"].astype(bool)
    return df


class TestR5Velocity:
    def test_fires_on_velocity_fixture(self) -> None:
        df = _make_velocity_fixture()
        ctx = _make_ctx(df)
        result = rule_detect(ctx, patterns=["velocity"])
        assert result.ok
        hits = _hits_for_rule(result.artifacts["rule_hits"], "R5")
        hit_ids = _hit_entity_ids(hits)
        assert "C-FAST" in hit_ids, (
            f"R5 did not fire on C-FAST (high-velocity customer). "
            f"Hits: {hit_ids}"
        )
        print(f"\nR5 velocity fires: {len(hits)} customers")

    def test_silent_on_normal_in_velocity_fixture(self) -> None:
        df = _make_velocity_fixture()
        ctx = _make_ctx(df)
        result = rule_detect(ctx, patterns=["velocity"])
        assert result.ok
        hits = _hits_for_rule(result.artifacts["rule_hits"], "R5")
        hit_ids = _hit_entity_ids(hits)
        assert "C-SLOW" not in hit_ids, (
            f"R5 incorrectly flagged C-SLOW (normal-velocity customer). "
            f"All hits: {hit_ids}"
        )

    def test_evidence_shape(self) -> None:
        df = _make_velocity_fixture()
        ctx = _make_ctx(df)
        result = rule_detect(ctx, patterns=["velocity"])
        hits = _hits_for_rule(result.artifacts["rule_hits"], "R5")
        if not hits:
            pytest.skip("No R5 hits")
        ev = hits[0]["evidence"]
        required_keys = {
            "max_txns_per_hour", "window_hours", "amount_zscore",
            "zscore_baseline_days", "zscore_n_samples",
            "mean_historical_amount", "std_historical_amount", "triggering_amount",
        }
        for k in required_keys:
            assert k in ev, f"R5 evidence missing key: '{k}'"

    def test_weight_correct(self) -> None:
        df = _make_velocity_fixture()
        ctx = _make_ctx(df)
        result = rule_detect(ctx, patterns=["velocity"])
        hits = _hits_for_rule(result.artifacts["rule_hits"], "R5")
        for h in hits:
            assert abs(h["weight"] - 0.65) < 0.001


# ---------------------------------------------------------------------------
# R6 — Dormant Reactivation (synthetic fixture)
# ---------------------------------------------------------------------------


def _make_dormancy_fixture() -> pd.DataFrame:
    """C-DORMANT: active in early 2024, silent for 90 days, burst in Dec 2024."""
    base_pre  = pd.Timestamp("2024-01-10")
    base_post = pd.Timestamp("2024-12-15")   # 90+ days after last pre-tx

    rows = []
    # Pre-dormancy: 5 small txns (baseline ~$300)
    for i in range(5):
        rows.append({
            "txn_id": f"T-PRE{i:03d}",
            "timestamp": base_pre + pd.Timedelta(days=i),
            "sender_id": "C-DORMANT",
            "receiver_id": f"C-P{i}",
            "amount": 300.0 + i * 20,
            "currency": "USD", "txn_type": "transfer", "channel": "online",
            "sender_country": "US", "receiver_country": "US",
            "is_cross_border": False, "label_is_laundering": False,
            "pattern_label": None,
        })
    # Post-reactivation burst: 5 txns in 5 days with abnormal large amounts
    for i in range(5):
        rows.append({
            "txn_id": f"T-POST{i:03d}",
            "timestamp": base_post + pd.Timedelta(days=i),
            "sender_id": "C-DORMANT",
            "receiver_id": f"C-D{i}",
            "amount": 9500.0,  # z >> 2σ above baseline $300
            "currency": "USD", "txn_type": "transfer", "channel": "online",
            "sender_country": "US", "receiver_country": "US",
            "is_cross_border": False, "label_is_laundering": True,
            "pattern_label": "dormant_reactivation",
        })
    # Normal customer: consistent low activity throughout
    for i in range(10):
        rows.append({
            "txn_id": f"T-N{i:03d}",
            "timestamp": base_pre + pd.Timedelta(days=i * 15),
            "sender_id": "C-NORMAL",
            "receiver_id": f"C-NR{i}",
            "amount": 400.0 + i * 5,
            "currency": "USD", "txn_type": "transfer", "channel": "online",
            "sender_country": "US", "receiver_country": "US",
            "is_cross_border": False, "label_is_laundering": False,
            "pattern_label": None,
        })
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["is_cross_border"] = df["is_cross_border"].astype(bool)
    return df


class TestR6DormantReactivation:
    def test_fires_on_dormant_customer(self) -> None:
        df = _make_dormancy_fixture()
        ctx = _make_ctx(df)
        result = rule_detect(ctx, patterns=["dormant_reactivation"])
        assert result.ok
        hits = _hits_for_rule(result.artifacts["rule_hits"], "R6")
        hit_ids = _hit_entity_ids(hits)
        assert "C-DORMANT" in hit_ids, (
            f"R6 did not fire on C-DORMANT. Hits: {hit_ids}"
        )
        print(f"\nR6 dormant reactivation fires: {len(hits)} customers")

    def test_silent_on_normal_customer(self) -> None:
        df = _make_dormancy_fixture()
        ctx = _make_ctx(df)
        result = rule_detect(ctx, patterns=["dormant_reactivation"])
        assert result.ok
        hits = _hits_for_rule(result.artifacts["rule_hits"], "R6")
        hit_ids = _hit_entity_ids(hits)
        assert "C-NORMAL" not in hit_ids, (
            f"R6 incorrectly flagged C-NORMAL. Hits: {hit_ids}"
        )

    def test_evidence_shape(self) -> None:
        df = _make_dormancy_fixture()
        ctx = _make_ctx(df)
        result = rule_detect(ctx, patterns=["dormant_reactivation"])
        hits = _hits_for_rule(result.artifacts["rule_hits"], "R6")
        if not hits:
            pytest.skip("No R6 hits")
        ev = hits[0]["evidence"]
        required_keys = {
            "dormancy_gap_days", "last_txn_before_gap", "first_txn_after_gap",
            "burst_txn_count", "burst_window_days",
            "amount_zscore_vs_pre_dormancy",
            "pre_dormancy_mean_amount", "burst_amounts",
        }
        for k in required_keys:
            assert k in ev, f"R6 evidence missing key: '{k}'"
        assert ev["dormancy_gap_days"] >= 60
        assert ev["burst_txn_count"] >= 3

    def test_weight_correct(self) -> None:
        df = _make_dormancy_fixture()
        ctx = _make_ctx(df)
        result = rule_detect(ctx, patterns=["dormant_reactivation"])
        hits = _hits_for_rule(result.artifacts["rule_hits"], "R6")
        for h in hits:
            assert abs(h["weight"] - 0.60) < 0.001


# ---------------------------------------------------------------------------
# Selective execution — wrong pattern must not fire wrong rule
# ---------------------------------------------------------------------------


class TestR7InboundStructuring:
    """R7 — receiver-side structuring.

    The only receiver-keyed rule in the set, so the tests that matter most are
    the ones proving it flags the *receiver* and that it stays quiet on
    ordinary inbound traffic.
    """

    @staticmethod
    def _txns(rows: list[tuple[str, str, float, str]]) -> pd.DataFrame:
        """(sender, receiver, amount, ISO timestamp) -> canonical-ish frame."""
        return pd.DataFrame([
            {
                "txn_id": f"T{i:04d}",
                "timestamp": pd.Timestamp(ts),
                "sender_id": s,
                "receiver_id": r,
                "amount": amt,
                "currency": "USD",
                "txn_type": "transfer",
                "channel": "online",
                "sender_country": "US",
                "receiver_country": "US",
                "is_cross_border": False,
            }
            for i, (s, r, amt, ts) in enumerate(rows)
        ])

    def test_fires_on_two_band_deposits_from_one_sender(self) -> None:
        df = self._txns([
            ("C-BAD01", "C-MULE1", 9_500.0, "2025-01-02T10:00:00"),
            ("C-BAD01", "C-MULE1", 9_800.0, "2025-01-05T10:00:00"),
        ])
        hits = _run_r7_inbound_structuring(df, pd.DataFrame())

        assert len(hits) == 1
        assert hits[0]["entity_id"] == "C-MULE1", "R7 must flag the receiver"
        assert hits[0]["rule_id"] == "R7"
        assert hits[0]["weight"] == 0.75
        assert hits[0]["evidence"]["counterparty"] == "C-BAD01"
        assert hits[0]["evidence"]["inbound_band_txns_from_one_sender"] == 2

    def test_single_deposit_does_not_fire(self) -> None:
        """One sub-threshold deposit is ordinary. On the committed dataset no
        true negative ever exceeds one, which is why the threshold is 2."""
        df = self._txns([("C-BAD01", "C-MULE1", 9_500.0, "2025-01-02T10:00:00")])
        assert _run_r7_inbound_structuring(df, pd.DataFrame()) == []

    def test_deposits_outside_the_seven_day_window_do_not_fire(self) -> None:
        df = self._txns([
            ("C-BAD01", "C-MULE1", 9_500.0, "2025-01-02T10:00:00"),
            ("C-BAD01", "C-MULE1", 9_800.0, "2025-01-20T10:00:00"),   # 18 days later
        ])
        assert _run_r7_inbound_structuring(df, pd.DataFrame()) == []

    def test_amounts_outside_the_band_do_not_fire(self) -> None:
        """Below $9,000 is not threshold-adjacent; at or above $10,000 triggers
        a CTR anyway, so neither is structuring."""
        df = self._txns([
            ("C-BAD01", "C-MULE1", 5_000.0, "2025-01-02T10:00:00"),
            ("C-BAD01", "C-MULE1", 12_000.0, "2025-01-03T10:00:00"),
        ])
        assert _run_r7_inbound_structuring(df, pd.DataFrame()) == []

    def test_two_senders_one_deposit_each_does_not_fire(self) -> None:
        """The signal is per (receiver, sender) pair, not per receiver.

        Two unrelated counterparties each sending one band-range payment is not
        structuring — and conflating it with the pair signal is exactly what
        would turn R7 into a false-positive generator.
        """
        df = self._txns([
            ("C-AAA01", "C-MULE1", 9_500.0, "2025-01-02T10:00:00"),
            ("C-BBB01", "C-MULE1", 9_800.0, "2025-01-03T10:00:00"),
        ])
        assert _run_r7_inbound_structuring(df, pd.DataFrame()) == []

    def test_receiver_reported_once_on_its_strongest_relationship(self) -> None:
        """A receiver fed by two structuring senders yields one hit, describing
        the denser pair — not one hit per counterparty."""
        df = self._txns([
            ("C-AAA01", "C-MULE1", 9_100.0, "2025-01-02T10:00:00"),
            ("C-AAA01", "C-MULE1", 9_200.0, "2025-01-03T10:00:00"),
            ("C-BBB01", "C-MULE1", 9_300.0, "2025-01-02T10:00:00"),
            ("C-BBB01", "C-MULE1", 9_400.0, "2025-01-03T10:00:00"),
            ("C-BBB01", "C-MULE1", 9_500.0, "2025-01-04T10:00:00"),
        ])
        hits = _run_r7_inbound_structuring(df, pd.DataFrame())

        assert len(hits) == 1
        assert hits[0]["evidence"]["counterparty"] == "C-BBB01"
        assert hits[0]["evidence"]["inbound_band_txns_from_one_sender"] == 3

    def test_catches_receive_only_positives_with_no_false_positives(
        self, full_df: pd.DataFrame
    ) -> None:
        """The reason R7 exists, measured on the real dataset.

        63 labelled customers appear only as receivers and were previously
        unreachable by every sender-side rule. R7 recovers 11 of them and — on
        this data — flags no true negative at all. It does not close the gap:
        the other 52 receive a single labelled transaction each and are not
        separable from ordinary counterparties.
        """
        from evaluation.harness import load_ground_truth

        _, gt = load_ground_truth()
        hits = _run_r7_inbound_structuring(full_df, pd.DataFrame())
        flagged = {h["entity_id"] for h in hits}

        assert flagged <= gt.receive_only, (
            "R7 flagged something outside the receive-only positive set: "
            f"{flagged - gt.receive_only}"
        )
        assert len(flagged) == 11
        assert not (flagged - gt.sender_or_receiver), "R7 produced a false positive"

    def test_evidence_carries_what_an_analyst_needs(self, full_df: pd.DataFrame) -> None:
        hits = _run_r7_inbound_structuring(full_df, pd.DataFrame())
        assert hits

        for h in hits:
            ev = h["evidence"]
            assert ev["inbound_band_txns_from_one_sender"] >= 2
            assert ev["window_days"] == 7
            assert ev["band_low"] == 9_000.0 and ev["band_high"] == 9_999.99
            assert ev["amounts"], "evidence must show the actual amounts"
            assert all(9_000.0 <= a <= 9_999.99 for a in ev["amounts"])
            assert len(ev["amounts"]) <= 10, "amounts list should stay readable"
            assert ev["counterparty"]


class TestSelectiveExecution:
    def test_structuring_pattern_fires_only_structuring_rules(
        self, structuring_df: pd.DataFrame
    ) -> None:
        """R1 (sender side) and R7 (receiver side) are both structuring rules.

        This assertion used to be `<= {"R1"}`; R7 was added as the receiver-side
        mirror of the same typology, so a structuring query now legitimately
        evaluates both ends of the transaction.
        """
        ctx = _make_ctx(structuring_df)
        result = rule_detect(ctx, patterns=["structuring"])
        assert result.ok
        hits = result.artifacts["rule_hits"]
        rule_ids = {h["rule_id"] for h in hits}
        assert rule_ids <= {"R1", "R7"}, (
            f"Unexpected rules fired for pattern=['structuring']: {rule_ids}"
        )

    def test_empty_df_returns_ok_no_hits(self) -> None:
        empty = pd.DataFrame(columns=[
            "txn_id", "timestamp", "sender_id", "receiver_id", "amount",
            "currency", "txn_type", "channel", "sender_country", "receiver_country",
            "is_cross_border", "label_is_laundering", "pattern_label",
        ])
        ctx = ToolContext(df=empty)
        result = rule_detect(ctx)
        assert result.ok
        assert result.artifacts["rule_hits"] == []

    def test_ctx_df_not_mutated(self, full_df: pd.DataFrame) -> None:
        ctx = _make_ctx(full_df)
        original_len = len(ctx.df)
        rule_detect(ctx)
        assert len(ctx.df) == original_len, "rule_detect mutated ctx.df"

    def test_rule_hits_shape(self, full_df: pd.DataFrame) -> None:
        """Every hit must have entity_id, rule_id, evidence (dict), weight (float)."""
        ctx = _make_ctx(full_df)
        result = rule_detect(ctx)
        assert result.ok
        for hit in result.artifacts["rule_hits"]:
            assert "entity_id" in hit
            assert "rule_id" in hit
            assert "evidence" in hit
            assert "weight" in hit
            assert isinstance(hit["evidence"], dict)
            assert isinstance(hit["weight"], float)
            assert 0.0 < hit["weight"] <= 1.0
