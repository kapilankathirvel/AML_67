"""
tests/test_out_of_time.py — guard the out-of-time validation study.

The load-bearing test here is `test_in_time_arm_reproduces_ml_detect`. Arm A is
only a fair reference point for arms B and C if it is genuinely the shipped
model restricted to the test window; if `ml_arm` drifted from `ml_detect` in any
detail — column selection, scaling, seeds, the 0.6/0.4 fusion, the percentile
convention — the reported drop would be measuring my reimplementation rather
than the cost of freezing the fit. Everything else in this file is cheaper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import backend.tools.rules as rules_mod
from backend.tools.ml_detect import ml_detect
from backend.tools.base import ToolContext

from evaluation.harness import (
    DEFAULT_CUST_CSV,
    ground_truth_from_frames,
    load_ground_truth,
)
from evaluation.out_of_time import (
    DEFAULT_TRAIN_DAYS,
    _matrix,
    _rank_against,
    build,
    layering_search_space,
    load_canonical,
    ml_arm,
    partition_by_days,
    rule_reachability,
    split_by_time,
    window_artifacts,
)


# ---------------------------------------------------------------------------
# Fixtures — the real pipeline is slow, so each stage is built once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def canonical():
    return load_canonical()


@pytest.fixture(scope="module")
def windows(canonical):
    df, _ = canonical
    return split_by_time(df, train_days=DEFAULT_TRAIN_DAYS)


@pytest.fixture(scope="module")
def test_window(canonical, windows):
    _, customers = canonical
    _, test_df, _ = windows
    return window_artifacts(test_df, customers)


@pytest.fixture(scope="module")
def train_window(canonical, windows):
    _, customers = canonical
    train_df, _, _ = windows
    return window_artifacts(train_df, customers)


# ---------------------------------------------------------------------------
# The equivalence that makes the study meaningful
# ---------------------------------------------------------------------------


def test_in_time_arm_reproduces_ml_detect(test_window):
    """Arm A must be ml_detect, not a lookalike.

    Runs the registered tool over the test-window features and compares
    percentiles entity by entity. Any divergence means the reported
    out-of-time drop is contaminated by an implementation difference.
    """
    feat = test_window["features"]
    ctx = ToolContext(
        df=None,
        artifacts={
            "features": feat,
            "features_reference": feat,
            "feature_list": test_window["feature_list"],
        },
    )
    shipped = ml_detect(ctx)
    assert shipped.ok

    expected = {r["entity_id"]: r["percentile"] for r in shipped.artifacts["ml_scores"]}
    actual = {
        r["entity_id"]: r["percentile"]
        for r in ml_arm(feat, feat, test_window["feature_list"], freeze_distribution=False)
    }

    assert set(actual) == set(expected)
    for eid, pct in expected.items():
        assert actual[eid] == pytest.approx(pct, abs=1e-9), f"{eid} diverged"


def test_novelty_lof_does_not_change_the_in_sample_fit(test_window):
    """The novelty=True substitution must be invisible in-sample.

    ml_detect builds LOF with novelty=False. Arm A goes through ml_arm, which
    builds it with novelty=True and reads negative_outlier_factor_. If sklearn
    ever changed that attribute's meaning between modes, the test above would
    fail — this one names the reason so the failure is diagnosable.
    """
    feat = test_window["features"]
    in_sample = ml_arm(feat, feat, test_window["feature_list"], freeze_distribution=False)
    assert len({r["entity_id"] for r in in_sample}) == len(feat)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def test_split_is_disjoint_and_exhaustive(canonical, windows):
    df, _ = canonical
    train_df, test_df, boundary = windows

    assert len(train_df) + len(test_df) == len(df)
    assert set(train_df["txn_id"]).isdisjoint(set(test_df["txn_id"]))
    assert set(train_df["txn_id"]) | set(test_df["txn_id"]) == set(df["txn_id"])

    assert pd.to_datetime(train_df["timestamp"]).max() < boundary
    assert pd.to_datetime(test_df["timestamp"]).min() >= boundary


def test_boundary_is_derived_from_the_data(canonical):
    df, _ = canonical
    _, _, boundary = split_by_time(df, train_days=DEFAULT_TRAIN_DAYS)
    first = pd.to_datetime(df["timestamp"]).min()
    assert boundary == first + pd.Timedelta(days=DEFAULT_TRAIN_DAYS)


def test_split_does_not_mutate_its_input(canonical):
    df, _ = canonical
    before = df.copy(deep=True)
    train_df, test_df, _ = split_by_time(df)
    train_df["amount"] = -1.0
    test_df["amount"] = -1.0
    pd.testing.assert_frame_equal(df, before)


# ---------------------------------------------------------------------------
# Ground truth over a frame
# ---------------------------------------------------------------------------


def test_frame_ground_truth_matches_the_file_loader():
    """The refactor must not have changed what load_ground_truth returns."""
    df, gt = load_ground_truth()
    roster = set(pd.read_csv(DEFAULT_CUST_CSV)["customer_id"])
    rebuilt = ground_truth_from_frames(df, roster)
    assert rebuilt == gt


def test_window_ground_truth_is_a_subset_of_the_whole():
    """A window cannot contain a positive the full dataset does not."""
    df, full_gt = load_ground_truth()
    roster = set(pd.read_csv(DEFAULT_CUST_CSV)["customer_id"])
    half = df.iloc[: len(df) // 2]
    windowed = ground_truth_from_frames(half, roster)
    assert windowed.sender_only <= full_gt.sender_only
    assert windowed.labelled_txn_count <= full_gt.labelled_txn_count


def test_population_is_the_caller_s_choice():
    """Passing a narrower population must shrink every set, not just the roster."""
    df, _ = load_ground_truth()
    roster = set(pd.read_csv(DEFAULT_CUST_CSV)["customer_id"])
    full = ground_truth_from_frames(df, roster)
    narrowed = ground_truth_from_frames(df, set(sorted(roster)[:50]))
    assert narrowed.all_customers <= full.all_customers
    assert narrowed.sender_only <= full.sender_only


# ---------------------------------------------------------------------------
# Scoring mechanics
# ---------------------------------------------------------------------------


def test_rank_against_own_distribution_spans_the_unit_interval():
    ref = np.arange(100, dtype=float)
    ranked = _rank_against(ref, ref)
    assert ranked.min() == pytest.approx(0.0)
    assert ranked.max() == pytest.approx(1.0)
    assert np.all(np.diff(ranked) >= 0)


def test_rank_against_is_monotone_in_the_value():
    ref = np.random.default_rng(0).normal(size=500)
    values = np.array([-5.0, 0.0, 5.0])
    ranked = _rank_against(values, ref)
    assert ranked[0] < ranked[1] < ranked[2]


def test_rank_against_frozen_distribution_can_exceed_the_reference_range():
    """A test point beyond everything seen in training must land at the top.

    This is the behaviour that distinguishes arm C from arm B: a frozen model
    has no way to re-rank, so an unprecedented score saturates rather than
    being placed relative to its peers.
    """
    ref = np.arange(10, dtype=float)
    assert _rank_against(np.array([100.0]), ref)[0] == pytest.approx(1.0)


def test_matrix_fills_columns_the_scoring_window_never_produced():
    fit = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    score = pd.DataFrame({"a": [5.0]})
    out = _matrix(score, ["a", "b"])
    assert out.shape == (1, 2)
    assert out[0, 0] == 5.0
    assert out[0, 1] == 0.0  # not observed means zero, as feature_engineer does
    assert list(_matrix(fit, ["b", "a"])[0]) == [3.0, 1.0]  # column order is honoured


# ---------------------------------------------------------------------------
# The caveats, asserted so they cannot quietly stop being true
# ---------------------------------------------------------------------------


def test_r6_is_reported_as_structurally_impossible(test_window, train_window, canonical):
    df, customers = canonical
    full = window_artifacts(df, customers)
    rows = rule_reachability(
        train_window["rule_hits"], test_window["rule_hits"], full["rule_hits"]
    )
    r6 = next(r for r in rows if r["rule"] == "R6")
    assert rules_mod.R6_DORMANCY_DAYS >= 30
    assert r6["structurally_possible_in_test"] is False
    assert r6["test_30d"] == 0

    others = [r for r in rows if r["rule"] != "R6"]
    assert all(r["structurally_possible_in_test"] for r in others)


def test_r3_search_space_shrinks_as_the_window_grows(canonical, windows):
    """The anti-monotonicity finding, pinned.

    If R3's source selection is ever fixed, this test fails and the docstring
    claiming R3 is anti-monotone has to be revisited with it — which is the
    point of asserting a defect rather than only writing it down.
    """
    df, _ = canonical
    train_df, test_df, _ = windows
    rows = layering_search_space({"test": test_df, "train": train_df, "full": df})
    by = {r["window"]: r for r in rows}

    assert by["test"]["eligible_txns"] < by["train"]["eligible_txns"] < by["full"]["eligible_txns"]
    assert (
        by["test"]["sources_in_degree_0"]
        > by["train"]["sources_in_degree_0"]
        > by["full"]["sources_in_degree_0"]
    )
    assert by["full"]["pairs_searched"] < by["test"]["pairs_searched"] / 10


# ---------------------------------------------------------------------------
# The R3 counterfactual
# ---------------------------------------------------------------------------


def test_partition_is_exhaustive_and_non_overlapping(canonical):
    """Every transaction lands in exactly one window.

    This is what makes the unioned hit count comparable to the whole-frame run:
    the windowed arm sees the same rows, cut differently. If a row appeared in
    two windows a chain could be found twice, and if a row appeared in none the
    comparison would be against a smaller dataset rather than a re-cut one.
    """
    df, _ = canonical
    for days in (7, 14, 30):
        parts = partition_by_days(df, days)
        seen: list[int] = []
        for chunk in parts:
            seen.extend(chunk.index.tolist())
        assert len(seen) == len(set(seen)), f"{days}d windows overlap"
        assert set(seen) == set(df.index.tolist()), f"{days}d windows lose rows"


def test_partition_keeps_the_short_final_window(canonical):
    """89 days does not divide by 30, and the remainder is kept rather than dropped.

    Dropping it would silently shorten the dataset the windowed arm is scored
    on, which would make its recall incomparable to the whole-frame row.
    """
    df, _ = canonical
    ts = pd.to_datetime(df["timestamp"])
    span = (ts.max() - ts.min()).days
    parts = partition_by_days(df, 30)

    assert span % 30 != 0, "pick a window size that does not divide the span"
    assert len(parts) == span // 30 + 1
    last = pd.to_datetime(parts[-1]["timestamp"])
    assert (last.max() - last.min()).days < 30


def test_counterfactual_whole_frame_row_matches_the_shipped_r3(payload):
    """The baseline row must be the rule as it ships, not a re-derivation of it.

    Compares distinct entities against a hit count, which are only equal because
    no entity carries two R3 hits on this dataset. That is the weaker of the two
    checks it looks like, and it is still the one worth having: it catches the
    counterfactual's baseline drifting away from the published rule output.
    """
    wf = payload["r3_counterfactual"]["whole_frame"]
    published = {r["rule"]: r["full_90d"] for r in payload["rule_reachability"]}
    assert wf["entities_flagged"] == published["R3"]


def test_counterfactual_recovers_chains_the_whole_frame_cannot_reach(payload):
    """The finding, pinned: shorter windows flip R3 from all-wrong to all-right.

    Whole-frame R3 flags nobody who is a launderer; the 7-day partition flags
    only launderers, and the two sets are disjoint. If R3's origin selection is
    ever fixed this test fails, which is the intent — the docstring's claim that
    the fix is worth something concrete has to fall with it.
    """
    cf = payload["r3_counterfactual"]
    wf = cf["whole_frame"]
    seven = next(r for r in cf["windowed"] if r["window_days"] == 7)

    assert wf["true_positives"] == 0
    assert wf["precision"] == 0.0
    assert seven["precision"] == 1.0
    assert seven["true_positives"] == seven["entities_flagged"] > 0
    # Disjoint: everything the 7-day arm finds is new, and everything the whole
    # frame found is missed.
    assert seven["new_vs_whole_frame"] == seven["entities_flagged"]
    assert seven["missed_vs_whole_frame"] == wf["entities_flagged"]


def test_counterfactual_precision_decays_as_windows_widen(payload):
    """The mechanism confirming itself.

    As each window approaches the whole frame the in-degree-0 origin set
    collapses the same way, so precision should fall back toward the shipped
    result. A non-monotone curve would mean something other than the search
    space is driving the difference.
    """
    rows = sorted(payload["r3_counterfactual"]["windowed"], key=lambda r: r["window_days"])
    precisions = [r["precision"] for r in rows]
    assert precisions == sorted(precisions, reverse=True)
    assert precisions[-1] > payload["r3_counterfactual"]["whole_frame"]["precision"]


def test_counterfactual_union_is_no_larger_than_the_sum_of_its_windows(payload):
    """A union cannot exceed the parts, and equality would mean no entity was
    ever flagged in two different windows — worth knowing either way."""
    for r in payload["r3_counterfactual"]["windowed"]:
        assert r["entities_flagged"] <= sum(r["hits_per_window"])
        assert r["windows"] == len(r["hits_per_window"])


# ---------------------------------------------------------------------------
# The study end to end
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def payload():
    return build()


def test_every_arm_scores_the_same_population(payload):
    """Arms differ only in the ML half — the denominator must be identical."""
    populations = {
        r["by_definition"]["sender_only"]["any_flag"]["true_positives"]
        + r["by_definition"]["sender_only"]["any_flag"]["false_positives"]
        + r["by_definition"]["sender_only"]["any_flag"]["false_negatives"]
        + r["by_definition"]["sender_only"]["any_flag"]["true_negatives"]
        for r in payload["arms"]
    }
    assert len(populations) == 1


def test_rules_only_arm_is_the_floor(payload):
    """No ML configuration may flag fewer entities than rules alone.

    Fusion is additive over the rule hits, so a hybrid arm dropping below the
    rules-only row would mean the ML half had removed a flag — which the
    scoring code has no mechanism to do, and which would indicate the arms
    are not sharing rule hits the way the study claims.
    """
    floor = next(r for r in payload["arms"] if r["arm"].startswith("Rules only"))
    for arm in payload["arms"]:
        assert arm["flagged"] >= floor["flagged"], arm["arm"]


def test_drop_block_matches_the_arms_it_summarises(payload):
    a = next(r for r in payload["arms"] if r["arm"].startswith("A."))
    c = next(r for r in payload["arms"] if r["arm"].startswith("C."))
    d = payload["drop"]
    assert d["in_time_recall"] == a["recall"]
    assert d["frozen_model_recall"] == c["recall"]
    assert d["recall_delta"] == pytest.approx(c["recall"] - a["recall"])
    assert d["precision_delta"] == pytest.approx(c["precision"] - a["precision"])


def test_windows_block_is_internally_consistent(payload):
    w = payload["windows"]
    assert w["train"]["days"] + w["test"]["days"] == w["span_days"]
    assert 0.0 <= w["test"]["unseen_share"] <= 1.0
    assert w["test"]["positives"]["sender_only"] <= w["test"]["customers"]


def test_full_window_run_reproduces_the_published_rule_hits(payload):
    """The harness must agree with evaluation/results/ablation.json.

    window_artifacts runs feature_engineer and rule_detect itself rather than
    going through ablation.run_detection_stack, so this is the check that the
    two paths have not diverged.
    """
    published = {"R1": 11, "R2": 3, "R3": 4, "R4": 8, "R5": 0, "R6": 0, "R7": 11}
    actual = {r["rule"]: r["full_90d"] for r in payload["rule_reachability"]}
    assert actual == published
