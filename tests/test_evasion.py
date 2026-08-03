"""Tests for the adversarial evasion study.

The study's whole claim rests on one property: at dial zero the perturbation
must be the identity, so every recall drop further along the curve is caused by
the evasion and nothing else. Most of what follows is that property, checked
per move, because a perturbation that quietly alters rows it was not supposed
to touch would produce a smooth, plausible, meaningless curve.

The second thing being protected is the committed sample. CLAUDE.md forbids
regenerating data/sample/aml_sample.csv, and a study that mutated its own
ground truth would invalidate every other number in the repo. The frames here
are copies, and one test holds the files to their hashes across a full set of
perturbations.
"""

import hashlib
from pathlib import Path

import pandas as pd
import pytest

import backend.tools.rules as rules_mod
from evaluation.evasion import (
    MOVES,
    _take,
    adversary_rows,
    move_amount_dodge,
    move_cashout_delay,
    move_combined,
    move_shrink_amounts,
    move_slow_down,
)
from evaluation.harness import DEFAULT_CUST_CSV, DEFAULT_TXN_CSV, load_ground_truth

_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def gt():
    _, ground_truth = load_ground_truth()
    return ground_truth


@pytest.fixture(scope="module")
def df():
    frame = pd.read_csv(DEFAULT_TXN_CSV)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


# ---------------------------------------------------------------------------
# What the adversary is modelled as controlling
# ---------------------------------------------------------------------------


def test_adversary_controls_only_their_own_labelled_transactions(df, gt):
    """The negative class must be held completely fixed.

    If an ordinary customer's transactions could be perturbed, a recall change
    would be confounded with a change to the false positives, and the study
    would be measuring the dataset rather than the adversary.
    """
    controlled = adversary_rows(df, gt)
    sub = df.loc[controlled]

    assert len(controlled) > 0
    assert sub["label_is_laundering"].astype(str).str.lower().isin({"true"}).all()
    assert set(sub["sender_id"]) <= gt.sender_only


def test_take_is_deterministic_and_nested(df, gt):
    """Ordering by txn_id rather than sampling buys two things the curve needs.

    Determinism, so a rerun cannot move a number; and nesting, so a recall
    change between two dial settings cannot be an artifact of a different
    subset having been drawn.
    """
    controlled = adversary_rows(df, gt)

    quarter = _take(controlled, df, 0.25)
    half = _take(controlled, df, 0.5)

    assert list(quarter) == list(_take(controlled, df, 0.25))
    assert set(quarter) < set(half)
    assert len(_take(controlled, df, 0.0)) == 0
    assert len(_take(controlled, df, 1.0)) == len(controlled)


# ---------------------------------------------------------------------------
# Dial zero is the identity — per move
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "move",
    [move_amount_dodge, move_slow_down, move_cashout_delay, move_shrink_amounts, move_combined],
    ids=lambda f: f.__name__,
)
def test_zero_dial_is_the_identity(df, gt, move):
    """Every published baseline row depends on this.

    The first row of each table is quoted as reproducing the shipped system's
    recall. If a move perturbed anything at dial zero, that row would be a
    different system and the whole comparison would be against the wrong
    reference.
    """
    perturbed, cost = move(df, gt, 0.0)

    pd.testing.assert_frame_equal(perturbed, df)
    # Every cost, in whatever unit the move prices itself in, must be zero.
    # Checked this way rather than on a single key because move_combined
    # aggregates three moves and has no one transaction count to report.
    assert all(v == 0 for v in cost.values()), cost


@pytest.mark.parametrize(
    "move,dial",
    [
        (move_amount_dodge, 1.0),
        (move_slow_down, 4.0),
        (move_cashout_delay, 24.0),
        (move_shrink_amounts, 0.25),
        (move_combined, 1.0),
    ],
    ids=lambda v: getattr(v, "__name__", str(v)),
)
def test_moves_never_mutate_the_input_frame(df, gt, move, dial):
    """Each move copies. Without this the moves would compose by accident and
    every table after the first would be measuring the previous one's damage."""
    before = df.copy()
    move(df, gt, dial)
    pd.testing.assert_frame_equal(df, before)


# ---------------------------------------------------------------------------
# Each move does what it claims, and only that
# ---------------------------------------------------------------------------


def test_amount_dodge_clears_r1s_band_but_not_r2s(df, gt):
    """The finding the move exists to demonstrate.

    Stepping one dollar under R1's floor puts the amount at $8,999, which is
    still inside R2's $7,000 band. Defeating the cheapest rule to defeat does
    not defeat the others, and that is the argument for having more than one.
    """
    perturbed, cost = move_amount_dodge(df, gt, 1.0)
    moved = perturbed["amount"] != df["amount"]

    assert cost["txns_moved"] > 0
    assert (perturbed.loc[moved, "amount"] < rules_mod.BAND_LOW).all()
    assert (perturbed.loc[moved, "amount"] >= rules_mod.R2_SMURFING_BAND_LOW).all()
    # A dollar a transaction is the entire price of evading R1.
    assert cost["dollars_per_txn"] < 1000.0


def test_amount_dodge_touches_only_in_band_transactions(df, gt):
    perturbed, _ = move_amount_dodge(df, gt, 1.0)
    changed = df.index[perturbed["amount"] != df["amount"]]
    original = df.loc[changed, "amount"]
    assert original.between(rules_mod.BAND_LOW, rules_mod.BAND_HIGH).all()


def test_slow_down_only_moves_timestamps_forward(df, gt):
    """A negative shift would be time travel, and would also silently pull
    transactions INTO a window rather than out of it."""
    perturbed, cost = move_slow_down(df, gt, 4.0)
    delta = perturbed["timestamp"] - df["timestamp"]

    assert (delta >= pd.Timedelta(0)).all()
    assert cost["max_delay_days"] > 0
    assert perturbed["amount"].equals(df["amount"]), "slow_down must not touch amounts"


def test_slow_down_leaves_the_first_transaction_of_each_sender_alone(df, gt):
    """The cost model says the i-th transaction is delayed by i * spacing, so
    the earliest one is free. If it moved, the reported mean delay would be
    understating what the adversary actually paid."""
    perturbed, _ = move_slow_down(df, gt, 4.0)
    controlled = adversary_rows(df, gt)

    sub = df.loc[controlled].sort_values(["sender_id", "timestamp", "txn_id"])
    firsts = sub.groupby("sender_id").head(1).index

    assert (perturbed.loc[firsts, "timestamp"] == df.loc[firsts, "timestamp"]).all()


def test_cashout_delay_shifts_only_cash_transactions(df, gt):
    perturbed, cost = move_cashout_delay(df, gt, 24.0)
    changed = df.index[perturbed["timestamp"] != df["timestamp"]]

    assert len(changed) == cost["txns_moved"] > 0
    is_cash = df.loc[changed, "txn_type"].astype(str).str.lower().isin(rules_mod.R4_CASH_TYPES)
    is_chan = df.loc[changed, "channel"].astype(str).str.lower().isin(rules_mod.R4_CASH_CHANNELS)
    assert (is_cash | is_chan).all()
    assert (perturbed.loc[changed, "timestamp"] - df.loc[changed, "timestamp"]).eq(
        pd.Timedelta(hours=24)
    ).all()


def test_shrink_amounts_reduces_value_and_reports_the_loss(df, gt):
    perturbed, cost = move_shrink_amounts(df, gt, 0.25)
    controlled = adversary_rows(df, gt)

    assert (perturbed.loc[controlled, "amount"] < df.loc[controlled, "amount"]).all()
    expected = float((df.loc[controlled, "amount"] * 0.25).sum())
    assert cost["dollars_forgone"] == pytest.approx(expected, rel=1e-6)


def test_combined_reports_a_cost_from_every_component(df, gt):
    """The combined move is the one an interviewer will ask about, so its cost
    line has to show all three prices rather than collapsing to one."""
    _, cost = move_combined(df, gt, 1.0)
    assert cost["dollars_forgone"] > 0
    assert cost["mean_delay_days"] > 0
    assert cost["cashout_delay_hours"] > 0


# ---------------------------------------------------------------------------
# The committed sample is never written
# ---------------------------------------------------------------------------


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_perturbations_never_touch_the_committed_sample(df, gt):
    """CLAUDE.md forbids regenerating data/sample/aml_sample.csv.

    Every published metric in this repo is measured against those two files.
    A study that rewrote them would not just break itself, it would silently
    invalidate the README, the ablation, and run_evaluation's baselines.
    """
    before = {p: _digest(p) for p in (DEFAULT_TXN_CSV, DEFAULT_CUST_CSV)}

    for _key, _label, fn, dials, _dial_name, _targets in MOVES:
        for dial in dials:
            fn(df, gt, dial)

    assert {p: _digest(p) for p in before} == before


def test_every_move_is_registered_with_a_zero_first_dial():
    """The renderer labels row one 'baseline', and _retained divides by it.

    A move whose dial list did not start at the identity would print a baseline
    that is already evaded and report retention against it, which would make a
    defeated detector look robust.
    """
    for key, _label, _fn, dials, _dial_name, _targets in MOVES:
        assert dials[0] == 0.0, f"{key} does not start from an unperturbed baseline"
        assert dials == sorted(dials), f"{key} dials are not monotonically increasing"
