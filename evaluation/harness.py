"""
evaluation/harness.py — ground truth and detection metrics.

Pure functions only: this module reads CSVs and does arithmetic. It imports
nothing from backend.agent.* or backend.tools.*, so it stays cheap to unit-test
and can be reused by both the CLI (evaluation/run_evaluation.py) and the
integration test that protects the README's headline claim.

Why this exists
---------------
README.md's Results section publishes precision/recall/FPR for the naive
baseline and for two tiers of our system. Before this module, nothing in the
repo computed them — they were produced by hand once and transcribed into
prose, which meant no later change to detection could be shown to have helped.

Three ground-truth definitions
------------------------------
Our system flags *customers*, but the dataset labels *transactions*, so the
labels have to be lifted to customer level. There is more than one defensible
way to do that and they give very different recall:

  sender_only        — a customer is positive if they SENT at least one
                       labelled transaction (51 of 270). This matches what the
                       detectors actually look at: every rule and rolling
                       feature is sender-side/outbound.

  sender_or_receiver — positive if they sent OR received one (114 of 270).
                       The extra 63 are receive-only participants, e.g. the
                       individual recipients in a fan-out. Only R7 evaluates
                       inbound behaviour, so most of these are structurally
                       uncatchable today.

  sender_or_repeat_receiver
                     — positive if they sent one, or RECEIVED AT LEAST TWO
                       (84 of 270). The middle ground, and arguably the most
                       honest target.

Why the third exists
--------------------
sender_or_receiver over-labels. 30 of its 63 receive-only positives receive
exactly ONE labelled transaction, which does not distinguish a participant from
an ordinary counterparty who happened to be paid once by a launderer. Scoring
against them measures whether the system can identify people the data gives it
no evidence about. Requiring repetition keeps the receive-only participants that
show a *pattern* (33 of them) and drops the incidental ones.

Two candidate spellings of "repeat" were measured and are identical on this
dataset: at-least-two-inbound-total, and at-least-two-from-a-single-sender (the
signal R7 keys on). No customer here receives two labelled transactions from two
different senders, so the two sets coincide exactly. The simpler total-count
form is implemented; if that ever stops holding on new data the distinction
would need revisiting, which is what REPEAT_RECEIVER_MIN_TXNS documents.

A third clause was considered and rejected: "or received from a flagged sender".
It is degenerate here — every labelled transaction's sender is a sender-side
positive by construction, so the clause selects all 91 receivers and collapses
straight back into sender_or_receiver.

Reporting all three is the honest thing to do: the first says how well the
system does at the job it was built for, the second says how much of the problem
that job leaves untouched, and the third says how much of that gap is actually
evidenced in the data.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

# Repo-root-relative defaults, resolved from this file so the CLI works from
# any working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TXN_CSV = _REPO_ROOT / "data" / "sample" / "aml_sample.csv"
DEFAULT_CUST_CSV = _REPO_ROOT / "data" / "sample" / "aml_sample_customers.csv"

# AML_LOGIC.md §6: the naive comparator the whole false-positive story is told
# against — "flag any customer who sent a transaction over $9,000".
NAIVE_THRESHOLD = 9_000.0

# How many labelled inbound transactions make a receiver a *participant* rather
# than an incidental counterparty, for the sender_or_repeat_receiver definition.
# 2 is the smallest value that expresses "more than once", which is the whole
# claim being made — anything higher would be tuning the ground truth to the
# detector, which is the mistake this definition exists to avoid.
REPEAT_RECEIVER_MIN_TXNS = 2


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def _as_bool(series: pd.Series) -> pd.Series:
    """Coerce a label column to real booleans.

    label_is_laundering arrives as dtype=object from read_csv. Comparing it
    with `== True` happens to work for the committed CSV, but silently returns
    all-False if the column is ever written as the strings "True"/"False" —
    which would zero out every metric without raising. Normalise explicitly
    rather than depending on how pandas guessed.
    """
    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "t"})
    )


@dataclass(frozen=True)
class GroundTruth:
    """Customer-level ground truth under both label-lifting definitions."""

    all_customers: set[str]
    sender_only: set[str]
    sender_or_receiver: set[str]
    sender_or_repeat_receiver: set[str]
    labelled_txn_count: int

    @property
    def receive_only(self) -> set[str]:
        """Positives reachable only through inbound behaviour.

        This set is the recall gap: only R7 evaluates inbound behaviour, so
        most of these cannot be flagged on their own evidence.
        """
        return self.sender_or_receiver - self.sender_only

    @property
    def incidental_receivers(self) -> set[str]:
        """Receive-only positives with a single labelled inbound transaction.

        The difference between the broad and the repeat-receiver definitions,
        and the reason the third one exists: one inbound payment does not
        distinguish a participant from someone a launderer happened to pay
        once.
        """
        return self.sender_or_receiver - self.sender_or_repeat_receiver

    def positives(self, definition: str) -> set[str]:
        if definition == "sender_only":
            return self.sender_only
        if definition == "sender_or_receiver":
            return self.sender_or_receiver
        if definition == "sender_or_repeat_receiver":
            return self.sender_or_repeat_receiver
        raise ValueError(
            f"unknown ground-truth definition {definition!r}; expected "
            "'sender_only', 'sender_or_receiver' or 'sender_or_repeat_receiver'"
        )


def ground_truth_from_frames(
    df: pd.DataFrame,
    all_customers: Iterable[str],
) -> GroundTruth:
    """Lift transaction labels in `df` to customer level over a fixed roster.

    Separated from `load_ground_truth` so a *subset* of transactions can be
    scored — a time window, an evasion-perturbed frame — against the same
    definitions, without writing a second copy of the lifting logic that could
    drift from this one.

    `all_customers` is the population, passed in rather than derived, because
    the caller decides what "everyone" means: the full roster for a whole-file
    run, or the customers active in a window for an out-of-time run. Every set
    is intersected with it, so a counterparty that appears in transactions but
    not in the population never inflates a denominator.
    """
    population = set(all_customers)
    labelled = df[_as_bool(df["label_is_laundering"])]

    senders = set(labelled["sender_id"]) & population
    receivers = set(labelled["receiver_id"]) & population

    # Receivers of MORE THAN ONE labelled transaction. value_counts() is over
    # the raw column, so intersect with the roster afterwards for the same
    # reason every other set here does.
    inbound_counts = labelled["receiver_id"].value_counts()
    repeat_receivers = set(
        inbound_counts[inbound_counts >= REPEAT_RECEIVER_MIN_TXNS].index
    ) & population

    return GroundTruth(
        all_customers=population,
        sender_only=senders,
        sender_or_receiver=senders | receivers,
        sender_or_repeat_receiver=senders | repeat_receivers,
        labelled_txn_count=int(len(labelled)),
    )


def load_ground_truth(
    txn_csv: str | Path = DEFAULT_TXN_CSV,
    cust_csv: str | Path = DEFAULT_CUST_CSV,
) -> tuple[pd.DataFrame, GroundTruth]:
    """Load the labelled dataset and lift its transaction labels to customers.

    Returns the raw transactions frame alongside the ground truth, because
    callers (the naive baseline) need the transactions too and there's no
    reason to read the file twice.
    """
    df = pd.read_csv(txn_csv)
    cust = pd.read_csv(cust_csv)
    return df, ground_truth_from_frames(df, set(cust["customer_id"]))


def naive_baseline(
    df: pd.DataFrame,
    all_customers: set[str],
    threshold: float = NAIVE_THRESHOLD,
) -> set[str]:
    """The comparator: every customer who sent a transaction over the threshold.

    Deliberately translated to customer level the same way our own flags are,
    so the comparison is like-for-like rather than flattering.
    """
    return set(df[df["amount"] > threshold]["sender_id"].unique()) & all_customers


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Metrics:
    """One confusion matrix and the rates derived from it."""

    flagged: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1: float
    false_positive_rate: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_div(numerator: float, denominator: float) -> float:
    """Rates are 0.0 rather than undefined when nothing lands in the denominator.

    A system that flags nobody has precision 0.0 here, not a crash and not 1.0 —
    treating "never wrong because never tried" as perfect precision would make
    the metric useless for exactly the degenerate case worth catching.
    """
    return numerator / denominator if denominator else 0.0


def evaluate(
    flagged: Iterable[str],
    positives: Iterable[str],
    all_customers: Iterable[str],
) -> Metrics:
    """Score a flagged set against a positive set over a fixed population."""
    flagged = set(flagged)
    positives = set(positives)
    population = set(all_customers)

    # Confine everything to the known population so a stray ID can't produce
    # a negative true-negative count.
    flagged &= population
    positives &= population
    negatives = population - positives

    tp = len(flagged & positives)
    fp = len(flagged & negatives)
    fn = len(positives - flagged)
    tn = len(negatives - flagged)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)

    return Metrics(
        flagged=len(flagged),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        precision=precision,
        recall=recall,
        f1=_safe_div(2 * precision * recall, precision + recall),
        false_positive_rate=_safe_div(fp, fp + tn),
    )
