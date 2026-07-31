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

Two ground-truth definitions
----------------------------
Our system flags *customers*, but the dataset labels *transactions*, so the
labels have to be lifted to customer level. There are two defensible ways to
do that and they give very different recall:

  sender_only        — a customer is positive if they SENT at least one
                       labelled transaction (51 of 270). This matches what the
                       detectors actually look at: every rule and rolling
                       feature is sender-side/outbound.

  sender_or_receiver — positive if they sent OR received one (114 of 270).
                       The extra 63 are receive-only participants, e.g. the
                       individual recipients in a fan-out. No rule currently
                       evaluates inbound behaviour, so these are structurally
                       uncatchable today.

Reporting both is the honest thing to do: the first says how well the system
does at the job it was built for, the second says how much of the problem that
job leaves untouched.
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
    labelled_txn_count: int

    @property
    def receive_only(self) -> set[str]:
        """Positives reachable only through inbound behaviour.

        This set is the recall gap: no current rule or feature evaluates
        fan-in, so nothing in the system can flag these on their own evidence.
        """
        return self.sender_or_receiver - self.sender_only

    def positives(self, definition: str) -> set[str]:
        if definition == "sender_only":
            return self.sender_only
        if definition == "sender_or_receiver":
            return self.sender_or_receiver
        raise ValueError(
            f"unknown ground-truth definition {definition!r}; "
            "expected 'sender_only' or 'sender_or_receiver'"
        )


def load_ground_truth(
    txn_csv: str | Path = DEFAULT_TXN_CSV,
    cust_csv: str | Path = DEFAULT_CUST_CSV,
) -> tuple[pd.DataFrame, GroundTruth]:
    """Load the labelled dataset and lift its transaction labels to customers.

    Returns the raw transactions frame alongside the ground truth, because
    callers (the naive baseline) need the transactions too and there's no
    reason to read the file twice.

    Every set is intersected with the customer roster, so a counterparty that
    appears in transactions but not in the customer table never inflates a
    denominator.
    """
    df = pd.read_csv(txn_csv)
    cust = pd.read_csv(cust_csv)

    all_customers = set(cust["customer_id"])
    labelled = df[_as_bool(df["label_is_laundering"])]

    senders = set(labelled["sender_id"]) & all_customers
    receivers = set(labelled["receiver_id"]) & all_customers

    return df, GroundTruth(
        all_customers=all_customers,
        sender_only=senders,
        sender_or_receiver=senders | receivers,
        labelled_txn_count=int(len(labelled)),
    )


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
