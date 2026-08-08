"""
evaluation/out_of_time.py — does the ML half generalise forward in time?

Usage
-----
    python -m evaluation.out_of_time
    python -m evaluation.out_of_time --train-days 60 --output <path>

Why this exists
---------------
`ml_detect` fits IsolationForest and LOF on the same rows it then scores
(`ml_detect.py:256`). That is a defensible choice for unsupervised transductive
scoring — nobody is claiming a trained artifact, and the percentile is a
statement about a population, not a prediction about a future customer.

It is also the first thing a bank's model-validation function will ask about,
and "it is transductive by design" is only a good answer if you can say what the
alternative would have cost. Nothing in the repo could say that. Every published
number fits and scores one 90-day span, so the honest summary of our ML evidence
was: it separates launderers from non-launderers *in a population it has already
seen in full*.

This study fits on the first 60 days and scores the last 30.

The three arms
--------------
All three are evaluated on the SAME test-window rows against the SAME
test-window ground truth. Only the ML half changes, and it changes one thing at
a time, so a difference between rows has exactly one cause:

  A. in_time      — fit on test features, rank within test. This is what ships,
                    restricted to the test window. The reference point.
  B. frozen_fit   — fit on TRAIN features, rank within test. Isolates the cost
                    of never having seen the scored rows.
  C. frozen_model — fit on train features, rank against the TRAIN score
                    distribution. What a deployed model actually does: the
                    scoring population is not available to it, so the cut points
                    have to be frozen at training time too.

C is the number to quote. B exists because if C degrades, B says whether the
damage came from the model or from the frozen thresholds, and those have
completely different fixes.

The rules half is stateless — no fitting — so it is identical in all three arms.
It is reported anyway, as the floor: it is the part of the test-window result
that cannot degrade out of time, and the hybrid's number is only interesting
relative to it.

Three things that are wrong with this study, stated up front
------------------------------------------------------------
1. R6 cannot fire in the test window. It requires 60 days of dormancy
    (`R6_DORMANCY_DAYS = 60`) and the window is 30 days long, so the rule is
    arithmetically incapable of a hit — not because it failed, but because the
    experiment does not contain enough time for it to be asked. §2 of the output
    measures this rather than asserting it.

2. `amount_zscore_90d` is not the same statistic in the two windows. It is
    computed over whatever history is present, so in the test window it is a
    30-day z-score wearing a 90-day name. The feature is not being held constant
    across the split; only the fitting population is.

3. 30 days is a small window and this is a small dataset. The test window holds
    a few hundred transactions. Treat the direction of the result as the finding
    and the third decimal place as noise.

What this study found that it was not looking for
-------------------------------------------------
R3 reports MORE layering chains on 29 days of data (6) than on the full 90 (4).
A hit count that falls as evidence accumulates is not a rounding artifact, so §3
of the output measures the cause: `rules.py:311` starts chain enumeration only
at nodes with **in-degree 0** in the wire/transfer subgraph. Measured on this
dataset — 46 such nodes in the 29-day window, 28 in 60 days, **6** in 90 days.
The searched (src, snk) pair count therefore collapses from 2,300 to 42 as the
window grows.

The rule is anti-monotone in data volume by construction. Extrapolated to a
production graph with years of history, virtually no customer has zero inbound
wires and R3 has almost nowhere to begin, so it would approach zero recall on
exactly the datasets it matters on. This is a defect in R3, not in the split,
and it is deliberately NOT fixed here: changing detection code would invalidate
every published baseline in `evaluation/results/`, which is a decision to take
on its own rather than as a side effect of adding a study.

§4 then asks what that decision would be worth, without taking it. The shipped
`rule_detect` is run unchanged over non-overlapping partitions of the same
transactions and the R3 hits are unioned — a counterfactual, not a proposed
implementation. The whole-frame R3 flags 4 entities and none of them is a
launderer; the 7-day partition flags 3 and all of them are, with **zero overlap
between the two sets**. Precision 0.000 → 1.000 on identical data. So the
in-degree-0 collapse is not merely shrinking R3's yield, it is leaving behind
precisely the wrong chains, and the fix is worth something concrete rather than
speculative.

The LOF finding
---------------
`ml_detect` constructs `LocalOutlierFactor(novelty=False)`, which has no
`score_samples` — it can only score the rows it was fitted on. **The shipped
ml_detect is structurally incapable of out-of-time scoring**, and this was found
by trying to do it rather than by reading the code. Arms B and C therefore fit
LOF with `novelty=True`, which changes nothing about the fit (the training
scores are identical either way) and only exposes the scoring method that
novelty=False withholds.

That substitution is a confound, so §5 reports an IF-only control alongside. If
the IF-only drop matches the IF+LOF drop, the substitution is not carrying the
result.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

import backend.tools.rules as rules_mod
from backend.agent import registry
from backend.config import settings
from backend.schemas import QueryIntent
from backend.tools.base import ToolContext
from backend.tools.ml_detect import (
    IF_CONTAMINATION,
    IF_N_ESTIMATORS,
    IF_RANDOM_STATE,
    IF_WEIGHT,
    LOF_MIN_SAMPLES,
    LOF_N_NEIGHBORS,
    LOF_WEIGHT,
    _percentile_rank,
    _select_feature_cols,
)

from evaluation.ablation import ALL_RULES, fuse
from evaluation.harness import (
    DEFAULT_CUST_CSV,
    Metrics,
    evaluate,
    ground_truth_from_frames,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = _REPO_ROOT / "evaluation" / "results" / "out_of_time.json"

# The dataset spans 90 days. 60/30 puts two thirds of the history in the fit and
# leaves a test window long enough for R1's 7-day and R4's 24-hour windows to be
# expressible. It is not tuned: it is the only split that keeps every rule except
# R6 structurally able to fire, and R6 cannot be saved by any split of 90 days.
DEFAULT_TRAIN_DAYS = 60

PRIMARY_DEFINITION = "sender_only"

DEFINITIONS = ("sender_only", "sender_or_receiver", "sender_or_repeat_receiver")


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_by_time(
    df: pd.DataFrame,
    train_days: int = DEFAULT_TRAIN_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split transactions at `train_days` after the first timestamp.

    The boundary is derived from the data rather than hard-coded, so the split
    stays meaningful if the dataset is ever regenerated over different dates.
    Rows exactly on the boundary go to test, which is the convention that keeps
    the two frames disjoint and exhaustive.
    """
    ts = pd.to_datetime(df["timestamp"])
    boundary = ts.min() + pd.Timedelta(days=train_days)
    return df[ts < boundary].copy(), df[ts >= boundary].copy(), boundary


# ---------------------------------------------------------------------------
# Running the real tools over a window
# ---------------------------------------------------------------------------


def _ctx(df: pd.DataFrame | None, customers: Any = None, **artifacts: Any) -> ToolContext:
    return ToolContext(
        df=df,
        customers=customers,
        intent=QueryIntent(
            raw_query="Analyse this dataset for suspicious activity",
            intent="full_analysis",
            parsed_by="rules",
            confidence=0.9,
        ),
        artifacts=dict(artifacts),
    )


def load_canonical(source: str = "synthetic") -> tuple[pd.DataFrame, Any]:
    """Run the real load_data so the split operates on the canonical frame.

    Splitting the raw CSV instead would fork the timestamp parsing and dtype
    normalisation away from what every other study sees.
    """
    settings.aml_use_mocks = False
    tools = registry.load_tools(use_mocks=False)
    ctx = _ctx(None)
    result = tools["load_data"](ctx, source=source)
    if not result.ok:
        raise RuntimeError(f"load_data failed: {result.error}")
    return result.df, ctx.customers


def window_artifacts(df: pd.DataFrame, customers: Any) -> dict[str, Any]:
    """feature_engineer + rule_detect over one window, via the registered tools.

    Returns the features frame, the feature list and the rule hits. No
    `transactions_reference` artifact is supplied, so feature_engineer treats
    this window as its own unfiltered population — which is exactly right here,
    since the window *is* the whole world for that arm.
    """
    tools = registry.load_tools(use_mocks=False)
    ctx = _ctx(df, customers=customers)

    feat = tools["feature_engineer"](ctx)
    if not feat.ok:
        raise RuntimeError(f"feature_engineer failed: {feat.error}")
    ctx.artifacts.update(feat.artifacts)

    hits = tools["rule_detect"](ctx)
    if not hits.ok:
        raise RuntimeError(f"rule_detect failed: {hits.error}")

    return {
        "features": feat.artifacts["features"],
        "feature_list": feat.artifacts["feature_list"],
        "rule_hits": list(hits.artifacts.get("rule_hits", [])),
    }


# ---------------------------------------------------------------------------
# The ML arms
# ---------------------------------------------------------------------------


def _rank_against(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Percentile of each value within a FIXED reference distribution.

    `_percentile_rank` ranks a population against itself, which a deployed model
    cannot do — it sees one customer at a time and has no cohort to sort them
    into. This is the frozen-threshold analogue: where does this score fall in
    the distribution the model was trained on?
    """
    ref = np.sort(reference)
    if len(ref) == 0:
        return np.zeros(len(values))
    # Divide by n-1 so a value at the top of the reference scores exactly 1.0,
    # matching _percentile_rank's convention. searchsorted can return n for a
    # value above everything seen in training, which is the whole point of a
    # frozen distribution — so clip, or the fused percentile exceeds 1.0 and
    # carries a risk score past the ceiling risk.py assumes.
    ranked = np.searchsorted(ref, values, side="left") / max(len(ref) - 1, 1)
    return np.clip(ranked, 0.0, 1.0)


def _matrix(feat_df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Feature matrix restricted to `cols`, in that order.

    Columns are chosen from the FITTING frame and imposed on the scoring frame.
    A column that the fit used but the test window never produced is filled with
    0.0 — the same "not observed means zero" convention feature_engineer uses
    (`features.py:657`) — rather than dropped, because dropping it would change
    the matrix the scaler and the models were fitted against.
    """
    return feat_df.reindex(columns=cols).fillna(0.0).values.astype(float)


def ml_arm(
    fit_feat: pd.DataFrame,
    score_feat: pd.DataFrame,
    feature_list: list[str],
    *,
    freeze_distribution: bool,
    use_lof: bool = True,
) -> list[dict[str, Any]]:
    """Fit on `fit_feat`, score `score_feat`, return ml_scores rows.

    When the two frames are the same object this reproduces `ml_detect` exactly:
    same column selection, same scaler, same models and seeds, same 0.6/0.4
    fusion, same `_percentile_rank`. That equivalence is asserted in
    tests/test_out_of_time.py rather than assumed, because it is the only thing
    making arm A a fair reference point for arms B and C.
    """
    cols = _select_feature_cols(fit_feat, feature_list)
    X_fit = StandardScaler().fit(_matrix(fit_feat, cols))
    X_train = X_fit.transform(_matrix(fit_feat, cols))
    X_score = X_fit.transform(_matrix(score_feat, cols))

    in_sample = fit_feat is score_feat

    iso = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=IF_RANDOM_STATE,
    )
    iso.fit(X_train)
    if_train = -iso.decision_function(X_train)
    if_raw = if_train if in_sample else -iso.decision_function(X_score)

    n_fit = len(X_train)
    lof_ok = use_lof and n_fit >= LOF_MIN_SAMPLES

    if lof_ok:
        lof = LocalOutlierFactor(
            n_neighbors=min(LOF_N_NEIGHBORS, n_fit - 1),
            # novelty=True is the whole reason this module cannot just call
            # ml_detect. It leaves negative_outlier_factor_ untouched, so the
            # in-sample path below is byte-identical to the shipped one, and it
            # additionally exposes score_samples for rows the fit never saw.
            novelty=True,
        )
        lof.fit(X_train)
        lof_train = -lof.negative_outlier_factor_
        lof_raw = lof_train if in_sample else -lof.score_samples(X_score)

    if freeze_distribution:
        if_pct = _rank_against(if_raw, if_train)
        lof_pct = _rank_against(lof_raw, lof_train) if lof_ok else None
    else:
        if_pct = _percentile_rank(if_raw)
        lof_pct = _percentile_rank(lof_raw) if lof_ok else None

    fused = IF_WEIGHT * if_pct + LOF_WEIGHT * lof_pct if lof_ok else if_pct

    return [
        {
            "entity_id": str(eid),
            "score": round(float(fused[i]), 4),
            "percentile": round(float(fused[i]), 4),
            "top_features": [],
        }
        for i, eid in enumerate(score_feat.index)
    ]


# ---------------------------------------------------------------------------
# The R3 counterfactual — what would the fix be worth?
# ---------------------------------------------------------------------------


def partition_by_days(df: pd.DataFrame, days: int) -> list[pd.DataFrame]:
    """Cut the frame into NON-OVERLAPPING windows of `days`.

    Non-overlapping matters. Sliding windows would let the same chain be found
    from several offsets and counted each time, which would manufacture the
    result this counterfactual is supposed to test. A partition re-cuts exactly
    the same transactions, so any extra hit is a chain the whole-frame run could
    not see rather than one it saw repeatedly.

    The final window is short if the span does not divide evenly. It is kept:
    dropping it would silently discard transactions and make the comparison
    unfair in the counterfactual's favour.
    """
    ts = pd.to_datetime(df["timestamp"])
    start, end = ts.min(), ts.max()
    out: list[pd.DataFrame] = []
    edge = start
    while edge <= end:
        stop = edge + pd.Timedelta(days=days)
        chunk = df[(ts >= edge) & (ts < stop)]
        if len(chunk):
            out.append(chunk.copy())
        edge = stop
    return out


def r3_counterfactual(
    df: pd.DataFrame,
    customers: Any,
    gt: Any,
    window_sizes: tuple[int, ...] = (7, 14, 30),
) -> dict[str, Any]:
    """What R3 would find if chain origin were a windowed property.

    NOT A PROPOSED IMPLEMENTATION, and nothing here touches detection code. The
    shipped `rule_detect` runs verbatim over each slice and the R3 hits are
    unioned. That is deliberately cruder than a real fix would be, but it tests
    the one thing worth testing: whether the in-degree-0 origin set is the
    binding constraint. If it is, re-cutting the same 90 days into shorter
    windows recovers chains the whole-frame run cannot reach.

    Reported with precision, not just hit counts. R3 currently finds 4 chains
    and zero true positives, so "the windowed variant finds more" is only good
    news if the extra chains are launderers. If precision stays at zero, the
    honest conclusion is that the search space is not the only thing wrong with
    R3 — which is a different finding, and a more useful one than a bigger
    number.
    """
    positives = gt.positives(PRIMARY_DEFINITION)
    population = gt.all_customers

    def _r3_entities(hits) -> set[str]:
        return {str(h["entity_id"]) for h in hits if str(h.get("rule_id")) == "R3"}

    whole = window_artifacts(df, customers)
    whole_entities = _r3_entities(whole["rule_hits"])
    baseline = evaluate(whole_entities, positives, population)

    rows: list[dict[str, Any]] = []
    for days in window_sizes:
        slices = partition_by_days(df, days)
        found: set[str] = set()
        per_window: list[int] = []
        for chunk in slices:
            hits = _r3_entities(window_artifacts(chunk, customers)["rule_hits"])
            per_window.append(len(hits))
            found |= hits
        metrics = evaluate(found, positives, population)
        rows.append({
            "window_days": days,
            "windows": len(slices),
            "entities_flagged": len(found),
            "hits_per_window": per_window,
            "new_vs_whole_frame": len(found - whole_entities),
            "missed_vs_whole_frame": len(whole_entities - found),
            "precision": metrics.precision,
            "recall": metrics.recall,
            "true_positives": metrics.true_positives,
        })

    return {
        "whole_frame": {
            "entities_flagged": len(whole_entities),
            "precision": baseline.precision,
            "recall": baseline.recall,
            "true_positives": baseline.true_positives,
        },
        "windowed": rows,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score(risk_rows: list[dict], gt: Any, definition: str) -> tuple[Metrics, Metrics]:
    any_flag = {r["entity_id"] for r in risk_rows}
    high_only = {r["entity_id"] for r in risk_rows if r["risk_level"] == "high"}
    positives = gt.positives(definition)
    return (
        evaluate(any_flag, positives, gt.all_customers),
        evaluate(high_only, positives, gt.all_customers),
    )


def _row(label: str, rule_hits, ml_scores, gt) -> dict[str, Any]:
    risk_rows = fuse(rule_hits, ml_scores)
    any_m, high_m = _score(risk_rows, gt, PRIMARY_DEFINITION)
    return {
        "arm": label,
        "flagged": any_m.flagged,
        "high": high_m.flagged,
        "precision": any_m.precision,
        "recall": any_m.recall,
        "f1": any_m.f1,
        "fpr": any_m.false_positive_rate,
        "by_definition": {
            d: {
                "any_flag": _score(risk_rows, gt, d)[0].as_dict(),
                "high_only": _score(risk_rows, gt, d)[1].as_dict(),
            }
            for d in DEFINITIONS
        },
    }


# ---------------------------------------------------------------------------
# Diagnostics — the caveats, measured
# ---------------------------------------------------------------------------


def rule_reachability(train_hits, test_hits, full_hits) -> list[dict[str, Any]]:
    """Per-rule hit counts in each window.

    R6 is the reason this table exists. It needs 60 days of dormancy before a
    burst, so in a 30-day test window it cannot fire however well it works — a
    zero here is a property of the experiment, not a verdict on the rule. Any
    other rule showing zero in the test window but hits in the full run deserves
    the same question asked of it, which is why all seven are listed.
    """
    def _count(hits, rule):
        return sum(1 for h in hits if str(h.get("rule_id")) == rule)

    rows = []
    for rule in ALL_RULES:
        rows.append({
            "rule": rule,
            "full_90d": _count(full_hits, rule),
            "train_60d": _count(train_hits, rule),
            "test_30d": _count(test_hits, rule),
            "structurally_possible_in_test": not (
                rule == "R6" and rules_mod.R6_DORMANCY_DAYS >= 30
            ),
        })
    return rows


def layering_search_space(windows: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Why R3 finds fewer chains as the window grows.

    R3 enumerates paths from in-degree-0 nodes to out-degree-0 nodes
    (`rules.py:311`). Both sets shrink as history accumulates — a node that
    looked like a chain origin over 29 days has received something by day 90 —
    so the search space contracts even though the evidence expands.

    Rebuilds the same wire/transfer subgraph R3 builds, deduplicated the same
    way, so these counts are the ones the rule actually searched.
    """
    rows = []
    for label, frame in windows.items():
        eligible = frame[frame["txn_type"].isin(rules_mod.R3_CHAIN_TXN_TYPES)]
        edges = eligible[["sender_id", "receiver_id"]].drop_duplicates()
        graph = nx.DiGraph()
        graph.add_edges_from(edges.itertuples(index=False, name=None))
        sources = sum(1 for n in graph.nodes() if graph.in_degree(n) == 0)
        sinks = sum(1 for n in graph.nodes() if graph.out_degree(n) == 0)
        rows.append({
            "window": label,
            "eligible_txns": int(len(eligible)),
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "sources_in_degree_0": sources,
            "sinks_out_degree_0": sinks,
            "pairs_searched": sources * sinks,
        })
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render(payload: dict[str, Any]) -> str:
    w = payload["windows"]
    out: list[str] = []

    out.append("### 1. Windows")
    out.append("")
    out.append("| Window | Days | Transactions | Customers (sender or receiver) |")
    out.append("|---|---|---|---|")
    for key, label in (("train", "Train (fit)"), ("test", "Test (scored)")):
        d = w[key]
        out.append(
            f"| {label} | {d['days']} | {d['transactions']:,} | {d['customers']:,} |"
        )
    out.append("")
    out.append(f"Boundary: {w['boundary']}.")
    out.append("")
    unseen = w["test"]["unseen_customers"]
    if unseen:
        out.append(
            f"**{unseen} of {w['test']['customers']} test-window customers "
            f"({w['test']['unseen_share']:.1%}) never appear in the train window** — the "
            "frozen model has no history for them and must score them cold."
        )
    else:
        out.append(
            f"**Every one of the {w['test']['customers']} test-window customers also "
            "transacts in the train window, so this study cannot measure cold-start "
            "degradation at all.** That is a property of the synthetic generator, which "
            "gives each customer activity across the whole span. Real out-of-time "
            "validation is mostly a question about customers the model has never seen, "
            "and this dataset contains none — so the drop reported in §5 is a *lower "
            "bound* on what a frozen model would cost in production."
        )
    out.append("")

    td, rd, sd = w["test"]["days"], w["train"]["days"], w["span_days"]
    out.append(f"### 2. Which rules can even fire in a {td}-day window")
    out.append("")
    out.append(f"| Rule | Hits, full {sd}d | Hits, train {rd}d | Hits, test {td}d | Can fire in test? |")
    out.append("|---|---|---|---|---|")
    for r in payload["rule_reachability"]:
        mark = "yes" if r["structurally_possible_in_test"] else "**no — needs 60d dormancy**"
        out.append(
            f"| {r['rule']} | {r['full_90d']} | {r['train_60d']} | {r['test_30d']} | {mark} |"
        )
    out.append("")

    out.append("### 3. Why R3 finds fewer chains as the window grows")
    out.append("")
    r3 = next(r for r in payload["rule_reachability"] if r["rule"] == "R3")
    out.append(
        f"R3 reports {r3['test_30d']} hits on {td} days and {r3['full_90d']} on the full "
        f"{sd}. It enumerates paths only from in-degree-0 to out-degree-0 nodes "
        "(`rules.py:311`), and both sets shrink as history accumulates:"
    )
    out.append("")
    out.append("| Window | Eligible txns | Nodes | Edges | Sources (in-deg 0) | Sinks (out-deg 0) | Pairs searched |")
    out.append("|---|---|---|---|---|---|---|")
    for r in payload["layering_search_space"]:
        out.append(
            f"| {r['window']} | {r['eligible_txns']:,} | {r['nodes']} | {r['edges']} | "
            f"{r['sources_in_degree_0']} | {r['sinks_out_degree_0']} | {r['pairs_searched']:,} |"
        )
    out.append("")
    out.append(
        "**R3 is anti-monotone in data volume.** More evidence gives it a smaller search "
        "space, not a larger one. On a production graph with years of history almost no "
        "customer has zero inbound wires, so R3 would approach zero recall on exactly the "
        "datasets that matter. Reported, not fixed — see the module docstring."
    )
    out.append("")

    cf = payload["r3_counterfactual"]
    wf = cf["whole_frame"]
    out.append("### 4. What fixing R3 would be worth — counterfactual, not shipped")
    out.append("")
    out.append(
        "The same shipped `rule_detect`, run over non-overlapping partitions of the same "
        "transactions, with the R3 hits unioned. No detection code is changed. If the "
        "in-degree-0 origin set is the binding constraint, re-cutting the identical data "
        "into shorter windows should recover chains the whole-frame run cannot reach."
    )
    out.append("")
    out.append("| Config | Windows | Flagged | True positives | Precision | Recall | New vs whole frame | Missed |")
    out.append("|---|---|---|---|---|---|---|---|")
    out.append(
        f"| Whole frame — **what ships** | 1 | {wf['entities_flagged']} | "
        f"{wf['true_positives']} | {wf['precision']:.3f} | {wf['recall']:.3f} | — | — |"
    )
    for r in cf["windowed"]:
        out.append(
            f"| {r['window_days']}-day windows | {r['windows']} | {r['entities_flagged']} | "
            f"{r['true_positives']} | {r['precision']:.3f} | {r['recall']:.3f} | "
            f"{r['new_vs_whole_frame']} | {r['missed_vs_whole_frame']} |"
        )
    out.append("")

    best = min(cf["windowed"], key=lambda r: r["window_days"])
    out.append(
        f"**The shipped R3's {wf['entities_flagged']} hits are all false positives; the "
        f"{best['window_days']}-day variant's {best['entities_flagged']} are all launderers, "
        f"and the two sets do not overlap** ({best['new_vs_whole_frame']} new, "
        f"{best['missed_vs_whole_frame']} of the originals dropped). Precision goes "
        f"{wf['precision']:.3f} → {best['precision']:.3f} on identical data. The decay across "
        "widening windows is the mechanism showing itself: as each window approaches the whole "
        "frame, precision falls back toward the shipped result."
    )
    out.append("")
    out.append(
        "**This is a measurement, not a proposed implementation.** Unioning hits across a "
        "partition is cruder than a real fix, which would evaluate chain origin within a "
        "rolling window rather than a hard partition, and would have to decide what a chain "
        "spanning a boundary means. Recall stays low throughout — R3 is one typology, not the "
        "system — so the claim is about precision only."
    )
    out.append("")

    out.append("### 5. Out-of-time degradation")
    out.append("")
    out.append("All rows: same test-window transactions, same ground truth. Only the ML half moves.")
    out.append("")
    out.append("| Arm | Flagged | HIGH | Precision | Recall | F1 | FPR |")
    out.append("|---|---|---|---|---|---|---|")
    for r in payload["arms"]:
        out.append(
            f"| {r['arm']} | {r['flagged']} | {r['high']} | {r['precision']:.3f} | "
            f"{r['recall']:.3f} | {r['f1']:.3f} | {r['fpr']:.3f} |"
        )
    out.append("")

    d = payload["drop"]
    out.append(
        f"**Headline:** going from the shipped transductive fit to a fully frozen model "
        f"moves recall {d['in_time_recall']:.3f} → {d['frozen_model_recall']:.3f} "
        f"({d['recall_delta']:+.3f}) and precision {d['in_time_precision']:.3f} → "
        f"{d['frozen_model_precision']:.3f} ({d['precision_delta']:+.3f})."
    )
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build(train_days: int = DEFAULT_TRAIN_DAYS, source: str = "synthetic") -> dict[str, Any]:
    df, customers = load_canonical(source=source)
    train_df, test_df, boundary = split_by_time(df, train_days=train_days)

    roster = set(pd.read_csv(DEFAULT_CUST_CSV)["customer_id"])

    def _active(frame: pd.DataFrame) -> set[str]:
        return (
            set(frame["sender_id"].astype(str)) | set(frame["receiver_id"].astype(str))
        ) & roster

    train_active, test_active = _active(train_df), _active(test_df)

    train_art = window_artifacts(train_df, customers)
    test_art = window_artifacts(test_df, customers)
    full_art = window_artifacts(df, customers)

    # The population is the customers active in the test window, not the whole
    # roster. Scoring against 270 customers when only a fraction transacted in
    # those 30 days would pad the true negatives with people the system was
    # never shown and never had the chance to flag.
    gt = ground_truth_from_frames(test_df, test_active)

    fit_feat, score_feat = train_art["features"], test_art["features"]
    flist = test_art["feature_list"]
    rule_hits = test_art["rule_hits"]

    arms = [
        _row("Rules only (no ML — the floor)", rule_hits, [], gt),
        _row(
            "A. in_time — fit on test, rank in test (shipped)",
            rule_hits,
            ml_arm(score_feat, score_feat, flist, freeze_distribution=False),
            gt,
        ),
        _row(
            "B. frozen_fit — fit on train, rank in test",
            rule_hits,
            ml_arm(fit_feat, score_feat, flist, freeze_distribution=False),
            gt,
        ),
        _row(
            "C. frozen_model — fit on train, rank against train",
            rule_hits,
            ml_arm(fit_feat, score_feat, flist, freeze_distribution=True),
            gt,
        ),
        _row(
            "control: A, IsolationForest only",
            rule_hits,
            ml_arm(score_feat, score_feat, flist, freeze_distribution=False, use_lof=False),
            gt,
        ),
        _row(
            "control: C, IsolationForest only",
            rule_hits,
            ml_arm(fit_feat, score_feat, flist, freeze_distribution=True, use_lof=False),
            gt,
        ),
    ]

    by_arm = {r["arm"]: r for r in arms}
    a = by_arm["A. in_time — fit on test, rank in test (shipped)"]
    c = by_arm["C. frozen_model — fit on train, rank against train"]

    span_days = int((pd.to_datetime(df["timestamp"]).max()
                     - pd.to_datetime(df["timestamp"]).min()).days)

    return {
        "_comment": (
            "Regenerate with: python -m evaluation.out_of_time — everything "
            "outside 'run_metadata' is deterministic and safe to diff."
        ),
        "run_metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "primary_definition": PRIMARY_DEFINITION,
            "source": source,
        },
        "windows": {
            "boundary": str(boundary),
            "span_days": span_days,
            "train": {
                "days": train_days,
                "transactions": int(len(train_df)),
                "customers": len(train_active),
                "scored_entities": int(len(fit_feat)),
            },
            "test": {
                "days": span_days - train_days,
                "transactions": int(len(test_df)),
                "customers": len(test_active),
                "scored_entities": int(len(score_feat)),
                "unseen_customers": len(test_active - train_active),
                "unseen_share": len(test_active - train_active) / max(len(test_active), 1),
                "positives": {d: len(gt.positives(d)) for d in DEFINITIONS},
                "labelled_txns": gt.labelled_txn_count,
            },
        },
        "rule_reachability": rule_reachability(
            train_art["rule_hits"], rule_hits, full_art["rule_hits"]
        ),
        "layering_search_space": layering_search_space({
            f"test {span_days - train_days}d": test_df,
            f"train {train_days}d": train_df,
            f"full {span_days}d": df,
        }),
        # Scored against the WHOLE dataset's ground truth over the full roster,
        # not the test window's — this asks what R3 would find across all 90
        # days, so restricting it to the out-of-time population would be the
        # wrong denominator.
        "r3_counterfactual": r3_counterfactual(
            df, customers, ground_truth_from_frames(df, roster)
        ),
        "arms": arms,
        "drop": {
            "in_time_recall": a["recall"],
            "frozen_model_recall": c["recall"],
            "recall_delta": c["recall"] - a["recall"],
            "in_time_precision": a["precision"],
            "frozen_model_precision": c["precision"],
            "precision_delta": c["precision"] - a["precision"],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit the ML half on the first 60 days, score the last 30, report the drop.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--train-days", type=int, default=DEFAULT_TRAIN_DAYS)
    parser.add_argument("--source", default="synthetic")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    print("Splitting the dataset and running each window...", file=sys.stderr)
    payload = build(train_days=args.train_days, source=args.source)

    print(render(payload))

    if not args.no_write:
        out = args.output.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        try:
            shown = out.relative_to(_REPO_ROOT).as_posix()
        except ValueError:
            shown = str(out)
        print(f"Wrote {shown}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
