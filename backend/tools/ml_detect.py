"""
Track B — ml_detect.py

Tool name  : ml_detect  (Contract 2 fixed list)
Input      : ctx.df                      — canonical transactions DataFrame
             ctx.artifacts["features"]  — DataFrame indexed by customer_id
                                          (from feature_engineer; required)
             pattern_types              — list[str] | None; passed through to
                                          select the right feature columns

Output     : ToolResult.artifacts["ml_scores"]
               list[{entity_id, score, percentile, top_features: list[str]}]

Design decisions (WORKPLAN.md H22-H28):

  Primary model  : IsolationForest (contamination=0.05)
    - Unsupervised; no labels needed for production use
    - Naturally suited to high-dimensional, mixed-scale AML feature spaces
    - sklearn default n_estimators=100, random_state=42 for reproducibility

  Secondary model: LocalOutlierFactor (n_neighbors=20)
    - Complements IF: catches local density anomalies IF misses
    - Fused score = 0.6 * IF_score + 0.4 * LOF_score (both percentile-ranked)
    - LOF is only run if n_samples >= LOF_MIN_SAMPLES (avoid degenerate KNN)

  Feature selection for ML:
    - Use the columns from ctx.artifacts["feature_list"] that are present in
      the features DataFrame (exclude metadata like zscore_n_samples)
    - Drop columns with zero variance (constant features add noise)
    - StandardScaler applied before both models

  Explainability (cheap, no SHAP):
    - For each flagged entity, top-3 contributing features by
      |value - column_median| / column_std (deviation from peer median)
    - This is what Contract 2 specifies: top_features: list[str]

  Reference population (what a percentile is measured against):
    - Both models are fitted and ranked on ctx.artifacts["features_reference"] —
      the unfiltered customer set — and the resulting percentile is looked up for
      whichever entities the query asked about.
    - Percentiles used to be ranked inside the query's filtered cohort, which made
      a customer's risk score depend on the analyst's filters: adding
      amount_min=5000 to a structuring search moved percentiles by up to 0.73 and
      pushed four customers across a risk band. A SAR-escalation threshold cannot
      float with the query, so the peer group is now fixed.
    - Trade-off, accepted deliberately: the ML term is now blind to the query
      window. A customer who is unremarkable over the full dataset but spikes
      inside a 30-day filter no longer stands out on the ML half. The rules still
      run on the filtered frame and still catch them.
    - When no filter narrowed the frame, features_reference is the same object as
      features, so full_analysis behaviour is unchanged.

  Sample size guard:
    - IF needs >= IF_MIN_SAMPLES. LOF needs >= LOF_MIN_SAMPLES.
    - Both are measured against the reference population, since that is what is
      fitted — a query scoped to five customers still gets real percentiles.
    - Below thresholds: return ok=True with empty ml_scores and a note.
    - Never crash; comply with the "never raise for expected condition" rule.

No tool may import from backend.agent.* or from another tool.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from backend.tools.base import ToolContext, ToolResult, tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IF_CONTAMINATION   = 0.05    # expected fraction of anomalies
IF_N_ESTIMATORS    = 100
IF_RANDOM_STATE    = 42
LOF_N_NEIGHBORS    = 20
LOF_MIN_SAMPLES    = 30      # LOF is unstable below this
IF_MIN_SAMPLES     = 10      # absolute floor for IF
IF_WEIGHT          = 0.60    # fused score weights
LOF_WEIGHT         = 0.40
TOP_N_FEATURES     = 3       # features reported per entity

# Metadata columns that are NOT ML features
_META_COLS = {"zscore_n_samples"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _select_feature_cols(
    feat_df: pd.DataFrame,
    feature_list: list[str],
) -> list[str]:
    """Return usable ML columns: in feature_list, present in df, non-zero variance.

    Excludes metadata columns (zscore_n_samples) and constant columns.
    """
    candidates = [
        c for c in feature_list
        if c in feat_df.columns and c not in _META_COLS
    ]
    # Drop zero-variance columns
    keep = [c for c in candidates if feat_df[c].std() > 0]
    return keep


def _top_features(
    entity_values: pd.Series,
    col_medians: pd.Series,
    col_stds: pd.Series,
    feature_cols: list[str],
    n: int = TOP_N_FEATURES,
) -> list[str]:
    """Return the top-n features by |value - median| / std for a single entity."""
    scores: dict[str, float] = {}
    for col in feature_cols:
        if col_stds[col] > 0:
            scores[col] = abs(entity_values[col] - col_medians[col]) / col_stds[col]
        else:
            scores[col] = 0.0
    return sorted(scores, key=lambda c: scores[c], reverse=True)[:n]


def _percentile_rank(scores: np.ndarray) -> np.ndarray:
    """Convert raw anomaly scores to percentile ranks in [0, 1].

    Higher percentile = more anomalous.
    """
    n = len(scores)
    if n == 0:
        return scores
    # argsort twice gives rank; divide by (n-1) to get [0,1]
    ranks = scores.argsort().argsort()
    return ranks / max(n - 1, 1)


# ---------------------------------------------------------------------------
# Registered tool
# ---------------------------------------------------------------------------


@tool(
    name="ml_detect",
    params={
        "pattern_types": (
            "list[str] | None — passed from the planner to scope feature selection. "
            "If None, all features are used."
        ),
        "min_samples": (
            "int | None — override the minimum sample threshold for ML models. "
            "Defaults to IF_MIN_SAMPLES (10)."
        ),
    },
    description=(
        "Unsupervised anomaly detection via IsolationForest + LocalOutlierFactor. "
        "Reads ctx.artifacts['features'] (must run feature_engineer first). "
        "Emits artifacts['ml_scores']: list[{entity_id, score, percentile, top_features}]."
    ),
)
def ml_detect(
    ctx: ToolContext,
    pattern_types: Optional[list[str]] = None,
    min_samples: Optional[int] = None,
    **kw,
) -> ToolResult:
    """Run unsupervised ML anomaly detection on per-customer features.

    Parameters
    ----------
    ctx          : ToolContext — features in ctx.artifacts["features"]
    pattern_types: list of AML patterns (passed through for logging; does not
                   change which features are used — that was decided by feature_engineer)
    min_samples  : override minimum sample count for models
    """
    try:
        feat_df: pd.DataFrame = ctx.artifacts.get("features", pd.DataFrame())
        feature_list: list[str] = ctx.artifacts.get("feature_list", [])

        if feat_df is None or len(feat_df) == 0:
            return ToolResult(
                ok=True,
                artifacts={"ml_scores": []},
                metrics={"ml_entities_scored": 0},
                notes=["ml_detect: no features available — run feature_engineer first"],
            )

        # The models are fitted and ranked on the *reference* population, not on the
        # query's cohort. feature_engineer emits features_reference as the unfiltered
        # customer set; when no filter narrowed the frame it is the same object, so
        # this is a no-op for full_analysis. Ranking inside the filtered cohort used
        # to make a customer's percentile a function of the analyst's filters rather
        # than of their own behaviour — an amount_min=5000 filter shifted percentiles
        # by up to 0.73 and moved four customers across a risk band.
        ref_df: pd.DataFrame = ctx.artifacts.get("features_reference")
        if ref_df is None or len(ref_df) == 0:
            ref_df = feat_df

        floor = min_samples if min_samples is not None else IF_MIN_SAMPLES
        n = len(ref_df)

        if n < floor:
            return ToolResult(
                ok=True,
                artifacts={"ml_scores": []},
                metrics={"ml_entities_scored": 0},
                notes=[
                    f"ml_detect: only {n} entities — below minimum {floor} for "
                    f"reliable anomaly detection; ml_scores is empty"
                ],
            )

        # ------------------------------------------------------------------
        # Feature matrix
        # ------------------------------------------------------------------
        # Column selection follows the reference frame, since that is what gets fitted.
        # Both frames come out of the same feature_engineer call with the same
        # pattern_types, so their columns are identical by construction.
        feature_cols = _select_feature_cols(ref_df, feature_list)

        if not feature_cols:
            # Fallback: use all numeric non-meta columns
            feature_cols = [
                c for c in ref_df.select_dtypes(include=[np.number]).columns
                if c not in _META_COLS and ref_df[c].std() > 0
            ]

        if not feature_cols:
            return ToolResult(
                ok=True,
                artifacts={"ml_scores": []},
                metrics={"ml_entities_scored": 0},
                notes=["ml_detect: no usable numeric features found after variance filter"],
            )

        X_raw = ref_df[feature_cols].fillna(0.0).values.astype(float)

        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        # ------------------------------------------------------------------
        # IsolationForest (primary)
        # ------------------------------------------------------------------
        iso = IsolationForest(
            n_estimators=IF_N_ESTIMATORS,
            contamination=IF_CONTAMINATION,
            random_state=IF_RANDOM_STATE,
        )
        iso.fit(X)
        # decision_function: lower = more anomalous; negate so higher = more anomalous
        if_raw = -iso.decision_function(X)
        if_pct = _percentile_rank(if_raw)

        # ------------------------------------------------------------------
        # LocalOutlierFactor (secondary) — only if enough samples
        # ------------------------------------------------------------------
        use_lof = n >= LOF_MIN_SAMPLES
        if use_lof:
            lof = LocalOutlierFactor(
                n_neighbors=min(LOF_N_NEIGHBORS, n - 1),
                novelty=False,
            )
            lof.fit(X)
            # negative_outlier_factor_: more negative = more anomalous; negate
            lof_raw = -lof.negative_outlier_factor_
            lof_pct = _percentile_rank(lof_raw)
            fused_pct = IF_WEIGHT * if_pct + LOF_WEIGHT * lof_pct
        else:
            lof_pct = np.zeros(n)
            fused_pct = if_pct  # 100% IF when LOF skipped

        # ------------------------------------------------------------------
        # Top-3 features per entity (deviation from peer median)
        # ------------------------------------------------------------------
        # "Unusual how?" is answered against the same reference peers the percentile
        # was ranked against — using the filtered cohort's medians here would explain
        # the score with a distribution it was not computed from.
        feat_sub = ref_df[feature_cols].fillna(0.0)
        col_medians = feat_sub.median()
        col_stds    = feat_sub.std().replace(0, 1.0)   # avoid divide-by-zero

        # Rank once over the reference population, then report the entities that are
        # in scope for this query.
        #
        # "In scope" is every customer appearing in the working frame as sender OR
        # receiver — not the feature frame's index, which holds senders only (see
        # features.py). Scoping to the feature index silently dropped receiver-side
        # entities: C-N0138 is an R7 (inbound structuring) hit that sends nothing
        # above $5,000, so under an amount_min=5000 filter it vanished from the
        # feature index, lost its ML score to risk.py's 0.0 default, and dropped from
        # 52.58 to 45.00 — a filter-dependent score, which is the bug being fixed.
        scored_by_entity: dict[str, dict[str, Any]] = {}
        for i, eid in enumerate(ref_df.index):
            scored_by_entity[str(eid)] = {
                "entity_id":    str(eid),
                "score":        round(float(fused_pct[i]), 4),
                "percentile":   round(float(fused_pct[i]), 4),
                "top_features": _top_features(feat_sub.iloc[i], col_medians, col_stds, feature_cols),
            }

        in_scope: set[str] = set(str(e) for e in feat_df.index)
        working = ctx.df
        if working is not None and len(working) > 0:
            for col in ("sender_id", "receiver_id"):
                if col in working.columns:
                    in_scope |= set(working[col].astype(str).unique())

        # Customers who only ever receive have no feature vector — features.py indexes
        # on senders — so they are simply unscoreable, not an anomaly. Only a *cohort*
        # entity missing from the reference would break the superset invariant, and
        # that is what gets reported.
        ml_scores = [scored_by_entity[e] for e in sorted(in_scope) if e in scored_by_entity]
        missing = sum(1 for e in feat_df.index if str(e) not in scored_by_entity)

        # Sort descending by score for easier downstream consumption
        ml_scores.sort(key=lambda r: r["score"], reverse=True)

        patterns_label = ", ".join(pattern_types) if pattern_types else "all"
        scoped = len(ref_df) != len(feat_df)
        note = (
            f"ml_detect: scored {len(ml_scores)} entities on {len(feature_cols)} features "
            f"(IF{'+ LOF' if use_lof else ' only'}), "
            f"pattern_types=[{patterns_label}]"
            + (f", ranked against the full population of {n:,} customers" if scoped else "")
        )
        notes = [note]
        if missing:
            notes.append(
                f"ml_detect: {missing} entities in the working cohort were absent from the "
                "reference population — expected zero, since every filter is a row filter"
            )

        return ToolResult(
            ok=True,
            artifacts={"ml_scores": ml_scores},
            metrics={
                "ml_entities_scored": len(ml_scores),
                "ml_reference_population": n,
                "feature_cols_used": len(feature_cols),
                "lof_used": use_lof,
            },
            notes=notes,
        )

    except Exception as exc:  # noqa: BLE001
        return ToolResult(ok=False, error=f"ml_detect failed: {exc}")
