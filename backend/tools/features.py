"""
Track B — features.py

Tool name  : feature_engineer  (Contract 2 fixed list)
Input      : ctx.df            — canonical transactions DataFrame
             ctx.artifacts["customers"] — optional; used for KYC-deviation features
             pattern_types     — list of AML patterns to compute features for;
                                 if None/empty → compute ALL features

Output     : ToolResult.artifacts["features"]     — DataFrame indexed by customer_id
             ToolResult.artifacts["feature_list"] — list[str] of computed feature names

Design decisions (documented per WORKPLAN.md H14-H22 requirement):

  Rolling windows (sender-side only):
    Computed using sender_id. Rationale: structuring, smurfing, and rapid cashout
    are sender behaviours. Inbound amounts are tracked separately only for
    rapid_cashout_ratio. See AML_LOGIC.md §5.4.

  pct_just_below_threshold band: $9,000.00 – $9,999.99 (inclusive).
    See AML_LOGIC.md §3 R1 and §1.1.

  amount_zscore_90d fallback: 0.0 when < 3 transactions in 90-day window.
    See AML_LOGIC.md §5.3.

  Night hours: UTC 22:00–05:59. Limitation: tz-naive data means non-UTC
    customers are approximated. See AML_LOGIC.md §5.1.

  round_amount: divisible by $500, remainder < $1. See AML_LOGIC.md §5.2.
    Added to both structuring and smurfing per user confirmation.

  pass_through_ratio: 48h sliding window, step=1h, formula=min(in,out)/max(in,out).
    The DEFINING layering signal. See AML_LOGIC.md §5.5.

Pattern → feature mapping (per AML_LOGIC.md §4):

  structuring:      rolling_1d_*, rolling_7d_*, rolling_30d_*,
                    pct_just_below_threshold, amount_zscore_90d, round_amount_ratio

  smurfing:         rolling_1d_*, rolling_7d_*,
                    pct_just_below_threshold, amount_zscore_90d,
                    velocity_txns_per_hour, velocity_counterparties_per_day,
                    new_counterparty_ratio, round_amount_ratio

  layering:         rolling_7d_*, rolling_30d_*,
                    amount_zscore_90d,
                    velocity_counterparties_per_day,
                    night_hours_ratio, new_counterparty_ratio,
                    cross_border_count, cross_border_ratio,
                    pass_through_ratio

  rapid_cashout:    rolling_1d_*,
                    amount_zscore_90d,
                    velocity_txns_per_hour,
                    rapid_cashout_ratio, night_hours_ratio

  velocity:         rolling_1d_*, velocity_txns_per_hour, amount_zscore_90d

  dormant_reactivation: rolling_7d_*, amount_zscore_90d
    (dormancy gap computed directly from tx timestamps, not in features table)

No tool may import from backend.agent.* or from another tool.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from backend.tools.base import ToolContext, ToolResult, tool

# ---------------------------------------------------------------------------
# Constants (from AML_LOGIC.md)
# ---------------------------------------------------------------------------

THRESHOLD_BAND_LOW  = 9_000.00
THRESHOLD_BAND_HIGH = 9_999.99
ROUND_AMOUNT_UNIT   = 500.0         # AML_LOGIC.md §5.2
NIGHT_HOURS_UTC     = set(range(22, 24)) | set(range(0, 6))  # 22,23,0,1,2,3,4,5
ZSCORE_MIN_SAMPLES  = 3             # AML_LOGIC.md §5.3
ZSCORE_WINDOW_DAYS  = 90
PASS_THROUGH_WINDOW_HOURS = 48
PASS_THROUGH_MAGNITUDE_TOL = 0.30  # ±30% of inbound amount

# ---------------------------------------------------------------------------
# Pattern → feature set mapping
# ---------------------------------------------------------------------------

_PATTERN_FEATURES: dict[str, set[str]] = {
    "structuring": {
        "rolling_1d_sum", "rolling_1d_count",
        "rolling_7d_sum", "rolling_7d_count",
        "rolling_30d_sum", "rolling_30d_count",
        "pct_just_below_threshold",
        "amount_zscore_90d",
        "round_amount_ratio",
    },
    "smurfing": {
        "rolling_1d_sum", "rolling_1d_count",
        "rolling_7d_sum", "rolling_7d_count",
        "pct_just_below_threshold",
        "amount_zscore_90d",
        "velocity_txns_per_hour",
        "velocity_counterparties_per_day",
        "new_counterparty_ratio",
        "round_amount_ratio",
    },
    "layering": {
        "rolling_7d_sum", "rolling_7d_count",
        "rolling_30d_sum", "rolling_30d_count",
        "amount_zscore_90d",
        "velocity_counterparties_per_day",
        "night_hours_ratio",
        "new_counterparty_ratio",
        "cross_border_count", "cross_border_ratio",
        "pass_through_ratio",
    },
    "rapid_cashout": {
        "rolling_1d_sum", "rolling_1d_count",
        "amount_zscore_90d",
        "velocity_txns_per_hour",
        "rapid_cashout_ratio",
        "night_hours_ratio",
    },
    "velocity": {
        "rolling_1d_sum", "rolling_1d_count",
        "velocity_txns_per_hour",
        "amount_zscore_90d",
    },
    "dormant_reactivation": {
        "rolling_7d_sum", "rolling_7d_count",
        "amount_zscore_90d",
    },
    "unknown": {
        # full set for unknown intent
        "rolling_1d_sum", "rolling_1d_count",
        "rolling_7d_sum", "rolling_7d_count",
        "rolling_30d_sum", "rolling_30d_count",
        "pct_just_below_threshold", "amount_zscore_90d",
        "velocity_txns_per_hour", "velocity_counterparties_per_day",
        "rapid_cashout_ratio", "round_amount_ratio",
        "night_hours_ratio", "new_counterparty_ratio",
        "cross_border_count", "cross_border_ratio",
        "pass_through_ratio",
    },
}

_ALL_FEATURES: set[str] = set().union(*_PATTERN_FEATURES.values())


def _requested_features(pattern_types: Optional[list[str]]) -> set[str]:
    """Return the union of features needed by the requested pattern_types.
    If pattern_types is None or empty, return ALL features.
    """
    if not pattern_types:
        return _ALL_FEATURES.copy()
    result: set[str] = set()
    for pt in pattern_types:
        result |= _PATTERN_FEATURES.get(pt, set())
    return result


# ---------------------------------------------------------------------------
# Individual feature computers
# Each returns a pd.Series indexed by customer_id (sender_id)
# ---------------------------------------------------------------------------


def _rolling_window_features(
    df: pd.DataFrame,
    want: set[str],
) -> pd.DataFrame:
    """Compute rolling 1d/7d/30d sum and count per sender_id.

    Uses the last transaction timestamp in the dataset as the reference point
    (i.e., features are computed over the window ending at dataset max timestamp).
    Sender-side only — see AML_LOGIC.md §5.4.
    """
    cols_needed = [
        ("rolling_1d_sum",   "1d",  "sum"),
        ("rolling_1d_count", "1d",  "count"),
        ("rolling_7d_sum",   "7d",  "sum"),
        ("rolling_7d_count", "7d",  "count"),
        ("rolling_30d_sum",  "30d", "sum"),
        ("rolling_30d_count","30d", "count"),
    ]
    requested = [(n, w, a) for (n, w, a) in cols_needed if n in want]
    if not requested:
        return pd.DataFrame()

    ref_time = df["timestamp"].max()
    tmp = df[["sender_id", "timestamp", "amount"]].copy()
    tmp = tmp.sort_values("timestamp")

    results: dict[str, pd.Series] = {}

    for feat_name, window_str, agg in requested:
        days = int(window_str.rstrip("d"))
        cutoff = ref_time - pd.Timedelta(days=days)
        window_df = tmp[tmp["timestamp"] >= cutoff]
        if agg == "sum":
            series = window_df.groupby("sender_id")["amount"].sum()
        else:
            series = window_df.groupby("sender_id")["amount"].count()
        results[feat_name] = series

    out = pd.DataFrame(results)
    out.index.name = "customer_id"
    return out


def _pct_just_below_threshold(df: pd.DataFrame) -> pd.Series:
    """Fraction of a customer's sent transactions in [$9000, $9999.99].

    AML_LOGIC.md §3 R1: the single most important structuring signal.
    """
    tmp = df[["sender_id", "amount"]].copy()
    in_band = (tmp["amount"] >= THRESHOLD_BAND_LOW) & (tmp["amount"] <= THRESHOLD_BAND_HIGH)
    band_count  = tmp[in_band].groupby("sender_id")["amount"].count()
    total_count = tmp.groupby("sender_id")["amount"].count()
    ratio = (band_count / total_count).fillna(0.0)
    ratio.name = "pct_just_below_threshold"
    ratio.index.name = "customer_id"
    return ratio


def _amount_zscore_90d(df: pd.DataFrame) -> pd.DataFrame:
    """Max z-score of sent amounts against each customer's own 90-day baseline.

    Baseline = all transactions within 90 days of the dataset max timestamp.
    Fallback: < 3 samples → zscore = 0.0.  See AML_LOGIC.md §5.3.
    """
    ref_time = df["timestamp"].max()
    cutoff = ref_time - pd.Timedelta(days=ZSCORE_WINDOW_DAYS)
    window = df[df["timestamp"] >= cutoff][["sender_id", "amount"]].copy()

    def _zscore_max(amounts: pd.Series) -> float:
        n = len(amounts)
        if n < ZSCORE_MIN_SAMPLES:
            return 0.0
        mean = amounts.mean()
        std = amounts.std()
        if std == 0 or np.isnan(std):
            return 0.0
        zscores = ((amounts - mean) / std).abs()
        return float(zscores.max())

    def _zscore_n(amounts: pd.Series) -> int:
        return len(amounts)

    agg = window.groupby("sender_id")["amount"].agg([_zscore_max, _zscore_n])
    agg.columns = ["amount_zscore_90d", "zscore_n_samples"]
    agg.index.name = "customer_id"
    return agg


def _velocity_features(df: pd.DataFrame, want: set[str]) -> pd.DataFrame:
    """velocity_txns_per_hour and velocity_counterparties_per_day.

    txns_per_hour:          max of (count in any 24h window / 24)
    counterparties_per_day: mean distinct receivers per calendar day
    """
    cols_out: dict[str, pd.Series] = {}
    need_tph = "velocity_txns_per_hour" in want
    need_cpd = "velocity_counterparties_per_day" in want
    if not need_tph and not need_cpd:
        return pd.DataFrame()

    tmp = df[["sender_id", "timestamp", "receiver_id"]].copy()
    tmp = tmp.sort_values(["sender_id", "timestamp"])

    if need_tph:
        # For each customer: find max transactions in any 24h window
        def _max_tph(grp: pd.DataFrame) -> float:
            ts = grp["timestamp"].values
            if len(ts) < 2:
                return len(ts) / 24.0
            ts_ns = ts.astype("int64")
            window_ns = int(24 * 3600 * 1e9)
            max_count = 1
            left = 0
            for right in range(len(ts_ns)):
                while ts_ns[right] - ts_ns[left] > window_ns:
                    left += 1
                max_count = max(max_count, right - left + 1)
            return max_count / 24.0

        tph = tmp.groupby("sender_id").apply(_max_tph, include_groups=False)
        tph.name = "velocity_txns_per_hour"
        tph.index.name = "customer_id"
        cols_out["velocity_txns_per_hour"] = tph

    if need_cpd:
        tmp2 = tmp.copy()
        tmp2["date"] = tmp2["timestamp"].dt.date
        daily = (
            tmp2.groupby(["sender_id", "date"])["receiver_id"]
            .nunique()
            .reset_index()
        )
        cpd = daily.groupby("sender_id")["receiver_id"].mean()
        cpd.name = "velocity_counterparties_per_day"
        cpd.index.name = "customer_id"
        cols_out["velocity_counterparties_per_day"] = cpd

    return pd.DataFrame(cols_out)


def _rapid_cashout_ratio(df: pd.DataFrame) -> pd.Series:
    """For each customer (as receiver), find large inbound then immediate cash/ATM outflows.

    Inbound threshold: $10,000. Window: 24 hours. See AML_LOGIC.md §3 R4.
    Ratio = total cash outflow within 24h / inbound amount. Max over all inbound events.
    """
    large_in = df[df["amount"] >= 10_000.0][["receiver_id", "timestamp", "amount"]].copy()
    large_in = large_in.rename(columns={"receiver_id": "customer_id", "amount": "inbound_amount"})

    # Cash/ATM outflows
    cash_types = {"cash"}
    cash_channels = {"atm", "branch"}
    cash_out = df[
        (df["txn_type"].isin(cash_types)) | (df["channel"].isin(cash_channels))
    ][["sender_id", "timestamp", "amount"]].copy()
    cash_out = cash_out.rename(columns={"sender_id": "customer_id", "amount": "outflow_amount"})

    if large_in.empty or cash_out.empty:
        return pd.Series(dtype=float, name="rapid_cashout_ratio")

    # For each large inbound, sum cash outflows within 24h
    window_ns = int(24 * 3600 * 1e9)
    ratios: dict[str, float] = {}

    for cid, inbound_grp in large_in.groupby("customer_id"):
        cust_cash = cash_out[cash_out["customer_id"] == cid]
        if cust_cash.empty:
            continue
        in_ts = inbound_grp["timestamp"].values.astype("int64")
        in_amt = inbound_grp["inbound_amount"].values
        out_ts = cust_cash["timestamp"].values.astype("int64")
        out_amt = cust_cash["outflow_amount"].values
        max_ratio = 0.0
        for i, (t_in, a_in) in enumerate(zip(in_ts, in_amt)):
            mask = (out_ts >= t_in) & (out_ts <= t_in + window_ns)
            total_out = out_amt[mask].sum()
            if a_in > 0:
                max_ratio = max(max_ratio, total_out / a_in)
        ratios[cid] = max_ratio

    result = pd.Series(ratios, name="rapid_cashout_ratio")
    result.index.name = "customer_id"
    return result


def _round_amount_ratio(df: pd.DataFrame) -> pd.Series:
    """Share of sent transactions with amount divisible by $500 (rem < $1).

    AML_LOGIC.md §5.2. Added to both structuring and smurfing.
    """
    tmp = df[["sender_id", "amount"]].copy()
    is_round = (tmp["amount"] % ROUND_AMOUNT_UNIT) < 1.0
    round_count = tmp[is_round].groupby("sender_id")["amount"].count()
    total_count = tmp.groupby("sender_id")["amount"].count()
    ratio = (round_count / total_count).fillna(0.0)
    ratio.name = "round_amount_ratio"
    ratio.index.name = "customer_id"
    return ratio


def _night_hours_ratio(df: pd.DataFrame) -> pd.Series:
    """Share of sent transactions in UTC night hours (22:00–05:59).

    AML_LOGIC.md §5.1. Limitation: tz-naive UTC timestamps.
    """
    tmp = df[["sender_id", "timestamp"]].copy()
    tmp["hour"] = tmp["timestamp"].dt.hour
    is_night = tmp["hour"].isin(NIGHT_HOURS_UTC)
    night_count = tmp[is_night].groupby("sender_id")["hour"].count()
    total_count = tmp.groupby("sender_id")["hour"].count()
    ratio = (night_count / total_count).fillna(0.0)
    ratio.name = "night_hours_ratio"
    ratio.index.name = "customer_id"
    return ratio


def _new_counterparty_ratio(df: pd.DataFrame) -> pd.Series:
    """Share of transactions sent to a counterparty not seen in the first 30 days.

    'New' = receiver_id not seen in the customer's own first 30 calendar days
    of sending activity. Proxy for customers rapidly expanding their counterparty
    network — a smurfing and layering signal.
    """
    tmp = df[["sender_id", "receiver_id", "timestamp"]].copy()
    tmp = tmp.sort_values(["sender_id", "timestamp"])

    ratios: dict[str, float] = {}
    for cid, grp in tmp.groupby("sender_id"):
        if len(grp) < 2:
            ratios[cid] = 0.0
            continue
        first_ts = grp["timestamp"].min()
        warmup_cutoff = first_ts + pd.Timedelta(days=30)
        warmup = grp[grp["timestamp"] <= warmup_cutoff]
        known = set(warmup["receiver_id"])
        later = grp[grp["timestamp"] > warmup_cutoff]
        if len(later) == 0:
            ratios[cid] = 0.0
            continue
        new_cp = later["receiver_id"].apply(lambda r: r not in known)
        ratios[cid] = float(new_cp.mean())

    result = pd.Series(ratios, name="new_counterparty_ratio")
    result.index.name = "customer_id"
    return result


def _cross_border_features(df: pd.DataFrame) -> pd.DataFrame:
    """cross_border_count and cross_border_ratio per sender."""
    tmp = df[["sender_id", "is_cross_border"]].copy()
    cb_count = tmp[tmp["is_cross_border"]].groupby("sender_id")["is_cross_border"].count()
    total_count = tmp.groupby("sender_id")["is_cross_border"].count()
    cb_ratio = (cb_count / total_count).fillna(0.0)

    out = pd.DataFrame({
        "cross_border_count": cb_count,
        "cross_border_ratio": cb_ratio,
    })
    out.index.name = "customer_id"
    return out


def _pass_through_ratio(df: pd.DataFrame) -> pd.Series:
    """Max 48h pass-through ratio per customer.

    For each customer C, finds the 48h window where pass-through is highest:
        ratio = min(total_received_48h, total_sent_48h) / max(total_received_48h, total_sent_48h)
    Maximum ratio across all 48h windows is the feature value.

    Implementation: vectorised using pandas rolling on per-customer hourly-resampled
    time series. Avoids a Python-level while loop for performance.
    Window step = 1 hour. See AML_LOGIC.md §5.5.
    """
    WINDOW_H = PASS_THROUGH_WINDOW_HOURS

    sent     = df[["sender_id",   "timestamp", "amount"]].rename(
        columns={"sender_id": "customer_id", "amount": "sent"})
    received = df[["receiver_id", "timestamp", "amount"]].rename(
        columns={"receiver_id": "customer_id", "amount": "received"})

    if df.empty:
        return pd.Series(dtype=float, name="pass_through_ratio")

    # Resample sent and received to 1h buckets per customer, then rolling-sum 48h
    def _max_ppt(s_grp: pd.Series, r_grp: pd.Series) -> float:
        """Given per-customer sorted sent/received series, return max 48h ratio."""
        # Align on 1h grid spanning both series
        ts_min = min(s_grp.index.min(), r_grp.index.min())
        ts_max = max(s_grp.index.max(), r_grp.index.max())
        idx = pd.date_range(ts_min.floor("h"), ts_max.ceil("h"), freq="1h")
        # Resample to hourly sums
        s_hr = s_grp.resample("1h").sum().reindex(idx, fill_value=0.0)
        r_hr = r_grp.resample("1h").sum().reindex(idx, fill_value=0.0)
        # Rolling sum over 48h window
        s_roll = s_hr.rolling(window=WINDOW_H, min_periods=1).sum()
        r_roll = r_hr.rolling(window=WINDOW_H, min_periods=1).sum()
        mins = s_roll.clip(lower=0).combine(r_roll.clip(lower=0), min)
        maxs = s_roll.clip(lower=0).combine(r_roll.clip(lower=0), max)
        valid = maxs > 0
        if not valid.any():
            return 0.0
        ratios = (mins[valid] / maxs[valid])
        return float(ratios.max())

    sent_by_cust     = {c: g.set_index("timestamp")["sent"]
                        for c, g in sent.groupby("customer_id")}
    received_by_cust = {c: g.set_index("timestamp")["received"]
                        for c, g in received.groupby("customer_id")}

    all_customers = set(sent_by_cust) | set(received_by_cust)
    ratios: dict[str, float] = {}

    for cid in all_customers:
        s_grp = sent_by_cust.get(cid, pd.Series(dtype=float))
        r_grp = received_by_cust.get(cid, pd.Series(dtype=float))
        if s_grp.empty or r_grp.empty:
            ratios[cid] = 0.0
            continue
        ratios[cid] = _max_ppt(s_grp, r_grp)

    result = pd.Series(ratios, name="pass_through_ratio")
    result.index.name = "customer_id"
    return result


# ---------------------------------------------------------------------------
# Registered tool
# ---------------------------------------------------------------------------


@tool(
    name="feature_engineer",
    params={
        "pattern_types": (
            "list[str] | None — which AML patterns to engineer features for. "
            "Valid values: 'structuring', 'smurfing', 'layering', 'rapid_cashout', "
            "'velocity', 'dormant_reactivation', 'unknown'. "
            "If None or empty, ALL features are computed."
        ),
    },
    description=(
        "Compute per-customer AML features from the working transaction DataFrame. "
        "Only computes features relevant to the requested pattern_types (adaptive). "
        "Returns artifacts['features'] (DataFrame indexed by customer_id) and "
        "artifacts['feature_list'] (list of computed feature names)."
    ),
)
def feature_engineer(
    ctx: ToolContext,
    pattern_types: Optional[list[str]] = None,
    **kw,
) -> ToolResult:
    """Compute per-customer AML features.

    Parameters
    ----------
    ctx          : ToolContext — working transactions in ctx.df
    pattern_types: list of AML patterns to compute features for.
                   None/empty → all features.
    """
    try:
        df = ctx.df

        if df is None or len(df) == 0:
            return ToolResult(
                ok=True,
                artifacts={"features": pd.DataFrame(), "feature_list": []},
                notes=["feature_engineer: working DataFrame is empty — no features computed"],
            )

        # Resolve which features to compute
        want = _requested_features(pattern_types)
        patterns_label = ", ".join(pattern_types) if pattern_types else "all"

        # All senders (and receivers for some features) seen in the dataset
        all_senders = df["sender_id"].unique()
        feat_df = pd.DataFrame(index=pd.Index(all_senders, name="customer_id"))

        computed: list[str] = []

        # ------------------------------------------------------------------
        # Rolling window features
        # ------------------------------------------------------------------
        rolling_want = want & {
            "rolling_1d_sum", "rolling_1d_count",
            "rolling_7d_sum", "rolling_7d_count",
            "rolling_30d_sum", "rolling_30d_count",
        }
        if rolling_want:
            rdf = _rolling_window_features(df, rolling_want)
            if not rdf.empty:
                feat_df = feat_df.join(rdf, how="left")
                computed.extend([c for c in rdf.columns if c in want])

        # ------------------------------------------------------------------
        # pct_just_below_threshold
        # ------------------------------------------------------------------
        if "pct_just_below_threshold" in want:
            s = _pct_just_below_threshold(df)
            feat_df = feat_df.join(s, how="left")
            computed.append("pct_just_below_threshold")

        # ------------------------------------------------------------------
        # amount_zscore_90d (+ zscore_n_samples as metadata)
        # ------------------------------------------------------------------
        if "amount_zscore_90d" in want:
            zdf = _amount_zscore_90d(df)
            feat_df = feat_df.join(zdf, how="left")
            computed.append("amount_zscore_90d")
            computed.append("zscore_n_samples")

        # ------------------------------------------------------------------
        # Velocity features
        # ------------------------------------------------------------------
        vel_want = want & {"velocity_txns_per_hour", "velocity_counterparties_per_day"}
        if vel_want:
            vdf = _velocity_features(df, vel_want)
            if not vdf.empty:
                feat_df = feat_df.join(vdf, how="left")
                computed.extend([c for c in vdf.columns if c in want])

        # ------------------------------------------------------------------
        # rapid_cashout_ratio
        # ------------------------------------------------------------------
        if "rapid_cashout_ratio" in want:
            s = _rapid_cashout_ratio(df)
            feat_df = feat_df.join(s, how="left")
            computed.append("rapid_cashout_ratio")

        # ------------------------------------------------------------------
        # round_amount_ratio
        # ------------------------------------------------------------------
        if "round_amount_ratio" in want:
            s = _round_amount_ratio(df)
            feat_df = feat_df.join(s, how="left")
            computed.append("round_amount_ratio")

        # ------------------------------------------------------------------
        # night_hours_ratio
        # ------------------------------------------------------------------
        if "night_hours_ratio" in want:
            s = _night_hours_ratio(df)
            feat_df = feat_df.join(s, how="left")
            computed.append("night_hours_ratio")

        # ------------------------------------------------------------------
        # new_counterparty_ratio
        # ------------------------------------------------------------------
        if "new_counterparty_ratio" in want:
            s = _new_counterparty_ratio(df)
            feat_df = feat_df.join(s, how="left")
            computed.append("new_counterparty_ratio")

        # ------------------------------------------------------------------
        # cross_border_count, cross_border_ratio
        # ------------------------------------------------------------------
        cb_want = want & {"cross_border_count", "cross_border_ratio"}
        if cb_want:
            cbdf = _cross_border_features(df)
            join_cols = [c for c in cbdf.columns if c in want]
            if join_cols:
                feat_df = feat_df.join(cbdf[join_cols], how="left")
                computed.extend(join_cols)

        # ------------------------------------------------------------------
        # pass_through_ratio (layering defining signal)
        # ------------------------------------------------------------------
        if "pass_through_ratio" in want:
            s = _pass_through_ratio(df)
            feat_df = feat_df.join(s, how="left")
            computed.append("pass_through_ratio")

        # ------------------------------------------------------------------
        # Fill NaN — missing means "not observed → zero"
        # ------------------------------------------------------------------
        feat_df = feat_df.fillna(0.0)
        feat_df.index.name = "customer_id"

        # Deduplicate computed list while preserving order
        seen: set[str] = set()
        feature_list: list[str] = []
        for f in computed:
            if f not in seen:
                seen.add(f)
                feature_list.append(f)

        note = (
            f"feature_engineer: {len(feat_df):,} customers × "
            f"{len(feature_list)} features "
            f"for pattern_types=[{patterns_label}]"
        )

        # ------------------------------------------------------------------
        # Reference features — the population ml_detect ranks percentiles against
        # ------------------------------------------------------------------
        # feature_engineer runs *after* filter_data, so feat_df above describes only
        # the customers who survived the analyst's filters. Ranking anomaly
        # percentiles inside that cohort made a customer's risk score depend on the
        # query: adding amount_min=5000 moved percentiles by up to 0.73 and pushed
        # four customers across a risk band. So we also compute the same features
        # over the unfiltered frame and hand that to ml_detect as the fixed peer
        # group.
        #
        # This re-invokes the tool rather than extracting the ~120-line computation
        # above, which guarantees both frames go through byte-identical logic. The
        # recursive call gets no "transactions_reference" artifact, so it computes
        # features once and stops — there is no second level of recursion.
        reference_df = ctx.artifacts.get("transactions_reference")
        ref_feat_df = feat_df
        if reference_df is not None and len(reference_df) > len(df):
            ref_result = feature_engineer(
                ToolContext(df=reference_df, customers=ctx.customers, intent=ctx.intent,
                            artifacts={"customers": ctx.artifacts.get("customers")}),
                pattern_types=pattern_types,
            )
            if ref_result.ok:
                ref_feat_df = ref_result.artifacts["features"]
                note += (
                    f"; anomaly peer group = {len(ref_feat_df):,} customers "
                    "(unfiltered population)"
                )

        return ToolResult(
            ok=True,
            artifacts={
                "features": feat_df,
                "feature_list": feature_list,
                "features_reference": ref_feat_df,
            },
            metrics={
                "customer_count": len(feat_df),
                "feature_count": len(feature_list),
                "reference_customer_count": len(ref_feat_df),
                "pattern_types": pattern_types or ["all"],
            },
            notes=[note],
        )

    except Exception as exc:  # noqa: BLE001
        return ToolResult(ok=False, error=f"feature_engineer failed: {exc}")
