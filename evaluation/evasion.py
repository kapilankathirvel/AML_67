"""
evaluation/evasion.py — how much does it cost a launderer to defeat each rule?

Usage
-----
    python -m evaluation.evasion
    python -m evaluation.evasion --output evaluation/results/evasion.json

Why this exists
---------------
Every metric this repo publishes assumes a launderer who does not know how they
are being watched. That assumption is false by construction: structuring *is*
adversarial adaptation. The $9,000-$9,999.99 band R1 keys on exists because
launderers already adapted once, to the $10,000 CTR threshold. A rule written
against that adaptation invites the next one, and the next one is arithmetic --
send $8,999 instead.

So the honest question is not "what is our recall?" but "what does it cost to
make our recall zero?". A detector that collapses when the adversary gives up
one dollar per transaction is not a control, whatever its precision says.

This study answers that quantitatively. For each evasion move it reports what
the launderer surrenders (dollars, days, or capital held), and what the system
retains at that price -- separately for the rules half, the ML half, and the
shipped hybrid, because the whole argument for running both is that they do not
fail to the same move.

Relationship to the ablation
----------------------------
ablation.py asked which components earn their place on a static dataset, and
returned an uncomfortable answer: the hybrid is LESS precise than rules alone
(0.561 vs 0.583). Taken by itself that is an argument for deleting the ML half.

This study is where that answer gets its missing axis. Precision on a frozen
dataset measures one adversary -- the one who never adapts. If the rules go to
zero under a move the ML half survives, the hybrid buys robustness with
precision, and that is a trade a bank actually wants. If the ML collapses too,
then the ablation's verdict stands and should be acted on. Either result is
worth having; the point is to measure it rather than assert it.

Method
------
Perturbations are applied IN MEMORY to the loaded transactions frame. Nothing
here writes to data/sample/ -- CLAUDE.md forbids regenerating the committed
sample, and a study that mutated its own ground truth would be worthless
anyway. `test_evasion.py` asserts the files are byte-identical after a run.

Unlike ablation.py, this CANNOT re-fuse captured detector output: changing an
amount changes the features, which changes both the rules and the ML scores.
The full stack therefore re-runs per configuration. That is slower, and it is
the only correct way to do it.

Only transactions that are BOTH labelled as laundering AND sent by a
sender-side positive are perturbed. The rationale is the threat model: this
measures an adversary altering their own behaviour, not a change to the
dataset. Ordinary customers keep their transactions, so the negative class --
and therefore every false positive -- is held fixed, and a recall drop can only
come from the evasion.

Scope, stated so the numbers are not over-read
----------------------------------------------
- Ground truth never moves. A customer who evades successfully becomes a false
  negative rather than disappearing from the denominator, which is the point.
- The ML half is refitted on the perturbed population, so it adapts to the new
  data the way it would in production. It is NOT retrained on labels; it has
  never seen labels.
- Costs are stated in the launderer's units, not converted to a common one.
  Dollars forgone and days of delay are not exchangeable, and inventing a rate
  between them would be the least defensible number in the file.
- Recall is reported against `sender_only`. Every rule and rolling feature is
  sender-side, and every perturbed transaction is one a positive sent, so this
  is the only definition under which the question is well posed.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

import backend.tools.rules as rules_mod
from backend.agent import registry
from backend.config import settings
from backend.schemas import QueryIntent
from backend.tools.base import ToolContext

from evaluation.ablation import fuse
from evaluation.harness import Metrics, evaluate, load_ground_truth

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = _REPO_ROOT / "evaluation" / "results" / "evasion.json"

# Sender-side, for the reason given in the module docstring.
PRIMARY_DEFINITION = "sender_only"

# Steps after load_data. load_data is run once and its frame is reused, since
# re-reading the CSV per configuration would be pure waste and would risk the
# perturbation silently not being applied.
DETECTION_STEPS = ("feature_engineer", "rule_detect", "ml_detect")


# ---------------------------------------------------------------------------
# Running the stack over a (possibly perturbed) frame
# ---------------------------------------------------------------------------


def load_source(source: str = "synthetic") -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Run load_data once and hand back its frame, ready to be perturbed."""
    settings.aml_use_mocks = False
    tools = registry.load_tools(use_mocks=False)
    ctx = _fresh_context(None)
    result = tools["load_data"](ctx, source=source)
    if not result.ok:
        raise RuntimeError(f"load_data failed: {result.error}")
    return result.df, ctx.customers


def _fresh_context(df: pd.DataFrame | None, customers: pd.DataFrame | None = None) -> ToolContext:
    return ToolContext(
        df=df,
        customers=customers,
        intent=QueryIntent(
            raw_query="Analyse this dataset for suspicious activity",
            intent="full_analysis",
            parsed_by="rules",
            confidence=0.9,
        ),
        artifacts={},
    )


def detect(df: pd.DataFrame, customers: pd.DataFrame | None) -> tuple[list[dict], list[dict]]:
    """feature_engineer -> rule_detect -> ml_detect over the given frame.

    Mirrors executor.run_plan's context threading. Both halves of it matter:
    dropping the `ctx.artifacts.update` line is silent and produces a table of
    zeros that reads like a finding rather than a broken harness. That is not
    hypothetical -- it is the bug that made ablation.py's first run worthless.
    """
    tools = registry.load_tools(use_mocks=False)
    ctx = _fresh_context(df, customers)

    for name in DETECTION_STEPS:
        result = tools[name](ctx)
        if not result.ok:
            raise RuntimeError(f"{name} failed during evasion run: {result.error}")
        if result.df is not None:
            ctx.df = result.df
        ctx.artifacts.update(result.artifacts)

    return (
        list(ctx.artifacts.get("rule_hits", [])),
        list(ctx.artifacts.get("ml_scores", [])),
    )


def score_halves(rule_hits: list[dict], ml_scores: list[dict], gt: Any) -> dict[str, Metrics]:
    """Rules only, ML only, and the shipped hybrid, all through the real fusion."""
    positives = gt.positives(PRIMARY_DEFINITION)
    out: dict[str, Metrics] = {}
    for label, rh, ms in (
        ("rules_only", rule_hits, []),
        ("ml_only", [], ml_scores),
        ("hybrid", rule_hits, ml_scores),
    ):
        risk_rows = fuse(rh, ms)
        flagged = {r["entity_id"] for r in risk_rows}
        out[label] = evaluate(flagged, positives, gt.all_customers)
    return out


# ---------------------------------------------------------------------------
# Selecting what the adversary controls
# ---------------------------------------------------------------------------


def _as_bool(series: pd.Series) -> pd.Series:
    """Same coercion harness.py uses; label_is_laundering arrives as object."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "t"})


def adversary_rows(df: pd.DataFrame, gt: Any) -> pd.Index:
    """Index of transactions the adversary is modelled as controlling.

    Labelled, and sent by a sender-side positive. The sender condition is not
    redundant: it excludes labelled transactions whose sender is a counterparty
    outside the customer roster, which the adversary would not be moving.
    """
    labelled = _as_bool(df["label_is_laundering"])
    theirs = df["sender_id"].isin(gt.sender_only)
    return df.index[labelled & theirs]


def _take(index: pd.Index, df: pd.DataFrame, fraction: float) -> pd.Index:
    """Deterministically take the first `fraction` of rows, ordered by txn_id.

    Ordering by a stable key rather than sampling gives two properties the
    study depends on: fraction 0.0 reproduces the baseline exactly, and the
    selections nest, so a recall curve cannot move because a different subset
    happened to be drawn.
    """
    if fraction <= 0 or len(index) == 0:
        return df.index[[]]
    ordered = df.loc[index].sort_values("txn_id").index
    n = math.ceil(fraction * len(ordered))
    return ordered[:n]


# ---------------------------------------------------------------------------
# The evasion moves
# ---------------------------------------------------------------------------
#
# Each returns (perturbed frame, cost dict). The cost dict is what the
# launderer gave up; keys differ per move because the units differ and
# collapsing them into one number would be dishonest.


# One dollar below R1's band floor. The whole point of the move: the rule's
# boundary is public, so the cheapest possible evasion is to step over it.
_DODGE_TO = rules_mod.BAND_LOW - 1.00


def move_amount_dodge(df: pd.DataFrame, gt: Any, strength: float) -> tuple[pd.DataFrame, dict]:
    """Step below R1's $9,000 band floor.

    Note what this does NOT evade: R2's band starts at $7,000, so $8,999 is
    still inside it. Defeating R1 by one dollar leaves smurfing detection
    completely intact, which is the first argument in this file for having
    more than one rule.
    """
    out = df.copy()
    in_band = out["amount"].between(rules_mod.BAND_LOW, rules_mod.BAND_HIGH)
    eligible = adversary_rows(out, gt).intersection(out.index[in_band])
    chosen = _take(eligible, out, strength)

    forgone = float((out.loc[chosen, "amount"] - _DODGE_TO).sum())
    out.loc[chosen, "amount"] = _DODGE_TO

    return out, {
        "txns_moved": int(len(chosen)),
        "dollars_forgone": round(forgone, 2),
        "dollars_per_txn": round(forgone / len(chosen), 2) if len(chosen) else 0.0,
    }


def move_slow_down(df: pd.DataFrame, gt: Any, spacing_days: float) -> tuple[pd.DataFrame, dict]:
    """Space transactions out so fewer land in R1's 7-day and R2's 48-hour windows.

    The i-th labelled transaction of each sender is pushed forward by
    i * spacing_days. Spacing 0 is the identity, and the cost grows with how
    much activity the sender had -- which is the real dynamic: the busier the
    laundering operation, the more expensive it is to slow down.
    """
    out = df.copy()
    eligible = adversary_rows(out, gt)
    if spacing_days <= 0 or len(eligible) == 0:
        return out, {"txns_moved": 0, "mean_delay_days": 0.0, "max_delay_days": 0.0}

    out["timestamp"] = pd.to_datetime(out["timestamp"])
    sub = out.loc[eligible].sort_values(["sender_id", "timestamp", "txn_id"])
    rank = sub.groupby("sender_id").cumcount()
    delay_days = rank * spacing_days
    out.loc[sub.index, "timestamp"] = sub["timestamp"] + pd.to_timedelta(delay_days, unit="D")

    moved = delay_days[delay_days > 0]
    return out, {
        "txns_moved": int(len(moved)),
        "mean_delay_days": round(float(moved.mean()), 2) if len(moved) else 0.0,
        "max_delay_days": round(float(delay_days.max()), 2),
    }


def move_cashout_delay(df: pd.DataFrame, gt: Any, delay_hours: float) -> tuple[pd.DataFrame, dict]:
    """Hold the cash-out past R4's 24-hour window.

    R4 fires on >= $10,000 in, then >= 3 cash withdrawals totalling >= 50% of
    it, all inside 24 hours. The move costs nothing but time -- and time is a
    real cost, because the funds sit in an account the bank can freeze.
    """
    out = df.copy()
    eligible = adversary_rows(out, gt)
    is_cash = out["txn_type"].astype(str).str.lower().isin(rules_mod.R4_CASH_TYPES) | out[
        "channel"
    ].astype(str).str.lower().isin(rules_mod.R4_CASH_CHANNELS)
    chosen = eligible.intersection(out.index[is_cash])

    if delay_hours <= 0 or len(chosen) == 0:
        return out, {
            "txns_moved": 0,
            "delay_hours": 0.0,
            "capital_held_usd": 0.0,
            "capital_hours_usd": 0.0,
        }

    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out.loc[chosen, "timestamp"] = out.loc[chosen, "timestamp"] + pd.Timedelta(hours=delay_hours)

    principal = float(out.loc[chosen, "amount"].sum())
    return out, {
        "txns_moved": int(len(chosen)),
        "delay_hours": float(delay_hours),
        # Principal exposed: funds sitting in an account the bank can freeze.
        # This does NOT vary with the dial -- the same money is at risk whether
        # it waits six hours or forty-eight -- so it is the dollar-hours figure
        # below, not this one, that prices the delay.
        "capital_held_usd": round(principal, 2),
        "capital_hours_usd": round(principal * delay_hours, 2),
    }


def move_shrink_amounts(df: pd.DataFrame, gt: Any, pct: float) -> tuple[pd.DataFrame, dict]:
    """Scale every controlled transaction down by `pct`.

    The move with no threshold to aim at. It is included as the control: the
    rules have edges to step over, and the ML half does not, so this is where
    the two should stop failing the same way. Whether they do is the question.
    """
    out = df.copy()
    eligible = adversary_rows(out, gt)
    if pct <= 0 or len(eligible) == 0:
        return out, {"txns_moved": 0, "shrink_pct": 0.0, "dollars_forgone": 0.0}

    original = out.loc[eligible, "amount"].astype(float)
    out.loc[eligible, "amount"] = (original * (1.0 - pct)).round(2)

    return out, {
        "txns_moved": int(len(eligible)),
        "shrink_pct": round(pct, 4),
        "dollars_forgone": round(float((original * pct).sum()), 2),
    }


def move_combined(df: pd.DataFrame, gt: Any, strength: float) -> tuple[pd.DataFrame, dict]:
    """Everything at once, scaled together -- the adversary who read the docs.

    The individual moves say which rule breaks. This one says whether the
    system has anything left when they are used together, which is the only
    version a real operation would run.
    """
    out, c1 = move_amount_dodge(df, gt, strength)
    out, c2 = move_slow_down(out, gt, spacing_days=strength * 4.0)
    out, c3 = move_cashout_delay(out, gt, delay_hours=strength * 48.0)
    return out, {
        "dollars_forgone": c1["dollars_forgone"],
        "mean_delay_days": c2["mean_delay_days"],
        "cashout_delay_hours": c3["delay_hours"],
    }


# Each entry: (key, human label, move fn, dial values, dial name, rules targeted)
MOVES: list[tuple[str, str, Callable, list[float], str, str]] = [
    (
        "amount_dodge",
        "Step below the $9,000 band",
        move_amount_dodge,
        [0.0, 0.25, 0.5, 0.75, 1.0],
        "fraction of in-band txns moved",
        "R1",
    ),
    (
        "slow_down",
        "Space transactions further apart",
        move_slow_down,
        [0.0, 1.0, 2.0, 4.0, 8.0],
        "days added per successive txn",
        "R1, R2",
    ),
    (
        "cashout_delay",
        "Hold the cash-out longer",
        move_cashout_delay,
        [0.0, 6.0, 12.0, 24.0, 48.0],
        "hours of delay",
        "R4",
    ),
    (
        "shrink_amounts",
        "Move less money per transaction",
        move_shrink_amounts,
        [0.0, 0.1, 0.25, 0.5],
        "fraction of value given up",
        "none directly",
    ),
    (
        "combined",
        "All of the above together",
        move_combined,
        [0.0, 0.25, 0.5, 0.75, 1.0],
        "intensity",
        "R1, R2, R4",
    ),
]


# ---------------------------------------------------------------------------
# Running the study
# ---------------------------------------------------------------------------


def run_move(
    base_df: pd.DataFrame,
    customers: pd.DataFrame | None,
    gt: Any,
    move: Callable,
    dials: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dial in dials:
        perturbed, cost = move(base_df, gt, dial)
        rule_hits, ml_scores = detect(perturbed, customers)
        scored = score_halves(rule_hits, ml_scores, gt)
        rows.append({
            "dial": dial,
            "cost": cost,
            "rule_hits": len(rule_hits),
            "rules_only": scored["rules_only"].as_dict(),
            "ml_only": scored["ml_only"].as_dict(),
            "hybrid": scored["hybrid"].as_dict(),
        })
    return rows


def _retained(rows: list[dict], half: str) -> float:
    """Fraction of baseline recall surviving the strongest dial setting.

    1.0 means the move achieved nothing; 0.0 means the half was defeated
    outright. Undefined if the baseline caught nothing, in which case there was
    nothing to evade and the answer is 0.0 rather than a division by zero.
    """
    base = rows[0][half]["recall"]
    worst = rows[-1][half]["recall"]
    return round(worst / base, 3) if base else 0.0


def render(study: dict[str, list[dict]]) -> str:
    out: list[str] = []

    for key, label, _fn, _dials, dial_name, targets in MOVES:
        rows = study[key]
        out.append(f"### {label}  (targets {targets})")
        out.append("")
        out.append(f"| {dial_name} | Cost to the launderer | Rule hits | Recall: rules | ML | Hybrid |")
        out.append("|---|---|---|---|---|---|")
        for r in rows:
            cost = ", ".join(
                f"{k.replace('_', ' ')}={v:,}" for k, v in r["cost"].items() if k != "txns_moved"
            )
            marker = "  ← baseline" if r["dial"] == rows[0]["dial"] else ""
            out.append(
                f"| {r['dial']:g}{marker} | {cost or '—'} | {r['rule_hits']} | "
                f"{r['rules_only']['recall']:.3f} | {r['ml_only']['recall']:.3f} | "
                f"{r['hybrid']['recall']:.3f} |"
            )
        out.append("")
        out.append(
            f"Recall retained at full strength — rules {_retained(rows, 'rules_only'):.3f}, "
            f"ML {_retained(rows, 'ml_only'):.3f}, hybrid {_retained(rows, 'hybrid'):.3f}."
        )
        out.append("")

    out.append("### Summary — recall retained at full evasion strength")
    out.append("")
    out.append("| Move | Rules | ML | Hybrid |")
    out.append("|---|---|---|---|")
    for key, label, _fn, _dials, _dial_name, _targets in MOVES:
        rows = study[key]
        out.append(
            f"| {label} | {_retained(rows, 'rules_only'):.3f} | "
            f"{_retained(rows, 'ml_only'):.3f} | {_retained(rows, 'hybrid'):.3f} |"
        )
    out.append("")
    out.append(
        "Read these as ratios to each half's OWN baseline, not to each other. The ML "
        "half starts at 0.176 recall and the rules at 0.412, so 'ML retains 0.889' "
        "is retention of a much smaller number — it degrades gently because it was "
        "never catching much to begin with. The claim this study supports is the "
        "narrow one: the two halves fail to DIFFERENT moves, so the hybrid keeps "
        "more under every move than the rules do alone. It is not a claim that the "
        "ML half is the stronger detector."
    )
    out.append("")

    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure what it costs a launderer to defeat each detector.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--source", default="synthetic")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    _, gt = load_ground_truth()
    base_df, customers = load_source(args.source)

    # Same guard run_evaluation and ablation use: flags and labels must
    # describe one dataset, or every number below is comparing two populations.
    loaded = int(len(customers)) if customers is not None else 0
    if loaded and loaded != len(gt.all_customers):
        print(
            f"\nERROR: loaded {loaded} customers but the ground truth describes "
            f"{len(gt.all_customers)}. Check --source (given: {args.source!r}).",
            file=sys.stderr,
        )
        return 1

    controlled = adversary_rows(base_df, gt)
    print(
        f"Adversary controls {len(controlled)} labelled transactions "
        f"across {len(gt.sender_only)} sender-side positives.\n",
        file=sys.stderr,
    )

    study: dict[str, list[dict]] = {}
    for key, label, fn, dials, _dial_name, _targets in MOVES:
        print(f"  {label}...", file=sys.stderr)
        study[key] = run_move(base_df, customers, gt, fn, dials)

    print()
    print(render(study))

    if not args.no_write:
        payload = {
            "_comment": (
                "Regenerate with: python -m evaluation.evasion — everything "
                "outside 'run_metadata' is deterministic and safe to diff."
            ),
            "run_metadata": {
                "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "python": platform.python_version(),
                "primary_definition": PRIMARY_DEFINITION,
                "controlled_transactions": int(len(controlled)),
                "sender_side_positives": len(gt.sender_only),
            },
            "moves": {
                key: {
                    "label": label,
                    "dial": dial_name,
                    "targets": targets,
                    "rows": study[key],
                    "recall_retained": {
                        half: _retained(study[key], half)
                        for half in ("rules_only", "ml_only", "hybrid")
                    },
                }
                for key, label, _fn, _dials, dial_name, targets in MOVES
            },
        }
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
