"""
evaluation/ablation.py — turn each component off and measure what breaks.

Usage
-----
    python -m evaluation.ablation
    python -m evaluation.ablation --output evaluation/results/ablation.json

Why this exists
---------------
run_evaluation.py answers "how well does the system do?". It cannot answer
"which parts of it are doing the work?", and that is the question every
component in this repo has been assumed rather than shown to deserve. The
hybrid design, the 0.6/0.4 fusion split, the 70/40/15 bands and all seven rules
were each chosen up front and never individually measured. Some of them may not
be earning their place.

What is ablated
---------------
1. Components   — rules only, ML only, both. If rules-only matches the hybrid,
                  the ML is decoration.
2. Rules        — each rule alone (is it precise?) and each rule left out (does
                  it add anything the others do not?). Both are needed: two
                  redundant rules each look worthless under leave-one-out while
                  being valuable as a pair, and only the alone-view separates
                  that from a rule that is genuinely dead.
3. Fusion split — RULE_WEIGHT_COEFF swept 0.0 to 1.0. 0.6/0.4 is currently a
                  documented constant with no measurement behind it.
4. Risk bands   — the HIGH threshold swept, since 70 is what gates SAR drafting
                  and is therefore the single most consequential constant here.

Method
------
The detection stack runs ONCE. Every configuration is then produced by re-fusing
the captured `rule_hits` and `ml_scores` through the real `risk_classify`, with
its module constants patched for the sweeps. Two reasons for this over re-running
the pipeline per configuration: it is ~40x faster, and it guarantees every
configuration sees byte-identical detector output, so a difference between rows
can only come from the fusion. It also means these numbers exercise the shipped
scoring code rather than a reimplementation of it that could drift from it.

Scope, stated so the numbers are not over-read
----------------------------------------------
This ablates the DETECTION stack — rules, ML, fusion, bands. It deliberately
does not run the agent layer: no planner, no executor re-planning, no narrator.
Those affect which tools run for a given *query*, whereas full_analysis runs all
of them by construction, so including them would add nondeterminism without
changing a single number below.

The sweeps re-fuse fixed detector output, so they answer "what would this
constant have produced on these hits?" — not "what if the system had been
tuned this way from the start", which would change which entities `ml_detect`
scores in the first place. The distinction matters for section 3: a fusion
weight of 0.0 does not give you the same thing as the ML-only row in section 1,
because the entity universe still includes every rule-hit entity.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import backend.tools.risk as risk_mod
from backend.agent import registry
from backend.config import settings
from backend.schemas import QueryIntent
from backend.tools.base import ToolContext

from evaluation.harness import (
    DEFAULT_TXN_CSV,
    NAIVE_THRESHOLD,
    Metrics,
    evaluate,
    load_ground_truth,
    naive_baseline,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = _REPO_ROOT / "evaluation" / "results" / "ablation.json"

# The detection half of the full_analysis plan (backend/agent/planner.py).
# eda_profile is omitted: it produces charts, not detections, and nothing
# downstream reads its artifacts.
DETECTION_STEPS = ("load_data", "feature_engineer", "rule_detect", "ml_detect")

ALL_RULES = ("R1", "R2", "R3", "R4", "R5", "R6", "R7")

# Reported against the sender-side definition. Every rule and rolling feature is
# sender-side, so this is the definition that asks whether the components do the
# job they were built for. The other two are carried in the JSON.
PRIMARY_DEFINITION = "sender_only"


# ---------------------------------------------------------------------------
# Running the detection stack once
# ---------------------------------------------------------------------------


def run_detection_stack(source: str = "synthetic") -> tuple[list[dict], list[dict], int]:
    """Run load_data -> feature_engineer -> rule_detect -> ml_detect once.

    Mirrors executor.run_plan's context threading (`ctx.df = result.df`) without
    importing the agent layer, so the captured artifacts are exactly what
    risk_classify would have received in a real run.
    """
    settings.aml_use_mocks = False
    tools = registry.load_tools(use_mocks=False)
    ctx = ToolContext(
        df=None,
        customers=None,
        intent=QueryIntent(
            raw_query="Analyse this dataset for suspicious activity",
            intent="full_analysis",
            parsed_by="rules",
            confidence=0.9,
        ),
        artifacts={},
    )

    for name in DETECTION_STEPS:
        params = {"source": source} if name == "load_data" else {}
        result = tools[name](ctx, **params)
        if not result.ok:
            raise RuntimeError(f"{name} failed during ablation setup: {result.error}")
        if result.df is not None:
            ctx.df = result.df
        # Both halves of the executor's context threading are load-bearing here.
        # Omitting this one is silent: feature_engineer's output never reaches
        # rule_detect, every detector returns nothing, and the whole study
        # reports zeros that look like a finding rather than a broken harness.
        ctx.artifacts.update(result.artifacts)

    customer_count = int(len(ctx.customers)) if ctx.customers is not None else 0
    return (
        list(ctx.artifacts.get("rule_hits", [])),
        list(ctx.artifacts.get("ml_scores", [])),
        customer_count,
    )


# ---------------------------------------------------------------------------
# Re-fusing under a configuration
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _patched(**constants: float):
    """Temporarily override risk.py module constants.

    risk_classify reads them as globals at call time, so this reaches the real
    scoring path rather than a copy of the formula.
    """
    originals = {k: getattr(risk_mod, k) for k in constants}
    for k, v in constants.items():
        setattr(risk_mod, k, v)
    try:
        yield
    finally:
        for k, v in originals.items():
            setattr(risk_mod, k, v)


def fuse(
    rule_hits: Iterable[dict],
    ml_scores: Iterable[dict],
    **constants: float,
) -> list[dict]:
    """Score one configuration through the real risk_classify."""
    ctx = ToolContext(
        df=None,
        artifacts={"rule_hits": list(rule_hits), "ml_scores": list(ml_scores)},
    )
    with _patched(**constants):
        result = risk_mod.risk_classify(ctx)
    if not result.ok:
        raise RuntimeError(f"risk_classify failed: {result.error}")
    return result.artifacts["risk_rows"]


def _flag_sets(risk_rows: list[dict]) -> tuple[set[str], set[str]]:
    """(any flag, HIGH only) — the same two tiers run_evaluation reports."""
    return (
        {r["entity_id"] for r in risk_rows},
        {r["entity_id"] for r in risk_rows if r["risk_level"] == "high"},
    )


def _score(risk_rows: list[dict], gt: Any, definition: str) -> tuple[Metrics, Metrics]:
    any_flag, high_only = _flag_sets(risk_rows)
    positives = gt.positives(definition)
    return (
        evaluate(any_flag, positives, gt.all_customers),
        evaluate(high_only, positives, gt.all_customers),
    )


def _all_definitions(risk_rows: list[dict], gt: Any) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for definition in ("sender_only", "sender_or_receiver", "sender_or_repeat_receiver"):
        any_m, high_m = _score(risk_rows, gt, definition)
        out[definition] = {"any_flag": any_m.as_dict(), "high_only": high_m.as_dict()}
    return out


# ---------------------------------------------------------------------------
# The four studies
# ---------------------------------------------------------------------------


def study_components(rule_hits, ml_scores, gt) -> list[dict[str, Any]]:
    """Rules only, ML only, both."""
    configs = [
        ("Rules only (ML disabled)", rule_hits, []),
        ("ML only (rules disabled)", [], ml_scores),
        ("Hybrid — the shipped system", rule_hits, ml_scores),
    ]
    rows = []
    for label, rh, ms in configs:
        risk_rows = fuse(rh, ms)
        any_m, high_m = _score(risk_rows, gt, PRIMARY_DEFINITION)
        rows.append({
            "config": label,
            "flagged": any_m.flagged,
            "high": high_m.flagged,
            "precision": any_m.precision,
            "recall": any_m.recall,
            "f1": any_m.f1,
            "fpr": any_m.false_positive_rate,
            "by_definition": _all_definitions(risk_rows, gt),
        })
    return rows


def study_rules(rule_hits, ml_scores, gt) -> list[dict[str, Any]]:
    """Each rule alone, and each rule left out of the full hybrid.

    Alone answers "is this rule precise?"; leave-one-out answers "does it catch
    anything the others miss?". A rule can score well on the first and zero on
    the second, which means it is real but redundant — a different problem from
    a rule that is simply dead, and one that only shows up by reporting both.

    Every rule is also scored under the repeat-receiver definition, because one
    of them cannot be judged without it. R7 is receiver-side by design: under
    `sender_only` it is not merely expected to score badly, it is arithmetically
    incapable of a true positive, since no entity it can flag is in that
    positive set unless it also sent. Reporting only the sender-side column
    would make a definition artifact look like a broken rule.
    """
    baseline_rows = fuse(rule_hits, ml_scores)
    base_any, _ = _score(baseline_rows, gt, PRIMARY_DEFINITION)
    base_repeat, _ = _score(baseline_rows, gt, "sender_or_repeat_receiver")

    rows = []
    for rule in ALL_RULES:
        only = [h for h in rule_hits if str(h.get("rule_id")) == rule]
        without = [h for h in rule_hits if str(h.get("rule_id")) != rule]

        alone_rows = fuse(only, [])
        loo_rows = fuse(without, ml_scores)

        alone_any, _ = _score(alone_rows, gt, PRIMARY_DEFINITION)
        loo_any, _ = _score(loo_rows, gt, PRIMARY_DEFINITION)
        alone_repeat, _ = _score(alone_rows, gt, "sender_or_repeat_receiver")
        loo_repeat, _ = _score(loo_rows, gt, "sender_or_repeat_receiver")

        rows.append({
            "rule": rule,
            "hits": len(only),
            "entities": len({str(h["entity_id"]) for h in only}),
            "alone_precision": alone_any.precision,
            "alone_recall": alone_any.recall,
            "loo_precision": loo_any.precision,
            "loo_recall": loo_any.recall,
            "delta_precision": loo_any.precision - base_any.precision,
            "delta_recall": loo_any.recall - base_any.recall,
            # Repeat-receiver view — the only one under which R7 can be right.
            "alone_precision_repeat": alone_repeat.precision,
            "delta_precision_repeat": loo_repeat.precision - base_repeat.precision,
            "delta_recall_repeat": loo_repeat.recall - base_repeat.recall,
        })
    return rows


def study_fusion_weights(rule_hits, ml_scores, gt) -> list[dict[str, Any]]:
    """Sweep RULE_WEIGHT_COEFF; ML takes the remainder."""
    rows = []
    for i in range(11):
        rule_coeff = i / 10.0
        risk_rows = fuse(
            rule_hits, ml_scores,
            RULE_WEIGHT_COEFF=rule_coeff,
            ML_PERCENTILE_COEFF=round(1.0 - rule_coeff, 10),
        )
        any_m, high_m = _score(risk_rows, gt, PRIMARY_DEFINITION)
        rows.append({
            "rule_coeff": rule_coeff,
            "ml_coeff": round(1.0 - rule_coeff, 10),
            "flagged": any_m.flagged,
            "high": high_m.flagged,
            "precision": any_m.precision,
            "recall": any_m.recall,
            "f1": any_m.f1,
            "high_precision": high_m.precision,
        })
    return rows


def study_bands(rule_hits, ml_scores, gt) -> list[dict[str, Any]]:
    """Sweep the HIGH threshold — the constant that gates SAR drafting."""
    rows = []
    for threshold in range(30, 95, 5):
        risk_rows = fuse(rule_hits, ml_scores, RISK_HIGH_THRESHOLD=float(threshold))
        _, high_m = _score(risk_rows, gt, PRIMARY_DEFINITION)
        rows.append({
            "high_threshold": threshold,
            "high_flagged": high_m.flagged,
            "precision": high_m.precision,
            "recall": high_m.recall,
            "f1": high_m.f1,
        })
    return rows


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render(
    components: list[dict],
    rules: list[dict],
    weights: list[dict],
    bands: list[dict],
    naive_m: Metrics,
    total: int,
) -> str:
    out: list[str] = []

    out.append("### 1. Component ablation")
    out.append("")
    out.append("| Configuration | Flagged | HIGH | Precision | Recall | F1 | FPR |")
    out.append("|---|---|---|---|---|---|---|")
    out.append(
        f"| Naive baseline (any txn > ${NAIVE_THRESHOLD:,.0f}) | {naive_m.flagged} / {total} | — | "
        f"{naive_m.precision:.3f} | {naive_m.recall:.3f} | {naive_m.f1:.3f} | "
        f"{naive_m.false_positive_rate:.3f} |"
    )
    for r in components:
        out.append(
            f"| {r['config']} | {r['flagged']} / {total} | {r['high']} | {r['precision']:.3f} | "
            f"{r['recall']:.3f} | {r['f1']:.3f} | {r['fpr']:.3f} |"
        )
    out.append("")

    out.append("### 2. Per-rule contribution")
    out.append("")
    out.append("Sender-side definition, then the repeat-receiver view. R7 is receiver-side,")
    out.append("so only the second pair of columns can say anything about it.")
    out.append("")
    out.append("| Rule | Hits | Entities | Prec. alone | ΔPrec. if removed | ΔRecall if removed | Prec. alone (repeat) | ΔPrec. repeat | ΔRecall repeat |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in rules:
        out.append(
            f"| {r['rule']} | {r['hits']} | {r['entities']} | {r['alone_precision']:.3f} | "
            f"{r['delta_precision']:+.3f} | {r['delta_recall']:+.3f} | "
            f"{r['alone_precision_repeat']:.3f} | {r['delta_precision_repeat']:+.3f} | "
            f"{r['delta_recall_repeat']:+.3f} |"
        )
    out.append("")

    out.append("### 3. Fusion weight sweep")
    out.append("")
    out.append("| Rule coeff | ML coeff | Flagged | HIGH | Precision | Recall | F1 |")
    out.append("|---|---|---|---|---|---|---|")
    for r in weights:
        marker = "  ← shipped" if abs(r["rule_coeff"] - 0.6) < 1e-9 else ""
        out.append(
            f"| {r['rule_coeff']:.1f}{marker} | {r['ml_coeff']:.1f} | {r['flagged']} | {r['high']} | "
            f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |"
        )
    out.append("")

    out.append("### 4. HIGH-band threshold sweep")
    out.append("")
    out.append("| HIGH threshold | Flagged HIGH | Precision | Recall | F1 |")
    out.append("|---|---|---|---|---|")
    for r in bands:
        marker = "  ← shipped" if r["high_threshold"] == 70 else ""
        out.append(
            f"| {r['high_threshold']}{marker} | {r['high_flagged']} | {r['precision']:.3f} | "
            f"{r['recall']:.3f} | {r['f1']:.3f} |"
        )
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ablate the detection stack: components, rules, fusion weights, bands.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--source", default="synthetic")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    df, gt = load_ground_truth()
    total = len(gt.all_customers)

    print(f"Running the detection stack once over {DEFAULT_TXN_CSV.name}...", file=sys.stderr)
    rule_hits, ml_scores, loaded = run_detection_stack(source=args.source)

    # Same guard as run_evaluation: flags and labels must describe one dataset.
    if loaded and loaded != total:
        print(
            f"\nERROR: loaded {loaded} customers but the ground truth describes "
            f"{total}. Check --source (given: {args.source!r}).",
            file=sys.stderr,
        )
        return 1

    print(f"  {len(rule_hits)} rule hits, {len(ml_scores)} ML-scored entities\n", file=sys.stderr)

    components = study_components(rule_hits, ml_scores, gt)
    rules = study_rules(rule_hits, ml_scores, gt)
    weights = study_fusion_weights(rule_hits, ml_scores, gt)
    bands = study_bands(rule_hits, ml_scores, gt)
    naive_m = evaluate(
        naive_baseline(df, gt.all_customers), gt.positives(PRIMARY_DEFINITION), gt.all_customers
    )

    print(render(components, rules, weights, bands, naive_m, total))

    if not args.no_write:
        payload = {
            "_comment": (
                "Regenerate with: python -m evaluation.ablation — everything "
                "outside 'run_metadata' is deterministic and safe to diff."
            ),
            "run_metadata": {
                "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "python": platform.python_version(),
                "primary_definition": PRIMARY_DEFINITION,
            },
            "detector_output": {
                "rule_hits": len(rule_hits),
                "ml_scored_entities": len(ml_scores),
                "customers": total,
            },
            "components": components,
            "rules": rules,
            "fusion_weights": weights,
            "bands": bands,
            "naive_baseline": naive_m.as_dict(),
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
