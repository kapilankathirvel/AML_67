"""
evaluation/run_evaluation.py — regenerate the README's Results table.

Usage
-----
    python -m evaluation.run_evaluation
    python -m evaluation.run_evaluation --output evaluation/results/after_r7.json

Runs the agent's full_analysis pipeline over the committed labelled dataset,
scores its flags against both ground-truth definitions, and prints a
paste-ready markdown table alongside a JSON record of the run.

Determinism
-----------
Three things in this pipeline are nondeterministic or make network calls, and
all three are pinned here. This mirrors the fixtures in
tests/test_integration.py, which had to solve the same problem:

  1. Tool selection — executor._TOOLS_CACHE is a module-level cache keyed on
     whatever settings.aml_use_mocks was at first call. Setting the flag alone
     is not enough; the cache must be cleared or the run silently evaluates
     the mock tools and reports meaningless numbers.

  2. Intent parsing — the LLM path returns different QueryIntents run to run.
     We construct the QueryIntent directly with parsed_by="rules" and never
     invoke the parser, which is also what the integration tests do.

  3. Explanation polish — narrator._explain() calls the LLM for every HIGH
     flag. On this dataset that is ~23 real network calls per run: slow,
     nondeterministic, and a documented cause of quota exhaustion in this repo.
     We stub complete_json to None, which the narrator already treats as
     "fall back to the deterministic template".

  4. Tool selection — settings.aml_llm_planner is pinned off. This harness
     calls build_plan() directly and so cannot reach backend/agent/llm_planner
     anyway, but an LLM-chosen plan would make the metrics depend on a model's
     output, which is exactly what a reproducible baseline cannot tolerate.
     Pinned explicitly rather than left to the module default, so the guarantee
     is stated in the code that depends on it.

ML seeding is already fixed (random_state=42 in ml_detect.py), so with the
above the run reproduces byte-for-byte.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import backend.agent.narrator as narrator_mod
import backend.agent.executor as executor_mod
from backend.agent.executor import run_plan
from backend.agent.planner import build_plan
from backend.config import settings
from backend.schemas import QueryIntent

from evaluation.harness import (
    DEFAULT_CUST_CSV,
    DEFAULT_TXN_CSV,
    NAIVE_THRESHOLD,
    Metrics,
    evaluate,
    load_ground_truth,
    naive_baseline,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = _REPO_ROOT / "evaluation" / "results" / "baseline.json"

FULL_ANALYSIS_QUERY = "Analyse this dataset for suspicious activity"

DEFINITIONS = ("sender_only", "sender_or_receiver")

_DEFINITION_LABEL = {
    "sender_only": "Sender-side ground truth",
    "sender_or_receiver": "Broader ground truth (sender or receiver)",
}


# ---------------------------------------------------------------------------
# Running the agent
# ---------------------------------------------------------------------------


def run_agent_flags(source: str = "synthetic") -> tuple[set[str], set[str], dict[str, Any]]:
    """Run full_analysis deterministically; return (any_flag, high_only, metrics).

    `source` is forced onto the load_data step rather than left to its default.
    This matters: load_data's signature defaults to source="synthetic_alt"
    (a second synthetic set with a different raw schema, 1,710 txns / 294
    customers), while the labelled dataset the README's Results section is
    computed against is "synthetic" (aml_sample.csv, 2,002 txns / 270
    customers). Scoring flags from one against labels from the other silently
    produces nonsense, because the customer IDs don't overlap and the
    intersection just drops flags on the floor.

    The planner emits load_data with empty params, so overriding here is the
    supported way to pin it.
    """
    # (1) real tools, not mocks — and clear the cache so the setting is honoured
    settings.aml_use_mocks = False
    executor_mod._TOOLS_CACHE = None

    # (3) template-only explanations: no network, no nondeterminism
    original_complete_json = narrator_mod.complete_json
    narrator_mod.complete_json = lambda *args, **kwargs: None

    # (4) deterministic tool selection
    original_llm_planner = settings.aml_llm_planner
    settings.aml_llm_planner = False

    try:
        # (2) construct the intent directly — never call the LLM parser
        intent = QueryIntent(
            raw_query=FULL_ANALYSIS_QUERY,
            intent="full_analysis",
            parsed_by="rules",
            confidence=0.9,
        )
        plan = build_plan(intent)
        for step in plan.steps:
            if step.tool == "load_data":
                step.params = {**step.params, "source": source}
        response = run_plan(intent, plan)
    finally:
        narrator_mod.complete_json = original_complete_json
        settings.aml_llm_planner = original_llm_planner
        executor_mod._TOOLS_CACHE = None

    any_flag = {f.entity_id for f in response.flags}
    high_only = {f.entity_id for f in response.flags if f.risk_level == "high"}
    return any_flag, high_only, response.metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_row(label: str, m: Metrics, total: int) -> str:
    return (
        f"| {label} | {m.flagged} / {total} | {m.precision:.3f} | "
        f"{m.recall:.3f} | {m.false_positive_rate:.3f} |"
    )


def render_markdown(results: dict[str, dict[str, Metrics]], total: int) -> str:
    """Produce the paste-ready table, one block per ground-truth definition."""
    out: list[str] = []
    for definition in DEFINITIONS:
        rows = results[definition]
        out.append(f"**{_DEFINITION_LABEL[definition]}**")
        out.append("")
        out.append("| | Flagged | Precision | Recall | False-positive rate |")
        out.append("|---|---|---|---|---|")
        out.append(_fmt_row(
            f"Naive baseline (any txn > ${NAIVE_THRESHOLD:,.0f})", rows["naive"], total))
        out.append(_fmt_row(
            "Our system — any flag (LOW/MEDIUM/HIGH)", rows["any_flag"], total))
        out.append(_fmt_row(
            "Our system — HIGH only (the SAR-draft tier)", rows["high_only"], total))
        out.append("")
    return "\n".join(out)


def build_payload(
    results: dict[str, dict[str, Metrics]],
    gt: Any,
    reproducible: bool,
) -> dict[str, Any]:
    """JSON record of the run.

    `generated_utc` and the environment block are deliberately kept OUT of the
    comparable section: they change every run, and a diff between two runs
    should show a detection change, not a timestamp. Everything under
    "results" and "ground_truth" is what should be byte-stable.
    """
    return {
        "_comment": (
            "Regenerate with: python -m evaluation.run_evaluation — "
            "'results' and 'ground_truth' are deterministic and safe to diff "
            "between runs; 'run_metadata' is not."
        ),
        "run_metadata": {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "deterministic_run": reproducible,
        },
        "dataset": {
            "transactions_csv": str(DEFAULT_TXN_CSV.relative_to(_REPO_ROOT).as_posix()),
            "customers_csv": str(DEFAULT_CUST_CSV.relative_to(_REPO_ROOT).as_posix()),
        },
        "ground_truth": {
            "customers_total": len(gt.all_customers),
            "labelled_transactions": gt.labelled_txn_count,
            "positives_sender_only": len(gt.sender_only),
            "positives_sender_or_receiver": len(gt.sender_or_receiver),
            "positives_receive_only": len(gt.receive_only),
        },
        "results": {
            definition: {system: m.as_dict() for system, m in rows.items()}
            for definition, rows in results.items()
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate the README Results table from the labelled dataset.",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"where to write the JSON record (default: {DEFAULT_OUTPUT.name})",
    )
    parser.add_argument(
        "--no-write", action="store_true",
        help="print the table but write no JSON",
    )
    parser.add_argument(
        "--source", default="synthetic",
        help=("load_data source to evaluate. Default 'synthetic' (aml_sample.csv) — "
              "the labelled set the README's Results section is computed against. "
              "Note load_data's own default is 'synthetic_alt', a different dataset."),
    )
    args = parser.parse_args(argv)

    # The table contains em-dashes and is meant to be copied into the README.
    # Windows consoles default to cp1252 and render them as replacement
    # characters, which would get pasted in verbatim.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    df, gt = load_ground_truth()
    total = len(gt.all_customers)

    print(f"Scoring against {DEFAULT_TXN_CSV.name} "
          f"({gt.labelled_txn_count} labelled txns, {total} customers)...",
          file=sys.stderr)

    any_flag, high_only, agent_metrics = run_agent_flags(source=args.source)
    naive = naive_baseline(df, gt.all_customers)

    # Guard the exact bug this harness hit during development: if the agent
    # analysed a different dataset than the labels describe, every metric below
    # is meaningless. Fail loudly instead of reporting a plausible-looking table.
    loaded_customers = agent_metrics.get("customer_count")
    if loaded_customers is not None and loaded_customers != total:
        print(
            f"\nERROR: the agent loaded {loaded_customers} customers but the ground "
            f"truth describes {total}. The flags and the labels are from different "
            f"datasets, so the metrics would be nonsense.\n"
            f"       Check --source (given: {args.source!r}) against "
            f"{DEFAULT_TXN_CSV.name}.",
            file=sys.stderr,
        )
        return 1

    flagged_sets = {"naive": naive, "any_flag": any_flag, "high_only": high_only}

    results: dict[str, dict[str, Metrics]] = {
        definition: {
            system: evaluate(flags, gt.positives(definition), gt.all_customers)
            for system, flags in flagged_sets.items()
        }
        for definition in DEFINITIONS
    }

    print()
    print(render_markdown(results, total))

    sender = results["sender_only"]
    broad = results["sender_or_receiver"]
    caught_receive_only = len(any_flag & gt.receive_only)

    print(f"Flag reduction vs naive: "
          f"{len(naive) / max(len(any_flag), 1):.1f}x fewer customers flagged")
    print(f"FPR reduction vs naive:  "
          f"{sender['naive'].false_positive_rate / max(sender['any_flag'].false_positive_rate, 1e-9):.0f}x lower")
    print(f"Recall by ground truth:  "
          f"{sender['any_flag'].recall:.3f} sender-side -> "
          f"{broad['any_flag'].recall:.3f} broader")
    print(f"Receive-only positives:  "
          f"{caught_receive_only} of {len(gt.receive_only)} caught "
          f"({len(gt.receive_only) - caught_receive_only} remain structurally unreachable)")

    if not args.no_write:
        payload = build_payload(results, gt, reproducible=True)
        out = args.output.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        # A user-supplied --output may sit outside the repo, so relative_to()
        # cannot be assumed here.
        try:
            shown = out.relative_to(_REPO_ROOT).as_posix()
        except ValueError:
            shown = str(out)
        print(f"\nWrote {shown}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
