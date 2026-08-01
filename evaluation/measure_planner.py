"""
evaluation/measure_planner.py — how often does a real model produce a usable plan?

Usage
-----
    python -m evaluation.measure_planner                      # local ollama
    python -m evaluation.measure_planner --provider groq
    python -m evaluation.measure_planner --repeats 3          # variance
    python -m evaluation.measure_planner --output results.json

Why this is a module and not a scratch script
---------------------------------------------
README.md quotes a planner acceptance figure. Like the detection metrics, it
has to be regenerable or it decays into prose nobody can check — the exact
problem evaluation/harness.py was written to solve for precision and recall.

What it measures, and what it does NOT
--------------------------------------
It measures the PLANNER in isolation: QueryIntent objects are constructed
directly rather than parsed, so the figure is not a mixture of parser variance
and planner variance.

The cost of that isolation is real and worth stating. The live API path runs
parse_intent first, producing different intent state, a different prompt, and
proposals this harness never generates — including the pattern_types=["risk"]
that motivated validator rule V13. So this measures the planner on a clean
bench, not in situ, and the two numbers are not interchangeable.

Two figures are reported because they can come apart badly:

  accepted — the validator said yes.
  useful   — accepted AND the plan contains the tool its intent needs to
             produce an answer.

They diverged once already: 60% accepted against 7% useful, because the model
had learned that shorter plans pass and a truncated plan satisfies every
ordering rule vacuously. Validator rule V12 closed that. Reporting only
"accepted" would have hidden it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import backend.agent.executor as executor_mod
import backend.agent.llm_planner as llm_planner
import backend.llm.client as llm_client
from backend.agent.plan_validator import _REQUIRED_TERMINAL, validate_proposal
from backend.config import settings
from backend.schemas import Filters, QueryIntent

_REPO_ROOT = Path(__file__).resolve().parent.parent

# 15 queries spanning all 7 intents. Fixed, so runs are comparable over time.
QUERIES: list[tuple[str, str, dict[str, Any]]] = [
    ("full_analysis", "Analyse this dataset for suspicious activity", {}),
    ("full_analysis", "Give me a complete review of everything going on", {}),
    ("pattern_search", "Find structuring patterns in the last 30 days",
     {"pattern_types": ["structuring"], "filters": Filters(amount_min=9000.0)}),
    ("pattern_search", "Show me any smurfing behaviour", {"pattern_types": ["smurfing"]}),
    ("pattern_search", "Look for layering through offshore accounts",
     {"pattern_types": ["layering"], "filters": Filters(countries=["KY"])}),
    ("threshold_query", "Which customers made 10+ transactions under $10,000?",
     {"filters": Filters(amount_max=10000.0, min_txn_count=10)}),
    ("threshold_query", "How many transactions were over $50,000?",
     {"filters": Filters(amount_min=50000.0)}),
    ("entity_investigation", "Is customer C-STR02 suspicious?", {"entities": ["C-STR02"]}),
    ("entity_investigation", "Investigate C-HUB01 for me", {"entities": ["C-HUB01"]}),
    ("entity_investigation", "What is going on with account C-RCO05?", {"entities": ["C-RCO05"]}),
    ("ranking", "Who are my 5 riskiest customers?", {"top_n": 5}),
    ("ranking", "Rank customers by risk", {"top_n": 10}),
    ("eda", "Show transaction distribution by country", {}),
    ("eda", "What does this dataset look like?", {}),
    ("explain_flag", "Why was C-STR02 flagged?", {"entities": ["C-STR02"]}),
]

# Buckets for rejection reasons, matched as substrings against the validator's
# messages. Ordered: first match wins, so specific patterns precede general.
_REASON_BUCKETS = (
    "needs", "requires", "must be the first step", "must follow load_data",
    "proposed more than once", "unknown tool", "undeclared param",
    "is not a valid", "may not set", "is not available to a plan",
    "missing reason", "malformed proposal", "max is", "no entity was extracted",
)


def _bucket(rejection: str) -> str:
    for pattern in _REASON_BUCKETS:
        if pattern in rejection:
            return pattern
    return rejection


def run_once(tools: dict, verbose: bool = True) -> list[dict[str, Any]]:
    """One pass over all 15 queries. Returns a row per query."""
    rows: list[dict[str, Any]] = []
    for i, (intent_name, query, kw) in enumerate(QUERIES, 1):
        intent = QueryIntent(raw_query=query, intent=intent_name,
                             parsed_by="rules", confidence=0.9, **kw)
        t0 = time.perf_counter()
        raw = llm_planner.propose_plan(intent, tools)
        elapsed = time.perf_counter() - t0

        proposed = llm_planner._proposed_names(raw)
        if raw is None:
            accepted, useful, rejections = False, False, ["no usable JSON returned"]
            selected: list[str] = []
        else:
            result = validate_proposal(raw, intent, tools)
            accepted = result.ok
            rejections = result.rejections
            selected = [s.tool for s in result.steps]
            need = _REQUIRED_TERMINAL.get(intent_name)
            useful = accepted and (need is None or need in selected)

        rows.append({
            "intent": intent_name, "query": query,
            "accepted": accepted, "useful": useful,
            "seconds": round(elapsed, 2),
            "proposed": proposed, "validated": selected,
            "rejections": rejections,
        })
        if verbose:
            mark = "OK  " if useful else ("THIN" if accepted else "FAIL")
            print(f"  [{i:2d}/{len(QUERIES)}] {mark} {elapsed:5.1f}s  "
                  f"{intent_name:22s} {query[:42]}")
            if useful:
                print(f"             -> {' -> '.join(selected)}")
            else:
                for r in rejections[:2]:
                    print(f"             x {r}")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--provider", default=None,
                        help="llm_provider override: ollama | gemini | groq | openai")
    parser.add_argument("--model", default=None, help="ollama model override")
    parser.add_argument("--repeats", type=int, default=1,
                        help="repeat the whole set N times, clearing the response "
                             "cache between repeats, to measure run-to-run variance")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    if args.provider:
        settings.llm_provider = args.provider
    if args.model:
        settings.ollama_model = args.model
    settings.aml_use_mocks = False
    settings.aml_llm_planner = True
    executor_mod._TOOLS_CACHE = None
    tools = executor_mod._get_tools()

    label = (settings.ollama_model if settings.llm_provider == "ollama"
             else settings.llm_provider)
    print(f"provider={settings.llm_provider}  model={label}  "
          f"tools={len(tools)}  repeats={args.repeats}\n")

    all_runs: list[list[dict]] = []
    rejection_counter: Counter[str] = Counter()
    for rep in range(1, args.repeats + 1):
        if args.repeats > 1:
            # llm/client caches on (prompt, schema_hint); without clearing it
            # every repeat would replay run 1 and report zero variance by
            # construction.
            llm_client._CACHE.clear()
            print(f"--- repeat {rep}/{args.repeats} ---")
        rows = run_once(tools)
        for r in rows:
            for rej in r["rejections"]:
                rejection_counter[_bucket(rej)] += 1
        all_runs.append(rows)

    useful_counts = [sum(1 for r in run if r["useful"]) for run in all_runs]
    accepted_counts = [sum(1 for r in run if r["accepted"]) for run in all_runs]
    n = len(QUERIES)
    pcts = [100.0 * c / n for c in useful_counts]

    print("\n" + "=" * 72)
    print(f"accepted per run : {accepted_counts} of {n}")
    print(f"USEFUL per run   : {useful_counts} of {n}   "
          f"({', '.join(f'{p:.0f}%' for p in pcts)})")
    if len(pcts) > 1:
        print(f"mean {statistics.mean(pcts):.1f}%   "
              f"range {min(pcts):.0f}-{max(pcts):.0f}%   "
              f"stdev {statistics.stdev(pcts):.1f} points")
    print(f"one query is worth {100.0 / n:.1f} points")

    # Per-query stability is more informative than the headline percentage:
    # it distinguishes a capability boundary from sampling noise.
    if len(all_runs) > 1:
        per_query: dict[str, list[bool]] = {}
        for run in all_runs:
            for r in run:
                per_query.setdefault(r["query"], []).append(r["useful"])
        always = [q for q, v in per_query.items() if all(v)]
        never = [q for q, v in per_query.items() if not any(v)]
        flaky = [q for q, v in per_query.items() if any(v) and not all(v)]
        print(f"\nalways useful {len(always)}   never useful {len(never)}   "
              f"unstable {len(flaky)}")
        for q in flaky:
            print(f"  unstable: {q}  {per_query[q]}")

    print("\nrejection reasons:")
    for reason, count in rejection_counter.most_common():
        print(f"  {count:3d}  {reason}")

    by_intent: dict[str, list[bool]] = {}
    for run in all_runs:
        for r in run:
            by_intent.setdefault(r["intent"], []).append(r["useful"])
    print("\nuseful by intent:")
    for name in sorted(by_intent):
        v = by_intent[name]
        print(f"  {name:22s} {sum(v)}/{len(v)}")

    times = [r["seconds"] for run in all_runs for r in run]
    print(f"\nlatency: median {statistics.median(times):.1f}s  "
          f"min {min(times):.1f}s  max {max(times):.1f}s")

    if args.output:
        payload = {
            "provider": settings.llm_provider, "model": label,
            "repeats": args.repeats,
            "useful_per_run": useful_counts, "accepted_per_run": accepted_counts,
            "total_queries": n,
            "rejections": dict(rejection_counter),
            "runs": all_runs,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
