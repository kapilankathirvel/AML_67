# CLAUDE.md — instructions for coding agents working in this repo

This is a **fork**, maintained by one person. The upstream repo was built by two people in parallel
under a Track A / Track B file-ownership protocol; that protocol **no longer applies here** and you
are not restricted by it. See [WORKPLAN.md](docs/WORKPLAN.md) for the original plan (useful history)
and [docs/CONTRACTS.md](docs/CONTRACTS.md) for the interface the whole system codes against.

## What you may edit

Every file in this fork is editable, including the detection modules that were Track B's
(`backend/tools/features.py`, `rules.py`, `ml_detect.py`, `risk.py`) and the frontend. The rules
that still hold:

- `backend/schemas.py`, `backend/tools/base.py`, and `docs/CONTRACTS.md` are **frozen** interface
  files. Changing them ripples through the agent core, every tool, and the UI at once — so treat
  them as read-only unless the task explicitly calls for a contract change, and say so plainly
  when it does.
- Never run `git add -A` or `git add .` — stage explicit paths only.
- **Do not regenerate `data/sample/aml_sample.csv`.** It is the labelled ground truth every
  published metric is computed against; regenerating it silently invalidates every number in
  `README.md` and `docs/`. `tests/test_evaluation.py` pins its shape (270 customers, 202 labelled
  transactions, 51/114 positives) and will fail if it changes.
- Leave every `__init__.py` empty — no re-exports, no `__all__`.

## Metrics are generated, never hand-written

`README.md`'s Results table comes from `python -m evaluation.run_evaluation`. If a change affects
detection, re-run it and update the docs from its output — do not edit the numbers by hand.

Note that `load_data`'s default source is `synthetic_alt` (a second dataset with a different raw
schema), while the labelled set the metrics are computed against is `synthetic`
(`aml_sample.csv`). Anything that scores flags against labels **must pin the source explicitly**,
or it silently compares two different populations.

## Behavioural rules

**[ANTI_HALLUCINATION_A.md](docs/ANTI_HALLUCINATION_A.md)** and
**[ANTI_HALLUCINATION_B.md](docs/ANTI_HALLUCINATION_B.md)** still apply, both of them now that the
track split is gone. In short: do only the requested task, do not redesign architecture, do not
touch files outside the requested scope, do not invent contract fields, and ask rather than guess
when information is missing.

## Project one-liner

A natural-language query goes in; the agent parses intent, builds a query-specific execution plan
(not a fixed pipeline), calls only the tools that plan needs, and returns risk-scored, explained,
escalation-tagged AML flags. See `WORKPLAN.md` §0 for the two-sentence pitch and §8 for the
plan-divergence test that is the project's core acceptance criterion.
