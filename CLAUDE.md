# CLAUDE.md — instructions for coding agents working in this repo

This repo is being built by two people in parallel: **Track A** (agent core + API) and **Track B**
(data, detection, UI). See [WORKPLAN.md](docs/WORKPLAN.md) for the full plan and
[docs/CONTRACTS.md](docs/CONTRACTS.md) for the frozen interface both tracks code against.

## File ownership — the rule that prevents merge conflicts

Every file has exactly one owner (full matrix in `WORKPLAN.md` §4). Whichever track you are helping:

- You may only **create or edit** files owned by your track.
- You may **read** any file in the repo.
- If a change looks necessary in a file owned by the other track — including `requirements.txt`,
  any `__init__.py`, `README.md`, or `backend/schemas.py` — **stop and report** what change is needed
  and why. Do not make the edit yourself, and do not work around it by duplicating the file.
- `backend/schemas.py`, `backend/tools/base.py`, and `docs/CONTRACTS.md` are **frozen** interface
  files (owned by Track A). Treat them as read-only ground truth and conform to them exactly.
- Never run `git add -A` or `git add .` — stage explicit paths only.
- Never edit or regenerate `data/sample/aml_sample.csv` unless you are Track B.
- Leave every `__init__.py` empty — no re-exports, no `__all__`.

## Behavioural rules

Also read and follow **[ANTI_HALLUCINATION_A.md](docs/ANTI_HALLUCINATION_A.md)** if you are working Track A,
or **[ANTI_HALLUCINATION_B.md](docs/ANTI_HALLUCINATION_B.md)** if you are working Track B. In short: do only
the requested task, do not redesign architecture, do not touch files outside the requested scope, do not
invent contract fields, and ask rather than guess when information is missing.

## Project one-liner

A natural-language query goes in; the agent parses intent, builds a query-specific execution plan
(not a fixed pipeline), calls only the tools that plan needs, and returns risk-scored, explained,
escalation-tagged AML flags. See `WORKPLAN.md` §0 for the two-sentence pitch and §8 for the
plan-divergence test that is the project's core acceptance criterion.
