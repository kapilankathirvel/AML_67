"""
scripts/check_baselines.py — do the published numbers still reproduce?

Usage
-----
    python -m scripts.check_baselines
    python -m scripts.check_baselines --only ablation

Why this exists
---------------
Three JSON files under evaluation/results/ are the provenance for every number
in README.md and docs/. They were each produced by a real run, and nothing
until now re-ran them. A change to detection could therefore land, pass the
entire test suite, and leave the documented metrics quietly describing a system
that no longer exists.

This regenerates each one into a temporary directory and compares it against
what is committed. It never overwrites the committed file -- a baseline should
move because someone decided it should, not as a side effect of a check.

What is compared, and what is not
---------------------------------
Each study writes a `run_metadata` block containing a timestamp and the Python
version. Those change every run by design and are excluded here; everything
else is deterministic and is compared exactly.

That exclusion is the whole reason the comparison can be strict. Without it the
only options would be a fuzzy match, which would hide small regressions, or a
guaranteed failure every night, which would train everyone to ignore the job.

Exit codes
----------
0 — every baseline reproduced
1 — at least one moved (the differing paths are printed)
2 — a study failed to run at all
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RESULTS = _REPO_ROOT / "evaluation" / "results"

# Keys excluded from comparison, by dotted path from the document root.
# Everything not listed here is compared exactly.
_VOLATILE = {"run_metadata"}

# (name, module, committed baseline). The baseline for run_evaluation is the
# most recent published record rather than a file named "baseline" -- see
# docs/AFTER_THE_DEADLINE.md for why the earlier ones are kept.
STUDIES: list[tuple[str, str, Path]] = [
    ("evaluation", "evaluation.run_evaluation", _RESULTS / "after_repeat_receiver_gt.json"),
    ("ablation", "evaluation.ablation", _RESULTS / "ablation.json"),
    ("evasion", "evaluation.evasion", _RESULTS / "evasion.json"),
]


def _comparable(doc: dict[str, Any]) -> dict[str, Any]:
    """The deterministic part of a study's output."""
    return {k: v for k, v in doc.items() if k not in _VOLATILE}


def _diff_paths(expected: Any, actual: Any, path: str = "") -> list[str]:
    """Dotted paths where two JSON documents disagree.

    Returns paths rather than a full diff because these documents run to a
    thousand lines: "moves.slow_down.rows.4.hybrid.recall" tells you what
    changed, and `git diff` after a regeneration tells you by how much.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        out: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            here = f"{path}.{key}" if path else key
            if key not in expected:
                out.append(f"{here} (added)")
            elif key not in actual:
                out.append(f"{here} (removed)")
            else:
                out.extend(_diff_paths(expected[key], actual[key], here))
        return out

    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path} (length {len(expected)} -> {len(actual)})"]
        out = []
        for i, (e, a) in enumerate(zip(expected, actual)):
            out.extend(_diff_paths(e, a, f"{path}.{i}"))
        return out

    return [] if expected == actual else [f"{path}: {expected!r} -> {actual!r}"]


def _shown(path: Path) -> str:
    """Repo-relative where possible, absolute otherwise.

    A caller may point this at a baseline outside the repo, and relative_to
    raises rather than returning the absolute path in that case — which would
    turn "your baseline is missing" into an unrelated traceback.
    """
    try:
        return path.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def check(name: str, module: str, baseline: Path, workdir: Path) -> tuple[bool, list[str]]:
    """Regenerate one study and compare it to its committed baseline."""
    if not baseline.exists():
        return False, [f"baseline missing: {_shown(baseline)}"]

    fresh = workdir / f"{name}.json"
    proc = subprocess.run(
        [sys.executable, "-m", module, "--output", str(fresh)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-15:]
        return False, [f"{module} exited {proc.returncode}"] + tail
    if not fresh.exists():
        return False, [f"{module} wrote nothing to {fresh}"]

    expected = _comparable(json.loads(baseline.read_text(encoding="utf-8")))
    actual = _comparable(json.loads(fresh.read_text(encoding="utf-8")))
    return (True, []) if expected == actual else (False, _diff_paths(expected, actual))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the published evaluation baselines still reproduce.",
    )
    parser.add_argument(
        "--only",
        choices=[name for name, _, _ in STUDIES],
        help="check a single study instead of all three",
    )
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    studies = [s for s in STUDIES if args.only is None or s[0] == args.only]
    failures: list[str] = []

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for name, module, baseline in studies:
            print(f"Checking {name} ({module})...", flush=True)
            ok, problems = check(name, module, baseline, workdir)
            if ok:
                print(f"  OK — reproduces {baseline.name}")
                continue
            failures.append(name)
            print(f"  MOVED — {len(problems)} difference(s) against {baseline.name}:")
            for line in problems[:25]:
                print(f"    {line}")
            if len(problems) > 25:
                print(f"    ... and {len(problems) - 25} more")

    if failures:
        print(
            f"\n{len(failures)} baseline(s) moved: {', '.join(failures)}.\n"
            "If the change was intended, regenerate the file(s) and commit them "
            "together with the change that moved them, so the published numbers "
            "and the code that produced them stay in one commit.",
            file=sys.stderr,
        )
        return 1

    print(f"\nAll {len(studies)} baseline(s) reproduce.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
