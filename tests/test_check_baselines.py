"""Tests for the published-baseline regression check.

The check itself is only useful if it fails when a number moves. That sounds
tautological, but the two ways it could quietly stop working are both easy to
write by accident: excluding too much as volatile (so real changes slip
through) and comparing floats loosely (same problem, harder to see).

Nothing here runs a study. The studies take ~40 minutes combined, which is why
they are a nightly job rather than a test; what is tested is the comparison
logic they feed.
"""

import json
from pathlib import Path

import pytest

from scripts.check_baselines import STUDIES, _comparable, _diff_paths, check

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# What counts as a difference
# ---------------------------------------------------------------------------


def test_identical_documents_have_no_differences():
    doc = {"results": {"sender_only": {"precision": 0.561, "recall": 0.451}}}
    assert _diff_paths(doc, json.loads(json.dumps(doc))) == []


def test_a_moved_metric_is_reported_with_its_path():
    """The failure mode the whole script exists to catch."""
    before = {"results": {"sender_only": {"precision": 0.561}}}
    after = {"results": {"sender_only": {"precision": 0.583}}}

    diffs = _diff_paths(before, after)
    assert len(diffs) == 1
    assert diffs[0].startswith("results.sender_only.precision")
    assert "0.561" in diffs[0] and "0.583" in diffs[0]


def test_float_comparison_is_exact():
    """No tolerance, deliberately.

    Every study is deterministic — fixed random_state, no sampling, no wall
    clock in the compared section. A tolerance would only ever hide a real
    change, since there is no source of legitimate jitter to absorb.
    """
    assert _diff_paths({"x": 0.451}, {"x": 0.4510000001}) != []


def test_added_and_removed_keys_are_both_reported():
    assert _diff_paths({"a": 1}, {"a": 1, "b": 2}) == ["b (added)"]
    assert _diff_paths({"a": 1, "b": 2}, {"a": 1}) == ["b (removed)"]


def test_a_changed_row_count_is_reported_as_a_length_change():
    """A study that gained or lost a configuration should say so once, rather
    than emitting a difference for every field of every shifted row."""
    diffs = _diff_paths({"rows": [1, 2, 3]}, {"rows": [1, 2]})
    assert diffs == ["rows (length 3 -> 2)"]


def test_nested_list_elements_are_compared_by_index():
    diffs = _diff_paths({"rows": [{"recall": 0.4}]}, {"rows": [{"recall": 0.2}]})
    assert diffs == ["rows.0.recall: 0.4 -> 0.2"]


# ---------------------------------------------------------------------------
# What is excluded from comparison
# ---------------------------------------------------------------------------


def test_run_metadata_is_excluded():
    """It holds a timestamp and the Python version, so a strict comparison
    including it would fail every single night and train everyone to ignore
    the job."""
    assert "run_metadata" not in _comparable({"run_metadata": {}, "results": {}})


def test_nothing_else_is_excluded():
    """The exclusion list is what makes strictness possible, so it must stay
    minimal. If a future change adds to it, this test should be the thing that
    forces the addition to be argued for rather than slipped in."""
    doc = {"run_metadata": 1, "results": 2, "ground_truth": 3, "dataset": 4, "_comment": 5}
    assert set(_comparable(doc)) == {"results", "ground_truth", "dataset", "_comment"}


# ---------------------------------------------------------------------------
# The studies it points at
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,module,baseline", STUDIES, ids=[s[0] for s in STUDIES])
def test_every_registered_baseline_exists_and_is_readable(name, module, baseline):
    """A missing or malformed baseline would make the nightly job pass by
    doing nothing, which is worse than failing."""
    assert baseline.exists(), f"{name}: {baseline} is missing"
    doc = json.loads(baseline.read_text(encoding="utf-8"))
    assert _comparable(doc), f"{name}: nothing left to compare after exclusions"


def test_missing_baseline_fails_rather_than_passing_vacuously(tmp_path):
    ok, problems = check("nope", "evaluation.ablation", tmp_path / "absent.json", tmp_path)
    assert not ok
    assert "missing" in problems[0]


def test_a_study_that_fails_to_run_is_reported_as_a_failure(tmp_path):
    """Distinguished from 'the numbers moved' on purpose: a crashed study and a
    changed metric need different responses, and reporting a crash as a diff
    would send someone looking for a detection change that never happened."""
    baseline = tmp_path / "b.json"
    baseline.write_text(json.dumps({"results": {}}), encoding="utf-8")

    ok, problems = check("broken", "evaluation.does_not_exist", baseline, tmp_path)
    assert not ok
    assert "exited" in problems[0]
