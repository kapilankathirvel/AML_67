"""Tests for the ablation study.

Two things are being protected here. The first is the study's own correctness:
it re-fuses captured detector output, so a mistake in the capture or the
re-fusion produces a table of plausible numbers with nothing behind them — the
first run of this module reported all zeros because the detection stack was
threaded incorrectly, and nothing about the output looked wrong.

The second is the findings themselves. Two of them are structural claims about
the scoring formula rather than observations about this dataset, so they can be
asserted directly and will fail loudly if someone retunes a constant without
realising what it does.
"""

import json
from pathlib import Path

import pytest

import backend.tools.risk as risk_mod
from evaluation.ablation import (
    ALL_RULES,
    _patched,
    fuse,
    run_detection_stack,
    study_fusion_weights,
    study_rules,
)
from evaluation.harness import load_ground_truth

_REPO_ROOT = Path(__file__).resolve().parent.parent
# The live baseline is the newest snapshot in evaluation/results/, not a file
# literally named "baseline" — the earlier ones record what the system produced
# at a point in time and are never regenerated. Keep this in step with
# scripts/check_baselines.py, which is what CI diffs against.
_BASELINE = _REPO_ROOT / "evaluation" / "results" / "after_r3_fix.json"


def _hits(*specs):
    """(entity, rule, weight) triples -> rule_hit dicts."""
    return [
        {"entity_id": e, "rule_id": r, "weight": w, "evidence": {}}
        for e, r, w in specs
    ]


def _scores(*specs):
    """(entity, percentile) pairs -> ml_score dicts."""
    return [
        {"entity_id": e, "percentile": p, "score": -0.5, "top_features": []}
        for e, p in specs
    ]


# ---------------------------------------------------------------------------
# The re-fusion machinery
# ---------------------------------------------------------------------------


def test_patched_restores_constants_even_on_error():
    before = risk_mod.RULE_WEIGHT_COEFF
    with pytest.raises(RuntimeError):
        with _patched(RULE_WEIGHT_COEFF=0.1):
            assert risk_mod.RULE_WEIGHT_COEFF == 0.1
            raise RuntimeError("boom")
    assert risk_mod.RULE_WEIGHT_COEFF == before


def test_fuse_reaches_the_real_scoring_code():
    """0.6 * 0.85 * 100 = 51. If this drifts, the study is measuring a copy."""
    rows = fuse(_hits(("C-1", "R1", 0.85)), [])
    assert rows[0]["risk_score"] == pytest.approx(51.0)


# ---------------------------------------------------------------------------
# Structural findings — true of the formula, not just of this dataset
# ---------------------------------------------------------------------------


def test_neither_component_alone_can_reach_the_high_band():
    """The SAR tier is definitionally hybrid.

    Highest rule weight is R1 at 0.85, so rules-only tops out at 0.6*0.85*100 =
    51. ML-only tops out at 0.4*1.0*100 = 40. Both sit under the HIGH band of
    70, so a HIGH flag is arithmetically impossible without both signals.
    """
    rules_only = fuse(_hits(("C-1", "R1", 0.85)), [])
    ml_only = fuse([], _scores(("C-2", 1.0)))

    assert all(r["risk_level"] != "high" for r in rules_only)
    assert all(r["risk_level"] != "high" for r in ml_only)

    both = fuse(_hits(("C-1", "R1", 0.85)), _scores(("C-1", 1.0)))
    assert both[0]["risk_level"] == "high"


def test_fusion_weights_cannot_change_who_gets_flagged():
    """The headline finding of section 3, asserted as a property.

    Membership in risk_rows is decided by the entity universe — has a rule hit,
    or an ML percentile above the floor — which does not consult the
    coefficients at all. They only redistribute severity within a fixed set, so
    no fusion split can move precision or recall.
    """
    rule_hits = _hits(("C-1", "R1", 0.85), ("C-2", "R4", 0.75))
    ml_scores = _scores(("C-1", 0.99), ("C-3", 0.97), ("C-4", 0.10))

    flagged_sets = set()
    for i in range(11):
        coeff = i / 10.0
        rows = fuse(
            rule_hits, ml_scores,
            RULE_WEIGHT_COEFF=coeff,
            ML_PERCENTILE_COEFF=round(1.0 - coeff, 10),
        )
        flagged_sets.add(frozenset(r["entity_id"] for r in rows))

    assert len(flagged_sets) == 1, "the flagged set moved with the fusion weights"
    # C-4 is below the ML-only floor and has no rule hit, so it is never scored.
    assert flagged_sets.pop() == {"C-1", "C-2", "C-3"}


def test_fusion_weights_do_change_severity():
    """The complement of the above — the weights are not inert, just confined
    to banding. Without this, the previous test would also pass if fuse() were
    ignoring the patched constants entirely."""
    rule_hits = _hits(("C-1", "R1", 0.85))
    ml_scores = _scores(("C-1", 0.10))

    rule_heavy = fuse(rule_hits, ml_scores, RULE_WEIGHT_COEFF=1.0, ML_PERCENTILE_COEFF=0.0)
    ml_heavy = fuse(rule_hits, ml_scores, RULE_WEIGHT_COEFF=0.0, ML_PERCENTILE_COEFF=1.0)

    assert rule_heavy[0]["risk_score"] > ml_heavy[0]["risk_score"]


# ---------------------------------------------------------------------------
# The per-rule study
# ---------------------------------------------------------------------------


def test_leaving_out_a_silent_rule_changes_nothing():
    """R5 and R6 fire zero times on this dataset, so their leave-one-out deltas
    must be exactly zero. A non-zero delta would mean the filtering is wrong."""
    _, gt = load_ground_truth()
    rule_hits = _hits(("C-1", "R1", 0.85), ("C-2", "R4", 0.75))
    rows = {r["rule"]: r for r in study_rules(rule_hits, _scores(("C-1", 0.99)), gt)}

    for silent in ("R5", "R6"):
        assert rows[silent]["hits"] == 0
        assert rows[silent]["delta_precision"] == 0.0
        assert rows[silent]["delta_recall"] == 0.0


def test_every_rule_is_reported_even_when_it_never_fires():
    """A rule that fires zero times is a finding, not a row to omit."""
    _, gt = load_ground_truth()
    rows = study_rules(_hits(("C-1", "R1", 0.85)), [], gt)
    assert [r["rule"] for r in rows] == list(ALL_RULES)


def test_rules_are_scored_under_both_definitions():
    """R7 is receiver-side and scores 0.000 under sender_only by construction.
    Reporting only that column would read as a broken rule rather than a
    definition artifact, so both must be present."""
    _, gt = load_ground_truth()
    rows = study_rules(_hits(("C-1", "R7", 0.75)), [], gt)
    r7 = next(r for r in rows if r["rule"] == "R7")
    assert "alone_precision" in r7
    assert "alone_precision_repeat" in r7


def test_fusion_sweep_covers_both_endpoints():
    _, gt = load_ground_truth()
    rows = study_fusion_weights(_hits(("C-1", "R1", 0.85)), _scores(("C-1", 0.99)), gt)
    assert [r["rule_coeff"] for r in rows][0] == 0.0
    assert [r["rule_coeff"] for r in rows][-1] == 1.0
    assert all(
        r["rule_coeff"] + r["ml_coeff"] == pytest.approx(1.0) for r in rows
    ), "the two coefficients must always partition 1.0"


# ---------------------------------------------------------------------------
# The real pipeline — slow, and the regression that matters most
# ---------------------------------------------------------------------------


def test_detection_stack_threads_artifacts_between_steps():
    """The bug that made the first run of this study report all zeros.

    The executor threads BOTH `ctx.df` and `ctx.artifacts` between steps. The
    first version of run_detection_stack copied only the first, so
    feature_engineer's output never reached rule_detect, every detector
    returned nothing, and the study printed a full table of zeros that looked
    like a result rather than a broken harness.
    """
    rule_hits, ml_scores, customers = run_detection_stack(source="synthetic")
    assert rule_hits, "no rule hits — artifacts are not being threaded between steps"
    assert ml_scores, "no ML scores — artifacts are not being threaded between steps"
    assert customers == 270


def test_ablation_agrees_with_the_published_evaluation():
    """The study and run_evaluation must describe the same system.

    They reach risk_classify by different routes — this one calls the tools
    directly, run_evaluation goes through build_plan and the executor — so
    agreement here is what licenses quoting ablation numbers alongside the
    README's.
    """
    baseline = json.loads(_BASELINE.read_text(encoding="utf-8"))
    expected = baseline["results"]["sender_only"]["any_flag"]

    _, gt = load_ground_truth()
    rule_hits, ml_scores, _ = run_detection_stack(source="synthetic")
    rows = fuse(rule_hits, ml_scores)

    assert len(rows) == expected["flagged"]
    # Read from the baseline rather than hardcoded: this assertion broke on the
    # R3 repair because the HIGH count was written into the test, so a correct
    # detection change looked like a test failure.
    assert (
        sum(1 for r in rows if r["risk_level"] == "high")
        == baseline["results"]["sender_only"]["high_only"]["flagged"]
    )
