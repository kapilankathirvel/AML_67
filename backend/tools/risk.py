"""
Track B — risk.py

Tool name  : risk_classify  (Contract 2 fixed list)
Input      : ctx.artifacts["rule_hits"]  — list[{entity_id, rule_id, evidence, weight}]
             ctx.artifacts["ml_scores"]  — list[{entity_id, score, percentile, top_features}]

Output     : ToolResult.artifacts["risk_rows"]
               list[{entity_id, risk_score, risk_level, escalation,
                     patterns, triggered_rules, evidence}]
             ToolResult.tables["risk_summary"] — flat table for the UI

Formula (Contract 5):
    risk_score = 100 * (0.6 * normalized_rule_weight + 0.4 * ml_percentile)

    normalized_rule_weight = max(hit.weight for hits on entity)
    ml_percentile          = ml_scores[entity].percentile  (0.0 if not scored)

Risk bands (Contract 5 / AML_LOGIC.md):
    ≥ 70  → high     → report   (SAR escalation path)
    40–69 → medium   → review
    15–39 → low      → monitor
     < 15 → none     → no_action

Design notes:
  - An entity only appears in risk_rows if it has ≥ 1 rule hit OR an ML score
    in the top contamination band (percentile ≥ 0.95).
  - If rule_hits is empty, risk_classify falls back to ML-only scoring with
    weight = 0 → risk_score = 40 * ml_percentile (max 40, capped at medium).
  - Rule weight is the MAX weight across all rules that fired on the entity
    (per AML_LOGIC.md: a customer triggering R1 + R3 is more dangerous, but
    we don't double-count the weight — both rules are still surfaced in evidence).
  - patterns list is derived from rule_id → pattern name mapping.
  - evidence list is the raw evidence dicts from rule_hits (for narrator).

No tool may import from backend.agent.* or from another tool.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from backend.tools.base import ToolContext, ToolResult, tool

# ---------------------------------------------------------------------------
# Constants (Contract 5 / AML_LOGIC.md §6)
# ---------------------------------------------------------------------------

RISK_HIGH_THRESHOLD   = 70.0
RISK_MEDIUM_THRESHOLD = 40.0
RISK_LOW_THRESHOLD    = 15.0

RULE_WEIGHT_COEFF  = 0.60
ML_PERCENTILE_COEFF = 0.40

# ML-only entities (no rule hits) are only included if above this percentile
ML_ONLY_PERCENTILE_FLOOR = 0.95

# Rule ID → AML pattern name
_RULE_TO_PATTERN: dict[str, str] = {
    "R1": "structuring",
    "R2": "smurfing",
    "R3": "layering",
    "R4": "rapid_cashout",
    "R5": "velocity",
    "R6": "dormant_reactivation",
    # R7 is receiver-side structuring — the beneficiary account of the same
    # scheme R1 detects on the sending side. It maps onto the existing
    # "structuring" pattern deliberately: it is the same typology viewed from
    # the other end of the transaction, and reusing the name avoids widening
    # the frozen PatternType literal in backend/schemas.py.
    "R7": "structuring",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _risk_band(score: float) -> tuple[str, str]:
    """Map a risk score to (risk_level, escalation)."""
    if score >= RISK_HIGH_THRESHOLD:
        return "high", "report"
    if score >= RISK_MEDIUM_THRESHOLD:
        return "medium", "review"
    if score >= RISK_LOW_THRESHOLD:
        return "low", "monitor"
    return "none", "no_action"


# ---------------------------------------------------------------------------
# Registered tool
# ---------------------------------------------------------------------------


@tool(
    name="risk_classify",
    params={},
    description=(
        "Fuse rule_hits and ml_scores into per-entity risk scores. "
        "Reads ctx.artifacts['rule_hits'] and ctx.artifacts['ml_scores']. "
        "Emits artifacts['risk_rows'] as list[{entity_id, risk_score, risk_level, "
        "escalation, patterns, triggered_rules, evidence}] and "
        "tables['risk_summary'] for the UI."
    ),
)
def risk_classify(
    ctx: ToolContext,
    **kw,
) -> ToolResult:
    """Fuse rule hits and ML scores into per-entity risk classifications.

    Contract 5 formula:
        risk_score = 100 * (0.6 * max_rule_weight + 0.4 * ml_percentile)
    """
    try:
        rule_hits: list[dict[str, Any]] = ctx.artifacts.get("rule_hits", [])
        ml_scores: list[dict[str, Any]] = ctx.artifacts.get("ml_scores", [])

        # ------------------------------------------------------------------
        # Index by entity_id
        # ------------------------------------------------------------------
        # rule hits: entity → {max_weight, rules, patterns, evidences}
        rule_by_entity: dict[str, dict[str, Any]] = {}
        for hit in rule_hits:
            eid = str(hit["entity_id"])
            w   = float(hit.get("weight", 0.0))
            rid = str(hit.get("rule_id", ""))
            ev  = hit.get("evidence", {})

            if eid not in rule_by_entity:
                rule_by_entity[eid] = {
                    "max_weight": w,
                    "rules":      [rid],
                    "patterns":   [_RULE_TO_PATTERN.get(rid, "unknown")],
                    "evidences":  [ev],
                }
            else:
                rule_by_entity[eid]["max_weight"] = max(rule_by_entity[eid]["max_weight"], w)
                if rid not in rule_by_entity[eid]["rules"]:
                    rule_by_entity[eid]["rules"].append(rid)
                pname = _RULE_TO_PATTERN.get(rid, "unknown")
                if pname not in rule_by_entity[eid]["patterns"]:
                    rule_by_entity[eid]["patterns"].append(pname)
                rule_by_entity[eid]["evidences"].append(ev)

        # ml scores: entity → {percentile, top_features}
        ml_by_entity: dict[str, dict[str, Any]] = {
            str(m["entity_id"]): m
            for m in ml_scores
        }

        # ------------------------------------------------------------------
        # Build the entity universe: all entities with rule hits + ML-only
        # entities above the floor percentile
        # ------------------------------------------------------------------
        all_entities: set[str] = set(rule_by_entity.keys())
        for eid, m in ml_by_entity.items():
            if m.get("percentile", 0.0) >= ML_ONLY_PERCENTILE_FLOOR:
                all_entities.add(eid)

        if not all_entities:
            return ToolResult(
                ok=True,
                artifacts={"risk_rows": []},
                tables={"risk_summary": []},
                metrics={
                    "total_flagged": 0,
                    "high": 0, "medium": 0, "low": 0, "none": 0,
                },
                notes=["risk_classify: no rule hits and no ML anomalies above floor — clean dataset"],
            )

        # ------------------------------------------------------------------
        # Score each entity
        # ------------------------------------------------------------------
        risk_rows: list[dict[str, Any]] = []
        band_counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0, "none": 0}

        for eid in sorted(all_entities):
            rule_info = rule_by_entity.get(eid, {})
            ml_info   = ml_by_entity.get(eid, {})

            max_rule_weight = rule_info.get("max_weight", 0.0)
            ml_percentile   = ml_info.get("percentile", 0.0)

            risk_score = 100.0 * (
                RULE_WEIGHT_COEFF * max_rule_weight
                + ML_PERCENTILE_COEFF * ml_percentile
            )
            risk_score = min(100.0, max(0.0, risk_score))  # clamp [0, 100]

            risk_level, escalation = _risk_band(risk_score)
            band_counts[risk_level] += 1

            risk_rows.append({
                "entity_id":       eid,
                "risk_score":      round(risk_score, 2),
                "risk_level":      risk_level,
                "escalation":      escalation,
                "patterns":        rule_info.get("patterns", []),
                "triggered_rules": rule_info.get("rules", []),
                "ml_score":        round(float(ml_percentile), 4),
                "ml_top_features": ml_info.get("top_features", []),
                "evidence":        rule_info.get("evidences", []),
            })

        # Sort by risk_score descending
        risk_rows.sort(key=lambda r: r["risk_score"], reverse=True)

        # ------------------------------------------------------------------
        # Flat summary table for UI
        # ------------------------------------------------------------------
        summary_table = [
            {
                "entity_id":       r["entity_id"],
                "risk_score":      r["risk_score"],
                "risk_level":      r["risk_level"],
                "escalation":      r["escalation"],
                "patterns":        ", ".join(r["patterns"]),
                "triggered_rules": ", ".join(r["triggered_rules"]),
                "ml_score":        r["ml_score"],
            }
            for r in risk_rows
        ]

        note = (
            f"risk_classify: {len(risk_rows)} entities scored — "
            f"HIGH={band_counts['high']}, MEDIUM={band_counts['medium']}, "
            f"LOW={band_counts['low']}, NONE={band_counts['none']}"
        )

        return ToolResult(
            ok=True,
            artifacts={"risk_rows": risk_rows},
            tables={"risk_summary": summary_table},
            metrics={
                "total_flagged": len(risk_rows),
                **band_counts,
            },
            notes=[note],
        )

    except Exception as exc:  # noqa: BLE001
        return ToolResult(ok=False, error=f"risk_classify failed: {exc}")
