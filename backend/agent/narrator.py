"""
Narrator: turns risk_rows (artifact from risk_classify) into Flags with
human-readable explanations and escalation actions. Owner: Track A.

Template layer always runs and is always accurate (built from each hit's
evidence). LLM polish is optional and only rewrites — it is never given
license to invent a number.
"""

from typing import Any

from backend.config import settings
from backend.llm.client import complete_json
from backend.schemas import Escalation, Evidence, Flag, RiskLevel

RULE_NAMES = {
    "R1": "Structuring",
    "R2": "Smurfing",
    "R3": "Layering",
    "R4": "Rapid cash-out",
    "R5": "Velocity spike",
    "R6": "Dormant reactivation",
    "R7": "Inbound structuring",
}

ESCALATION_BY_LEVEL: dict[RiskLevel, Escalation] = {
    "high": "report",
    "medium": "review",
    "low": "monitor",
    "none": "no_action",
}


def build_flags(risk_rows: list[dict[str, Any]]) -> list[Flag]:
    flags: list[Flag] = []
    polished_count = 0
    for row in risk_rows:
        evidence = _build_evidence(row)
        risk_level: RiskLevel = row["risk_level"]
        # Cap LLM polish to the first N HIGH-risk rows (risk_classify already
        # emits risk_rows sorted by risk_score descending, so this is "top N"),
        # not "every HIGH-risk row" — a full_analysis run can produce 20+ HIGH
        # flags, and at several seconds per call that multiplies past the
        # frontend's request timeout (measured live: 23 HIGH flags -> 144s
        # total against local Ollama, vs. a 60s frontend timeout). Every row
        # still gets an accurate template-based explanation regardless.
        allow_llm_polish = risk_level == "high" and polished_count < settings.llm_polish_max_flags
        if allow_llm_polish:
            polished_count += 1
        explanation = _explain(row, evidence, allow_llm_polish=allow_llm_polish)
        escalation: Escalation = row.get("escalation") or ESCALATION_BY_LEVEL[risk_level]
        flags.append(
            Flag(
                entity_type=row.get("entity_type", "customer"),
                entity_id=row["entity_id"],
                risk_score=row["risk_score"],
                risk_level=risk_level,
                escalation=escalation,
                patterns=row.get("patterns", []),
                triggered_rules=row.get("triggered_rules", []),
                ml_score=row.get("ml_score"),
                evidence=evidence,
                explanation=explanation,
                sar_draft=_sar_draft(row, explanation) if risk_level == "high" else None,
            )
        )
    return flags


def _build_evidence(row: dict[str, Any]) -> list[Evidence]:
    """Adapt risk_classify's raw per-rule evidence into the frozen Evidence shape.

    rule_detect's evidence dicts are rule-specific (structuring's fields differ
    from layering's, etc.) — Contract 2 documents them as free-form, only
    Contract 1's Evidence model is fixed. risk_classify pairs `evidence[i]`
    positionally with `triggered_rules[i]` (see backend/tools/risk.py), so we
    use that pairing to synthesize a valid Evidence per hit. Already-conformant
    Evidence dicts (e.g. from the mock tools) pass through unchanged.
    """
    raw_list = row.get("evidence", [])
    rule_ids = row.get("triggered_rules", [])
    items: list[Evidence] = []

    for i, raw in enumerate(raw_list):
        if isinstance(raw, Evidence):
            items.append(raw)
            continue
        try:
            items.append(Evidence(**raw))
            continue
        except Exception:
            pass

        raw = raw if isinstance(raw, dict) else {}
        rule_id = rule_ids[i] if i < len(rule_ids) else None
        label = RULE_NAMES.get(rule_id, rule_id or "Rule")
        formatted = _format_raw_evidence(raw)
        note = f"{label} — {formatted}" if formatted else f"{label} triggered."
        items.append(Evidence(rule_id=rule_id, feature=None, value=formatted or (rule_id or "n/a"),
                               threshold=None, note=note))
    return items


def _format_raw_evidence(raw: dict[str, Any]) -> str:
    parts = []
    for k, v in raw.items():
        if isinstance(v, float):
            v_str = f"{v:.1%}" if 0 <= v <= 1 else f"{v:,.2f}"
        elif isinstance(v, list):
            v_str = ", ".join(str(x) for x in v[:5])
        else:
            v_str = str(v)
        parts.append(f"{k.replace('_', ' ')}={v_str}")
    return "; ".join(parts)


def _explain(row: dict[str, Any], evidence: list[Evidence], allow_llm_polish: bool = False) -> str:
    parts: list[str] = []
    for rule_id in row.get("triggered_rules", []):
        ev = next((e for e in evidence if e.rule_id == rule_id), None)
        if ev and ev.note:
            parts.append(ev.note)
        else:
            parts.append(f"{RULE_NAMES.get(rule_id, rule_id)} rule triggered.")

    if not parts and row.get("ml_score") is not None:
        feats = row.get("ml_top_features") or []
        feat_txt = f" Top contributing features: {', '.join(feats)}." if feats else ""
        parts.append(
            f"Flagged by anomaly detection (percentile {row['ml_score']:.0%}) — no single rule matched, "
            f"but the transaction pattern deviates significantly from this entity's baseline.{feat_txt}"
        )
    if not parts:
        parts.append("Flagged for review based on the query's risk criteria.")

    text = " ".join(parts)

    # LLM polish is capped by build_flags() to the top settings.llm_polish_max_flags
    # HIGH-risk rows (not "every HIGH-risk row") — see that function for why.
    # MEDIUM/LOW/NONE, and any HIGH row past the cap, ship the template text,
    # which is already specific and accurate.
    if not allow_llm_polish:
        return text

    polished = complete_json(
        f"Rewrite this AML compliance evidence into one clear analyst-facing paragraph. "
        f"Use only the facts given, never invent numbers: {text}",
        schema_hint='Return JSON: {"paragraph": "..."}',
    )
    if polished and isinstance(polished.get("paragraph"), str) and polished["paragraph"].strip():
        return polished["paragraph"].strip()
    return text


def _sar_draft(row: dict[str, Any], explanation: str) -> str:
    patterns = ", ".join(row.get("patterns", [])) or "unspecified pattern"
    rules = ", ".join(row.get("triggered_rules", [])) or "anomaly detection"
    return (
        f"Suspicious Activity Report (draft) — Entity {row['entity_id']}. "
        f"Risk score {row['risk_score']:.0f}/100 (HIGH). Pattern(s): {patterns}. "
        f"Detection basis: {rules}. {explanation} Recommended action: file SAR / escalate to compliance for review."
    )
