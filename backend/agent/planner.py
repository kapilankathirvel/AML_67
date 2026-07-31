"""
Planner: QueryIntent -> ExecutionPlan. Owner: Track A.

Implements the intent -> tool mapping table in docs/CONTRACTS.md Contract 4
exactly. Every step carries a `reason`; every deliberately-omitted tool is
logged in `tools_considered_but_skipped` with a reason of its own.
"""

import uuid

from backend.schemas import ExecutionPlan, Filters, QueryIntent, ToolCall


def _filter_kwargs(filters: Filters) -> dict:
    """Flatten Filters into the individual kwargs backend/tools/filters.py::filter_data
    actually takes (it mirrors schemas.Filters field-for-field, but as separate
    params, not one nested dict) — see docs/CONTRACTS.md Contract 2."""
    return {
        "date_from": filters.date_from.isoformat() if filters.date_from else None,
        "date_to": filters.date_to.isoformat() if filters.date_to else None,
        "countries": filters.countries or None,
        "txn_types": filters.txn_types or None,
        "amount_min": filters.amount_min,
        "amount_max": filters.amount_max,
        "min_txn_count": filters.min_txn_count,
        "customer_segment": filters.customer_segment,
    }


def build_plan(intent: QueryIntent) -> ExecutionPlan:
    steps: list[ToolCall] = []
    skipped: list[str] = []
    decisions: list[str] = []

    def add(tool: str, reason: str, **params) -> None:
        steps.append(ToolCall(tool=tool, params={k: v for k, v in params.items() if v is not None}, reason=reason))

    def skip(tool: str, reason: str) -> None:
        skipped.append(f"{tool}: {reason}")

    filter_kwargs = _filter_kwargs(intent.filters)

    if intent.intent == "full_analysis":
        add("load_data", "full analysis requires the complete working dataset")
        add("eda_profile", "broad exploration requested — profile the dataset before detection")
        add("feature_engineer", "compute all AML features for a full sweep")
        add("rule_detect", "apply all rule-based detectors")
        add("ml_detect", "apply anomaly detection to catch patterns the rules miss")
        add("risk_classify", "fuse rule + ML signals into a final risk score")

    elif intent.intent == "pattern_search":
        add("load_data", "load the working dataset")
        add("filter_data", "narrow to the requested filters before detection", **filter_kwargs)
        add("feature_engineer", f"compute only the features needed for {intent.pattern_types or 'the requested pattern'}",
            pattern_types=intent.pattern_types)
        add("rule_detect", "apply rule-based detectors scoped to the requested pattern(s)", patterns=intent.pattern_types)
        add("ml_detect", "widen the net with anomaly detection alongside the targeted rules")
        add("risk_classify", "fuse rule + ML signals")
        skip("eda_profile", "user asked for a specific pattern, not exploration")

    elif intent.intent == "threshold_query":
        add("load_data", "load the working dataset")
        add("filter_data", "apply the query's explicit filters (amount/date/etc.) — including min_txn_count, "
            "so the frame handed to aggregate_query already contains only qualifying senders' transactions",
            **filter_kwargs)
        add("aggregate_query", "count each sender's (already-filtered) transactions and keep those meeting the threshold",
            group_by=["sender_id"], agg_func="count", threshold=intent.filters.min_txn_count)
        skip("feature_engineer", "no derived features needed for a direct count")
        skip("ml_detect", "a deterministic count answers this exactly — no anomaly detection needed")
        skip("eda_profile", "user asked a specific aggregation question, not exploration")

    elif intent.intent == "entity_investigation":
        entity_id = intent.entities[0] if intent.entities else None
        add("load_data", "load the working dataset")
        add("filter_data", "apply any date/country/etc. filters — filter_data has no entity dimension, "
            "so per-entity scoping is applied after risk_classify instead", **filter_kwargs)
        add("entity_lookup", "fetch the entity's profile and transaction summary", entity_id=entity_id)
        add("feature_engineer", "compute features across the population (required for a comparable risk score)",
            pattern_types=intent.pattern_types)
        add("rule_detect", "check all entities against rule-based detectors; result is filtered to this entity after",
            patterns=intent.pattern_types)
        # ml_detect used to be skipped here as "one entity is too small a sample".
        # That reasoning was wrong about this very plan: feature_engineer above runs
        # across the whole population precisely so the score is comparable, so
        # ml_detect receives the full customer set, not one row. Skipping it zeroed
        # the ML term and made every single-entity query score 100*0.6*max_weight —
        # C-STR02 came back 51.00 MEDIUM here while full_analysis called the same
        # customer 89.84 HIGH. Genuinely small samples are still handled, twice:
        # executor.py drops ml_detect under 50 rows, and ml_detect itself no-ops
        # below IF_MIN_SAMPLES.
        add("ml_detect", "score this entity against the whole population, so its risk score "
            "matches what a full sweep would give it")
        add("risk_classify", "compute risk scores; executor filters the result down to this entity")
        skip("eda_profile", "single-entity investigation, not exploration")

    elif intent.intent == "ranking":
        add("load_data", "load the working dataset")
        add("filter_data", "apply any filters before ranking", **filter_kwargs)
        add("feature_engineer", "compute features across the population to rank on")
        add("rule_detect", "apply rule-based detectors")
        add("ml_detect", "apply anomaly detection to catch patterns the rules miss")
        add("risk_classify", f"fuse signals; executor truncates to the top {intent.top_n} by risk score")
        skip("eda_profile", "ranking query, not exploration")

    elif intent.intent == "eda":
        add("load_data", "load the working dataset")
        add("filter_data", "apply any filters before profiling", **filter_kwargs)
        add("eda_profile", "user asked to look at the data, not to flag it")
        skip("feature_engineer", "no detection requested")
        skip("rule_detect", "no detection requested")
        skip("ml_detect", "no detection requested")
        skip("risk_classify", "no detection requested")

    elif intent.intent == "explain_flag":
        # NOTE: docs/CONTRACTS.md Contract 4 originally described this as "reuse a
        # cached run" — that was never wired to anything (no mechanism connects an
        # explain_flag query to a prior /query's plan_id), so it always returned
        # empty. Changed to compute risk fresh for just this entity, same shape as
        # entity_investigation, so the feature actually answers the question asked.
        entity_id = intent.entities[0] if intent.entities else None
        add("load_data", "load the working dataset — no cached run exists to reuse")
        add("entity_lookup", "fetch the entity's profile and transaction summary", entity_id=entity_id)
        add("feature_engineer", "compute features across the population (required for a comparable risk score)",
            pattern_types=intent.pattern_types)
        add("rule_detect", "check all entities against rule-based detectors; result is filtered to this entity after",
            patterns=intent.pattern_types)
        # Same fix as entity_investigation above. The old skip reason ("explaining an
        # existing rule-based flag, not re-scoring") contradicted the NOTE at the top
        # of this branch: there is no cached flag to explain, so this plan *does*
        # re-score from scratch. Omitting ml_detect meant the explanation quoted a
        # different risk score than the flag it was explaining.
        add("ml_detect", "the score being explained includes an ML term — recompute it "
            "rather than explain a number we did not produce")
        add("risk_classify", "compute risk scores; executor filters the result down to this entity")
        skip("eda_profile", "explaining a flag, not exploring")

    else:
        add("load_data", "unrecognised intent — falling back to full analysis on a sample")
        add("eda_profile", "fallback full analysis")
        add("feature_engineer", "fallback full analysis")
        add("rule_detect", "fallback full analysis")
        add("ml_detect", "fallback full analysis")
        add("risk_classify", "fallback full analysis")
        decisions.append(f"intent '{intent.intent}' unrecognised — defaulted to full_analysis")

    if intent.confidence and intent.confidence < 0.4:
        decisions.append(f"low parser confidence ({intent.confidence:.2f}) — plan may be revised if results look sparse")

    return ExecutionPlan(
        plan_id=uuid.uuid4().hex[:12],
        steps=steps,
        decisions=decisions,
        tools_considered_but_skipped=skipped,
    )
