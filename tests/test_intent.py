from datetime import date

import pytest

from backend.agent.intent_parser import parse_intent


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    monkeypatch.setattr("backend.agent.intent_parser.complete_json", lambda *a, **kw: None)


CASES = [
    ("Analyse this dataset for suspicious activity", "full_analysis"),
    ("Give me a full analysis of the transactions", "full_analysis"),
    ("Find structuring patterns in the last 30 days", "pattern_search"),
    ("Look for smurfing activity", "pattern_search"),
    ("Check for layering in wire transfers", "pattern_search"),
    ("Which customers made 10+ transactions under $10,000?", "threshold_query"),
    ("Find customers with at least 5 transactions under $9,000", "threshold_query"),
    ("Is customer ID 4521 suspicious?", "entity_investigation"),
    ("Investigate customer C-01187", "entity_investigation"),
    ("Top 10 highest-risk customers", "ranking"),
    ("Show me the top 5 riskiest accounts", "ranking"),
    ("Show transaction distribution by country", "eda"),
    ("Give me a breakdown of transactions by type", "eda"),
    ("Why was transaction T-8891 flagged?", "explain_flag"),
    ("Why is customer 4521 flagged?", "explain_flag"),
]


@pytest.mark.parametrize("query,expected_intent", CASES)
def test_intent_classification(query, expected_intent):
    result = parse_intent(query)
    assert result.intent == expected_intent, f"'{query}' -> {result.intent}, expected {expected_intent}"
    assert result.parsed_by == "rules"


def test_entity_extraction_bare_id():
    result = parse_intent("Is customer ID 4521 suspicious?")
    assert "C-04521" in result.entities


def test_entity_extraction_prefixed_id():
    result = parse_intent("Investigate customer C-01187")
    assert "C-01187" in result.entities


def test_entity_extraction_alphanumeric_real_id():
    """Real customer IDs aren't purely numeric (Track B's generator scheme:
    C-STR02, C-N0001, C-HUB01, ...) — the entity regex must recognise these,
    not just C-#####. Regression test for a bug found in Phase 7 hardening
    where 'Is customer C-STR02 suspicious?' misclassified as full_analysis
    because the ID failed to extract as an entity at all."""
    result = parse_intent("Is customer C-STR02 suspicious?")
    assert result.entities == ["C-STR02"]
    assert result.intent == "entity_investigation"


def test_date_filter_extraction():
    result = parse_intent("Find structuring patterns in the last 30 days")
    assert result.filters.date_from is not None
    assert result.filters.date_to is not None


def test_relative_date_anchored_to_dataset_not_wallclock():
    """'last 30 days' must resolve relative to the dataset's own max date
    (2025-03-31 in the committed sample), not date.today() — regression test
    for a bug found in Phase 7 hardening where this made the brief's own
    example query ('Find structuring patterns in the last 30 days') return
    zero results, since the real wall-clock date is nowhere near the
    dataset's 2025 date range."""
    result = parse_intent("Find structuring patterns in the last 30 days")
    assert result.filters.date_to < date(2026, 1, 1)
    assert result.filters.date_to != date.today()


def test_amount_and_count_filters():
    result = parse_intent("Which customers made 10+ transactions under $10,000?")
    assert result.filters.min_txn_count == 10
    assert result.filters.amount_max == 10000.0


def test_pattern_type_extraction():
    result = parse_intent("Find structuring patterns in the last 30 days")
    assert "structuring" in result.pattern_types


def test_top_n_extraction():
    result = parse_intent("Top 5 highest-risk customers")
    assert result.top_n == 5


def test_llm_result_with_relative_date_shorthand_and_none_fields_still_parses_as_llm(monkeypatch):
    """Regression test for a real bug found live: both Gemini ("-30d"/"now")
    and Groq ("1 month ago") return relative-date shorthand instead of ISO
    dates for filters.date_from/date_to, and Groq also returns explicit
    `None` for confidence/top_n/countries/txn_types — all of which used to
    fail Filters/QueryIntent's strict validation and silently discard the
    ENTIRE LLM-parsed result (intent, entities, patterns included) in favour
    of the regex fallback. This is the exact dict captured live from Groq for
    "who are the top 10 suspicious customers in the last month"."""
    captured_groq_output = {
        "intent": "ranking",
        "filters": {"date_from": "1 month ago", "date_to": "now", "countries": None, "txn_types": None,
                    "amount_min": None, "amount_max": None, "min_txn_count": None, "customer_segment": None},
        "entities": [],
        "pattern_types": ["structuring", "smurfing", "layering", "rapid_cashout", "velocity", "dormant_reactivation"],
        "top_n": 10,
        "confidence": None,
    }
    monkeypatch.setattr("backend.agent.intent_parser.complete_json", lambda *a, **kw: captured_groq_output)

    result = parse_intent("who are the top 10 suspicious customers in the last month",
                           reference_date=date(2025, 3, 31))

    assert result.parsed_by == "llm"
    assert result.intent == "ranking"
    assert result.top_n == 10
    assert result.confidence == 0.0  # field default, not a validation crash
    assert result.filters.date_from == date(2025, 3, 1)  # "1 month ago" resolved against reference_date
    assert result.filters.date_to == date(2025, 3, 31)   # "now" resolved against reference_date
    assert result.filters.countries == []
    assert result.filters.txn_types == []


@pytest.mark.parametrize("supplied,expected", [
    (["all"], []),
    (["any"], []),
    (["*"], []),
    (["ALL"], []),
    (["all", "wire"], ["wire"]),
    (["wire", "cash"], ["wire", "cash"]),
])
def test_select_everything_placeholders_are_dropped_from_filters(monkeypatch, supplied, expected):
    """Observed live: asked for smurfing with no type restriction, the model
    returned txn_types=['all']. It is not a transaction type, matches nothing,
    and filter_data correctly ignored it — but the execution-plan trace then
    showed `Filters: txn_types=['all']`, which reads as though a filter had
    been applied. An empty list says the same thing truthfully."""
    monkeypatch.setattr(
        "backend.agent.intent_parser.complete_json",
        lambda *a, **kw: {
            "intent": "pattern_search",
            "filters": {"txn_types": supplied},
            "pattern_types": ["smurfing"],
            "confidence": 0.9,
        },
    )
    result = parse_intent("show me any smurfing behaviour", reference_date=date(2025, 3, 31))

    assert result.parsed_by == "llm"
    assert result.filters.txn_types == expected


def test_llm_result_with_gemini_style_shorthand_dates(monkeypatch):
    """Same bug class, Gemini's shorthand form ("-30d") rather than Groq's
    ("1 month ago") — both must resolve through the same coercion path."""
    llm_output = {
        "intent": "pattern_search",
        "filters": {"date_from": "-30d", "date_to": "now"},
        "entities": [], "pattern_types": ["structuring"], "top_n": 10, "confidence": 0.98,
    }
    monkeypatch.setattr("backend.agent.intent_parser.complete_json", lambda *a, **kw: llm_output)

    result = parse_intent("Find structuring patterns in the last 30 days",
                           reference_date=date(2025, 3, 31))

    assert result.parsed_by == "llm"
    assert result.filters.date_from == date(2025, 3, 1)
    assert result.filters.date_to == date(2025, 3, 31)


def test_unrecognisable_date_string_drops_field_without_failing_the_parse(monkeypatch):
    """An LLM date value that isn't ISO, "now"/"today", or a recognised
    relative-shorthand pattern should be dropped (None) rather than crash the
    whole QueryIntent construction and lose the LLM's otherwise-good parse."""
    llm_output = {"intent": "eda", "filters": {"date_from": "sometime in the spring"},
                  "entities": [], "pattern_types": [], "top_n": 10, "confidence": 0.7}
    monkeypatch.setattr("backend.agent.intent_parser.complete_json", lambda *a, **kw: llm_output)

    result = parse_intent("show me spring transactions")

    assert result.parsed_by == "llm"
    assert result.filters.date_from is None
