"""
Intent parsing: natural-language query -> QueryIntent. Owner: Track A.

LLM path first (backend.llm.client.complete_json); on any failure or invalid
response, falls back to a deterministic regex/keyword parser that alone must
cover all 7 intents well enough to demo on (see docs/CONTRACTS.md Contract 4).
"""

import functools
import re
from datetime import date, timedelta
from typing import Any

import pandas as pd

from backend.config import settings
from backend.llm.client import complete_json
from backend.schemas import Filters, PatternType, QueryIntent

PATTERN_KEYWORDS: dict[str, PatternType] = {
    "structuring": "structuring",
    "smurfing": "smurfing",
    "smurf": "smurfing",
    "layering": "layering",
    "rapid cash": "rapid_cashout",
    "cash-out": "rapid_cashout",
    "cash out": "rapid_cashout",
    "velocity": "velocity",
    "dormant": "dormant_reactivation",
}

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

# Real customer IDs aren't purely numeric (e.g. C-STR02, C-N0001 — Track B's
# generator scheme), so the ID portion accepts any 2-8 alphanumeric chars
# after the C-/T- prefix, not just digits.
ENTITY_RE = re.compile(r"\b(?:customer|cust|account|acct)?\s*(?:id\s*)?([CT]-[A-Z0-9]{2,8})\b", re.I)
BARE_ID_RE = re.compile(r"\b(\d{4,6})\b")
COUNT_RE = re.compile(r"(\d+)\s*\+?\s*(?:or more\s*)?transactions?", re.I)
AMOUNT_RE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)")
TOPN_RE = re.compile(r"top\s+(\d+)", re.I)
LAST_N_RE = re.compile(r"last\s+(\d+)\s+(day|week|month)s?", re.I)
SINCE_RE = re.compile(r"since\s+(\d{4}-\d{2}-\d{2})", re.I)
IN_MONTH_RE = re.compile(r"\bin\s+(" + "|".join(MONTHS) + r")\b", re.I)

_SCHEMA_HINT = (
    "Return JSON with keys: intent (one of full_analysis, pattern_search, threshold_query, "
    "entity_investigation, ranking, eda, explain_flag), filters (object with optional date_from, "
    "date_to, countries, txn_types, amount_min, amount_max, min_txn_count, customer_segment), "
    "entities (list of customer/transaction IDs mentioned, normalised like C-04521 or T-008891), "
    "pattern_types (list from structuring, smurfing, layering, rapid_cashout, velocity, "
    "dormant_reactivation), top_n (int, default 10), confidence (0-1 float). "
    "IMPORTANT: countries must be ISO-3166 alpha-2 codes (e.g. 'DE' not 'Germany', "
    "'GB' not 'United Kingdom', 'US' not 'United States'). "
    "IMPORTANT: amount_min is a lower bound, amount_max is an upper bound — never set them to "
    "the same value unless the query asks for an exact amount. 'under/below/less than $X' means "
    "amount_max=X only (leave amount_min unset). 'over/above/more than/at least $X' means "
    "amount_min=X only (leave amount_max unset). Only set both when the query names two distinct "
    "bounds, e.g. 'between $X and $Y'."
)


@functools.lru_cache(maxsize=1)
def _dataset_reference_date() -> date:
    """'Today', for relative date phrases like 'last 30 days'.

    Anchored to the working dataset's own max transaction date, not
    wall-clock time — the demo dataset is dated 2025-01-01..2025-03-31 and
    will never be "recent" relative to whenever this actually runs. Falls
    back to date.today() if the dataset can't be read (e.g. mocks-only runs
    with no CSV on disk). Cached for the process lifetime — the CSV is
    static, no reason to re-read it on every query.
    """
    try:
        df = pd.read_csv(settings.aml_dataset_path, usecols=["timestamp"], parse_dates=["timestamp"])
        return df["timestamp"].max().date()
    except Exception:
        return date.today()


def parse_intent(raw_query: str, reference_date: date | None = None) -> QueryIntent:
    reference_date = reference_date or _dataset_reference_date()

    llm_result = complete_json(f'Classify this AML compliance query: "{raw_query}"', _SCHEMA_HINT)
    if llm_result is not None:
        try:
            sanitized = _sanitize_llm_result(llm_result, reference_date, raw_query)
            return QueryIntent(raw_query=raw_query, parsed_by="llm", **sanitized)
        except Exception:
            pass  # malformed LLM output -> fall through to the regex parser

    return _parse_with_rules(raw_query, reference_date)


_RELATIVE_SHORTHAND_RE = re.compile(r"^-?(\d+)\s*(d|day|days|w|week|weeks|m|month|months|y|year|years)$", re.I)
_RELATIVE_AGO_RE = re.compile(r"^(\d+)\s+(day|days|week|weeks|month|months|year|years)\s+ago$", re.I)
_UNIT_DAYS = {"d": 1, "w": 7, "m": 30, "y": 365}


def _coerce_relative_date(value: Any, reference_date: date) -> str | None:
    """LLM providers commonly return relative-date shorthand for filters
    (Gemini: "-30d"/"now"; Groq: "1 month ago") instead of ISO dates. Filters'
    strict `date` type rejects these, which previously threw away the *entire*
    LLM-parsed QueryIntent (intent, entities, patterns — everything) over one
    bad field, silently falling back to the regex parser. Coerce what can be
    recognised, anchored to the same dataset reference date the regex fallback
    uses; drop (return None) anything unrecognisable rather than fail the parse.
    """
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        date.fromisoformat(text)
        return text
    except ValueError:
        pass
    lowered = text.lower()
    if lowered in ("now", "today"):
        return reference_date.isoformat()
    m = _RELATIVE_SHORTHAND_RE.match(lowered) or _RELATIVE_AGO_RE.match(lowered)
    if m:
        n = int(m.group(1))
        unit_days = _UNIT_DAYS[m.group(2)[0]]
        return (reference_date - timedelta(days=n * unit_days)).isoformat()
    return None


# Full country name → ISO-3166 alpha-2 normalisation.
# LLMs commonly return full names even when the prompt says ISO-2.
# This map covers every country that appears in the synthetic and IBM datasets;
# extend it if new datasets add new countries.
_COUNTRY_NAME_TO_ISO2: dict[str, str] = {
    # Full English names
    "united states": "US", "united states of america": "US",
    "united kingdom": "GB", "great britain": "GB",
    "germany": "DE", "deutschland": "DE",
    "singapore": "SG",
    "japan": "JP",
    "canada": "CA",
    "australia": "AU",
    "france": "FR",
    # Common LLM variants / abbreviations
    "usa": "US", "uk": "GB", "aus": "AU", "jpn": "JP",
    "can": "CA", "fra": "FR", "deu": "DE", "sgp": "SG",
}


def _normalise_country(value: str) -> str:
    """Convert a country name or 3-letter code to ISO-2, pass through ISO-2 as-is."""
    stripped = value.strip()
    if len(stripped) == 2:
        return stripped.upper()          # already ISO-2
    return _COUNTRY_NAME_TO_ISO2.get(stripped.lower(), stripped.upper())


_UNDER_WORDS = ("under $", "below $", "less than $")
_OVER_WORDS = ("over $", "above $", "more than $", "at least $")


def _fix_last_n_days_window(filters: dict, raw_query: str, reference_date: date) -> None:
    """Explicit "last N days/weeks/months" phrasing is unambiguous — it always
    means [reference_date - N, reference_date]. Observed the LLM invert this:
    e.g. for "last 30 days" it returned date_from='' and date_to='30 days ago',
    which _coerce_relative_date faithfully turns into an upper bound of 30
    days ago with NO lower bound at all — "everything up to 30 days ago",
    the opposite of the intended trailing window. Since this phrasing has one
    unambiguous meaning, just compute it directly (same formula the regex
    fallback in _parse_with_rules already uses) and override whatever the LLM
    proposed for these two fields.
    """
    m = LAST_N_RE.search(raw_query.lower())
    if not m:
        return
    n, unit = int(m.group(1)), m.group(2)
    days = n if unit == "day" else n * 7 if unit == "week" else n * 30
    filters["date_from"] = (reference_date - timedelta(days=days)).isoformat()
    filters["date_to"] = reference_date.isoformat()


def _fix_one_sided_amount_bound(filters: dict, raw_query: str) -> None:
    """Some LLM providers (observed with weaker/local models) mishandle a
    one-sided amount phrase like "under $10,000" in two ways: (a) collapsing
    it into amount_min=amount_max=X, which then filters for transactions
    equal to exactly X and empties the result, or (b) putting X in the wrong
    bound entirely (amount_min instead of amount_max for "under"). Detect the
    one-sided phrasing in the raw query text and correct the bound(s)
    accordingly, mirroring the equivalent regex-fallback logic in
    _parse_with_rules.
    """
    amount_min = filters.get("amount_min")
    amount_max = filters.get("amount_max")
    q = raw_query.lower()
    is_under = any(w in q for w in _UNDER_WORDS)
    is_over = any(w in q for w in _OVER_WORDS)
    if is_under and not is_over:
        if amount_min is not None and amount_max is None:
            filters["amount_max"] = filters.pop("amount_min")
        elif amount_min is not None and amount_max is not None and amount_min == amount_max:
            filters.pop("amount_min", None)
    elif is_over and not is_under:
        if amount_max is not None and amount_min is None:
            filters["amount_min"] = filters.pop("amount_max")
        elif amount_min is not None and amount_max is not None and amount_min == amount_max:
            filters.pop("amount_max", None)


def _sanitize_llm_result(llm_result: dict, reference_date: date, raw_query: str = "") -> dict:
    result = dict(llm_result)
    # confidence/top_n are non-Optional QueryIntent fields — an explicit None
    # from the LLM must be dropped so the field default applies, not passed
    # through (Pydantic rejects None for a plain `float`/`int` field).
    for key in ("confidence", "top_n"):
        if result.get(key) is None:
            result.pop(key, None)
    filters = result.get("filters")
    if isinstance(filters, dict):
        for key in ("date_from", "date_to"):
            if key in filters:
                filters[key] = _coerce_relative_date(filters[key], reference_date)
        _fix_last_n_days_window(filters, raw_query, reference_date)
        # countries/txn_types are non-Optional `list[str] = []` Filters fields —
        # same class of bug as confidence/top_n above, but nested.
        for key in ("countries", "txn_types"):
            if filters.get(key) is None:
                filters.pop(key, None)
        # Drop "select everything" placeholders. Asked for smurfing behaviour
        # with no type restriction, the model returns txn_types=['all'] — which
        # is not a transaction type, matches nothing, and is correctly ignored
        # downstream ("filter_data: no filters applied"). Harmless, but it shows
        # in the execution-plan trace as `Filters: txn_types=['all']`, which
        # reads like a filter was applied when none was. An empty list says the
        # same thing truthfully.
        for key in ("countries", "txn_types"):
            values = filters.get(key)
            if isinstance(values, list):
                cleaned = [
                    v for v in values
                    if str(v).strip().lower() not in {"all", "any", "*", ""}
                ]
                if cleaned:
                    filters[key] = cleaned
                else:
                    filters.pop(key, None)
        # Normalise country names → ISO-2 codes so they match the dataset columns.
        # LLMs often return 'Germany' even when told to use 'DE'.
        if filters.get("countries"):
            filters["countries"] = [
                _normalise_country(c) for c in filters["countries"]
            ]
        _fix_one_sided_amount_bound(filters, raw_query)
        # customer_segment must be a plain string — smaller/weaker models
        # (observed with a local 3B Ollama model) sometimes wrap it in a list.
        seg = filters.get("customer_segment")
        if isinstance(seg, list):
            filters["customer_segment"] = seg[0] if seg and isinstance(seg[0], str) else None
            seg = filters["customer_segment"]
        elif seg is not None and not isinstance(seg, str):
            filters.pop("customer_segment", None)
            seg = None
        # customer_segment only supports 'business' | 'pep' | 'high_risk'
        # (see backend/tools/filters.py _filter_customer_segment). Observed
        # the LLM sometimes stuffing an entity ID in here instead (e.g.
        # 'C-STR02') when it also puts the same ID in `entities` — drop it
        # rather than pass a value filter_data will reject with a warning.
        if isinstance(seg, str) and seg.lower().strip() not in ("business", "pep", "high_risk"):
            filters.pop("customer_segment", None)

    # entities must be plain ID strings — smaller models sometimes wrap them in
    # descriptive objects instead (e.g. {"entity_type": "customer", "value": "..."})
    # or invent an "entity" out of a non-ID phrase entirely (e.g. "3 sketchiest
    # customers" for a ranking query that has no real entity at all).
    if isinstance(result.get("entities"), list):
        result["entities"] = _sanitize_entities(result["entities"])

    # Guard: a query that names a real entity ID and asks an investigative
    # question about it ("is X suspicious", "investigate X", "X's risk") is
    # always entity_investigation, regardless of what the LLM guessed —
    # observed the LLM misroute these to pattern_search (sometimes biased by
    # an unrelated substring in the ID, e.g. "STR" in "C-STR02" reading like
    # "structuring"). Mirrors the equivalent, already-correct regex-fallback
    # logic in _classify(). Explicit "why ... flagged" phrasing is left alone
    # so explain_flag queries aren't reclassified.
    q_lower = raw_query.lower()
    if (
        result.get("entities")
        and result.get("intent") not in ("entity_investigation", "explain_flag")
        and not q_lower.startswith("why")
        and "why was" not in q_lower
        and "why is" not in q_lower
        and any(w in q_lower for w in ("suspicious", "risk", "investigate", "flagged", "flag"))
    ):
        result["intent"] = "entity_investigation"

    # Guard: entity_investigation without a real entity_id always errors in
    # entity_lookup ("entity_id is required").  If the LLM classified the
    # intent as entity_investigation but extracted zero valid entity IDs,
    # downgrade to ranking — the correct intent for population-level queries
    # like "show me risk scores of customers from Germany".
    if result.get("intent") == "entity_investigation" and not result.get("entities"):
        result["intent"] = "ranking"
        result["confidence"] = min(float(result.get("confidence") or 0.6), 0.7)

    return result


def _sanitize_entities(entities: list) -> list[str]:
    cleaned: list[str] = []
    for e in entities:
        candidate: str | None = None
        if isinstance(e, str):
            candidate = e
        elif isinstance(e, dict):
            for key in ("value", "id", "entity_id", "customer_id"):
                if isinstance(e.get(key), str):
                    candidate = e[key]
                    break
        if not candidate:
            continue
        # only keep it if it actually looks like an ID (C-/T- prefixed, or a
        # bare 4-6 digit number) — rejects invented descriptive phrases like
        # "3 sketchiest customers" rather than passing them through as if they
        # were a real entity to look up.
        m = ENTITY_RE.search(candidate)
        if m:
            cleaned.append(m.group(1).upper())
            continue
        m = BARE_ID_RE.search(candidate)
        if m:
            cleaned.append(_normalise_bare_number(m.group(1)))
    return cleaned


def _parse_with_rules(raw_query: str, reference_date: date) -> QueryIntent:
    q = raw_query.lower()
    filters = Filters()
    confidence = 0.6

    m = LAST_N_RE.search(q)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        days = n if unit == "day" else n * 7 if unit == "week" else n * 30
        filters.date_from = reference_date - timedelta(days=days)
        filters.date_to = reference_date

    m = SINCE_RE.search(q)
    if m:
        filters.date_from = date.fromisoformat(m.group(1))

    m = IN_MONTH_RE.search(q)
    if m:
        month = MONTHS[m.group(1).lower()]
        year = reference_date.year
        filters.date_from = date(year, month, 1)
        next_month = date(year + (1 if month == 12 else 0), 1 if month == 12 else month + 1, 1)
        filters.date_to = next_month - timedelta(days=1)

    if "$" in q and any(w in q for w in ["under $", "below $", "less than $"]):
        dollar_idx = q.find("$")
        am = AMOUNT_RE.search(q[dollar_idx:])
        if am:
            filters.amount_max = float(am.group(1).replace(",", ""))

    m = COUNT_RE.search(q)
    if m:
        filters.min_txn_count = int(m.group(1))

    top_n = 10
    m = TOPN_RE.search(q)
    if m:
        top_n = int(m.group(1))

    entities: list[str] = [m.group(1).upper() for m in ENTITY_RE.finditer(raw_query)]
    if not entities:
        # no C-/T- prefixed token found — fall back to a bare number and
        # construct a plausible ID; backend.agent.executor._resolve_entities()
        # reconciles this against the real dataset's customer_ids by numeric
        # id once load_data runs, so an exact guess isn't required here.
        entities = [_normalise_bare_number(m.group(1)) for m in BARE_ID_RE.finditer(raw_query)]

    pattern_types: list[PatternType] = []
    for kw, pt in PATTERN_KEYWORDS.items():
        if kw in q and pt not in pattern_types:
            pattern_types.append(pt)

    intent = _classify(q, filters, entities, pattern_types)
    if intent == "full_analysis" and not any(
        w in q for w in ["analyse", "analyze", "suspicious activity", "full analysis", "overview"]
    ):
        confidence = 0.3  # true fallback, not a confident full_analysis read

    return QueryIntent(
        raw_query=raw_query,
        intent=intent,
        filters=filters,
        entities=entities,
        pattern_types=pattern_types,
        top_n=top_n,
        confidence=confidence,
        parsed_by="rules",
    )


def _classify(q: str, filters: Filters, entities: list[str], pattern_types: list[PatternType]) -> str:
    if q.startswith("why") or "why was" in q or "why is" in q:
        return "explain_flag"
    if entities and any(w in q for w in ["suspicious", "risk", "investigate", "flagged", "flag"]):
        return "entity_investigation"
    if TOPN_RE.search(q) or any(w in q for w in ["highest risk", "riskiest", "top risk"]):
        return "ranking"
    if filters.min_txn_count is not None:
        return "threshold_query"
    if pattern_types:
        return "pattern_search"
    if any(w in q for w in ["distribution", "breakdown", "show me", "chart", "how many", "by country", "by type"]):
        return "eda"
    return "full_analysis"


def _normalise_bare_number(raw_digits: str) -> str:
    return f"C-{raw_digits.zfill(5)}"
