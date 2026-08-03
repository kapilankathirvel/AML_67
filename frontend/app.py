"""
frontend/app.py

AML Suspicious Activity Detection — Streamlit UI.
Owner: Track B.

Talks to the backend through frontend/api_client.py, which is the only module
here that knows whether the backend is a separate process or this one. This
file still imports nothing from backend.agent.* or backend.tools.*.

OPERATION MODES
---------------
LIVE mode   : the backend answers /health.
              All queries go to /query. Sidebar shows live dataset summary.
              Two transports, chosen by api_client on the AML_API_URL env var:
                set   -> HTTP to that API. Two processes, the original design.
                unset -> the backend runs inside the Streamlit process, which
                         is what makes single-process hosts (Streamlit
                         Community Cloud) viable at all.
FIXTURE mode: backend not reachable. Queries are matched to a saved fixture JSON
              (frontend/fixtures/full_analysis.json) so the demo is never blocked.
              A banner clearly labels that fixture data is being shown.
              The client code path is IDENTICAL in both modes — fixtures are
              only loaded if the call fails, not instead of it.

Switching modes: start the API (uvicorn backend.main:app), set AML_API_URL to
it, and refresh. Unset AML_API_URL to run everything in one process.

Per WORKPLAN.md §7:
  "B never waits for A. Test every tool directly with pytest;
   the UI can render a saved AgentResponse JSON fixture until the API is live."
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# --- repo root on sys.path, before any frontend.* or backend.* import -------
#
# `streamlit run frontend/app.py` puts only the SCRIPT'S folder on sys.path
# (streamlit/web/bootstrap.py:60 does sys.path.insert(0, dirname(script))), not
# the repo root. So `from frontend import ...` and `from backend import ...`
# both fail under the launcher every host actually uses.
#
# It works locally purely by accident: run_demo.py invokes
# `python -m streamlit`, and -m puts the working directory on sys.path. Run the
# same app with the `streamlit` console script and it dies on the first import.
# Measured, not guessed — with the repo root removed and only frontend/ added,
# all three of frontend.api_client, frontend.components.plan_trace and
# backend.main raise ModuleNotFoundError.
#
# This must stay above the imports below, which is why it violates the usual
# "imports at the top" shape. tests/test_app_imports.py pins it.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# ---------------------------------------------------------------------------

import streamlit as st

from frontend import api_client
from frontend.components.plan_trace import render_plan_trace
from frontend.components.flag_cards import render_flag_cards
from frontend.components.charts import (
    render_charts,
    render_tables,
    render_kpi_row,
    render_risk_distribution,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _prime_secrets() -> None:
    """Force st.secrets to load before anything reads the environment.

    Streamlit copies every top-level string/int/float secret into os.environ,
    which is how a deployment configures backend/config.py without any code
    knowing about st.secrets. But it does that lazily, inside the first access
    to st.secrets — and backend/config.py builds its Settings object at IMPORT
    time.

    So the ordering is load-bearing. api_client imports the backend on the
    first query, and if nothing has touched st.secrets by then, Settings reads
    an environment the secrets have not been written to yet: the deployed app
    silently runs on defaults, analysing synthetic_alt with mocks on, and looks
    like it is working.

    Touching st.secrets here, before the first backend import, is what makes
    the secrets file actually take effect. Absent locally, where there is no
    secrets.toml and .env covers the same ground.
    """
    try:
        _ = st.secrets  # noqa: B018 — the access itself is the point
        len(_)          # AttrDict is lazy too; force the parse
    except Exception:
        # No secrets file (normal locally). Environment and .env still apply.
        pass


_prime_secrets()

# Kept for display only. The actual transport choice lives in api_client:
# AML_API_URL set means HTTP to that API, unset means the backend runs inside
# this process (which is what makes a single-process host like Streamlit
# Community Cloud viable). See frontend/api_client.py.
API_BASE_URL  = os.getenv("AML_API_URL", "").strip() or "in-process"
FIXTURE_DIR   = Path(__file__).parent / "fixtures"
REQUEST_TIMEOUT = 120   # seconds — must exceed LLM_TIMEOUT_SECONDS (50s) + pipeline (~5s)

# ---------------------------------------------------------------------------
# Example queries — covers all plan-divergence test cases (WORKPLAN §8)
# ---------------------------------------------------------------------------

EXAMPLE_QUERIES: list[dict] = [
    {
        "label": "🔍 Full analysis",
        "query": "Analyse this dataset for suspicious activity",
        "intent": "full_analysis",
    },
    {
        "label": "🧑 Entity investigation",
        "query": "Is customer 4521 suspicious?",
        "intent": "entity_investigation",
    },
    {
        "label": "📊 Threshold query",
        "query": "Which customers made 10+ transactions under $10,000?",
        "intent": "threshold_query",
    },
    {
        "label": "🎯 Pattern search",
        "query": "Find structuring patterns in the last 30 days",
        "intent": "pattern_search",
    },
    {
        "label": "🏆 Ranking",
        "query": "Rank the top 10 highest-risk customers",
        "intent": "ranking",
    },
    {
        "label": "📈 Exploratory analysis",
        "query": "Show me a breakdown of transaction types and countries",
        "intent": "eda",
    },
]

# ---------------------------------------------------------------------------
# HTTP client helpers — same code in LIVE and FIXTURE mode
# ---------------------------------------------------------------------------


def _check_health() -> dict | None:
    """Return /health payload, or None if unreachable."""
    return api_client.check_health()


def _get_dataset_summary() -> dict | None:
    return api_client.get_dataset_summary()


def _post_query(query_text: str, dataset: str | None = None) -> dict | None:
    """POST to /query and return the AgentResponse dict, or None on failure."""
    return api_client.post_query(query_text, dataset)


def _load_fixture(intent_hint: str | None = None) -> dict:
    """Load the best-matching fixture JSON for the query intent.

    Falls back to full_analysis.json if no specific fixture found.
    """
    intent_to_file = {
        "full_analysis":        "full_analysis.json",
        "entity_investigation": "full_analysis.json",
        "threshold_query":      "full_analysis.json",
        "pattern_search":       "full_analysis.json",
        "ranking":              "full_analysis.json",
        "eda":                  "full_analysis.json",
    }
    filename = intent_to_file.get(intent_hint or "full_analysis", "full_analysis.json")
    fixture_path = FIXTURE_DIR / filename
    with open(fixture_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AML Detection Agent",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(90deg, #1e40af, #2563eb, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -1px;
        margin-bottom: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("AML Agent", divider="grey")

    health = _check_health()
    api_live = health is not None

    if api_live:
        st.success("API Online", icon="✅")
        llm_ok = health.get("llm_available", False)
        mocks  = health.get("mocks", False)
        st.markdown(f"**LLM:** {'✅ Available' if llm_ok else '⚠️ Offline (fallback mode)'}")
        st.markdown(f"**Mocks:** `{'on' if mocks else 'off'}`")
        # Which transport served this — HTTP to a separate API process, or the
        # backend running inside this one. Worth surfacing rather than hiding:
        # on a single-process host the answer is always "in-process", and
        # somebody debugging a deployment should not have to guess.
        st.markdown(f"**Backend:** `{health.get('mode', API_BASE_URL)}`")

        summary = _get_dataset_summary()
        if summary:
            st.subheader("Dataset", divider="grey")
            with st.container(border=True):
                st.metric("Transactions", f"{summary.get('row_count', 0):,}")
            with st.container(border=True):
                st.metric("Customers", f"{summary.get('customer_count', 0):,}")
            date_min = summary.get("date_min")
            date_max = summary.get("date_max")
            if date_min and date_max:
                st.markdown(f"**Date range:** `{date_min}` → `{date_max}`")
            cols = summary.get("columns", [])
            if cols:
                with st.expander("Schema columns"):
                    for c in cols:
                        st.markdown(f"- `{c}`")
    else:
        st.warning("API Offline", icon="⚠️")
        st.caption("Track A's API is not running. Showing fixture data — results are illustrative.")
        st.subheader("Fixture Dataset", divider="grey")
        with st.container(border=True):
            st.metric("Transactions", "2,000")
        with st.container(border=True):
            st.metric("Customers", "270")

    st.subheader("About", divider="grey")
    st.caption(
        "An agentic AI compliance analyst that translates natural language queries into adaptive "
        "execution plans, combining rule-based AML patterns and unsupervised ML anomaly detection "
        "to surface risk-scored suspicious entities.\n\nBuilt by **Sesenta y Siete**"
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.markdown('<h1 class="main-header">AML Suspicious Activity Detection</h1>', unsafe_allow_html=True)
st.caption(
    "Natural-language queries over financial transaction data · "
    "Adaptive execution plans · Risk-scored flags with escalation actions"
)

# Fixture mode banner
if not api_live:
    st.info(
        "**Fixture mode** — Track A's API is not reachable at `localhost:8000`. "
        "Results below are from a pre-computed fixture that matches the live AgentResponse schema exactly. "
        "Start `uvicorn backend.main:app` and refresh to switch to live mode.",
        icon="⚡",
    )

# ---------------------------------------------------------------------------
# Query box
# ---------------------------------------------------------------------------

st.markdown("### 💬 Query")

# Pre-fill from example button clicks (stored in session state)
if "query_prefill" not in st.session_state:
    st.session_state["query_prefill"] = ""

query_input = st.text_area(
    label="Enter your query:",
    value=st.session_state["query_prefill"],
    placeholder="e.g. 'Analyse this dataset for suspicious activity'",
    height=80,
    key="query_text",
    label_visibility="collapsed",
)

submit_col, _ = st.columns([1, 4])
with submit_col:
    run_query = st.button("🔍 Run Query", type="primary", use_container_width=True)

# Example query buttons — one row
st.markdown("**Quick queries:**")
btn_cols = st.columns(len(EXAMPLE_QUERIES))
for i, ex in enumerate(EXAMPLE_QUERIES):
    with btn_cols[i]:
        if st.button(ex["label"], key=f"ex_{i}", use_container_width=True):
            st.session_state["query_prefill"] = ex["query"]
            st.session_state["pending_intent"] = ex.get("intent")
            st.rerun()

# ---------------------------------------------------------------------------
# Execute query
# ---------------------------------------------------------------------------

response: dict | None = None
using_fixture = False

if run_query and query_input.strip():
    intent_hint = st.session_state.pop("pending_intent", None)

    with st.status("Running analysis…", expanded=True) as _status:
        st.write("📤 Sending query to backend…")
        t0 = time.time()
        response = _post_query(query_input.strip())
        elapsed = time.time() - t0
        if response is not None:
            st.write(f"✅ Response received in {elapsed:.1f}s")
            _status.update(label=f"Analysis complete — {elapsed:.1f}s", state="complete", expanded=False)
        else:
            st.write("⚠️ API call failed — loading fixture data")
            _status.update(label="API unavailable — showing fixture data", state="error", expanded=False)

    if response is None:
        # Live call failed — fall back to fixture
        using_fixture = True
        response = _load_fixture(intent_hint)
        # Patch the query field so the trace shows what was actually asked
        response = dict(response)
        response["query"] = query_input.strip()
        st.warning(
            f"⚠️ API call failed or timed out after {elapsed:.1f}s. Showing fixture data instead."
        )

elif "pending_intent" in st.session_state:
    # Example button was just clicked — show fixture immediately so the
    # user sees something without needing to press Run Query
    intent_hint = st.session_state["pending_intent"]
    using_fixture = True
    response = _load_fixture(intent_hint)
    response = dict(response)
    response["query"] = st.session_state.get("query_prefill", response.get("query", ""))

# ---------------------------------------------------------------------------
# Render results
# ---------------------------------------------------------------------------

if response:
    # Header: query + summary
    st.divider()

    summary_text = response.get("summary", "")
    warnings     = response.get("warnings", [])

    with st.container(border=True):
        st.markdown(f"**Query:** *{response.get('query', '')}*")
        if summary_text:
            st.markdown(f"**Summary:** {summary_text}")

    for w in warnings:
        st.warning(w)

    # KPI row + risk distribution — resolves both live-tool and fixture key
    # names (see frontend/components/theme.py::resolve_metric), so these no
    # longer silently disagree or go blank depending on mode.
    metrics = response.get("metrics", {})
    render_kpi_row(metrics)
    render_risk_distribution(metrics)

    # 1 — Execution plan trace (highest-value component, above results)
    render_plan_trace(response)

    # 2/3/4 — Flags / Charts / Tables, grouped into tabs below the trace panel
    tab_flags, tab_charts, tab_tables = st.tabs(
        ["🚩 Flagged Entities", "📊 Charts", "📋 Tables & Export"]
    )

    with tab_flags:
        flags = response.get("flags", [])
        if flags:
            render_flag_cards(flags)
        else:
            # Graceful empty-result handling
            no_flag_msg = summary_text or "No suspicious entities were flagged by this query."
            st.success(no_flag_msg, icon="✅")
            for w in warnings:
                st.info(w)

    with tab_charts:
        render_charts(response.get("charts", {}))

    with tab_tables:
        render_tables(response.get("tables", {}), response)

elif not run_query:
    # Landing state — prompt the user
    st.divider()
    st.info(
        "Enter a query or click an example above. The agent will build a custom execution plan, "
        "run only the relevant tools, and return risk-scored flags with escalation actions.",
        icon="🔎",
    )
