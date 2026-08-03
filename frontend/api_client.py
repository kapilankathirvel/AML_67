"""
frontend/api_client.py — the three calls the UI makes, over HTTP or in-process.

Why this exists
---------------
frontend/app.py talks to the FastAPI backend over HTTP, which means the product
is two processes. That is correct for a bank and wrong for a free demo host:
Streamlit Community Cloud runs one process and gives you no way to start a
second, so the app as written can only ever reach FIXTURE mode there — a live
URL that quietly serves canned JSON, which is worse than no URL at all.

This module removes the assumption without changing the architecture. The UI
calls three functions; each dispatches on whether an API is configured:

    AML_API_URL set   -> HTTP, exactly as before. Two processes, unchanged.
    AML_API_URL unset -> import the backend and call it directly. One process.

The in-process path calls backend.main's own endpoint functions rather than
re-running intent_parser -> planner -> executor itself. That is the whole point
of the design: there is one implementation of "what a query does", so the
deployed demo and a local two-process run cannot drift apart. A reimplementation
here would be a second system that looks identical until the day it doesn't.

What this is NOT
----------------
Not a fallback for a backend that is down. If AML_API_URL is set and the API is
unreachable, these functions return None and app.py drops into FIXTURE mode as
it always has. Silently switching to in-process when a configured API fails
would hide an outage behind a working-looking demo.

Import cost
-----------
The backend is imported lazily, inside the in-process branch. Importing
backend.main pulls in pandas, sklearn and networkx, which is several seconds and
a few hundred MB — pure waste for an HTTP-mode run, and on a memory-capped host
it is the difference between starting and not.
"""

from __future__ import annotations

import functools
import os
from typing import Any

import requests

# Must be exactly AML_API_URL — that is what frontend/app.py has always read,
# and what every deployment note and README instruction refers to.
_API_URL_ENV = "AML_API_URL"

REQUEST_TIMEOUT = 120  # matches app.py: must exceed LLM timeout + pipeline time
HEALTH_TIMEOUT = 3
SUMMARY_TIMEOUT = 10


def api_base_url() -> str | None:
    """The configured API, or None when the UI should run the backend itself.

    Read at call time rather than import time so a test can set the variable
    without reloading the module, and so Streamlit's script re-runs pick up a
    change to st.secrets without a restart.
    """
    url = os.getenv(_API_URL_ENV, "").strip()
    return url or None


def in_process() -> bool:
    """True when queries run inside this process instead of over HTTP."""
    return api_base_url() is None


@functools.lru_cache(maxsize=1)
def _backend():
    """Import backend.main once, on first in-process use.

    Cached because the import is expensive and Streamlit re-runs the whole
    script on every interaction — without this, each button click would pay for
    re-importing pandas and sklearn.
    """
    from backend.main import QueryRequest, dataset_summary, health, query

    return QueryRequest, dataset_summary, health, query


# ---------------------------------------------------------------------------
# The three calls
# ---------------------------------------------------------------------------


def check_health() -> dict | None:
    """GET /health, or its in-process equivalent. None if unreachable."""
    base = api_base_url()
    if base is None:
        try:
            _, _, health, _ = _backend()
            payload = dict(health())
            payload["mode"] = "in-process"
            return payload
        except Exception:
            # An in-process failure here means the backend cannot even import,
            # which is a broken deployment rather than a transient outage. It
            # is still reported as None so the UI degrades to fixtures rather
            # than showing a stack trace to whoever opened the link.
            return None

    try:
        r = requests.get(f"{base}/health", timeout=HEALTH_TIMEOUT)
        r.raise_for_status()
        payload = dict(r.json())
        payload["mode"] = "http"
        return payload
    except Exception:
        return None


def get_dataset_summary() -> dict | None:
    """GET /dataset/summary, or its in-process equivalent."""
    base = api_base_url()
    if base is None:
        try:
            _, dataset_summary, _, _ = _backend()
            return dict(dataset_summary())
        except Exception:
            return None

    try:
        r = requests.get(f"{base}/dataset/summary", timeout=SUMMARY_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def post_query(query_text: str, dataset: str | None = None) -> dict | None:
    """POST /query, or its in-process equivalent. Returns the AgentResponse dict.

    The in-process branch serialises through the same Pydantic model the HTTP
    response is built from, so both paths hand the UI the same shape. Returning
    the model itself would work until the first component that does
    `payload.get(...)` on it.
    """
    base = api_base_url()
    if base is None:
        try:
            QueryRequest, _, _, query = _backend()
            response = query(QueryRequest(query=query_text, dataset=dataset))
            return response.model_dump(mode="json")
        except Exception:
            return None

    try:
        payload: dict[str, Any] = {"query": query_text}
        if dataset:
            payload["dataset"] = dataset
        r = requests.post(f"{base}/query", json=payload, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None
