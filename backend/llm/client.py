"""
Provider-agnostic LLM client. Owner: Track A.

complete_json() returns None on ANY failure (no key, timeout, rate limit, bad
JSON) so every caller has a defined non-LLM fallback path — never assume the
LLM is available.
"""

import json
import logging
import threading
from typing import Any

from backend.config import settings

logger = logging.getLogger(__name__)

# Cache SUCCESSFUL completions only, keyed on the exact (prompt, schema_hint)
# pair. Deliberately not a plain @lru_cache on the whole function: today's
# actual failure mode is transient rate-limiting (429s that clear up later),
# and caching a None failure would poison that exact query for the rest of
# the process — the one case this cache must never produce. Re-running the
# same query during demo rehearsal costs zero additional API/inference calls
# once it has succeeded once; free-tier quotas are small enough that repeated
# identical testing alone can exhaust them in one session.
_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
_CACHE_MAXSIZE = 256


def complete_json(prompt: str, schema_hint: str = "") -> dict[str, Any] | None:
    cache_key = (prompt, schema_hint)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    result: dict[str, Any] | None = None
    try:
        if settings.llm_provider == "gemini" and settings.gemini_api_key:
            result = _complete_gemini(prompt, schema_hint)
        elif settings.llm_provider == "openai" and settings.openai_api_key:
            result = _complete_openai(prompt, schema_hint)
        elif settings.llm_provider == "groq" and settings.groq_api_key:
            result = _complete_groq(prompt, schema_hint)
        elif settings.llm_provider == "ollama":
            result = _complete_ollama(prompt, schema_hint)
        else:
            # No branch matched: either no provider configured, or the
            # configured one has no key. Worth saying once — a run where the
            # LLM never fires looks identical to one where it fires and adds
            # nothing, and the difference matters when reading a plan trace.
            logger.debug(
                "complete_json: no usable provider (llm_provider=%r) — using fallbacks",
                settings.llm_provider,
            )
    except Exception as exc:
        # Returning None here is the contract (every caller has a non-LLM
        # path), but swallowing the reason silently is how a misconfigured
        # model — say, ollama_model naming something that was never pulled —
        # looks exactly like "the LLM had nothing to add". Log it; the caller
        # still gets None and still falls back.
        logger.warning(
            "complete_json failed (provider=%s, model=%s): %s: %s — falling back",
            settings.llm_provider,
            settings.ollama_model if settings.llm_provider == "ollama" else "-",
            type(exc).__name__,
            exc,
        )
        result = None

    if result is not None:
        if len(_CACHE) >= _CACHE_MAXSIZE:
            _CACHE.pop(next(iter(_CACHE)))  # crude FIFO eviction, fine at this scale
        _CACHE[cache_key] = result

    return result


def _complete_gemini(prompt: str, schema_hint: str) -> dict[str, Any] | None:
    import google.generativeai as genai

    genai.configure(api_key=settings.gemini_api_key)
    # "latest" alias, not a dated version string — Google periodically retires
    # specific model versions (gemini-1.5-flash 404s now; gemini-2.0-flash has
    # zero free-tier quota on new accounts), so pinning to a version drifts out
    # of the free tier over time. The alias tracks whatever's current.
    model = genai.GenerativeModel("gemini-flash-latest")
    full_prompt = f"{prompt}\n\n{schema_hint}\nRespond with strict JSON only, no markdown fences."
    response = model.generate_content(
        full_prompt,
        generation_config={"response_mime_type": "application/json", "temperature": 0.0},
        request_options={"timeout": settings.llm_timeout_seconds},
    )
    return json.loads(response.text)


def _complete_openai(prompt: str, schema_hint: str) -> dict[str, Any] | None:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key, timeout=settings.llm_timeout_seconds)
    full_prompt = f"{prompt}\n\n{schema_hint}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": full_prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(response.choices[0].message.content)


def _complete_groq(prompt: str, schema_hint: str) -> dict[str, Any] | None:
    from groq import Groq

    client = Groq(api_key=settings.groq_api_key)
    full_prompt = f"{prompt}\n\n{schema_hint}\nRespond with strict JSON only, no markdown fences."
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": full_prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
        timeout=settings.llm_timeout_seconds,
    )
    return json.loads(response.choices[0].message.content)


def _complete_ollama(prompt: str, schema_hint: str) -> dict[str, Any] | None:
    # Local, no API key. Uses Ollama's native /api/chat endpoint (not the
    # OpenAI-compatibility layer) — format="json" is Ollama's own documented
    # JSON-mode flag, simpler and more robust than routing through a
    # compatibility shim. `requests` is already a project dependency.
    import requests

    full_prompt = f"{prompt}\n\n{schema_hint}\nRespond with strict JSON only, no markdown fences."
    response = requests.post(
        f"{settings.ollama_base_url}/api/chat",
        json={
            "model": settings.ollama_model,
            "messages": [{"role": "user", "content": full_prompt}],
            "format": "json",
            "stream": False,
            "keep_alive": settings.ollama_keep_alive,
            "options": {"temperature": 0.0},
        },
        timeout=settings.llm_timeout_seconds,
    )
    response.raise_for_status()
    return json.loads(response.json()["message"]["content"])


def warm_ollama() -> None:
    """Fire a throwaway completion to load the Ollama model into VRAM at startup.

    Called once from the FastAPI lifespan handler.  If Ollama is down or the
    model is not pulled, logs a warning and returns — startup is NOT blocked.
    The real requests will still fall back to the rules parser on any failure,
    per Contract 1.
    """
    if settings.llm_provider != "ollama":
        return
    import requests

    try:
        response = requests.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": settings.ollama_model,
                "messages": [{"role": "user", "content": "ping"}],
                "format": "json",
                "stream": False,
                "keep_alive": settings.ollama_keep_alive,
                "options": {"temperature": 0.0},
            },
            timeout=120,  # allow up to 2 min for model load on first warm-up
        )
        response.raise_for_status()
        logger.info(
            "Ollama pre-warm complete: model=%s keep_alive=%s",
            settings.ollama_model,
            settings.ollama_keep_alive,
        )
    except Exception as exc:
        # The overwhelmingly common cause is a model named in config that was
        # never pulled — ollama_model defaults to a 7B while a smaller one may
        # be what is actually on disk. Ollama answers that with a 404 whose
        # body names the model, so say the exact command that fixes it rather
        # than leaving the reader to infer it from a stack trace.
        detail = str(exc)
        hint = ""
        if "404" in detail or "not found" in detail.lower():
            hint = (
                f"  ->  that model is not pulled. Either run "
                f"`ollama pull {settings.ollama_model}`, or point OLLAMA_MODEL at "
                f"one you already have (`ollama list` to see them)."
            )
        logger.warning(
            "Ollama pre-warm failed (model=%s): %s — LLM calls will fall back to "
            "the rules parser and the deterministic planner.%s",
            settings.ollama_model,
            exc,
            hint,
        )

# ---------------------------------------------------------------------------
# Module-level startup hook: fires warm_ollama() in a daemon thread the first
# time this module is imported (i.e. at uvicorn startup).  Non-blocking —
# startup completes normally while the model loads in the background.
# ---------------------------------------------------------------------------
def _start_prewarm() -> None:
    if settings.llm_provider == "ollama":
        t = threading.Thread(target=warm_ollama, daemon=True, name="ollama-prewarm")
        t.start()


_start_prewarm()
