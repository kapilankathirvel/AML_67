# How the deployed demo actually runs

**Where the code executes, what happens when someone else runs a query, and why no API key is
involved.**

Short answer to the three questions this document exists for:

| Question | Answer |
|---|---|
| Where does my friend's query run? | On **Streamlit's servers**. Never on your machine. |
| Does it use my LLM API keys? | **No.** There are no keys in the deployment, and every LLM call site falls back cleanly. |
| Does it use my local Ollama? | **No, and it cannot.** Physically impossible — see [§4](#4-why-your-local-ollama-can-never-be-reached). |

Everything below is verified by running the deployed configuration, not inferred from reading the
code. The evidence is in [§6](#6-the-evidence).

---

## 1. Where the code runs

When you deployed, Streamlit **cloned your GitHub repository onto their server** and started a Python
process there. That process is the entire application.

```
Your friend's browser                Streamlit's server (Linux container)
────────────────────                 ───────────────────────────────────
  types a query        ──────────▶   frontend/app.py      (Streamlit UI)
                                        │
                                     frontend/api_client.py
                                        │  AML_API_URL is blank,
                                        │  so it calls the backend directly
                                        ▼
                                     backend/main.py      (query endpoint)
                                        │
                                     intent_parser → planner → validator
                                        → executor → 9 tools → narrator
                                        │
                                     data/sample/aml_sample.csv
                                        (committed to the repo, cloned with it)
                                        │
  sees flags + plan    ◀──────────   AgentResponse rendered as HTML
```

**Your laptop is not in this diagram.** It was involved exactly once — when you ran `git push`. You
could shut it down permanently and the demo would keep working.

### Why there is only one process

Normally this app is two processes: Streamlit talking HTTP to a FastAPI server. Streamlit Community
Cloud runs one process and gives you no way to start a second, so
[`frontend/api_client.py`](../frontend/api_client.py) detects that `AML_API_URL` is blank and
**imports the backend directly** instead of making HTTP calls to it.

It calls `backend.main`'s own endpoint functions — the same code the FastAPI server would run. There
is one implementation of what a query does, so the deployed demo cannot answer differently from a
local two-process run.

---

## 2. What happens when your friend runs a query

Step by step, for `Which customers made 10+ transactions under $10,000?`:

1. **Their browser sends the text** over a WebSocket to Streamlit's server. That is all their browser
   does — it has no code, no data, no logic.
2. **`intent_parser` classifies it.** It tries an LLM first and gets nothing (no key), so the regex
   parser handles it: `intent=threshold_query`, `min_txn_count=10`, `amount_max=10000`.
3. **`build_plan` produces a plan** — three steps: `load_data → filter_data → aggregate_query`. It
   deliberately omits `feature_engineer`, `rule_detect` and `ml_detect`, and records *why* in
   `tools_considered_but_skipped`.
4. **The executor runs those three tools**, reading the CSV that was cloned with the repo.
5. **The narrator turns results into text** using templates built from each hit's evidence.
6. **The page updates** in their browser with results and the full plan trace.

Total: about 4 seconds. `Analyse this dataset for suspicious activity` follows the same path but runs
all six detection steps and takes ~60 seconds.

### Everyone shares one process

Community Cloud runs a single Python process for all viewers. Practically:

- **Same dataset for everyone.** Nobody uploads anything; the CSV is baked into the repo.
- **Sessions are separate**, so one person's query does not appear on another's screen.
- **CPU is shared.** If two people click *Full analysis* simultaneously, both get slower — there is
  one container's worth of CPU between them. Fine for a demo, and worth knowing before you send the
  link to twenty people at once.
- **Nothing your friend does is saved.** No database, no writes. Refreshing loses the results, and the
  next visitor starts clean.

---

## 3. The LLM question, precisely

This is the part worth understanding properly, because the code has **four** places an LLM could be
called and only two of them are controlled by the flags you set.

| # | Call site | Gated by a flag? | What happens in the deployment |
|---|---|---|---|
| 1 | `intent_parser.py:87` — classify the query | **No flag** | Tries, gets `None`, uses the regex parser |
| 2 | `llm_planner.py:142` — choose the tools | `AML_LLM_PLANNER` | Never called; flag is `0` |
| 3 | `replanner.py:217` — revise the plan mid-run | `AML_LLM_REPLANNER` | Never called; flag is `0` |
| 4 | `narrator.py:145` — polish explanations | **No flag** | Tries, gets `None`, uses templates |

Sites 1 and 4 are **not** switched off by `AML_LLM_PLANNER=0`. They try every time. So why does
nothing happen?

### The actual gate is the missing key

Every call goes through `complete_json()` in [`backend/llm/client.py`](../backend/llm/client.py),
which begins:

```python
if settings.llm_provider == "gemini" and settings.gemini_api_key:
    ...
elif settings.llm_provider == "openai" and settings.openai_api_key:
    ...
elif settings.llm_provider == "groq" and settings.groq_api_key:
    ...
elif settings.llm_provider == "ollama":
    ...
else:
    # no usable provider — caller falls back
```

The deployment has `llm_provider = "gemini"` (the default) and **no key**, so the first branch is
false, none of the others match, and the function returns `None` without making a network call. Every
caller has a defined non-LLM path, so everything continues normally.

**The design principle worth naming:** `complete_json` returns `None` on *any* failure — no key,
timeout, rate limit, malformed JSON. Callers never assume the LLM is available. That is why removing
the key degrades the app instead of breaking it.

### So it costs nothing

No key means no calls, which means **no quota consumed and no bill**, no matter how many people use
the link. There is nothing to exhaust, nothing to rotate, and no key sitting in a public app's
configuration.

---

## 4. Why your local Ollama can never be reached

Worth spelling out because the instinct is natural and the reasoning is a genuinely useful thing to
understand.

Ollama runs on **your laptop** at `http://localhost:11434`. `localhost` means *"this machine, the one
executing the code."*

The deployed code executes on **Streamlit's server**. If it ever asked for `localhost:11434`, it would
be asking *Streamlit's container* for an Ollama that is not there, get `connection refused`, log a
warning, return `None`, and fall back. It would not reach across the internet to your desk — nothing
in `localhost` means "Kapilan's laptop".

For the deployment to use your Ollama, your machine would have to be running, publicly reachable on
the internet, with port 11434 exposed through your router and firewall, and `OLLAMA_BASE_URL` pointing
at your public IP. **Do not do this.** Exposing an unauthenticated model server to the internet means
anyone who finds it can use your GPU, and your home IP becomes a dependency of your demo.

The relevant secret is already correct:

```toml
LLM_PROVIDER  # not set → defaults to "gemini", which has no key → no calls
```

---

## 5. What the demo loses without an LLM — measured

Not "nothing". Being precise about the gap is more useful than claiming there isn't one.

**What is unaffected:** all seven intents parse, all nine tools run, all seven rules fire, both ML
models score, risk fusion and banding are identical, the plan trace renders fully, and SAR drafts
generate. **Every published metric in `README.md` comes from this exact path** — `run_evaluation.py`
pins both LLM flags off regardless of configuration, so the numbers describe the deterministic system,
not an LLM-assisted one.

**What degrades — one thing, and it is real.** The regex parser handles clean phrasing well and slang
less well. Measured on the deployed configuration:

| Query | With an LLM key | Deployed (no key) |
|---|---|---|
| `Which customers made 10+ transactions under $10,000?` | `threshold_query` | ✅ `threshold_query` |
| `Analyse this dataset for suspicious activity` | `full_analysis` | ✅ `full_analysis` |
| `who r my 3 sketchiest customers rn` | `ranking`, `top_n=3` | ❌ `full_analysis` |

The third row is the honest cost. Slangy phrasing falls through the keyword patterns and lands on the
`full_analysis` default — which still returns useful results, just not the ranking that was asked for.

**Two reasons this is the right trade for a public demo:**

1. A free-tier quota behind a public URL is a shared resource anyone can exhaust. The first visitor to
   burn it would silently degrade the demo for everyone after — possibly including whoever you most
   wanted to see it.
2. A key pasted into a public app's configuration is a key you have to rotate.

**If you are demonstrating live and want the LLM path**, run it locally with your `.env` instead. You
get the full behaviour and the quota is yours alone.

---

## 6. The evidence

Run of the deployed configuration — no keys, both flags off, with outbound HTTP intercepted:

```
llm_provider      = 'gemini'
gemini_api_key    = ''
aml_llm_planner   = False
aml_llm_replanner = False

complete_json() directly -> None   (None means no provider fired)
health.llm_available     -> False

'Analyse this dataset for suspicious activi'   intent=full_analysis    parsed_by=rules  flags=41
'who r my 3 sketchiest customers rn'           intent=full_analysis    parsed_by=rules  flags=41
'Which customers made 10+ transactions unde'   intent=threshold_query  parsed_by=rules  flags=0

OUTBOUND HTTP CALLS MADE: none
```

Three things this establishes:

- **`parsed_by=rules`** on every query — the regex parser did the work, not a model.
- **`OUTBOUND HTTP CALLS MADE: none`** — the strongest line here. `requests.get` and `requests.post`
  were both wrapped in spies, so a call that happened and failed would still have been recorded. None
  were made at all.
- **41 flags** — identical to the run with a key present. Detection does not depend on the LLM.

---

## 7. Quick reference

| | |
|---|---|
| Where it runs | Streamlit's Linux container, cloned from GitHub |
| Your machine's role | None after `git push` |
| LLM calls | **Zero** |
| API cost | **Zero** |
| Uses your Ollama | **No** — impossible; `localhost` there is their container |
| Dataset | `data/sample/aml_sample.csv`, committed, same for everyone |
| Data your friend uploads | None — nothing to upload |
| Anything stored | Nothing. No database, no writes |
| Concurrency | One shared process; simultaneous queries compete for CPU |
| Full analysis | ~60s, 41 flags |
| Threshold query | ~4s, 3-step plan |
| Cold start after idle | ~30s to wake |

**If someone asks "is your API key in there?"** — no, and the reason is worth giving: the demo does not
need one. The regex parser covers all seven intents and the deterministic planner is the floor, so the
system works with no model at all. That is a design property, not a limitation of the deployment.

---

## Related

- [DEPLOYMENT.md](DEPLOYMENT.md) — how to deploy it, and what breaks
- [PROJECT_GUIDE.md](PROJECT_GUIDE.md) — how the system works, layer by layer
- [`.streamlit/secrets.toml.example`](../.streamlit/secrets.toml.example) — the deployed configuration
