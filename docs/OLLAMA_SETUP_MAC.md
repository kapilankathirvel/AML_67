# Running the agent's LLM locally on your Mac (Ollama)

## Why

Both cloud keys we've been using (Gemini, Groq) are quota-exhausted from testing — Groq's daily token
quota and Gemini's free-tier request limit are both essentially spent for today. The repo now supports a
fourth LLM provider, **Ollama**, which runs entirely on your own machine: no key, no quota, no external
network dependency at all. The code change is already in the shared repo (`git pull` gets it) — everything
below is just what's specific to setting it up on your Mac.

Both of the agent's LLM uses are bounded, structured tasks (classify a query into one of 7 intents +
extract simple fields; rewrite an already-correct template into one analyst paragraph, only for HIGH-risk
flags). Neither needs a frontier model — a 7-14B instruction-tuned model handles both well.

## 1. Install Ollama

```bash
brew install ollama
```
(or download the `.dmg` from https://ollama.com/download). Metal acceleration is automatic — no separate
driver/toolkit step, unlike the NVIDIA/Windows side of this.

Ollama runs a local server automatically on `http://localhost:11434` once installed.

## 2. Check your unified memory first

 → About This Mac → note the memory figure. Model choice below depends on it.

## 3. Pull a model

| Unified memory | Recommended pull | Notes |
|---|---|---|
| 16GB+ | `ollama pull qwen2.5:14b-instruct` | ~9GB download. Noticeably better instruction-following/JSON quality than 7B — this is where your M3's unified memory has an actual edge over a 6GB discrete GPU, which can't fit a 14B model at all. |
| 8GB | `ollama pull qwen2.5:7b-instruct` | ~4-5GB download. Same model the Windows/RTX-3050 side of this project uses — safe, well-tested for this project's tasks. |

Honest note on M3 vs. a discrete NVIDIA GPU (e.g. the 3050 6GB used elsewhere on this project): for the
*same* model size, Metal is typically a bit slower per-token than CUDA. The M3's actual advantage is
memory *capacity* — it can run bigger models that simply won't fit on a 6GB GPU. For this project's
bounded tasks, that's a nice-to-have, not a requirement — 7B is already enough; 14B is a quality bonus if
your memory allows it.

Since this is a multi-GB download, kick it off and let it run in the background (or overnight) rather
than waiting on it.

## 4. Activate it

In your own local `.env` (not committed — copy `.env.example` if you don't have one yet):
```
LLM_PROVIDER=ollama
```
Only add these if you picked the 14B model above (7B is already the default in `config.py`):
```
OLLAMA_MODEL=qwen2.5:14b-instruct
```

## 5. Verify

```bash
curl http://localhost:11434/api/tags        # should list your pulled model
```
Then start the backend as usual (`uvicorn backend.main:app`) and check:
```bash
curl http://localhost:8000/health           # should show "llm_available": true
```
Try a query through the running app — the execution plan trace should show `parsed_by: "llm"`, not
`"rules"`, confirming Ollama is actually being used.

No quota, no rate limits, no network dependency — safe to run every query you want during testing or
rehearsal without worrying about running out.
