# Deploying the demo

**A step-by-step guide to putting this app on a public URL, assuming nothing.**

The app is normally **two processes** — Streamlit talking HTTP to FastAPI. Free hosts run one. This
guide covers the single-process path, which [`frontend/api_client.py`](../frontend/api_client.py)
exists to make possible.

Target: **Streamlit Community Cloud**. It is free, it deploys straight from GitHub, and it needs no
credit card. The fallback if it does not fit is in [§7](#7-resource-limits-and-other-platforms).

**Time:** about 15 minutes of your attention, plus 5–10 minutes of waiting for the first build.

---

## Contents

1. [Before you start](#1-before-you-start)
2. [Check the repository is ready](#2-check-the-repository-is-ready)
3. [Create the Streamlit Cloud account](#3-create-the-streamlit-cloud-account)
4. [Deploy the app](#4-deploy-the-app)
5. [Add the secrets](#5-add-the-secrets)
6. [Verify it actually works](#6-verify-it-actually-works)
7. [Resource limits, and other platforms](#7-resource-limits-and-other-platforms)
8. [Troubleshooting](#8-troubleshooting)
9. [After it is live](#9-after-it-is-live)

---

## 1. Before you start

You need three things. Nothing else.

| | What | How to check |
|---|---|---|
| 1 | A **GitHub account** that owns or can access `kapilankathirvel/AML_67` | Open https://github.com/kapilankathirvel/AML_67 while logged in |
| 2 | The repository is **public** | On that page, look next to the repo name — it should say `Public`, not `Private` |
| 3 | An **email address** you can receive mail at | For the Streamlit sign-up |

**On point 2:** Community Cloud can deploy private repos, but the free tier allows only one private
app and adds an authorisation step. Public is simpler. If the repo is private and you want to keep it
that way, everything below still works — you will just grant one extra permission in
[§3](#3-create-the-streamlit-cloud-account).

**You do not need:** a credit card, a domain name, Docker, an API key, or a Kaggle token. The demo runs
entirely on the committed synthetic dataset with no LLM calls at all.

---

## 2. Check the repository is ready

Everything here is already committed. This section is so you can confirm it rather than trust me.

### 2.1 Confirm the files exist

Open the repo on GitHub and check these five paths exist:

```
frontend/app.py                      ← the entry point Streamlit runs
frontend/api_client.py               ← chooses HTTP vs in-process
requirements-deploy.txt              ← the trimmed dependency list
.streamlit/secrets.toml.example      ← the config you will paste in §5
data/sample/aml_sample.csv           ← the dataset, committed, ~2,002 rows
```

The last one matters more than it looks. The demo reads a **committed** CSV, so there is no database
to provision, no storage bucket, no data upload step. That is the entire reason this deploys in
fifteen minutes.

### 2.2 Confirm the branch is current

Your local work must be pushed, or Streamlit will deploy an older commit.

```bash
cd "C:\Users\Kapilan Kathirvel\Desktop\soc"
git status
```

You want to see `nothing to commit, working tree clean` and
`Your branch is up to date with 'origin/main'`.

If it says you have unpushed commits:

```bash
git push origin main
```

If it lists modified files you want to keep, commit them first. If it lists
`evaluation/results/_run_a.json` and `_run_b.json` as untracked — leave them. They are scratch files
from an old determinism check and are deliberately not committed.

---

## 3. Create the Streamlit Cloud account

1. Go to **https://share.streamlit.io**
2. Click **Continue with GitHub** (this is the path that works; email sign-up leaves you unable to see
   your repos)
3. GitHub will ask you to authorise **Streamlit**. Click **Authorize streamlit**.
   - It asks for read access to your repositories. That is how it clones the code to build the app.
   - If your repo is **private**, you must also grant repository access when prompted — choose
     **All repositories**, or **Only select repositories** and pick `AML_67`.
4. You land on the Streamlit Cloud workspace. It will be empty.

---

## 4. Deploy the app

1. Click the **Create app** button (top right). Older versions of the UI call it **New app**.
2. When asked where your code lives, choose **Deploy a public app from GitHub**.
3. Fill in the three fields:

   | Field | Value |
   |---|---|
   | **Repository** | `kapilankathirvel/AML_67` |
   | **Branch** | `main` |
   | **Main file path** | `frontend/app.py` |

   Type the main file path exactly. It is `frontend/app.py`, **not** `app.py` and not
   `/frontend/app.py`.

4. Click **Advanced settings**. Set:

   | Field | Value |
   |---|---|
   | **Python version** | **3.11** |

   > ### ⚠️ Do not skip this. It is the one step that has actually broken a real deploy.
   >
   > Community Cloud's default is whatever is newest — a first attempt at this app landed on
   > **Python 3.14.6** and the build failed. It only lets you choose the version **when the app is
   > created**; there is no way to change it afterwards. Getting it wrong means deleting the app and
   > starting this section again.
   >
   > The failure is also thoroughly misleading, so it is worth recognising. The error is:
   >
   > ```
   > The headers or library files could not be found for zlib,
   > a required dependency when compiling Pillow from source.
   > ```
   >
   > That sends you looking for missing system libraries, which is a dead end — you cannot install
   > system packages on Community Cloud anyway. **zlib is not the problem.** The problem is three
   > lines earlier in the log: `Using Python 3.14.6 environment`. `pandas==2.2.3` and the `pillow`
   > that `streamlit==1.39.0` depends on publish no prebuilt wheels for 3.14, so pip fell back to
   > *compiling them from source*, and compiling pillow needs zlib headers the container does not
   > have.
   >
   > On 3.11 all of them install as prebuilt wheels and no compiler is involved. Nothing is wrong
   > with the code or the dependency list.
   >
   > **If 3.11 is not in the dropdown, choose 3.12** — pandas 2.2.3 and pillow both ship 3.12 wheels.
   > Do not accept 3.13 or 3.14.

5. **Do not click Deploy yet.** The Advanced settings panel also contains the **Secrets** box, and it
   is much less annoying to fill that in now than to fix a broken first boot. Go to
   [§5](#5-add-the-secrets), paste the secrets, and then come back and click **Deploy**.

### A note on the requirements file

Community Cloud **auto-detects** your dependency file and offers no field to name one. Confirmed from
a real build log, which read `/mount/src/aml_67/requirements.txt`.

So what it uses is **`requirements.txt`** in the repo root. That works. It installs four packages the
running app never imports (`jupyter`, `kaggle`, `kagglehub`, `pytest`), which makes the build slower
but does **not** meaningfully affect runtime memory, because memory is driven by what gets *imported*
and nothing imports them.

`requirements-deploy.txt` exists for the case where that is not good enough — a build timeout, or a
disk limit. If you hit one, the fix is:

```bash
cd "C:\Users\Kapilan Kathirvel\Desktop\soc"
cp requirements-deploy.txt frontend/requirements.txt
git add frontend/requirements.txt
git commit -m "Add a trimmed requirements file next to the Streamlit entry point"
git push origin main
```

Community Cloud looks for a dependency file next to the entry point as well as in the repo root, so
this takes precedence for this app without changing what developers install. **Only do this if the
default build actually fails** — a duplicated dependency list is a real maintenance cost and is not
worth paying pre-emptively.

---

## 5. Add the secrets

In **Advanced settings**, find the **Secrets** box. Paste this in, exactly:

```toml
AML_DATA_SOURCE = "synthetic"
AML_USE_MOCKS = "0"
AML_API_URL = ""
AML_LLM_PLANNER = "0"
AML_LLM_REPLANNER = "0"
```

That is the whole configuration. It is also in the repo at
[`.streamlit/secrets.toml.example`](../.streamlit/secrets.toml.example) with the reasoning inline.

**What each line does, because pasting config you do not understand is how deployments break:**

- **`AML_DATA_SOURCE = "synthetic"`** — which dataset the demo analyses. The application default is
  `synthetic_alt`, which is a **different population**: 1,710 transactions / 294 customers, against
  synthetic's 2,002 / 270, with no overlapping customer IDs. Every number in `README.md` is computed
  against `synthetic`. Leave this out and your demo answers questions about one dataset while your
  README reports another — which is exactly the sort of thing an interviewer notices.

- **`AML_USE_MOCKS = "0"`** — use the real detectors, not the mock tools used for developing the agent
  core in isolation. Get this wrong and the app runs, looks fine, and returns fabricated flags for
  three hardcoded customers.

- **`AML_API_URL = ""`** — **the line that makes single-process mode happen.** Blank means the UI
  imports the backend and runs it in its own process. Set it to anything and the UI will try to reach
  a FastAPI server that does not exist on Community Cloud, fail every call, and fall back to showing
  canned fixture data behind a warning banner.

- **`AML_LLM_PLANNER = "0"` and `AML_LLM_REPLANNER = "0"`** — no LLM calls. This costs the demo
  nothing worth having: the regex intent parser covers all seven intents on its own and the
  deterministic planner is the floor, so **every button works with no API key at all**. Turning them on
  in public would put a free-tier quota behind a URL anyone can click, and the first visitor to exhaust
  it would silently degrade the demo for everyone after them — possibly including whoever you sent the
  link to. Every published metric comes from the deterministic path anyway.

**Do not put an API key in this box.** You do not need one, and a key pasted into a public app's
configuration is a key you have to rotate.

Now click **Deploy**.

---

### What happens next

The build takes **3–7 minutes** on the first run, because on Python 3.11 every dependency installs as
a prebuilt wheel and no compiler runs. You will see a log streaming. It goes:

1. Cloning the repository
2. Installing dependencies — the slow part; `scipy`, `pandas` and `pyarrow` are large
3. `You can now view your Streamlit app in your browser`

**If you are past ten minutes and still watching install logs, something is wrong.** Scroll to the top
of the log and check the Python version — the failed build that prompted this warning ran for
**45 minutes** before dying, because it was compiling pandas and pillow from C source on Python 3.14.
See [§8.2](#82-the-build-fails-installing-dependencies).

After the build, the first page load adds roughly 30 seconds of cold start, and the first
**Full analysis** click takes about 60 seconds. Both are expected; see [§6](#6-verify-it-actually-works).

If the log stops with a red error, go to [§8](#8-troubleshooting).

---

## 6. Verify it actually works

A page that loads is not a demo that works. Check all six of these.

### 6.0 Anyone can actually open it

**Do this first, and do it from a browser that is not signed in to Streamlit** — a private window, or
your phone with wifi off.

A newly created app can be restricted to invited viewers, and the symptom is easy to miss because
*you* are signed in and it works perfectly for you. Everyone else gets a Streamlit sign-in page.

This happened on the first deploy of this app. From an unauthenticated client the URL bounces:

```
antimoneylaundering67.streamlit.app/           -> 303
share.streamlit.io/-/auth/app?redirect_uri=... -> 303
antimoneylaundering67.streamlit.app/-/login    -> back to the start
```

Note the repository being public does not make the app public; they are separate settings.

**Fix:** **Manage app → Settings → Sharing**, and set viewer access to public / anyone with the link.
It takes effect immediately, with no rebuild.

**Why this matters more than it looks:** an interviewer who clicks the link and is asked to create an
account will not create the account. They will conclude the demo does not work, and you will never
find out that is what happened. A link that silently requires a login is worse than no link.

### 6.1 The sidebar says the right things

Look at the left sidebar:

- ✅ **API Online** — green. If it says *API Offline* with a fixture warning, the backend failed to
  start; see [§8.4](#84-the-app-loads-but-says-api-offline).
- **Backend:** `in-process` — confirms single-process mode.
- **Mocks:** `off` — confirms real detectors.
- **Transactions: 2,002** and **Customers: 270**

**Those last two numbers are the check that matters most.** If you see **1,710** and **294**, the
`AML_DATA_SOURCE` secret did not take effect and the demo is running on the wrong dataset. Fix it
before showing anyone.

### 6.2 The slow query returns

Click **🔍 Full analysis**, or type `Analyse this dataset for suspicious activity`.

- Expect **~60 seconds** on the first run. This is genuinely slow and it is not broken — it runs the
  entire detection stack: features over 2,002 transactions, seven rules, IsolationForest and LOF, then
  fusion. Measured locally at 68s.
- Expect **41 flags**. That is the published figure, so it is also a check that the deployment matches
  the documentation.

### 6.3 The fast query returns

Type `Which customers made 10+ transactions under $10,000?`

- Expect **~4 seconds** and a **3-step plan**: `load_data → filter_data → aggregate_query`.
- **This is the demo's best single moment.** The plan visibly contains no `ml_detect` and no
  `rule_detect` — the agent decided a deterministic count answers this exactly and skipped them. Point
  at that when showing anyone. It is the difference between an agent and a pipeline with a text box.

### 6.4 The plan trace renders

On any result, the execution plan should show each step, its reason, and a *tools considered but
skipped* section with reasons. If the plan is empty or missing, something is wrong — that panel is the
core of what this project is.

### 6.5 Cold start

Community Cloud puts apps to sleep after inactivity. Close the tab, wait ten minutes, and open the URL
again. It should wake up within about a minute and show a *waking up* screen rather than an error.

Do this **before** you send the link to anyone, so you know what they will see.

---

## 7. Resource limits, and other platforms

### What the app actually uses

Measured, not estimated — peak resident memory through a full session on Python 3.11:

| Stage | Peak RSS |
|---|---|
| pandas + numpy | 77 MB |
| + scikit-learn | 152 MB |
| + networkx + plotly | 165 MB |
| + streamlit | 179 MB |
| + backend imported, dataset loaded | 213 MB |
| **+ full_analysis (41 flags)** | **252 MB** |
| + two further queries | **254 MB** |

**Peak 254 MB, and it levels off** — a third `full_analysis` used exactly the same as the second, so
there is no leak across queries. With Streamlit's own server and session overhead on top, call it
~350 MB against Community Cloud's ~1 GB. That is roughly 3× headroom.

An earlier draft of this guide called memory "the real risk". That was caution without evidence and it
was wrong. The thing that actually breaks this deploy is the Python version
([§8.2](#82-the-build-fails-installing-dependencies)).

Disk is not a constraint either, and the trimmed requirements file barely helps: `jupyter`,
`jupyterlab` and `notebook` come to about 40 MB, `kaggle` 1 MB, `pytest` under 1 MB. The bulk is
unavoidable — scipy 113 MB, plotly 87 MB, pyarrow 85 MB, pandas 67 MB, scikit-learn 45 MB, numpy
34 MB. **`requirements-deploy.txt` saves ~50 MB of disk and zero RAM.** Use the root
`requirements.txt`.

### Why Streamlit Community Cloud rather than somewhere else

| Platform | Verdict | Why |
|---|---|---|
| **Streamlit Community Cloud** | ✅ **Best fit** | Purpose-built for Streamlit. ~1 GB limit against a measured 254 MB. Free, deploys from GitHub, no card |
| **Hugging Face Spaces** | ✅ Best fallback | Free tier gives **16 GB RAM** and 2 vCPU, and supports Docker — so it could run both processes as originally designed. More setup: a Dockerfile and a Space |
| **Render** | ⚠️ Works, badly | Free tier is **512 MB RAM and 0.1 CPU**. Memory fits; the CPU does not — a query that takes 60s here would take many minutes. Also sleeps after 15 minutes idle |
| **Vercel** | ❌ Architecturally impossible | Serverless, with a **250 MB** function size limit against ~430 MB of core dependencies, and a request timeout below this app's 60s query. Streamlit also needs a persistent WebSocket server, which serverless is not |
| **GitHub Pages** | ❌ Architecturally impossible | Static file hosting. No Python execution of any kind |

The last two are worth being clear about: they are not *tight*, they are the wrong shape. Neither can
run a long-lived Python process, and that is what Streamlit fundamentally is.

If you do end up needing Hugging Face Spaces, ask and I will write the Dockerfile and Space
configuration. Do not start there — the measurements say Community Cloud is comfortable.

---

## 8. Troubleshooting

### 8.1 `ModuleNotFoundError: No module named 'frontend'` or `'backend'`

This should not happen — `frontend/app.py` puts the repository root on `sys.path` before its first
package import, and `tests/test_app_imports.py` pins that behaviour.

It is worth knowing why the guard exists, because it is a genuinely non-obvious trap:
`streamlit run frontend/app.py` puts only the **script's own folder** on `sys.path`, never the repo
root. Locally the app worked anyway, purely because `run_demo.py` launches it via `python -m streamlit`
and `-m` adds the working directory. Run the same app through the `streamlit` console script — which is
what every host uses — and it dies on the very first import. It was found by reproducing the host's
`sys.path` rather than by anything the test suite did.

If you somehow see it anyway: confirm you deployed the current `main`, and that the main file path is
`frontend/app.py`.

### 8.2 The build fails installing dependencies

**This has happened for real, so start here rather than reading the error.**

Scroll the build log up to near the top and find the line reading
`Using Python 3.X.Y environment at /home/adminuser/venv`.

If that version is **3.13 or newer, that is your problem**, whatever the error at the bottom says. The
observed failure was:

```
Using Python 3.14.6 environment at /home/adminuser/venv
...
× Failed to download and build `pillow==10.4.0`
...
The headers or library files could not be found for zlib,
a required dependency when compiling Pillow from source.
```

Nothing there is about your code. `pandas==2.2.3` and the `pillow` pulled in by `streamlit==1.39.0`
publish no wheels for 3.14, so pip tried to **compile them from source**, and that needs zlib headers
the container lacks. Chasing zlib is wasted effort — you cannot install system packages on Community
Cloud, and you would not need to on a supported Python.

**The fix, and it is annoying:** the Python version can only be set when an app is created. There is no
setting to change it afterwards.

1. **Manage app** → **⋮** → **Delete app**. This costs nothing; all the code is in GitHub.
2. Redo [§4](#4-deploy-the-app), and this time set **Python version → 3.11** in Advanced settings
   before deploying.

Two other symptoms of the same root cause, so you can recognise it early:

- The log shows `Downloading pandas-2.2.3.tar.gz` — a **`.tar.gz` is source**, not a wheel. Any large
  package arriving as a tarball means that version has no wheels for that Python.
- The build runs far longer than the usual 5–10 minutes before failing. Compiling pandas from source
  is tens of minutes of work that was never supposed to happen.

If the version is right and a *specific* package still fails, read which one. `kaggle` and `jupyter`
are not needed by the running app — that is the case where the `requirements-deploy.txt` route at the
end of [§4](#4-deploy-the-app) applies.

### 8.3 Other people are asked to sign in

The app is restricted to invited viewers. See [§6.0](#60-anyone-can-actually-open-it) — the fix is
**Manage app → Settings → Sharing**, and it is not the same setting as the repository's visibility.

You will not notice this from your own browser, because you are signed in. Always check the link from
a private window before sending it to anyone.

### 8.4 The app loads but says "API Offline"

The UI could not reach a backend, so it is showing fixture data behind a warning banner.

Check `AML_API_URL` in your secrets is **blank** (`""`), not unset-but-present-with-a-value and not
`http://localhost:8000`. A localhost URL on a cloud host points at the container itself, where nothing
is listening.

If it is blank and you still see this, the backend failed to import. Open **Manage app** and read the
log for a traceback.

### 8.5 The sidebar shows 1,710 transactions / 294 customers

`AML_DATA_SOURCE` is not reaching the app. Two causes:

1. The secret is missing or misspelled. It is `AML_DATA_SOURCE`, all caps, underscores.
2. You edited secrets after the app started. Streamlit reads secrets at startup —
   **Manage app → Reboot app**.

### 8.6 A query hangs or times out

Full analysis takes ~60 seconds legitimately. Beyond about three minutes, suspect memory — see
[§7](#7-resource-limits-and-other-platforms).

### 8.7 Changes to the repo do not appear

Community Cloud redeploys on push to the tracked branch, but not instantly. Force it with
**Manage app → Reboot app**. Confirm you pushed to `main` and not another branch.

---

## 9. After it is live

### 9.1 Send me the URL

I will run the cold-load check: one query per intent, confirming the plan trace renders with no key
configured, and that the numbers match the README.

### 9.2 Put it in the README

Add the link at the top so anyone landing on the repo can click through. Tell me the URL and I will
write it in.

### 9.3 Know what to say about it

Two things worth having ready, because both will come up:

**"Why is it slow?"** Because it runs the real detection stack on every query rather than serving
precomputed results — features over 2,002 transactions, seven rules, two ML models, then fusion. The
threshold query returns in ~4 seconds precisely because the agent *plans differently* for it and skips
the expensive tools. The slowness and the plan divergence are the same fact.

**"Is this real data?"** No, and say so first rather than being asked. It is a synthetic dataset the
project generated for itself. The IBM AML ingestion path is built and has never been run because it
needs a Kaggle token. There is a fuller answer to this question in your private briefing —
it is the criticism most worth being ready for.
