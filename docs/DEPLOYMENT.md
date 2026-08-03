# Deploying the demo

**A step-by-step guide to putting this app on a public URL, assuming nothing.**

The app is normally **two processes** — Streamlit talking HTTP to FastAPI. Free hosts run one. This
guide covers the single-process path, which [`frontend/api_client.py`](../frontend/api_client.py)
exists to make possible.

Target: **Streamlit Community Cloud**. It is free, it deploys straight from GitHub, and it needs no
credit card. The fallback if it does not fit is in [§7](#7-if-streamlit-cloud-does-not-work-out).

**Time:** about 15 minutes of your attention, plus 5–10 minutes of waiting for the first build.

---

## Contents

1. [Before you start](#1-before-you-start)
2. [Check the repository is ready](#2-check-the-repository-is-ready)
3. [Create the Streamlit Cloud account](#3-create-the-streamlit-cloud-account)
4. [Deploy the app](#4-deploy-the-app)
5. [Add the secrets](#5-add-the-secrets)
6. [Verify it actually works](#6-verify-it-actually-works)
7. [If Streamlit Cloud does not work out](#7-if-streamlit-cloud-does-not-work-out)
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

   3.11 is what the repo is developed and CI-tested against. Leaving it on a different version is the
   single most likely cause of a dependency failing to install, because the pinned versions in
   `requirements.txt` were resolved for 3.11.

5. **Do not click Deploy yet.** The Advanced settings panel also contains the **Secrets** box, and it
   is much less annoying to fill that in now than to fix a broken first boot. Go to
   [§5](#5-add-the-secrets), paste the secrets, and then come back and click **Deploy**.

### A note on the requirements file

Community Cloud **auto-detects** your dependency file. It does not, as far as I can determine, offer a
field to choose one by name — I could not verify its server-side behaviour from this repo, so treat
this section as the thing most likely to need a small adjustment on the day.

What it will find is **`requirements.txt`** in the repo root. That works. It installs four packages the
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

The build takes **5–10 minutes** on the first run. You will see a log streaming. It goes:

1. Cloning the repository
2. Installing dependencies — this is the slow part, `scikit-learn` and `pandas` are large
3. `You can now view your Streamlit app in your browser`

If the log stops with a red error, go to [§8](#8-troubleshooting).

---

## 6. Verify it actually works

A page that loads is not a demo that works. Check all five of these.

### 6.1 The sidebar says the right things

Look at the left sidebar:

- ✅ **API Online** — green. If it says *API Offline* with a fixture warning, the backend failed to
  start; see [§8.3](#83-the-app-loads-but-says-api-offline).
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

## 7. If Streamlit Cloud does not work out

The realistic failure is **memory**. Community Cloud caps a container at roughly 1 GB, and
pandas + scikit-learn + networkx + plotly is most of that before your data is loaded. If the app keeps
restarting or dies mid-query, that is what is happening.

**Fallback: Hugging Face Spaces.** It supports Docker, which means you can run both processes the way
the app was designed, and the free tier gives 16 GB of RAM. It is more setup — you write a Dockerfile
and push to a Space — but it removes the constraint entirely rather than working around it.

Ask me and I will write the Dockerfile and the Space configuration if you get to that point. Do not
start there; try the fifteen-minute path first.

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

Almost always the Python version. Go to **Manage app → Settings → Python version** and set **3.11**,
then **Reboot app**.

If it is a specific package failing, read which one. `kaggle` and `jupyter` are not needed by the
running app — that is the case where you use the `requirements-deploy.txt` route described at the end
of [§4](#4-deploy-the-app).

### 8.3 The app loads but says "API Offline"

The UI could not reach a backend, so it is showing fixture data behind a warning banner.

Check `AML_API_URL` in your secrets is **blank** (`""`), not unset-but-present-with-a-value and not
`http://localhost:8000`. A localhost URL on a cloud host points at the container itself, where nothing
is listening.

If it is blank and you still see this, the backend failed to import. Open **Manage app** and read the
log for a traceback.

### 8.4 The sidebar shows 1,710 transactions / 294 customers

`AML_DATA_SOURCE` is not reaching the app. Two causes:

1. The secret is missing or misspelled. It is `AML_DATA_SOURCE`, all caps, underscores.
2. You edited secrets after the app started. Streamlit reads secrets at startup —
   **Manage app → Reboot app**.

### 8.5 A query hangs or times out

Full analysis takes ~60 seconds legitimately. Beyond about three minutes, suspect memory — see
[§7](#7-if-streamlit-cloud-does-not-work-out).

### 8.6 Changes to the repo do not appear

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
