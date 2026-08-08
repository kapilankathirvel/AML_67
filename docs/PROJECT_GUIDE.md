# Project guide

**One pass through the whole system, layer by layer.**

Every layer below answers the same four questions: what it does, why it was built that way, what was
measured, and what is honestly wrong with it. The fourth question is the reason this document exists —
the numbers are all in [README.md](../README.md), and repeating them would not help anyone understand
the system. What is harder to find is which decisions were forced, which were chosen, and which are
still unjustified.

Read [AFTER_THE_DEADLINE.md](AFTER_THE_DEADLINE.md) alongside this if you want the chronology: what
was in the hackathon submission versus what came after.

**Contents**

1. [The shape of the thing](#1-the-shape-of-the-thing)
2. [Data](#2-data)
3. [Features](#3-features)
4. [Rules R1–R7](#4-rules-r1r7)
5. [The ML half](#5-the-ml-half)
6. [Fusion and bands](#6-fusion-and-bands)
7. [The agent layer](#7-the-agent-layer)
8. [Evaluation](#8-evaluation)
9. [What I would do next](#9-what-i-would-do-next)

---

## 1. The shape of the thing

A natural-language question goes in. The system parses it into a structured intent, **builds a plan
specific to that intent** rather than running a fixed pipeline, executes only the tools the plan
contains, and returns risk-scored, explained, escalation-tagged flags.

```
query ─▶ intent_parser ─▶ planner ─▶ plan_validator ─▶ executor ─▶ narrator ─▶ flags
                                          ▲                │
                                          └─── replanner ◀──┘
                                              (after each step)
```

The tools the executor can call are a **fixed list of nine**: `load_data`, `filter_data`,
`eda_profile`, `feature_engineer`, `rule_detect`, `ml_detect`, `aggregate_query`, `entity_lookup`,
`risk_classify`. Each is a pure function `(ToolContext, **params) -> ToolResult`. No tool imports
another tool, and no tool imports from `backend.agent.*`. That constraint is what makes the plan
meaningful: if tools could call each other, the plan would be a suggestion rather than a description
of what ran.

**The one thing to understand about the architecture:** the plan is data, not control flow. It is
constructed, validated, and only then executed, and the executed trace is returned to the user. That
is what makes it auditable — a compliance function can read what was decided and why without reading
the code.

The UI reaches all of this through [`frontend/api_client.py`](../frontend/api_client.py), which chooses
between HTTP to a separate FastAPI process and importing the backend into the Streamlit process. The
second exists because free hosts run one process. It matters here only for one reason: the in-process
path calls `backend.main`'s endpoint functions rather than re-running the pipeline itself, so there is
exactly one implementation of what a query does and a deployed demo cannot answer differently from a
local run.

---

## 2. Data

**Two synthetic datasets, and a third path that has never run.**

| Source | What it is | Where it is used |
|---|---|---|
| `synthetic` | `data/sample/aml_sample.csv` — 2,002 transactions, 270 customers, 202 labelled | **Every published metric** |
| `synthetic_alt` | A second generated set with a different raw schema | `load_data`'s **runtime default** |
| `ibm` / `ibm_stratified` | IBM AML (Kaggle) via a stratified sampler and parquet cache | **Never run** |

### Why the default is not the measured set

`load_data` defaults to `synthetic_alt`, but everything scored against labels pins `source="synthetic"`
explicitly. This looks like an inconsistency and is deliberate: `synthetic_alt` has a *different raw
schema*, so it exercises the loader's column-normalisation path in ordinary use. If the default were
the measured set, that path would only ever run in tests.

The cost is a real trap, and it is called out in [CLAUDE.md](../CLAUDE.md): **anything that compares
flags to labels must pin the source, or it silently scores one population against another's labels.**
`run_evaluation`, `ablation` and `evasion` all pin it and all assert the loaded customer count is 270
before reporting anything.

### The honest weakness

**The data is synthetic and self-generated. This is the fairest criticism of the project.** Every
number in the README describes data the project produced for itself, which means the detectors were
built and evaluated against the same author's idea of what laundering looks like. Recall in particular
should be read as "recall against patterns we knew to plant".

The IBM ingestion path is fully built — `data/build_ibm_cache.py`, a stratified sampler, a parquet
cache — and has never executed, because it needs a Kaggle API token. That is the single highest-value
thing outstanding on the project.

---

## 3. Features

`feature_engineer` produces a frame indexed by `customer_id` with **18 columns**. Seventeen are
behavioural features; the eighteenth, `zscore_n_samples`, is a support count that `ml_detect` excludes
as metadata (`ml_detect.py:90`). So "17 features" and "18 columns" are both correct, about different
things.

| Group | Columns |
|---|---|
| Rolling volume | `rolling_1d_sum`, `rolling_1d_count`, `rolling_7d_sum`, `rolling_7d_count`, `rolling_30d_sum`, `rolling_30d_count` |
| Threshold behaviour | `pct_just_below_threshold`, `round_amount_ratio` |
| Deviation | `amount_zscore_90d` (+ `zscore_n_samples`) |
| Velocity | `velocity_txns_per_hour`, `velocity_counterparties_per_day` |
| Counterparties | `new_counterparty_ratio`, `cross_border_count`, `cross_border_ratio` |
| Timing | `night_hours_ratio` |
| Directional | `rapid_cashout_ratio`, `pass_through_ratio` |

Features are computed **per pattern**, not all at once: a structuring query computes nine features, not
seventeen. The mapping lives in `_PATTERN_FEATURES` (`features.py:92`). This is one of the places the
plan actually changes what executes.

### The weakness that matters most

**Sixteen of the eighteen columns are computed per *sender*.** Only `rapid_cashout_ratio` and
`pass_through_ratio` consult inbound flows at all.

Be precise about what this does and does not mean, because it is easy to overstate — I did, more than
once, before measuring it. It does **not** mean receive-only customers are excluded: all 270 customers
enter the feature frame and all 270 receive an ML score. What it means is that **what gets measured
about them is their ordinary outbound behaviour**, which is unremarkable. The 63 receive-only positives
have a median ML percentile of **0.486** — dead centre — and only **2 of 63** clear the 0.95 ML-only
floor.

R7 (inbound structuring) recovers 12 of them. Closing the rest needs features that describe inbound or
graph structure, which is a different kind of work from tuning a threshold.

### One bug worth knowing about

`velocity_txns_per_hour` was originally *(max count in any 24-hour window) ÷ 24* — **a daily average
wearing an hourly name**. R5's documented threshold of 2.0 txns/hour therefore silently meant *48
transactions in one day*, and the busiest sender in the dataset has 25 transactions in total. **R5 was
unreachable by construction, and no amount of threshold tuning could have found it, because the bug
was in the feature rather than the rule.**

Fixing it made the published numbers *worse* — sender-side precision fell 0.590 → 0.561. That
regression is reported rather than tuned away.

---

## 4. Rules R1–R7

Seven deterministic detectors, all thresholds justified in [AML_LOGIC.md](AML_LOGIC.md) §3. This table
is the summary an interviewer would want: the rule, its regulatory basis, whether it earns its place,
and what it costs an adversary to defeat.

| Rule | Threshold | Basis | Ablation verdict | Evasion cost |
|---|---|---|---|---|
| **R1** structuring | ≥3 txns in $9,000–$9,999.99 per 7 days | Below the $10k CTR (Form 112) | **Load-bearing.** 11 hits, precision 1.000; removing it costs 0.075 precision, 0.118 recall | ~$497/txn to step under the band |
| **R2** smurfing | ≥5 distinct receivers in 48h, median $7k–$10k | Fan-out to mules | **Redundant, not wrong.** Precision 1.000 alone; marginal contribution exactly zero | Survives the R1 dodge — $8,999 is still in its band |
| **R3** layering | ≥3-hop chain, pass-through ≥0.70, ≥1 cross-border hop | FATF layering typology | **Not earning its place.** 4 hits, **zero** true positives under every definition; removing it *improves* precision | — |
| **R4** rapid cash-out | ≥$10k in, then ≥3 cash-outs ≥50% within 24h | Placement→integration | **Load-bearing.** Precision 1.000; removing it costs 0.106 precision, 0.157 recall | 48h delay on $1.30M of principal |
| **R5** velocity | ≥2.0 txns/hour **and** amount z-score ≥3.0 | Behavioural break | **Fires on nobody** — but now a measured fact: 15 senders clear the rate gate, all 15 fail the z-score gate (max 2.29 vs 3.0) | — |
| **R6** dormant reactivation | 60d dormant, then ≥3 txns in 7d at z ≥2.0 | Account takeover / mule activation | **Fires on nobody** | — |
| **R7** inbound structuring | ≥2 sub-threshold txns from one sender per 7d | R1 from the receiving end | **The trap.** See below | — |

### R7 is the most instructive row in the table

Scored sender-side, R7 looks like the worst rule in the system: precision **0.000**, and deleting it
*gains* 0.181 precision. Acting on that would have deleted one of the better rules.

R7 is receiver-side by design, so under a sender-only ground truth it is **arithmetically incapable**
of a true positive — no entity it can flag is in the positive set unless that entity also sent. Under
the repeat-receiver definition its precision is **1.000**, and removing it costs 0.055 precision and
0.119 recall.

**The lesson generalises past this repo: a metric can be computed correctly and still be measuring the
wrong thing.** Reporting only the sender-side column would have been defensible, reproducible, and
wrong. Every rule is now scored under both.

### R3 is the one genuinely underperforming rule

4 hits, zero true positives, and removing it improves precision at no recall cost. It is **kept**
anyway: 4 hits is far too small a sample to retire a rule on, and the layering typology is real. But
it is on the record rather than hidden inside an aggregate.

**And the out-of-time study found out why it only gets 4.** R3 enumerates chains only from nodes with
**in-degree 0** in the wire/transfer subgraph (`rules.py:311`). That set shrinks as history
accumulates, because a node that looked like a chain origin over 29 days has received something by day
90:

| Window | Eligible txns | Sources (in-deg 0) | Sinks | Pairs searched | R3 hits |
|---|---|---|---|---|---|
| 29 days | 362 | 46 | 50 | 2,300 | 6 |
| 60 days | 659 | 28 | 21 | 588 | 12 |
| 89 days | 1,021 | **6** | 7 | **42** | 4 |

**R3 is anti-monotone in data volume.** More evidence gives it a smaller search space. The 4 hits are
not a rule that is bad at layering — they are a rule that had 42 candidate pairs left to search out of
a possible 2,300. On a production graph with years of history, effectively nobody has zero inbound
wires and R3 would have nowhere to start at all.

If you are asked one question about the rules, this is the one worth volunteering. It is a concrete,
mechanical, scale-dependent defect, and the fix is not a threshold — it is that "chain origin" cannot
be defined as a global graph property when the graph is a growing accumulation of history. It needs
to be defined within a time window, which is a design change rather than a tuning change.

**And the counterfactual says what that change would be worth.** The shipped `rule_detect`, run
unchanged over non-overlapping partitions of the same transactions with R3's hits unioned:

| Config | Flagged | True positives | Precision |
|---|---|---|---|
| Whole frame — what ships | 4 | **0** | **0.000** |
| 7-day windows | 3 | **3** | **1.000** |
| 14-day windows | 7 | 4 | 0.571 |
| 30-day windows | 15 | 4 | 0.267 |

The two sets are **disjoint**: every entity the whole-frame run flags is a false positive, every
entity the 7-day partition flags is a launderer, and no entity appears in both. The origin collapse
is not merely reducing R3's yield — it is selecting the wrong chains. Precision decays monotonically
as the windows widen back toward the whole frame, which is the mechanism confirming itself.

That is a counterfactual, not an implementation: unioning over a hard partition is cruder than a real
fix, and recall stays low throughout because R3 is one typology rather than the system. The claim is
about precision only.

It is **deliberately not fixed yet**: changing detection code invalidates every baseline under
`evaluation/results/`, and that is a decision to take on its own rather than as a side effect of
adding a study. The counterfactual does not change that sequencing argument, but it does mean the fix
is now a costed item rather than a hunch.

---

## 5. The ML half

`IsolationForest(contamination=0.05, n_estimators=100, random_state=42)` as primary,
`LocalOutlierFactor(n_neighbors=20)` as secondary, fused `0.6 × IF + 0.4 × LOF`, both percentile-ranked,
`StandardScaler` first. Explainability is the top-3 features by `|value − median| / std` — deliberately
cheap, no SHAP.

None of that is the interesting part.

### The reference population — the best decision in the codebase

Percentiles used to be ranked **inside the cohort the analyst's query had selected**. So a customer's
risk score was partly a function of what was typed into the search box.

Measured: adding `amount_min=5000` to a structuring search moved percentiles by up to **0.73** and
pushed **four customers across a risk band**. A number that decides whether a SAR gets drafted cannot
move because somebody narrowed their search.

Both models are now fitted and ranked on `features_reference` — the unfiltered customer set — and the
resulting percentile is looked up for whichever entities the query asked about. The implementation
detail is worth noting: `feature_engineer` computes the reference frame by **recursively calling
itself** on the unfiltered transactions (`features.py:692`), rather than extracting the ~120-line
computation into a shared helper. That guarantees both frames go through byte-identical logic. The
recursive call receives no `transactions_reference` artifact, so there is no second level.

**The trade-off, accepted deliberately:** the ML term is now blind to the query window. A customer who
is unremarkable across the full dataset but spikes inside a 30-day filter no longer stands out on the
ML half. The rules still run on the filtered frame and still catch them. For a score attached to an
escalation decision, stability was judged the more important property — but it *is* a trade, not a
free improvement.

The same fix closed a related hole: single-entity queries used to skip `ml_detect` entirely on the
reasoning that one row is too small a sample. That reasoning was wrong about its own plan —
`feature_engineer` runs across the whole population precisely so the score is comparable — so
"is customer C-STR02 suspicious?" returned **51.00 MEDIUM** while a full sweep called the same
customer **89.84 HIGH**.

### The honest weakness

**It is transductive: `ml_detect` fits and scores the same rows.** Defensible for unsupervised scoring
— there is no label leakage, because there are no labels — but a bank's model-validation function will
ask how it generalises, and "by design" is only a good answer if you know what the alternative costs.

`evaluation/out_of_time.py` measures it: fit on the first 60 days, score the last 29. **The frozen
model costs 0.036 precision and no recall.** That is a small number, and the useful reading is that
the transductive shortcut is buying almost nothing — the design could be changed for the price of one
false positive, and the reason not to is simplicity rather than accuracy.

Two limits on that number, both the dataset's fault:

- **No cold starts exist.** All 268 test-window customers also appear in the training window, so the
  expensive part of out-of-time validation — customers the model has never seen — goes unmeasured.
- **The shipped code cannot actually do this.** `LocalOutlierFactor` is built with `novelty=False`,
  which has no `score_samples` and can only score rows it was fitted on. The study substitutes
  `novelty=True` — identical fit, plus the scoring method the other mode withholds — and reports an
  IsolationForest-only control so the substitution can be checked rather than trusted. Found by trying
  to run the study, not by reading the code.

---

## 6. Fusion and bands

```
risk_score = 100 × (0.6 × max_rule_weight + 0.4 × ml_percentile)

  ≥ 70  → high    → report     (SAR path)
  40–69 → medium  → review
  15–39 → low     → monitor
   < 15 → none    → no_action
```

Rule weight is the **maximum** across rules that fired, not the mean. A customer triggering R1 and R3
is more dangerous, but the weight is not double-counted; both rules are still surfaced in the evidence.

An entity appears at all only if it has ≥1 rule hit **or** an ML percentile ≥ 0.95.

### Two structural facts, both found by ablation

**Neither half can produce a HIGH flag alone.** The largest rule weight is R1 at 0.85, so rules-only
tops out at `0.6 × 0.85 × 100 = 51`. ML-only tops out at `0.4 × 1.0 × 100 = 40`. Both sit under the
HIGH band of 70. **The SAR-drafting tier is arithmetically unreachable without corroboration from both
signals.** That is a defensible design for a compliance system — and it was never a stated one. A test
now pins it.

**The 0.6/0.4 split cannot affect precision or recall at all.** Not weakly — exactly. Sweeping
`RULE_WEIGHT_COEFF` from 0.0 to 1.0 leaves precision at 0.561 and recall at 0.451 throughout, because
membership in `risk_rows` is decided by the entity universe, which never consults the coefficients.
They only redistribute severity inside a fixed set. **So 0.6/0.4 is purely a banding decision, and any
claim that it was tuned for detection quality would be false.**

### The honest weakness

**The formula, the weights and the bands are all hand-set.** They are documented and defensible, but
they are not calibrated. Ablation shows a HIGH threshold of **75 dominates 70** on this data — higher
precision (0.840 vs 0.778), higher F1, *identical* recall, because the two flags it drops are both
false positives. **It was left at 70 deliberately.** Moving it would be tuning a published constant
against the same 270 customers it is evaluated on, which is the circularity the study exists to expose
rather than exploit. It is worth revisiting on IBM data, where the tuning and evaluation sets can
differ.

---

## 7. The agent layer

This is what makes the project agentic rather than a pipeline with a text box on the front.

### Intent parser

Natural language → `QueryIntent` (intent, filters, entities, pattern types, top-n, confidence). LLM
first, with a **deterministic regex/keyword parser that alone covers all seven intents well enough to
demo.** No API key is required for the system to work.

The messy details that turned out to matter: providers routinely return relative-date shorthand
(`"-30d"`, `"1 month ago"`) instead of ISO dates, so `_coerce_relative_date()` resolves them against
the dataset's own reference date rather than failing validation. And free-tier quotas are small enough
to exhaust in one testing session, so `complete_json()` caches on `(prompt, schema_hint)` — which means
any prompt containing a changing counter must include it in the digest, or the cache returns a stale
answer.

### Two planners, with a deterministic floor

`build_plan` (`planner.py`) is a deterministic if/elif over the seven intents. `plan_query`
(`llm_planner.py`) asks a model to select tools. **The LLM planner is gated behind `aml_llm_planner`
and falls back to the deterministic one on any failure or rejection.** Both are pinned OFF in
`run_evaluation.py`, so every published metric describes the deterministic path.

Having both is the point. The deterministic planner guarantees the system always works; the LLM
planner is what makes tool selection genuinely model-driven when enabled. Neither alone would be
honest — a pure-LLM system that fails without a key is a demo, and a pure-deterministic one is a
pipeline calling itself an agent.

### The validator — the piece I would point at first

**Fifteen rules, V0 through V14**, that a model-proposed plan must satisfy before anything executes.
Violations are collected rather than short-circuited, so a rejection states everything wrong with the
proposal.

- **V0–V11 — safety and dependency legality.** V5–V7 mirror real preconditions in the tool bodies, so
  a plan that passes cannot fail on a missing artifact.
- **V12 — answerability.** The plan must contain the tool that produces the response its intent
  promises. Added after measuring: with V0–V11 alone a local model hit **60% acceptance while only 1
  plan in 15 was useful.** It had learned that *shorter plans pass*, and a truncated plan satisfies
  every ordering rule vacuously. "Who are my riskiest customers?" came back as
  `load_data → filter_data → feature_engineer` — perfectly legal, and it returns nothing.
- **V13 — closed-set parameter *values*, not just names.** Found live: a plan proposed
  `pattern_types=["risk"]`. `"risk"` is not a `PatternType`. Every tool was real, every dependency
  satisfied, the terminal tool present — and `feature_engineer` computed **0 features**, `rule_detect`
  ran nothing, and the query returned empty with no warning.
- **V14 — authority.** Some capabilities are not the model's to invoke *at any value*. `load_data`
  declares `source` and `force_rebuild`, so a proposal of `{"source": "ibm", "force_rebuild": true}`
  was fully legal under V0–V13 — it would have switched the product onto a different dataset mid-request
  and triggered a parquet cache rebuild, **a filesystem write, on a model's say-so.**

*"A plan may choose how to analyse, but not which dataset the product runs on"* is a compliance
instinct, and V14 is what it looks like in code.

Two design boundaries worth defending:

**Repair versus rejection.** Defects with exactly one correct fix and no judgement in applying it — a
missing `load_data`, empty `filter_data` params, a missing `entity_id` — are repaired and logged.
Anything involving a real choice (which detectors run, which patterns to test) is **never** repaired,
because silently rewriting those would make "the LLM chose this plan" untrue.

**V12 constrains output, not route.** A ranking query that cannot return a ranking is broken, not
suboptimal. Everything upstream stays the model's call. A legal-but-clumsy plan still passes; judging
elegance is not a whitelist's job.

### Executor and the observe→decide→act loop

The executor runs the plan step by step, threading **both** `ctx.df` and `ctx.artifacts` between steps.
After each step the re-planner may revise the *remaining* plan based on what the executed steps
actually produced.

It was originally **failure-blind**: unknown tools, raised exceptions and `ok=False` results each hit a
`continue` in the main loop, which skipped every decision point below — **including the one whose
entire job is deciding what to do when something goes wrong.** Fixing it surfaced a second real defect:
`MAX_REPLANS = 2` was being consumed by *successful* steps, so a failure at step four found no budget
left. Failures now have a separate `MAX_FAILURE_REPLANS`.

### Narrator

`risk_rows` → `Flag` objects with explanations and escalation actions. **The template layer always
runs and is always accurate**, built from each hit's evidence dict. LLM polish is optional, capped to
the first few HIGH rows, and **only rewrites** — it is never given licence to invent a number.

### The honest weakness

**The re-planner declines to intervene.** Across five queries through the full pipeline it produced a
decision on 5/5 and **revised on 0/5**; outcomes differed on 0/5. That is arguably correct behaviour —
the model plans well enough up front that by the time it sees the observation there is nothing to fix —
but it means the loop is unproven in practice. Manufacturing a scenario where it fires would have been
easy and dishonest.

---

## 8. Evaluation

### Three ground truths

The system flags **customers**; the dataset labels **transactions**. Lifting one to the other has more
than one defensible answer, and they disagree by more than 2× on recall — which is why all three are
reported rather than whichever flatters.

| Definition | Positives | What it asks |
|---|---|---|
| `sender_only` | 51 / 270 | How well does the system do the job it was built for? |
| `sender_or_receiver` | 114 / 270 | How much of the problem does that job leave untouched? |
| `sender_or_repeat_receiver` | 84 / 270 | How much of that gap is actually evidenced in the data? |

The third exists because the second **over-labels**: 30 of its 63 receive-only positives receive
exactly *one* labelled transaction, which does not distinguish a participant from someone a launderer
happened to pay once. Scoring against them measures whether the system can identify people the data
gives it no evidence about.

Two spellings of "repeat" were measured and coincide exactly on this dataset (≥2 inbound total, versus
≥2 from a single sender). The simpler one is implemented, and `REPEAT_RECEIVER_MIN_TXNS` documents that
the distinction would need revisiting if that ever stops holding.

### The four studies

- **`run_evaluation.py`** — how well does the system do? Precision/recall/FPR against all three
  definitions, plus a naive baseline (flag anyone who sent >$9,000) that the whole false-positive story
  is told against.
- **`ablation.py`** — which parts are doing the work? Runs the detection stack **once**, then produces
  every configuration by re-fusing the captured `rule_hits`/`ml_scores` through the real
  `risk_classify` with patched module constants. ~40× faster than re-running, and it guarantees every
  row sees byte-identical detector output, so a difference between rows can only come from the fusion.
- **`evasion.py`** — what does it cost to defeat us? Perturbs the launderers' own transactions in
  memory and re-runs the full stack per configuration. It **cannot** re-fuse, because changing an
  amount changes the features, which changes both halves.
- **`out_of_time.py`** — does the ML half generalise forward? Fits on the first 60 days and scores the
  last 29, holding the test rows and the ground truth fixed so that only the fitting population moves.
  Three arms, changing one thing at a time: fit-on-test (what ships), fit-on-train, and fit-on-train
  with the percentile cut points frozen at training time too — the last being what a deployed model
  actually has available to it.

### Why the evasion study is the one that matters

The ablation returned an uncomfortable result: **the hybrid is less precise than either half alone**
(0.561 vs 0.583 rules-only and 0.692 ML-only). It takes the union of their flags and inherits both sets
of false positives. Taken alone, that is an argument for deleting the ML half.

That verdict is correct and incomplete, because a static dataset models an adversary who never adapts —
and in AML that adversary does not exist. **Structuring is itself an adaptation**, to the $10,000 CTR
threshold. The $9,000 band R1 keys on is the *previous* move in a game still being played.

| Move | Rules retain | ML retain | Hybrid retain |
|---|---|---|---|
| Space transactions further apart | **0.048** | **0.889** | 0.391 |
| All moves combined | **0.095** | 0.778 | 0.391 |

Timing evasion takes rule recall from 0.412 to 0.020 and the ML half barely registers it. **The hybrid
retains more than the rules alone under every move tested.** That is what the 0.022 of precision buys,
and it is a measured robustness claim rather than an assertion that "we used both".

Three caveats travel with it, because without them the table says more than it should:

1. **The retention ratios are not comparable across columns.** Each is against its own baseline, and
   the ML half starts at 0.176 recall against the rules' 0.412. It degrades gently partly because it
   was never catching much. The defensible claim is narrow: **the two halves fail to different moves.**
2. **Everything degrades in absolute terms.** Hybrid recall under the combined move is 0.176 — 9 of 51.
   The system is more robust than its rules, not robust.
3. **The cheap evasion is not as cheap as the threshold implies.** Stepping under the band costs a mean
   of $497 per transaction, not $1, because in-band amounts average ~$9,495. That is an upper bound — a
   launderer who re-split would pay less.

### Things that were measured and *not* built

The most useful output of the evaluation work is the list of ideas that turned out not to be worth
implementing.

- **Fan-in detection.** Before building a funnel-account rule I measured the premise: receive-only
  positives average **7.6** distinct inbound counterparties against a population average of **6.9**, and
  within any 48-hour window both top out at 4. There is no separation to threshold on.
- **Noisy-OR fusion.** Combining rule weights probabilistically only helps when rules corroborate each
  other, so I counted first: 37 rule hits land on 36 distinct entities, and exactly **one** (C-HUB01, on
  R1 and R2) triggers more than one rule. Zero customers would move across a band.
- **A larger local model.** The same planner scores **27% on a local 3B and 93% on a hosted model** —
  same validator, same prompts, same queries. The 3B result was kept deliberately: pushing it from 20%
  to 27% is what produced V12, V13 and the `load_data` repair. **A weak model is a better validator test
  than a strong one.**

---

## 9. What I would do next

In priority order, with the reason rather than just the task:

1. **Run the IBM path.** It is built and blocked only on a Kaggle token. It converts every metric in
   the repo from "measured on data we generated" to "measured on data we did not", and it is the one
   criticism I cannot currently answer.
2. **Fix R3's chain-origin definition.** The out-of-time study showed it is anti-monotone in data
   volume — 46 in-degree-0 nodes over 29 days, 6 over 90 — so on production data it would find nearly
   nothing. "Chain origin" has to be evaluated inside a time window rather than over the whole graph.
   This is now the most concrete known defect in the detection code, and unlike the receive-only gap it
   has a fix that does not need data the dataset lacks. It is also **costed**: a windowed counterfactual
   over the same transactions takes R3 from 4 hits at 0.000 precision to 3 hits at 1.000, with no
   overlap between the two sets. The bill is re-baselining four JSON files under `evaluation/results/`
   and rewriting the numbers in three documents.
3. **Inbound and graph features.** The 51 unreachable receive-only positives are a feature-coverage
   problem, not a threshold problem, and no amount of tuning will move them.
4. **Revisit the HIGH threshold on real data**, where the tuning set and the evaluation set can
   differ and the choice is not circular.
5. **Make the re-planner earn its place.** It currently declines on 5/5. Either find the query class
   where mid-run revision genuinely helps, or report that the deterministic plan is good enough and the
   loop is insurance rather than a feature.
