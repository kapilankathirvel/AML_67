# After the deadline

**What changed between the hackathon submission and this repository, and why.**

This project was built for a 48-hour hackathon. Work continued after the deadline, and most of what
now makes the system interesting was not in the submission. This document draws that line honestly,
because the difference is the point: the submission worked, and it was also wrong in two ways nobody
had noticed.

Every claim here is checkable with `git log`.

---

## The two phases

| | Hackathon | After the deadline |
|---|---|---|
| **Window** | 25 Jul 11:03 → 26 Jul 19:47 (~57h) | 30 Jul → 2 Aug |
| **Commits** | 36 | 14 |
| **Contributors** | 2 — 23 mine, 13 Kalyanv007's | 1 — all 14 mine |
| **Test functions** | 132 | 221 |
| **Backend Python** | 5,779 lines | 7,265 lines |

The hackathon build was a genuine two-person effort on a split ownership model documented in
[WORKPLAN.md](WORKPLAN.md): I owned **Track A** — the frozen schemas and contracts, intent parser,
planner, executor, narrator, and the FastAPI layer. Kalyanv007 owned **Track B** — the data loader,
synthetic generator, the seven detection rules, feature engineering, the ML detector, and the
Streamlit frontend. Neither half would have shipped without the other.

The post-deadline work is solo and crosses both tracks.

---

## What was wrong with the submission

This section comes first deliberately. Both defects were live in the code that was judged, both were
found afterwards by re-reading my own work, and neither produced an error message.

### The risk score depended on how the analyst phrased their query

`399cd12` — *Make risk scores independent of the query*

The ML half of the risk score is a percentile, and percentiles were being ranked **inside the
filtered cohort the query had selected**. So a customer's score was partly a function of the
analyst's search box.

Adding `amount_min=5000` to a structuring search moved percentiles by up to **0.73** and pushed
**four customers across a risk band**. A number that decides whether a SAR is drafted cannot change
because somebody narrowed their search.

Percentiles are now ranked against a fixed unfiltered reference population
([AML_LOGIC.md §5.5](AML_LOGIC.md)). The accepted cost is stated there too: the ML term is now blind
to the query window, so a customer anomalous only within a 30-day slice no longer registers on that
half. Rules R1–R7 still evaluate the filtered frame. For a score attached to an escalation decision,
stability was judged the more important property.

The same commit fixed a related hole: single-entity queries skipped `ml_detect` entirely, so
*"is customer 4521 suspicious?"* was scored on a different formula than the same customer appearing
in a full sweep.

### R5 could never have fired, at any threshold

`08d9f1a` — *Fix velocity_txns_per_hour unit*

`velocity_txns_per_hour` was computed as *(max count in any 24-hour window) ÷ 24* — a **daily average
wearing an hourly name**. AML_LOGIC.md documented a bar of 2.0 transactions per hour, which therefore
silently meant *48 transactions inside one 24-hour window*.

The busiest sender in the dataset has 25 transactions **in total**. The observed maximum was 0.542.
The rule was unreachable by construction, and no amount of threshold tuning could have revealed it —
the bug was in the feature, not the rule.

The feature now computes a true peak 1-hour rate. R5 still fires on nobody, but that is now a
**measured fact rather than an artifact**: 15 senders clear the corrected rate gate, all 15 fail the
z-score gate, and the highest self-deviation among them is 2.29 against a threshold of 3.0.

Fixing it changed the published numbers for the worse — sender-side precision fell 0.590 → 0.561 as
two ML-only negatives crossed the percentile floor. That regression is reported rather than tuned
away. The alternative was shipping a feature that contradicts its own name in order to protect a
metric.

---

## What was added

| Commit | Change |
|---|---|
| `884a2f0` | An **LLM planner** that genuinely selects tools, gated by a validator. Until this, the "agent" chose tools from a deterministic `if`/`elif` table. |
| `d51f581` | **Measured** that planner against a real model, and fixed what the measurement found. |
| `45e81b1` | A **third ground-truth definition** — sender, or received more than once — because the existing two disagreed by more than 2× on recall. |
| `47e1323` | Made LLM failures visible. `complete_json` had been swallowing every exception silently. |
| `28c6275` | **V13** — validate closed-set parameter *values*, not just names. Found via a plan that was legal, accepted, and returned zero flags. |
| `0d02260` | **V14** plus the **observe→decide→act loop**: the model can now revise the remaining plan after seeing what the executed steps produced. |
| `a1ba821` | Made that loop able to see **failures**, not only successes — it had been skipping the case it exists for. |
| `4ef3f2a` | The **ablation study**: components, per-rule contribution, fusion weights, risk bands. |

The validator is the piece I would point at first. It is fifteen rules (V0–V14) that a model-proposed
plan must satisfy before anything executes, and the interesting ones are not the shape checks. V13
exists because a plan can be structurally perfect and still answer the wrong question. V14 exists
because `load_data` accepts `source` and `force_rebuild`, so a legal plan could have switched the
product onto a different dataset and rewritten a cache on a model's say-so. *"A plan may choose how to
analyse, not which dataset the product runs on"* is a compliance instinct, and it belongs in code.

---

## What was measured and disproved

The most useful output of the post-deadline work is the list of things that turned out not to be true.

- **Fan-in detection would not have worked.** Before building a funnel-account rule, I measured the
  premise: receive-only positives average **7.6 distinct inbound counterparties against a population
  average of 6.9**, and within any 48-hour window both top out at 4. There is no separation to
  threshold on. The rule was never built.
- **Noisy-OR fusion would have changed nothing.** Combining rule weights probabilistically only helps
  when rules corroborate each other, so I counted first: 37 rule hits land on 36 distinct entities,
  and exactly **one** — C-HUB01, on R1 and R2 — triggers more than one rule. There is nothing to
  combine, and zero customers would move across a band. Also never built.
- **The re-planning loop declines to intervene.** Across five queries through the full pipeline it
  produced a decision on 5/5 and **declined to revise on every one**; outcomes differed on 0/5. That
  is correct behaviour — the model plans well enough up front that by the time it sees the
  observation there is nothing to fix — and manufacturing a scenario where it fires would have been
  easy and dishonest.
- **The fusion weights cannot affect precision or recall.** Not weakly — *exactly*. Sweeping
  `RULE_WEIGHT_COEFF` from 0.0 to 1.0 leaves precision at 0.561 and recall at 0.451 throughout,
  because membership in the flagged set is decided by the entity universe and never consults the
  coefficients. The documented 0.6/0.4 split is a **banding** decision alone.
- **The hybrid is less precise than either half.** 0.561, against 0.583 for rules-only and 0.692 for
  ML-only. It takes the union of their flags and inherits both sets of false positives. It wins on
  recall and F1 — that is the actual trade, and it is more defensible than claiming the hybrid is
  simply better. See the next section for the axis on which that trade is actually won.
- **A small local model is not the bottleneck the architecture is.** The same planner scores **27% on
  a local 3B and 93% on a hosted model**, same validator, same prompts, same queries. The 3B result
  was kept deliberately: pushing it from 20% to 27% is what produced V12, V13 and the `load_data`
  repair. A weak model is a better validator test than a strong one.

---

## What was measured and confirmed

One thing survived the attempt to disprove it, and it is the finding I would lead with.

The ablation left the hybrid design looking indefensible: it is *less* precise than rules alone, so
on a static dataset it buys recall with false positives and nothing else. That verdict is correct
and incomplete, because a static dataset models an adversary who never adapts — and in AML that
adversary does not exist. Structuring is itself an adaptation, to the $10,000 CTR threshold. The
$9,000 band R1 keys on is the *previous* move in a game that is still being played.

So the evasion study perturbs the launderers' own transactions in memory and asks what each rule
costs to defeat. **Timing evasion effectively destroys the rules half** — spacing transactions
further apart takes rule recall from 0.412 to 0.020, retaining 4.8% — **and the ML half barely
registers it**, retaining 88.9%. Against all the moves used together the rules keep 9.5% and the
hybrid keeps 39.1%. The hybrid retains more than the rules alone under every move tested.

That is what the 0.022 of precision buys, and it is now a measured number rather than an assertion.
Three caveats are published with it rather than left for someone else to find: the retention ratios
are computed against each half's own baseline and are not comparable to each other (the ML half
starts at 0.176 recall, the rules at 0.412); everything degrades in absolute terms, with hybrid
recall under the combined move at 9 of 51 customers; and stepping under the $9,000 band costs a mean
of $497 per transaction rather than the $1 the threshold's placement implies, because the in-band
amounts average around $9,495.

---

## What is still wrong

Stated here so it does not have to be discovered:

- **The data is synthetic and self-generated.** This is the fairest criticism of the project. The
  ingestion path for the real IBM AML dataset is fully built — `data/build_ibm_cache.py` plus a
  stratified sampler and parquet cache in `data_loader.py` — and has **never been run**, because it
  needs a Kaggle account. Every published metric therefore comes from data the project generated for
  itself.
- **51 of 63 receive-only positives are structurally unreachable.** Not because they are excluded —
  all 270 customers enter the feature frame and all 270 are scored — but because 16 of the 18 feature
  columns are computed per *sender*, so what gets measured about these customers is their ordinary
  outbound behaviour. Their median ML percentile is **0.486**, and only 2 of the 63 clear the 0.95
  floor. R7 recovers 12. Closing the rest needs features that describe inbound or graph structure,
  which is a different thing from tuning a threshold.
- **R3 (layering) is broken in a way that gets worse with more data.** 4 hits, zero true positives,
  and removing it *improves* precision at no recall cost — but the out-of-time study found the
  mechanism, and it is not that layering is hard to detect. R3 enumerates chains only from nodes with
  in-degree 0, and that set collapses as history accumulates: **46 such nodes over 29 days, 28 over
  60, 6 over the full 90.** The searched pair count falls from 2,300 to 42. The rule is anti-monotone
  in data volume, so on a production graph where nobody has zero inbound wires it would find nothing
  at all. And a counterfactual — the shipped rule run unchanged over non-overlapping partitions of
  the same transactions — shows the collapse is picking the *wrong* chains rather than merely fewer:
  the whole-frame run's 4 hits contain **zero** launderers, the 7-day partition's 3 hits are **all**
  launderers, and the two sets do not overlap. Precision 0.000 → 1.000 on identical data. Reported
  and pinned by tests, deliberately not yet fixed — changing detection code invalidates every
  baseline under `evaluation/results/`, which should be a decision rather than a side effect of
  adding a study. It is now a costed decision rather than a speculative one.
- **The ML half is transductive, and the cost of that is now measured rather than unknown.** Fitting
  on the first 60 days and scoring the last 29 costs **0.036 precision and no recall**, so the
  shortcut is buying very little. Two things bound that number: this dataset has no customers absent
  from the training window, so cold-start cost goes unmeasured; and the shipped `ml_detect` cannot do
  out-of-time scoring at all, because `LocalOutlierFactor` is constructed with `novelty=False` and has
  no `score_samples`. The study substitutes `novelty=True` and publishes an IsolationForest-only
  control so the substitution can be checked rather than trusted.
- **The risk formula, bands, and thresholds are hand-set.** They are defensible and documented, but
  they are not calibrated, and calibrating them against the same 270 customers they are evaluated on
  would be circular.
