# AML_LOGIC.md — Domain Logic, Rule Definitions, and Business Justification

**Owner:** Track B  
**Last updated:** 2026-08-03  
**Read by:** `backend/tools/rules.py`, `backend/tools/features.py`, Track A's narrator templates

This document is the single source of truth for every rule threshold, business justification, and
escalation mapping used by the AML detection engine. CONTRACTS.md §5 explicitly requires this file.
**Do not invent thresholds in code — if a threshold is not documented here, add it here first.**

---

## 1. Regulatory Context

### 1.1 Bank Secrecy Act (BSA) — Currency Transaction Reports
US banks must file a **Currency Transaction Report (CTR)** for any cash transaction ≥ $10,000.
**Structuring** (31 U.S.C. § 5324) is the deliberate act of splitting transactions into amounts
below $10,000 to avoid the CTR trigger. It is a federal crime independent of whether the underlying
funds are illicit.

> *"No person shall for the purpose of evading the reporting requirements of section 5313(a)...
> structure or assist in structuring...any transaction with one or more domestic financial
> institutions."* — 31 U.S.C. § 5324(a)(3)

The $9,000–$9,999.99 band is not the only concern — FinCEN guidance flags $9,000–$9,999 as the
highest-risk sub-band because it maximises per-transaction value while remaining below the threshold.

### 1.2 FATF 40 Recommendations
Recommendations 1, 3, and 10 require financial institutions to apply enhanced due diligence (EDD)
to transactions exhibiting layering, smurfing, or structuring patterns. FATF explicitly names:
- Rapid succession of transactions shortly below a reporting threshold (structuring)
- Use of multiple accounts to disperse funds (smurfing)
- Multi-jurisdictional wire chains with no apparent economic purpose (layering)
- Immediate cash withdrawal after a large electronic receipt (rapid cashout)

### 1.3 Suspicious Activity Reports (SARs)
US banks file a SAR (**FinCEN Form 111**) for any transaction or pattern suspected to involve money
laundering, regardless of amount. The AML agent generates a SAR draft only for `risk_level=high`
flags — consistent with the threshold in CONTRACTS.md Contract 5.

For reference, since the three are easy to confuse: **Form 111** is the SAR, **Form 112** is the CTR
described in §1.1, and Form 114 is the FBAR (foreign account report), which is unrelated to either.

---

## 2. Risk Band Justification (Contract 5)

| Score | Level | Escalation | Rationale |
|---|---|---|---|
| ≥ 70 | `high` | `report` | Multiple corroborating signals (rule hit + ML anomaly) → affirmative SAR filing obligation |
| 40–69 | `medium` | `review` | Single strong rule hit or borderline ML anomaly → manual analyst review required |
| 15–39 | `low` | `monitor` | Weak or single-signal hit → enhanced monitoring, no case opened |
| < 15 | `none` | `no_action` | Baseline behaviour — no action warranted |

**Rule weight normalisation:** `normalized_rule_weight` is the **maximum** weight across the rules
that fired on an entity, not their mean — see `backend/tools/risk.py`, which takes
`max(hit.weight)`. Multiple hits are all surfaced in the evidence, but they do not compound the
score; a customer triggering R1 and R3 is more suspicious, and that is expressed by the evidence
list rather than by inflating the number.

The heaviest rule is R1 at 0.85, so a rule-only score cannot exceed:

```
100 × 0.6 × 0.85 = 51   → medium
```

Two corroborating rules at 0.85 and 0.75 give `max = 0.85`, so also **51** — the second rule adds
evidence, not score. An ML-only entity cannot exceed `100 × 0.4 × 1.0 = 40`, also medium.

**Neither half of the system can produce a HIGH flag on its own**, and that is arithmetic rather than
a tuning choice: with R1's 0.85, reaching 70 requires an ML percentile of at least
`(0.70 − 0.51) / 0.4 = 0.475`. The SAR-drafting tier is therefore unreachable without corroboration
from both signals, which is the intended design — a single rule reaches an analyst's review queue,
never an automatic filing. This was later confirmed empirically: the component ablation
(`evaluation/ablation.py`) records 0 HIGH flags for rules-only and 0 for ML-only, against 27 for the
hybrid. `tests/test_ablation.py` pins it so that retuning a rule weight cannot silently remove the
property.

---

## 3. Rules R1–R7

### R1 — Structuring

**Pattern:** `structuring`  
**Weight:** `0.85`  
**Rule ID:** `R1`

**Definition:** A customer sends **≥ 3 transactions** with amounts in **[$9,000.00, $9,999.99]**
(inclusive both ends) within any **rolling 7-day window**. The $9,000 lower bound is set 10%
below the CTR threshold to catch near-threshold structuring; the $9,999.99 upper bound is the
highest amount strictly below $10,000 (no $10,000 transaction triggers a CTR).

**Thresholds:**
- Band: `$9,000.00 ≤ amount ≤ $9,999.99`
- Minimum transaction count in band: `3`
- Rolling window: `7 calendar days`
- Feature signal: `pct_just_below_threshold ≥ 0.30` (at least 30% of customer's transactions in band)

**Business justification:**  
Three transactions in the same week all landing in the same narrow band ($9k–$10k) has a
negligible probability of being random (Poisson arrival in a $9,000 band out of a plausible
$100 range is p ≈ 0.09 per transaction; three such events in 7 days p ≈ 7×10⁻⁴). A single
$9,500 transaction is not actionable; a pattern of three in a week is.

**Evidence dict shape:**
```python
{
    "txn_count_in_band": 4,
    "window_days": 7,
    "amounts": [9500.0, 9200.0, 9800.0, 9100.0],
    "band_low": 9000.0,
    "band_high": 9999.99,
    "total": 37600.0,
    "pct_just_below_threshold": 0.67
}
```

---

### R2 — Smurfing (Fan-out)

**Pattern:** `smurfing`  
**Weight:** `0.75`  
**Rule ID:** `R2`

**Definition:** A customer sends funds to **≥ 5 distinct counterparties** within a **48-hour
window**, where the **median outbound transaction amount is in [$7,000, $9,999.99]** (the wider
smurfing sub-threshold band — smurfs may also use lower amounts to stay further below radar).

**Thresholds:**
- Distinct receivers in 48h: `≥ 5`
- Median outbound amount: `$7,000 ≤ median ≤ $9,999.99`
- Feature signal: `velocity_counterparties_per_day ≥ 3.0`

**Business justification:**  
Smurfing (placement-stage layering via multiple couriers) is characterised by a hub account
distributing funds to many accounts in a short window. The 48h window matches the typical
operational tempo of a smurfing ring. The $7k–$10k median band captures the sub-threshold
distribution strategy. Five counterparties is a conservative threshold — the typical smurfing ring
uses 6–20 accounts (FATF typologies report 2014).

**round_amount_ratio added to smurfing:** Round amounts (divisible by $500) are a smurfing tell
because couriers are given a round amount to deposit. If `round_amount_ratio ≥ 0.5`, this
strengthens the signal and is included in evidence.

**Evidence dict shape:**
```python
{
    "distinct_receivers_48h": 8,
    "window_hours": 48,
    "median_outbound_amount": 9200.0,
    "amounts": [9500.0, 8800.0, 9200.0, 9100.0, 7500.0, 9000.0, 8500.0, 9300.0],
    "band_low": 7000.0,
    "band_high": 9999.99,
    "round_amount_ratio": 0.625
}
```

---

### R3 — Layering (Multi-hop Wire Chain)

**Pattern:** `layering`  
**Weight:** `0.80`  
**Rule ID:** `R3`

**Definition:** A **sequence of ≥ 3 hops** (i.e., ≥ 4 nodes: A→B→C→D) of `wire` or `transfer`
transactions in which each hop occurs **strictly after** the hop before it, **within 48 hours** of
it, and carries an amount **within ±30%** of it. **Each intermediate node must have
pass_through_ratio ≥ 0.70**, and at least **1 hop must be cross-border**
(`is_cross_border=True`). Funds must **enter the chain at the anchor**: the first sender received
nothing in the 48 hours before sending.

**Thresholds:**
- Minimum chain length: `3 hops` (4 nodes)
- Maximum chain length searched: `5 hops`
- Transaction types allowed in chain: `wire`, `transfer`
- `pass_through_ratio` per intermediate node: `≥ 0.70`
- Hop-to-hop window: `48 hours`, strictly forward
- Hop-to-hop magnitude tolerance: `±30%` of the preceding hop's amount
- Chain origin: no inbound wire/transfer to the anchor in the preceding `48 hours`
- Minimum cross-border hops: `1`
- Feature signal: `pass_through_ratio ≥ 0.70` and `cross_border_ratio > 0`

> **These constraints were documented here long before they were implemented.** Until the
> forward-walk rewrite, R3 searched a static graph with time collapsed out of it and enforced
> neither the window nor the tolerance, and it selected chain origins as nodes with in-degree 0
> over the whole graph. All 4 chains it reported on `aml_sample.csv` ran out of chronological
> order, spanned 32–71 days, drifted up to 5252% in amount, and none was a launderer, while all 5
> generated layering chains went unsearched. See `evaluation/out_of_time.py` §3–§4.

**pass_through_ratio definition:**  
For a customer C in a 48h sliding window:
```
pass_through_ratio = min(total_received_48h, total_sent_48h) / max(total_received_48h, total_sent_48h)
```
Ranges [0, 1]; 1.0 = perfect pass-through (received exactly as much as sent). Computed as
the **maximum** of this ratio over all 48h windows in the customer's history.

**Business justification:**  
Layering is the obfuscation stage of money laundering — moving funds through multiple accounts
and jurisdictions to sever the audit trail. Three hops is the minimum to create meaningful
complexity. The cross-border requirement filters domestic transfers (often legitimate payroll
chains) from international laundering circuits. The pass-through requirement ensures intermediate
nodes are not accumulating wealth — they're purely transiting funds, which is economically
anomalous without a business purpose.

**Evidence dict shape:**
```python
{
    "chain": ["C-HUB01", "C-LAY01", "C-LAY02", "C-LAY03"],
    "chain_length": 3,
    "cross_border_hops": 2,
    "pass_through_ratios": [0.92, 0.88, 0.95],
    "hop_amounts": [150000.0, 142000.0, 138000.0],
    "hop_types": ["wire", "wire", "wire"],
    "total_elapsed_hours": 36.0
}
```

**Implementation search-safety bounds** *(not part of the rule definition — purely computational)*:
- Maximum search depth (cutoff): **5 hops** (6 nodes). The rule minimum is 3 hops; 5 gives one
  extra level of headroom without the O(E^8) worst-case of the original undocumented cutoff=8.
- Hard path cap per (source, sink) pair: **50 paths** (via `itertools.islice`). Prevents runaway
  enumeration on densely-connected subgraphs.
- Per-pair wall-clock budget: **200 ms**. If a single (src, snk) pair exceeds this budget, the
  pair is aborted and a warning note is emitted in `ToolResult.notes`.
- Graph-size guard: if the wire/transfer subgraph contains **> 500 unique nodes**, the path search
  is skipped entirely and a warning is emitted. Prevents hangs on large real-world datasets.

---

### R4 — Rapid Cash-Out

**Pattern:** `rapid_cashout`  
**Weight:** `0.75`  
**Rule ID:** `R4`

**Definition:** A customer receives an inbound transaction of **≥ $10,000** (any type) and then
makes **≥ 3 cash/ATM outflow transactions** (`txn_type ∈ {cash}` or `channel ∈ {atm, branch}`)
with **total outflow ≥ 50% of the inbound amount** within **24 hours** of the inbound.

**Thresholds:**
- Minimum inbound amount: `$10,000`
- Minimum cash outflow count: `3`
- Minimum total outflow as % of inbound: `50%`
- Window: `24 hours` after inbound transaction
- Feature signal: `rapid_cashout_ratio ≥ 0.50`

**Business justification:**  
Rapid cash-out (also called "cashing out" in FATF terminology) is the integration stage — the
launderer converts electronic funds to untraceable physical currency. The 24h window reflects
operational urgency; the 50% threshold ensures the behaviour is systematic and not incidental.
The $10,000 minimum inbound anchors to the CTR threshold — below this, cash-out is less
suspicious absent other indicators.

**Evidence dict shape:**
```python
{
    "inbound_amount": 45000.0,
    "inbound_txn_id": "T-000142",
    "inbound_timestamp": "2025-01-15T14:30:00",
    "cash_outflow_count": 4,
    "cash_outflow_total": 38000.0,
    "cashout_ratio": 0.844,
    "window_hours": 24,
    "outflow_amounts": [9500.0, 9500.0, 9500.0, 9500.0],
    "elapsed_to_first_cashout_hours": 1.5
}
```

---

### R5 — High Velocity

**Pattern:** `velocity`  
**Weight:** `0.65`  
**Rule ID:** `R5`

**Definition:** A customer's **outbound transaction rate exceeds 2.0 transactions/hour** in any
**24-hour window**, AND their **amount z-score against their own 90-day baseline is ≥ 3.0**
(i.e., the amounts are anomalous relative to their own history, not just the population).

**Thresholds:**
- `velocity_txns_per_hour ≥ 2.0` in a 24h window
- `amount_zscore_90d ≥ 3.0`
- Feature signal: both above thresholds

> **`velocity_txns_per_hour` is a peak 1-hour rate.** It was previously computed as
> (max count in any 24h window) ÷ 24 — a daily average — which made the 2.0 bar above silently mean
> "48 transactions inside one 24h window". No customer in `aml_sample.csv` can reach that (the busiest
> sender has 25 transactions in total), so R5 could not fire at any threshold. The evidence field name
> `max_txns_per_hour` and the "rate" wording in the definition both describe the peak, which is what the
> feature now computes.
>
> **R5 still fires on 0 customers in the committed dataset, and this is measured, not assumed.** The
> corrected rate admits 15 senders at the velocity gate — 12 of them labelled positives, against a 19%
> base rate, so the velocity signal itself discriminates well. All 15 then fail the z-score gate: the
> highest self-deviation z-score among them is 2.29 against the 3.0 threshold, and 4 have too little
> pre-burst history to score at all. The z-score gate is kept: dropping it would fire on all 15, adding
> 2 true positives and 3 false positives. See the Results section of [README.md](../README.md).

**Business justification:**  
High velocity alone (many small legitimate transactions, e.g. a payroll system) is not suspicious.
The combination with a self-deviation z-score of ≥ 3σ means the activity is both unusually fast
AND unusually large relative to that customer's own baseline — a pattern consistent with
account takeover or a sudden activation of a previously dormant/low-activity account for
laundering purposes.

**Evidence dict shape:**
```python
{
    "max_txns_per_hour": 3.5,
    "window_hours": 24,
    "amount_zscore": 4.2,
    "zscore_baseline_days": 90,
    "zscore_n_samples": 45,
    "mean_historical_amount": 1200.0,
    "std_historical_amount": 800.0,
    "triggering_amount": 4560.0
}
```

---

### R6 — Dormant Account Reactivation

**Pattern:** `dormant_reactivation`  
**Weight:** `0.60`  
**Rule ID:** `R6`

**Definition:** A customer has **≥ 60 days of inactivity** (no sent transactions) followed by
**≥ 3 sent transactions within 7 days** of reactivation, with an amount z-score ≥ 2.0 (amounts
significantly different from the customer's pre-dormancy baseline).

**Thresholds:**
- Dormancy gap: `≥ 60 days` with no outbound transactions
- Burst after reactivation: `≥ 3 outbound transactions within 7 days`
- Amount z-score: `≥ 2.0` (compared to pre-dormancy baseline)
- Feature signal: dormancy gap + burst count

> **R6 is inapplicable to the committed dataset and should not be retuned to fire.**
> `aml_sample.csv` spans 89 days, while R6 needs ~70 days of structure (60-day gap, then a 7-day burst,
> with ≥ 3 pre-gap transactions to compute the z-score against). Measured funnel: of 268 senders with
> ≥ 2 transactions, **2** have a gap ≥ 60 days (the largest gap in the whole dataset is 64.5 days),
> **1** clears the burst gate, and **0** clear the z-score gate. Median largest-gap per sender is 26.8
> days. Lowering the dormancy threshold destroys the rule rather than rescuing it — a 30-day gap admits
> 108 of 268 senders (40%), which is ordinary cadence, not dormancy. The thresholds above are correct
> for real data; this dataset simply contains no dormancy typology.

**Business justification:**  
Dormant account reactivation is a classic money laundering technique — accounts opened with
KYC years ago are reactivated when needed, bypassing newer customer due diligence requirements.
The 60-day gap is standard in BSA compliance systems. The burst + z-score combination separates
returning legitimate customers (who typically resume at their historical pace and amounts) from
accounts reactivated for laundering (unusual amounts, sudden high frequency).

**Evidence dict shape:**
```python
{
    "dormancy_gap_days": 95,
    "last_txn_before_gap": "2024-09-20",
    "first_txn_after_gap": "2024-12-24",
    "burst_txn_count": 5,
    "burst_window_days": 7,
    "amount_zscore_vs_pre_dormancy": 3.1,
    "pre_dormancy_mean_amount": 500.0,
    "burst_amounts": [9500.0, 9200.0, 9800.0, 9100.0, 9400.0]
}
```

---

### R7 — Structuring, Receiver Side

**Weight:** `0.75`

**Condition:** an account **receives** ≥ 2 transactions in the $9,000–$9,999.99 band from a
**single counterparty** within a 7-day window.

**Business justification:**
R1 detects the person making structured deposits. R7 detects the account they are being made
*into*. Under 31 U.S.C. § 5324 the offence attaches to the structuring itself, and the beneficiary
account is where the aggregated proceeds land — it is exactly what a SAR narrative would describe.
Without it every rule here is sender-side, which leaves the receiving half of the same scheme
invisible.

**Why the threshold is 2, not R1's 3:**
The signal is measured per *(receiver, sender) pair*, which is far narrower than R1's per-sender
aggregate. On the committed dataset no true negative ever exceeds one such pair-window
transaction, so 2 already separates cleanly — measured, not assumed.

**Why this is not fan-in detection:**
The intuitive receiver-side rule is a funnel account: many distinct senders converging on one
account. That was tested against this dataset and rejected. Customers who appear only as receivers
of labelled transactions average **7.6 distinct inbound counterparties** against a population
average of **6.9**, and within any 48-hour window both top out at 4 — there is no separation to
threshold on, and the highest fan-in accounts in the data are negatives. The discriminating signal
is the repeated pair relationship, not the breadth of counterparties.

**Coverage, honestly:**
R7 recovers 12 of the 63 receive-only positives with no measured false positives. The other 51 are
not reachable by any inbound rule: 29 of them receive exactly one labelled transaction, which is
indistinguishable from being an innocent counterparty of a bad actor. Closing that remainder would
need a signal this dataset does not carry — shared account ownership, KYC linkage, or device
overlap.

That "exactly one inbound" group is also why `evaluation/harness.py` reports a third ground-truth
definition, `sender_or_repeat_receiver`: 30 of the 63 receive-only positives have a single labelled
inbound transaction (29 of them among the 51 R7 misses), so counting them as positives at all is
arguably over-labelling. See the Results section of [README.md](../README.md).

**Weight rationale:**
0.75, below R1's 0.85. Receiving structured deposits is a strong signal, but attribution is weaker
than for the sender: the account holder may be a willing mule or an unwitting recipient. At 0.75 a
rule-only hit scores 45 — MEDIUM, "review" — so it reaches an analyst without auto-drafting a SAR,
which is the appropriate confidence level for a passive-side signal.

**Evidence dict shape:**
```python
{
    "inbound_band_txns_from_one_sender": 9,
    "counterparty": "C-STR05",
    "window_days": 7,
    "amounts": [9930.35, 9927.0, 9923.28, 9754.48, 9636.59],
    "band_low": 9000.0,
    "band_high": 9999.99,
    "total": 85418.7,
    "pair_band_txns_overall": 9
}
```

---

## 4. Feature → Rule Cross-Reference

| Feature | R1 | R2 | R3 | R4 | R5 | R6 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `rolling_1d_sum` / `rolling_1d_count` | ✓ | ✓ | | ✓ | | |
| `rolling_7d_sum` / `rolling_7d_count` | ✓ | ✓ | ✓ | | | ✓ |
| `rolling_30d_sum` / `rolling_30d_count` | ✓ | | ✓ | | | |
| `pct_just_below_threshold` | ✓ | ✓ | | | | |
| `amount_zscore_90d` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `velocity_txns_per_hour` | | ✓ | | ✓ | ✓ | |
| `velocity_counterparties_per_day` | | ✓ | ✓ | | | |
| `rapid_cashout_ratio` | | | | ✓ | | |
| `round_amount_ratio` | ✓ | ✓ | | | | |
| `night_hours_ratio` | | | ✓ | ✓ | | |
| `new_counterparty_ratio` | | ✓ | ✓ | | | |
| `cross_border_count` / `cross_border_ratio` | | | ✓ | | | |
| `pass_through_ratio` | | | ✓ | | | |

---

## 5. Definitions and Limitations

### 5.1 "Night hours" (UTC-based)
Night is defined as **UTC hours 22:00–05:59** (inclusive). Transactions timestamped in this
window are counted toward `night_hours_ratio`. **Limitation:** all timestamps in the canonical
schema are tz-naive UTC (per Contract 0). For customers in non-UTC time zones, this definition
is inaccurate. This is a known approximation — local time data is unavailable. The feature is
still meaningful at population level (laundering activity globally skews toward UTC night hours
relative to US/EU banking hours).

### 5.2 "Round amount" definition
An amount is considered **round** if it is divisible by **$500** with remainder < $1.00.
This covers $500, $1,000, $5,000, $9,000, $9,500, etc. The $500 granularity reflects the
smallest denomination at which structuring via ATM deposits is commonly observed.

### 5.3 Amount z-score fallback (thin history)
If a customer has **fewer than 3 transactions** in their 90-day baseline window, the
`amount_zscore_90d` is set to **0.0** (not suspicious by this feature). This prevents false
positives from new accounts with 1–2 transactions. The fallback is noted in feature output
via `zscore_n_samples` in the evidence dict.

### 5.4 Rolling windows — sender-only vs. combined
**Rolling 1d/7d/30d sum and count** are computed **sender-side only** (outbound transactions).
Rationale: structuring, smurfing, and rapid cashout are sender behaviours. Inbound amounts
are tracked separately only for `rapid_cashout_ratio` (where the trigger is an inbound event).
Computing sender-only reduces false positives from high-volume recipients (e.g. merchant accounts).

### 5.5 ML percentiles are ranked against a fixed population
The ML half of Contract 5's score is a **percentile**, which is meaningless without saying
percentile *of what*. The population is the **full customer set, unfiltered** — never the
subset the analyst's query happened to select.

This is a correction, not an original design choice. Ranking inside the query's own cohort
made a customer's score depend on the query rather than on their behaviour: adding
`amount_min=5000` to a structuring search moved percentiles by up to **0.73** and pushed
**four customers across a risk band**. A number that decides whether a SAR is drafted cannot
change because an analyst narrowed their search.

The accepted cost: the ML term is now blind to the query window. A customer who is
unremarkable across the whole dataset but anomalous within a 30-day slice no longer registers
on the ML half. Rules R1–R7 still evaluate the filtered frame and still fire on them. For a
score attached to an escalation decision, stability was judged the more important property.

One subtlety worth recording, because it reintroduced the bug once already: ML output is
scoped to every customer appearing in the working frame as **sender or receiver**. Features
are indexed on senders only (§5.4), so scoping ML output to the feature index dropped
receiver-side R7 hits — C-N0138 sends nothing above $5,000, so under an `amount_min=5000`
filter it lost its ML score to the 0.0 default and fell from 52.58 to 45.00.

### 5.6 pass_through_ratio computation
Window: sliding 48-hour windows, step = 1 hour.  
Formula: `min(received_48h, sent_48h) / max(received_48h, sent_48h)`  
If both are zero: ratio = 0.0. The **maximum** ratio across all 48h windows is the feature value.  
Magnitude tolerance for R3's chain detection: the outbound amount must be within ±30% of the
inbound amount for the hop to count as pass-through. That tolerance is applied by R3 itself, per
hop, alongside the 48-hour forward-ordering constraint — see §3 R3. It was documented here for a
long time before the rule enforced it; the feature above is a per-customer summary and was never
a substitute for checking the chain.

---

## 6. False Positive Control Claims

The WORKPLAN §8 requires a comparison table: "naive rule vs. ours." The baseline for comparison
is:

> **Naive rule:** flag any transaction with `amount > $9,000` → extremely high FP rate because
> all legitimate large transactions are flagged.

Our R1 requires **3 such transactions in a 7-day window from the same sender**, which filters
out customers who occasionally make a large transfer (e.g., monthly rent, salary). This is the
core false-positive reduction mechanism for the structuring detector.

The combination of **rule hits + ML anomaly score** (Contract 5 formula) ensures that no single
weak rule alone reaches `high` escalation — requiring corroboration before a SAR is drafted.

**Measured, not asserted.** The comparison table is generated from the labelled dataset by
`python -m evaluation.run_evaluation`, which writes `evaluation/results/`. Against the naive rule
the system flags **6.6× fewer customers at a 13× lower false-positive rate**, reaching **0.897
precision** under the broader ground truth where the naive rule manages 0.421. Do not hand-edit
those figures anywhere in the docs — regenerate them.
