# DATA_CARD.md — AML Agent Dataset Documentation

**Owner:** Track B  
**Last updated:** 2026-07-25  
**Governed by:** WORKPLAN.md §4, docs/CONTRACTS.md Contract 0

---

## 1. Datasets Used

### 1.1 IBM Transactions for Anti-Money Laundering (Primary)

| Field | Value |
|---|---|
| **Source** | Kaggle: [ealtman2019/ibm-transactions-for-anti-money-laundering-aml](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) |
| **Subset used** | HI-Small (High Illicit ratio, small size) |
| **License** | [Kaggle dataset license — research/non-commercial](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) |
| **Citation** | E. Altman, B. Baeck, M. Gerlach. "Realistic Synthetic Financial Transactions for Anti-Money Laundering Models." NeurIPS 2023 Datasets and Benchmarks. |
| **Files** | `HI-Small_Trans.csv` (transactions), `HI-Small_accounts.csv` (accounts) |
| **Download** | Via `kagglehub.dataset_download("ealtman2019/ibm-transactions-for-anti-money-laundering-aml")` — cached at `~/.cache/kagglehub/` (outside repo) |

#### Raw dataset statistics (HI-Small)

| Metric | Value |
|---|---|
| Total transactions | 5,078,345 |
| Total accounts (customers) | 518,581 |
| Laundering transactions | 5,177 (0.102%) |
| Normal transactions | 5,073,168 (99.898%) |
| Date range | 2022-09-01 to 2022-09-30 (1 month) |
| Raw columns | 11 (see mapping below) |

### 1.2 Synthetic Dataset (aml_sample.csv) — Committed

| Field | Value |
|---|---|
| **File** | `data/sample/aml_sample.csv` (transactions), `data/sample/aml_sample_customers.csv` (customers) |
| **Generator** | `data/generate_synthetic.py` |
| **Seed** | 42 (fixed — do not regenerate unless you own Track B) |
| **Purpose** | Demo data that requires no Kaggle download; committed once |

#### Synthetic dataset statistics

| Metric | Value |
|---|---|
| Total transactions | 2,002 |
| Total customers | 270 |
| Labelled laundering | 202 (10.1%) |
| Normal (unlabelled) | 1,800 |
| Date range | 2025-01-01 to 2025-04-01 (90 days) |
| Patterns | structuring (91), smurfing (51), rapid_cashout (40), layering (20) |

---

## 2. Canonical Schema (Contract 0)

Both datasets are adapted to the canonical schema defined in `docs/CONTRACTS.md` Contract 0.  
All detection code downstream touches only this schema — datasets are fully swappable.

### 2.1 Transactions

| Column | Type | Source strategy |
|---|---|---|
| `txn_id` | str | Sequential `T-000001` (IBM has no native ID) |
| `timestamp` | datetime64[ns] | Parsed tz-naive UTC |
| `sender_id` | str | `C-` + raw account number |
| `receiver_id` | str | `C-` + raw account number |
| `amount` | float | Receiving-side amount (see §3.1) |
| `currency` | str (ISO 4217) | Mapped from full name (see §3.2) |
| `txn_type` | str (enum) | Derived from Payment Format (see §3.3) |
| `channel` | str (enum) | Derived from Payment Format (see §3.3) |
| `sender_country` | str | `"UNK"` — IBM has no per-transaction country (see §3.4) |
| `receiver_country` | str | `"UNK"` — same |
| `is_cross_border` | bool | `False` — UNK vs UNK treated as same/unknown (see §3.4) |
| `label_is_laundering` | bool | IBM's `Is Laundering` 0/1 cast to bool; null for synthetic normal rows |
| `pattern_label` | str\|null | Null for IBM; synthetic-only field populated by generator |

### 2.2 Customers

| Column | Type | Source strategy |
|---|---|---|
| `customer_id` | str | `C-` + Account Number |
| `name` | str | IBM: Entity Name directly; synthetic: seeded generation |
| `account_open_date` | date | Synthesised (see §3.5) |
| `customer_type` | str | Derived from Entity Name prefix (see §3.6) |
| `country` | str | Extracted from Bank Name (see §3.7) |
| `occupation` | str | Derived from entity type (see §3.8) |
| `risk_rating` | str | Synthesised: 80% low / 15% medium / 5% high (see §3.9) |
| `kyc_status` | str | Synthesised: 90% verified / 7% pending / 3% incomplete (see §3.10) |
| `is_pep` | bool | Synthesised: ~1.5% True (see §3.11) |
| `expected_monthly_volume` | float | Median of actual monthly sent volume (see §3.12) |

---

## 3. Field-by-Field Preprocessing Decisions

### 3.1 Amount and Currency Selection

IBM provides two sides: `Amount Paid` + `Payment Currency` (sender) and `Amount Received` + `Receiving Currency` (beneficiary). We use the **receiver side** for both amount and currency.

**Rationale:** The received amount is what arrives in the beneficiary account — the quantity directly relevant to structuring detection (e.g., the $9,000–$9,999 band just below the $10,000 Bank Secrecy Act CTR threshold). For cross-currency wire transfers (~0.1% of rows), the paid and received amounts differ due to FX conversion; using the received side captures what was actually deposited. This is consistent with how compliance teams analyse inbound flows.

### 3.2 Currency Name → ISO 4217 Mapping

IBM stores currency as full English names. All 15 names present in HI-Small are mapped:

| IBM name | ISO code | Count |
|---|---|---|
| US Dollar | USD | 1,879,341 |
| Euro | EUR | 1,172,017 |
| Swiss Franc | CHF | 237,884 |
| Yuan | CNY | 206,551 |
| Shekel | ILS | 194,988 |
| Rupee | INR | 192,065 |
| UK Pound | GBP | 181,255 |
| Ruble | RUB | 157,361 |
| Yen | JPY | 156,319 |
| Bitcoin | BTC | 148,151 |
| Canadian Dollar | CAD | 141,357 |
| Australian Dollar | AUD | 138,511 |
| Mexican Peso | MXN | 111,030 |
| Saudi Riyal | SAR | 89,971 |
| Brazil Real | BRL | 71,544 |

Zero unmapped currencies. Any future unknown currency maps to `"UNK"`.

### 3.3 Payment Format → txn_type + channel

Contract 0 separates IBM's single `Payment Format` into two enums. Mapping:

| IBM `Payment Format` | `txn_type` | `channel` | Rationale |
|---|---|---|---|
| Reinvestment | `transfer` | `online` | Internal capital movement; online platform |
| Cheque | `deposit` | `branch` | Physical instrument, branch-processed |
| Credit Card | `transfer` | `online` | Card-rail electronic transfer |
| ACH | `transfer` | `online` | Automated Clearing House; electronic |
| Cash | `cash` | `branch` | Physical currency; requires branch/ATM |
| Wire | `wire` | `wire` | SWIFT/FEDWIRE; dedicated wire channel |
| Bitcoin | `transfer` | `online` | Crypto transfer; mapped to online channel |

All 7 IBM Payment Format values are covered. Any unknown value falls back to `transfer` / `online`.

### 3.4 Country and Cross-Border Flag

IBM's transaction records contain `From Bank` and `To Bank` as numeric Bank IDs — not country names. The accounts file (`HI-Small_accounts.csv`) provides bank names (e.g., "Canada Bank #27") but these link via Bank ID, and join would be complex without guaranteed uniqueness. Per task specification:

- `sender_country` = `"UNK"` for all IBM transactions
- `receiver_country` = `"UNK"` for all IBM transactions
- `is_cross_border` = `False` — **UNK vs UNK is treated as same/unknown, not cross-border**

**Implication:** The `is_cross_border` feature and cross-border filters will produce no signal on IBM data. They function correctly on the synthetic dataset (which has explicit country codes) and on any future adapter that resolves country.

Country at the customer level (`customers.country`) is extracted from the `Bank Name` field using keyword matching (e.g., "Canada Bank #27" → `CA`, "National Bank of Harrisburg" → `US`). Coverage is ~97% — unmatched banks → `"UNK"`.

### 3.5 account_open_date (Synthesised)

IBM has no account open date. Generated deterministically using `MD5(customer_id + "open_date") % range_days`, uniformly distributed in **[2015-01-01, 2021-12-31]**. This ensures every customer pre-dates the transaction dataset (2022-09) and provides realistic account age variance for dormancy features.

### 3.6 customer_type (Derived)

IBM's `Entity Name` field encodes entity type in its prefix:

| Entity Name prefix | `customer_type` |
|---|---|
| `Individual #...` | `individual` |
| `Corporation #...` | `business` |
| `Partnership #...` | `business` |
| `Sole Proprietorship #...` | `business` |
| `Country #...` (sovereign) | `business` |
| `Direct ...` | `business` |
| Any other | 85% `individual` / 15% `business` (hash-seeded) |

Distribution in HI-Small: **business 99.86%** (517,841), **individual 0.14%** (740). The IBM dataset is heavily skewed toward business entities — consistent with the paper's description of the simulated banking network.

### 3.7 customer country (Derived from Bank Name)

Country extracted using keyword matching on bank name (e.g., `"Canada"`, `"UK"`, `"Japan"` → respective ISO codes). US city-bank names ("National Bank of Harrisburg", "Savings Bank of Omaha") mapped to `"US"`. Coverage tested on the full accounts file; unmatched → `"UNK"`.

### 3.8 occupation (Derived)

Mapped from `Entity Name` type:
- `Corporation` → `"corporate banking"`
- `Partnership` → `"business partnership"`
- `Sole Proprietorship` → `"sole proprietor"`
- `Country` → `"government/sovereign"`
- `Individual` → `"individual"`
- `Direct` → `"direct payment entity"`

### 3.9 risk_rating (Synthesised)

IBM has no KYC risk rating. Distribution chosen:  
**80% `low` / 15% `medium` / 5% `high`**  
Generated via `MD5(customer_id + "risk") % 100` — deterministic, reproducible.

Rationale: reflects a realistic bank portfolio where the vast majority of customers are low-risk; medium-risk triggers enhanced due diligence; high-risk is a small population under SAR monitoring. This distribution means the `customer_segment=high_risk` filter has enough true members to be testable (~25,700 in IBM, ~14 in synthetic).

### 3.10 kyc_status (Synthesised)

**90% `verified` / 7% `pending` / 3% `incomplete`**  
Generated via `MD5(customer_id + "kyc") % 100`. The `pending` and `incomplete` segments are intentionally small but non-zero so that segment filters work during testing.

### 3.11 is_pep (Synthesised)

**~1.5% `True`** via `MD5(customer_id + "pep") % 1000 < 15`.  
IBM has no PEP flag. The 1.5% rate is consistent with industry estimates of PEP concentration in bank customer bases. Without variance the `customer_segment=pep` filter returns zero results and is untestable.

### 3.12 expected_monthly_volume (Computed + Synthesised)

For IBM data: computed as the **median of actual monthly sent amounts** per account (groupby `sender_id` and calendar month, sum `amount`, then take median). Accounts with zero transaction history in the dataset (exist in accounts file but not in transactions) fall back to **5,000.0**.  
For synthetic: median of generated monthly sent amounts; accounts with no history use a seed-based uniform draw in [1,000, 20,000].

---

## 4. Synthetic Generation Parameters

All parameters are constants in `data/generate_synthetic.py` and printed on every run. Reproduce with `python data/generate_synthetic.py` — output is deterministic.

| Parameter | Value | Description |
|---|---|---|
| `SEED` | 42 | Global random seed |
| `TOTAL_ROWS` | 1,800 | Normal (unlabelled) transactions |
| `NORMAL_CUSTOMERS` | 200 | Customers in the normal population |
| `STRUCTURING_CUSTOMERS` | 10 | Customers performing structuring |
| `STRUCTURING_TXN_PER_CUST` | 8 | Avg txns per structuring customer |
| `STRUCTURING_WINDOW_DAYS` | 7 | All structuring txns within 7 days |
| `STRUCTURING_AMOUNT` | [8,800 – 9,999] | Just-below-threshold amounts |
| `SMURFING_HUBS` | 3 | Hub accounts in smurfing rings |
| `SMURFING_RING` | 8 | Smurf accounts per hub |
| `SMURFING_TXN_PER_SMURF` | 2 | Deposits per smurf after hub transfer |
| `SMURFING_HUB_AMOUNT` | [50,000 – 200,000] | Initial hub wire |
| `SMURFING_SUB_AMOUNT` | [7,000 – 9,500] | Sub-threshold smurf deposits |
| `LAYERING_CHAINS` | 5 | Distinct layering chains |
| `LAYERING_HOPS` | 4 | Intermediate accounts per chain |
| `LAYERING_AMOUNT` | [20,000 – 500,000] | Starting wire amount |
| `LAYERING_HOP_DELAY_HOURS` | 12 | Hours between layering hops |
| `RAPID_CASHOUT_CUSTOMERS` | 8 | Customers performing rapid cashout |
| `RAPID_CASHOUT_IN` | [15,000 – 80,000] | Inbound wire amount |
| `RAPID_CASHOUT_SPLITS` | 4 | ATM/cash withdrawals after receipt |
| `RAPID_CASHOUT_WINDOW_HOURS` | 20 | All cashouts within 20 hours of receipt |

### Customer pool breakdown (synthetic)

| Cohort | Count | customer_id prefix |
|---|---|---|
| Normal | 200 | `C-N0001` – `C-N0200` |
| Structuring | 10 | `C-STR01` – `C-STR10` |
| Smurfing hubs | 3 | `C-HUB01` – `C-HUB03` |
| Smurfs | 24 | `C-SMF01` – `C-SMF24` |
| Layering | 25 | `C-LAY01` – `C-LAY25` |
| Rapid cashout | 8 | `C-RCO01` – `C-RCO08` |
| **Total** | **270** | |

### Synthesised customer attribute distributions (synthetic)

| Attribute | Distribution |
|---|---|
| `customer_type` | `individual` 85%, `business` 15% (cohort-overridden: HUB/LAY/RCO → business; STR/SMF → individual) |
| `risk_rating` | 80% low, 15% medium, 5% high (hash-seeded per customer_id) |
| `kyc_status` | 90% verified, 7% pending, 3% incomplete |
| `is_pep` | ~1.5% True |
| `name` | First + Last from seeded 30×20 name grid |
| `occupation` | 10 occupations, uniform random (individuals); "corporate banking" (business) |
| `account_open_date` | Uniform random in [2015-01-01, 2022-12-31], seeded |

---

## 5. What Is NOT in the Data (and Why It Matters)

| Missing field | IBM source | Synthetic source | Impact |
|---|---|---|---|
| Per-txn country | Not provided | Fully populated | `is_cross_border` is always False for IBM; cross-border detection only works on synthetic |
| Customer KYC fields | Not provided | Synthesised | Risk/KYC segment filters require synthesised data to have real variance — achieved via hash distributions |
| Pattern labels | Not provided (only `Is Laundering` binary) | Fully labelled | Pattern-specific detection (`pattern_label`) works only on synthetic in the demo; IBM provides binary ground truth |
| Transaction IDs | Not provided | Sequential T-XXXXXX | IBM IDs are synthetic sequential — no semantic meaning |

---

## 6. Gitignore and Data Governance

- Raw IBM files cached at `~/.cache/kagglehub/` — **outside the repo**, never staged
- `data/raw/` and `data/processed/` are in `.gitignore` (Track A, already present)
- `data/sample/aml_sample.csv` and `data/sample/aml_sample_customers.csv` are **committed exactly once** with fixed seed 42 — only Track B may regenerate
- No PII in any committed file (all synthetic or anonymised)
