"""
Track B — data_loader.py

Adapters that convert raw source data into the canonical schema defined in
docs/CONTRACTS.md Contract 0.  Each adapter is a pure function:
    raw input(s)  →  (transactions_df, customers_df)

The `load_data` tool selects the adapter via the `source` parameter.

Canonical transactions columns
    txn_id, timestamp, sender_id, receiver_id, amount, currency,
    txn_type, channel, sender_country, receiver_country, is_cross_border,
    label_is_laundering, pattern_label

Canonical customers columns
    customer_id, name, account_open_date, customer_type, country,
    occupation, risk_rating, kyc_status, is_pep, expected_monthly_volume

Imports: no backend.agent.*, no other backend.tools.*
"""

from __future__ import annotations

import hashlib
import os
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backend.tools.base import ToolContext, ToolResult, tool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# kagglehub default cache path (outside the repo — confirmed during setup)
_KAGGLE_CACHE = Path.home() / ".cache" / "kagglehub" / "datasets" / \
    "ealtman2019" / "ibm-transactions-for-anti-money-laundering-aml" / \
    "versions" / "8"

_IBM_TRANS_FILE = _KAGGLE_CACHE / "HI-Small_Trans.csv"
_IBM_ACCTS_FILE = _KAGGLE_CACHE / "HI-Small_accounts.csv"

# Synthetic sample — committed in the repo under data/sample/
_SYNTHETIC_FILE = Path(__file__).parent.parent.parent / "data" / "sample" / \
    "aml_sample.csv"
_SYNTHETIC_CUSTOMERS_FILE = Path(__file__).parent.parent.parent / "data" / \
    "sample" / "aml_sample_customers.csv"

# Alt-schema synthetic sample — same patterns, different raw column names/
# encodings (data/generate_synthetic_alt.py). Used to prove the canonical
# schema is reachable from a differently-named raw source via an adapter.
_SYNTHETIC_ALT_FILE = Path(__file__).parent.parent.parent / "data" / "sample" / \
    "aml_sample_alt.csv"
_SYNTHETIC_ALT_CUSTOMERS_FILE = Path(__file__).parent.parent.parent / "data" / \
    "sample" / "aml_sample_alt_customers.csv"

# Parquet cache for the stratified IBM sample (data/processed/ is gitignored).
# load_data(source='ibm_stratified') reads this if present, writes it on first build.
_PROCESSED_DIR        = Path(__file__).parent.parent.parent / "data" / "processed"
_IBM_STRAT_TX_CACHE   = _PROCESSED_DIR / "ibm_stratified_sample.parquet"
_IBM_STRAT_CUST_CACHE = _PROCESSED_DIR / "ibm_stratified_customers.parquet"
_IBM_STRAT_META_CACHE = _PROCESSED_DIR / "ibm_stratified_cache_meta.json"

# ---------------------------------------------------------------------------
# Currency name → ISO 4217
# ---------------------------------------------------------------------------

_CURRENCY_MAP: dict[str, str] = {
    # IBM AML actual values (from HI-Small_Trans.csv full scan)
    "US Dollar": "USD",
    "Euro": "EUR",
    "Swiss Franc": "CHF",
    "Yuan": "CNY",
    "Shekel": "ILS",
    "Rupee": "INR",
    "UK Pound": "GBP",
    "Ruble": "RUB",
    "Yen": "JPY",
    "Bitcoin": "BTC",
    "Canadian Dollar": "CAD",
    "Australian Dollar": "AUD",
    "Mexican Peso": "MXN",
    "Saudi Riyal": "SAR",
    "Brazil Real": "BRL",
    # Additional common names (for PaySim / synthetic robustness)
    "British Pound": "GBP",
    "Chinese Yuan": "CNY",
    "Russian Ruble": "RUB",
    "Brazilian Real": "BRL",
    "Swedish Krona": "SEK",
    "Norwegian Krone": "NOK",
    "Danish Krone": "DKK",
    "South Korean Won": "KRW",
    "Singapore Dollar": "SGD",
    "Hong Kong Dollar": "HKD",
    "New Zealand Dollar": "NZD",
    "South African Rand": "ZAR",
    "Turkish Lira": "TRY",
    "Polish Zloty": "PLN",
    "Czech Koruna": "CZK",
    "Hungarian Forint": "HUF",
    "Romanian Leu": "RON",
    "UAE Dirham": "AED",
    "Israeli New Shekel": "ILS",
    "Pakistani Rupee": "PKR",
    "Bangladesh Taka": "BDT",
}

# ---------------------------------------------------------------------------
# Payment Format → (txn_type, channel) mapping
# IBM's single field maps to two separate canonical enums.
# Choices documented in DATA_CARD.md.
# ---------------------------------------------------------------------------

_FORMAT_MAP: dict[str, tuple[str, str]] = {
    "Reinvestment": ("transfer", "online"),
    "Cheque": ("deposit", "branch"),
    "Credit Card": ("transfer", "online"),
    "ACH": ("transfer", "online"),
    "Cash": ("cash", "branch"),
    "Wire": ("wire", "wire"),
    "Bitcoin": ("transfer", "online"),
}

_DEFAULT_TXN_TYPE = "transfer"
_DEFAULT_CHANNEL = "online"

# ---------------------------------------------------------------------------
# Bank Name → ISO 3166 alpha-2 country heuristic
# ---------------------------------------------------------------------------

_COUNTRY_KEYWORDS: list[tuple[str, str]] = [
    ("portugal", "PT"), ("canada", "CA"), ("uk", "GB"), ("united kingdom", "GB"),
    ("germany", "DE"), ("spain", "ES"), ("brazil", "BR"), ("mexico", "MX"),
    ("russia", "RU"), ("croatia", "HR"), ("japan", "JP"), ("italy", "IT"),
    ("israel", "IL"), ("france", "FR"), ("netherlands", "NL"), ("sweden", "SE"),
    ("switzerland", "CH"), ("australia", "AU"), ("china", "CN"), ("india", "IN"),
    ("singapore", "SG"), ("hong kong", "HK"), ("south korea", "KR"),
    ("south africa", "ZA"), ("turkey", "TR"), ("poland", "PL"),
    ("czech", "CZ"), ("hungary", "HU"), ("romania", "RO"),
    ("new zealand", "NZ"), ("argentina", "AR"), ("colombia", "CO"),
    ("chile", "CL"), ("peru", "PE"), ("nigeria", "NG"), ("kenya", "KE"),
    ("egypt", "EG"), ("uae", "AE"), ("saudi", "SA"), ("pakistan", "PK"),
    ("bangladesh", "BD"), ("indonesia", "ID"), ("thailand", "TH"),
    ("vietnam", "VN"), ("philippines", "PH"), ("malaysia", "MY"),
    ("denmark", "DK"), ("norway", "NO"), ("finland", "FI"), ("austria", "AT"),
    ("belgium", "BE"), ("greece", "GR"), ("ukraine", "UA"),
    # US-named banks (city names, national names without country keyword)
    ("national bank", "US"), ("savings bank", "US"), ("community bank", "US"),
    ("bank of new york", "US"), ("willows", "US"), ("acme", "US"),
    ("harrisburg", "US"), ("omaha", "US"), ("cleveland", "US"),
]


def _bank_name_to_country(bank_name: str) -> str:
    """Heuristically extract ISO-3166-alpha-2 from IBM's bank name string.

    'Canada Bank #27'  → 'CA'
    'National Bank of Harrisburg' → 'US'
    Anything unrecognised → 'UNK'
    """
    lower = bank_name.lower()
    for keyword, iso in _COUNTRY_KEYWORDS:
        if keyword in lower:
            return iso
    return "UNK"


# ---------------------------------------------------------------------------
# Deterministic per-customer attribute synthesis
# ---------------------------------------------------------------------------

_RNG_SEED = 42
_OPEN_DATE_START = date(2015, 1, 1)
_OPEN_DATE_END = date(2021, 12, 31)
_DATE_RANGE_DAYS = (_OPEN_DATE_END - _OPEN_DATE_START).days


def _stable_hash(value: str, modulus: int) -> int:
    """Return a stable integer in [0, modulus) derived from value."""
    digest = int(hashlib.md5(value.encode()).hexdigest(), 16)
    return digest % modulus


def _synthesise_account_open_date(customer_id: str) -> date:
    """Deterministic random open date in [2015-01-01, 2021-12-31]."""
    offset = _stable_hash(customer_id + "open_date", _DATE_RANGE_DAYS)
    return _OPEN_DATE_START + timedelta(days=offset)


def _synthesise_risk_rating(customer_id: str) -> str:
    """80% low, 15% medium, 5% high — seeded by customer_id."""
    val = _stable_hash(customer_id + "risk", 100)
    if val < 80:
        return "low"
    if val < 95:
        return "medium"
    return "high"


def _synthesise_kyc_status(customer_id: str) -> str:
    """90% verified, 7% pending, 3% incomplete."""
    val = _stable_hash(customer_id + "kyc", 100)
    if val < 90:
        return "verified"
    if val < 97:
        return "pending"
    return "incomplete"


def _synthesise_is_pep(customer_id: str) -> bool:
    """~1.5% True, seeded per customer_id."""
    val = _stable_hash(customer_id + "pep", 1000)
    return val < 15  # 1.5%


def _entity_name_to_type(entity_name: str) -> str:
    """Map IBM entity name to canonical customer_type.

    'Individual #...'          → 'individual'
    'Corporation #...'         → 'business'
    'Partnership #...'         → 'business'
    'Sole Proprietorship #...' → 'business'
    'Country #...'             → 'business'
    'Direct ...'               → 'business'
    anything else              → 85/15 split by hash
    """
    lower = entity_name.lower()
    if lower.startswith("individual"):
        return "individual"
    for prefix in ("corporation", "partnership", "sole proprietorship",
                   "country", "direct"):
        if lower.startswith(prefix):
            return "business"
    # Fallback: 85% individual, 15% business
    val = _stable_hash(entity_name + "ctype", 100)
    return "individual" if val < 85 else "business"


def _entity_name_to_occupation(entity_name: str, customer_type: str) -> str:
    """Derive occupation from entity type."""
    lower = entity_name.lower()
    if lower.startswith("corporation"):
        return "corporate banking"
    if lower.startswith("partnership"):
        return "business partnership"
    if lower.startswith("sole proprietorship"):
        return "sole proprietor"
    if lower.startswith("country"):
        return "government/sovereign"
    if lower.startswith("individual"):
        return "individual"
    if lower.startswith("direct"):
        return "direct payment entity"
    return "business" if customer_type == "business" else "individual"


# ---------------------------------------------------------------------------
# IBM AML Adapter
# ---------------------------------------------------------------------------

def _adapt_ibm(
    trans_path: str = str(_IBM_TRANS_FILE),
    accts_path: str = str(_IBM_ACCTS_FILE),
    nrows: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pure adapter: IBM AML HI-Small CSV → (transactions_df, customers_df).

    No side effects. No I/O outside reading the two CSV files.

    Parameters
    ----------
    trans_path : path to HI-Small_Trans.csv
    accts_path : path to HI-Small_accounts.csv
    nrows      : if set, read only this many transaction rows (for testing)
    """
    # ------------------------------------------------------------------
    # 1. Load raw transactions
    # ------------------------------------------------------------------
    raw = pd.read_csv(trans_path, nrows=nrows)

    # ------------------------------------------------------------------
    # 2. txn_id: T-000001 format, sequential
    # ------------------------------------------------------------------
    raw["txn_id"] = [f"T-{i:06d}" for i in range(1, len(raw) + 1)]

    # ------------------------------------------------------------------
    # 3. timestamp: parse 'YYYY/MM/DD HH:MM', tz-naive
    # ------------------------------------------------------------------
    raw["timestamp"] = pd.to_datetime(raw["Timestamp"], format="%Y/%m/%d %H:%M")

    # ------------------------------------------------------------------
    # 4. sender_id / receiver_id: prefix 'C-'
    # ------------------------------------------------------------------
    raw["sender_id"] = "C-" + raw["Account"].astype(str)
    raw["receiver_id"] = "C-" + raw["Account.1"].astype(str)

    # ------------------------------------------------------------------
    # 5. amount / currency: use Amount Received / Receiving Currency
    #    Rationale: receiver side captures what landed in the account —
    #    the quantity relevant to structuring detection. Cross-currency
    #    rows (~<0.1%) differ only due to FX; we take the received value.
    # ------------------------------------------------------------------
    raw["amount"] = raw["Amount Received"].astype(float)
    raw["currency"] = raw["Receiving Currency"].map(
        lambda x: _CURRENCY_MAP.get(x, "UNK")  # unknown currencies → UNK, not a truncation
    )

    # ------------------------------------------------------------------
    # 6. txn_type + channel: from Payment Format
    # ------------------------------------------------------------------
    fmt_tuples = raw["Payment Format"].map(
        lambda x: _FORMAT_MAP.get(x, (_DEFAULT_TXN_TYPE, _DEFAULT_CHANNEL))
    )
    raw["txn_type"] = fmt_tuples.map(lambda t: t[0])
    raw["channel"] = fmt_tuples.map(lambda t: t[1])

    # ------------------------------------------------------------------
    # 7. sender_country / receiver_country: no country data in IBM raw
    #    → 'UNK' for all. Per task spec: UNK vs UNK = not cross-border.
    # ------------------------------------------------------------------
    raw["sender_country"] = "UNK"
    raw["receiver_country"] = "UNK"
    raw["is_cross_border"] = False  # UNK vs UNK treated as same/unknown

    # ------------------------------------------------------------------
    # 8. label_is_laundering: IBM's 'Is Laundering' 0/1 int → bool
    # ------------------------------------------------------------------
    raw["label_is_laundering"] = raw["Is Laundering"].astype(bool)

    # ------------------------------------------------------------------
    # 9. pattern_label: null for IBM (synthetic-only field)
    # ------------------------------------------------------------------
    raw["pattern_label"] = None

    # ------------------------------------------------------------------
    # 10. Select and order canonical columns
    # ------------------------------------------------------------------
    tx_canonical = raw[[
        "txn_id", "timestamp", "sender_id", "receiver_id",
        "amount", "currency", "txn_type", "channel",
        "sender_country", "receiver_country", "is_cross_border",
        "label_is_laundering", "pattern_label",
    ]].copy()

    # ------------------------------------------------------------------
    # 11. Build customers dataframe from accounts file
    # ------------------------------------------------------------------
    cust_df = _build_customers_from_ibm(accts_path, tx_canonical)

    return tx_canonical, cust_df


# ---------------------------------------------------------------------------
# IBM AML — Stratified Sampler
# ---------------------------------------------------------------------------

def _stratified_sample_ibm(
    trans_path: str = str(_IBM_TRANS_FILE),
    accts_path: str = str(_IBM_ACCTS_FILE),
    target_size: int = 200_000,
    max_pos_customers: int = 500,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified sample of the IBM AML HI-Small dataset for real-data testing.

    Unlike the plain IBM adapter (which can read nrows from the file in order),
    this function deliberately over-represents laundering-positive customers to
    produce a sample where the positive rate is meaningfully above the raw 0.1%
    baseline — necessary for rule and ML detection to have signal to work with.

    Strategy
    --------
    1. Read only 'Account' and 'Is Laundering' from the full CSV (fast, ~1 GB
       but usecols limits memory).
    2. Identify every unique sender account that appears in at least one
       laundering-labelled row.  Sample up to max_pos_customers of them (using
       the fixed seed so results are reproducible) — all of each selected
       customer's transactions are included, never truncated.
    3. Compute the remaining row budget after the positive customers fill their
       share.  Randomly sample clean (never-positive) customers to fill it,
       again including each selected customer's full history.
    4. Concatenate, reassign sequential txn_ids, and pass through the exact
       same canonical mapping already built in _adapt_ibm (reused, not
       duplicated).

    Parameters
    ----------
    trans_path      : path to HI-Small_Trans.csv
    accts_path      : path to HI-Small_accounts.csv
    target_size     : approximate total transaction count to aim for (default
                      200,000).  Actual count may exceed this slightly because
                      customer histories are kept whole — never truncated mid-slice.
    max_pos_customers : hard cap on how many positive customers to include.
                      The IBM dataset has 3,376; at ~150 txns each their
                      collective history (508k rows) already exceeds any
                      reasonable target_size, so this cap keeps the sample
                      balanced.  Default 500 (~75k txns).
    seed            : random seed for the clean-customer sampling step, so
                      identical inputs produce identical outputs across runs.
    """
    rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Step 1: scan for positive customers using only two columns (fast)
    # ------------------------------------------------------------------
    scan = pd.read_csv(trans_path, usecols=["Account", "Is Laundering"])
    all_pos_accounts: set = set(
        scan.loc[scan["Is Laundering"] == 1, "Account"].unique()
    )
    all_neg_accounts: set = set(scan["Account"].unique()) - all_pos_accounts
    del scan  # free memory before loading the full CSV

    # ------------------------------------------------------------------
    # Step 2: sample positive customers (capped)
    # ------------------------------------------------------------------
    pos_accounts_list = sorted(all_pos_accounts)  # deterministic ordering
    rng.shuffle(pos_accounts_list)
    selected_pos = set(pos_accounts_list[:max_pos_customers])

    # ------------------------------------------------------------------
    # Step 3: load the full CSV and slice positive customer rows
    # ------------------------------------------------------------------
    raw_full = pd.read_csv(trans_path)
    pos_rows = raw_full[raw_full["Account"].isin(selected_pos)].copy()

    # ------------------------------------------------------------------
    # Step 4: fill the remainder from clean customers (whole histories)
    # ------------------------------------------------------------------
    remaining_budget = target_size - len(pos_rows)

    selected_neg: set = set()
    if remaining_budget > 0:
        neg_accounts_list = sorted(all_neg_accounts)
        rng.shuffle(neg_accounts_list)
        budget_left = remaining_budget
        for acct in neg_accounts_list:
            acct_rows = (raw_full["Account"] == acct).sum()
            if budget_left <= 0:
                break
            selected_neg.add(acct)
            budget_left -= acct_rows

    neg_rows = raw_full[raw_full["Account"].isin(selected_neg)].copy()
    sampled_raw = pd.concat([pos_rows, neg_rows], ignore_index=True)
    del raw_full, pos_rows, neg_rows  # free memory

    # ------------------------------------------------------------------
    # Step 5: reassign sequential txn_ids and apply canonical mapping
    # ------------------------------------------------------------------
    sampled_raw["txn_id"] = [f"T-{i:06d}" for i in range(1, len(sampled_raw) + 1)]
    sampled_raw["timestamp"] = pd.to_datetime(
        sampled_raw["Timestamp"], format="%Y/%m/%d %H:%M"
    )
    sampled_raw["sender_id"] = "C-" + sampled_raw["Account"].astype(str)
    sampled_raw["receiver_id"] = "C-" + sampled_raw["Account.1"].astype(str)
    sampled_raw["amount"] = sampled_raw["Amount Received"].astype(float)
    sampled_raw["currency"] = sampled_raw["Receiving Currency"].map(
        lambda x: _CURRENCY_MAP.get(x, "UNK")
    )
    fmt_tuples = sampled_raw["Payment Format"].map(
        lambda x: _FORMAT_MAP.get(x, (_DEFAULT_TXN_TYPE, _DEFAULT_CHANNEL))
    )
    sampled_raw["txn_type"] = fmt_tuples.map(lambda t: t[0])
    sampled_raw["channel"] = fmt_tuples.map(lambda t: t[1])
    sampled_raw["sender_country"] = "UNK"
    sampled_raw["receiver_country"] = "UNK"
    sampled_raw["is_cross_border"] = False
    sampled_raw["label_is_laundering"] = sampled_raw["Is Laundering"].astype(bool)
    sampled_raw["pattern_label"] = None

    tx_canonical = sampled_raw[[
        "txn_id", "timestamp", "sender_id", "receiver_id",
        "amount", "currency", "txn_type", "channel",
        "sender_country", "receiver_country", "is_cross_border",
        "label_is_laundering", "pattern_label",
    ]].copy()

    cust_df = _build_customers_from_ibm(accts_path, tx_canonical)
    return tx_canonical, cust_df



def _build_customers_from_ibm(
    accts_path: str,
    tx_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build canonical customers table from IBM accounts file.

    customer_id = 'C-' + Account Number  (matches sender_id/receiver_id)
    All synthesised fields use stable hash-based generation (seed 42).
    """
    acc = pd.read_csv(accts_path)

    # customer_id from Account Number (must match transaction sender/receiver IDs)
    acc["customer_id"] = "C-" + acc["Account Number"].astype(str)

    # name: IBM's Entity Name is descriptive ('Corporation #33520')
    acc["name"] = acc["Entity Name"].astype(str)

    # account_open_date: deterministic synthetic
    acc["account_open_date"] = acc["customer_id"].apply(_synthesise_account_open_date)

    # customer_type: derived from Entity Name prefix
    acc["customer_type"] = acc["Entity Name"].apply(
        lambda x: _entity_name_to_type(str(x))
    )

    # country: extracted from Bank Name
    acc["country"] = acc["Bank Name"].apply(_bank_name_to_country)

    # occupation: derived from entity type
    acc["occupation"] = acc.apply(
        lambda row: _entity_name_to_occupation(str(row["Entity Name"]), row["customer_type"]),
        axis=1,
    )

    # risk_rating: 80% low, 15% medium, 5% high — seeded
    acc["risk_rating"] = acc["customer_id"].apply(_synthesise_risk_rating)

    # kyc_status: 90% verified, 7% pending, 3% incomplete
    acc["kyc_status"] = acc["customer_id"].apply(_synthesise_kyc_status)

    # is_pep: ~1.5% True
    acc["is_pep"] = acc["customer_id"].apply(_synthesise_is_pep)

    # expected_monthly_volume: median of actual monthly tx volume per account
    # Use sender_id to compute; fall back to 5000.0 for accounts with no history
    monthly_vol = _compute_expected_monthly_volume(tx_df)
    acc["expected_monthly_volume"] = acc["customer_id"].map(monthly_vol).fillna(5000.0)

    # Select canonical columns
    customers_canonical = acc[[
        "customer_id", "name", "account_open_date", "customer_type",
        "country", "occupation", "risk_rating", "kyc_status",
        "is_pep", "expected_monthly_volume",
    ]].copy()

    return customers_canonical


def _compute_expected_monthly_volume(tx_df: pd.DataFrame) -> pd.Series:
    """Compute the median monthly sent volume per customer.

    Groups transactions by sender and calendar month, sums amounts,
    then returns the median of those monthly sums.  Returns a Series
    indexed by customer_id (sender_id format).
    """
    if tx_df.empty:
        return pd.Series(dtype=float)

    tmp = tx_df[["sender_id", "timestamp", "amount"]].copy()
    tmp["month"] = tmp["timestamp"].dt.to_period("M")
    monthly = tmp.groupby(["sender_id", "month"])["amount"].sum().reset_index()
    medians = monthly.groupby("sender_id")["amount"].median()
    return medians


# ---------------------------------------------------------------------------
# Synthetic Adapter
# ---------------------------------------------------------------------------

def _adapt_synthetic(
    trans_path: str = str(_SYNTHETIC_FILE),
    customers_path: str = str(_SYNTHETIC_CUSTOMERS_FILE),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pure adapter: load the committed synthetic CSV pair into canonical DFs.

    The synthetic generator (data/generate_synthetic.py) already writes
    canonical-schema CSVs; this adapter handles type coercions.
    """
    tx = pd.read_csv(trans_path)
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    tx["is_cross_border"] = tx["is_cross_border"].astype(bool)
    tx["label_is_laundering"] = tx["label_is_laundering"].where(
        tx["label_is_laundering"].notna(), None
    )
    # Ensure label_is_laundering is bool where not null
    mask = tx["label_is_laundering"].notna()
    tx.loc[mask, "label_is_laundering"] = tx.loc[mask, "label_is_laundering"].astype(bool)

    cust = pd.read_csv(customers_path)
    cust["account_open_date"] = pd.to_datetime(cust["account_open_date"]).dt.date
    cust["is_pep"] = cust["is_pep"].astype(bool)

    return tx, cust


# ---------------------------------------------------------------------------
# Synthetic Adapter — alt raw schema
# ---------------------------------------------------------------------------

_ALT_TXN_TYPE_MAP = {"DEP": "deposit", "WD": "withdrawal", "XFER": "transfer", "WIRE": "wire", "CSH": "cash"}
_ALT_CHANNEL_MAP = {"ATM": "atm", "BRN": "branch", "ONL": "online", "MOB": "mobile", "WR": "wire"}
_ALT_CUSTOMER_TYPE_MAP = {"RETAIL": "individual", "CORP": "business"}
_ALT_RISK_MAP = {"L": "low", "M": "medium", "H": "high"}
_ALT_KYC_MAP = {"V": "verified", "P": "pending", "I": "incomplete"}


def _adapt_synthetic_alt(
    trans_path: str = str(_SYNTHETIC_ALT_FILE),
    customers_path: str = str(_SYNTHETIC_ALT_CUSTOMERS_FILE),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pure adapter: data/generate_synthetic_alt.py's raw CSVs → canonical DFs.

    Unlike `_adapt_synthetic`, this source uses a deliberately different raw
    column-naming/encoding convention (renamed headers, coded enums, Y/N
    flags, 'ACC-' id prefix, no `is_cross_border` column) to prove the
    canonical schema is reachable from a differently-shaped raw dataset via
    an adapter — real mapping, not just type coercion.
    """
    raw = pd.read_csv(trans_path)

    tx = pd.DataFrame()
    tx["txn_id"] = raw["ref_no"]
    tx["timestamp"] = pd.to_datetime(raw["event_ts"])
    tx["sender_id"] = raw["debit_acct"].str.replace("^ACC-", "C-", regex=True)
    tx["receiver_id"] = raw["credit_acct"].str.replace("^ACC-", "C-", regex=True)
    tx["amount"] = raw["txn_value"].astype(float)
    tx["currency"] = raw["ccy"]
    tx["txn_type"] = raw["activity_code"].map(_ALT_TXN_TYPE_MAP)
    tx["channel"] = raw["channel_cd"].map(_ALT_CHANNEL_MAP)
    tx["sender_country"] = raw["orig_ctry"]
    tx["receiver_country"] = raw["dest_ctry"]
    tx["is_cross_border"] = tx["sender_country"] != tx["receiver_country"]
    tx["label_is_laundering"] = raw["aml_flag"].map({"Y": True, "N": False}).where(
        raw["aml_flag"].isin(["Y", "N"]), None
    )
    tx["pattern_label"] = raw["typology"].map(
        lambda v: v.lower() if isinstance(v, str) and v else None
    )

    cust_raw = pd.read_csv(customers_path)
    cust = pd.DataFrame()
    cust["customer_id"] = cust_raw["acct_id"].str.replace("^ACC-", "C-", regex=True)
    cust["name"] = cust_raw["cust_name"]
    cust["account_open_date"] = pd.to_datetime(cust_raw["open_dt"]).dt.date
    cust["customer_type"] = cust_raw["segment"].map(_ALT_CUSTOMER_TYPE_MAP)
    cust["country"] = cust_raw["domicile"]
    cust["occupation"] = cust_raw["job_title"]
    cust["risk_rating"] = cust_raw["risk_tier"].map(_ALT_RISK_MAP)
    cust["kyc_status"] = cust_raw["kyc_stat"].map(_ALT_KYC_MAP)
    cust["is_pep"] = cust_raw["pep_ind"].map({"Y": True, "N": False})
    cust["expected_monthly_volume"] = cust_raw["exp_vol_monthly"].astype(float)

    return tx, cust


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _validate_canonical(tx_df: pd.DataFrame, cust_df: pd.DataFrame) -> list[str]:
    """Check canonical schema compliance; return list of violation strings."""
    violations: list[str] = []

    # --- transactions ---
    tx_required = {
        "txn_id", "timestamp", "sender_id", "receiver_id", "amount",
        "currency", "txn_type", "channel", "sender_country", "receiver_country",
        "is_cross_border", "label_is_laundering", "pattern_label",
    }
    missing_tx = tx_required - set(tx_df.columns)
    if missing_tx:
        violations.append(f"transactions missing columns: {missing_tx}")

    if "amount" in tx_df.columns:
        neg = (tx_df["amount"] <= 0).sum()
        if neg > 0:
            violations.append(f"transactions: {neg} non-positive amount rows")

    if "is_cross_border" in tx_df.columns:
        if tx_df["is_cross_border"].dtype != bool:
            violations.append("is_cross_border is not bool dtype")

    valid_txn_types = {"deposit", "withdrawal", "transfer", "wire", "cash"}
    valid_channels = {"atm", "branch", "online", "mobile", "wire"}
    if "txn_type" in tx_df.columns:
        bad = ~tx_df["txn_type"].isin(valid_txn_types)
        if bad.any():
            violations.append(f"invalid txn_type values: {tx_df.loc[bad, 'txn_type'].unique().tolist()}")
    if "channel" in tx_df.columns:
        bad = ~tx_df["channel"].isin(valid_channels)
        if bad.any():
            violations.append(f"invalid channel values: {tx_df.loc[bad, 'channel'].unique().tolist()}")

    # --- customers ---
    cust_required = {
        "customer_id", "name", "account_open_date", "customer_type", "country",
        "occupation", "risk_rating", "kyc_status", "is_pep", "expected_monthly_volume",
    }
    missing_cust = cust_required - set(cust_df.columns)
    if missing_cust:
        violations.append(f"customers missing columns: {missing_cust}")

    if "risk_rating" in cust_df.columns:
        valid_rr = {"low", "medium", "high"}
        bad = ~cust_df["risk_rating"].isin(valid_rr)
        if bad.any():
            violations.append(f"invalid risk_rating: {cust_df.loc[bad,'risk_rating'].unique()}")

    if "kyc_status" in cust_df.columns:
        valid_kyc = {"verified", "pending", "incomplete"}
        bad = ~cust_df["kyc_status"].isin(valid_kyc)
        if bad.any():
            violations.append(f"invalid kyc_status: {cust_df.loc[bad,'kyc_status'].unique()}")

    if "customer_type" in cust_df.columns:
        valid_ct = {"individual", "business"}
        bad = ~cust_df["customer_type"].isin(valid_ct)
        if bad.any():
            violations.append(f"invalid customer_type: {cust_df.loc[bad,'customer_type'].unique()}")

    return violations


# ---------------------------------------------------------------------------
# Registered tool
# ---------------------------------------------------------------------------

@tool(
    name="load_data",
    params={
        "source": (
            "str — 'ibm' | 'ibm_stratified' | 'synthetic' | 'synthetic_alt'. "
            "'ibm_stratified' loads a stratified sample of the IBM HI-Small dataset "
            "with over-represented laundering-positive customers for real-data testing. "
            "Uses a parquet cache (data/processed/) when available for fast loads (~2s vs ~153s). "
            "'synthetic_alt' loads a second synthetic dataset with a different raw column-naming/"
            "encoding convention (aml_sample_alt.csv), adapted to the same canonical schema."
        ),
        "nrows": "int | None — if set, load only this many transaction rows (ibm source only, for testing).",
        "target_size": (
            "int — approximate total transaction count for ibm_stratified mode (default 200,000). "
            "Actual count may exceed slightly to avoid truncating customer histories."
        ),
        "max_pos_customers": (
            "int — max laundering-positive customers to include in ibm_stratified mode (default 500). "
            "Each positive customer's FULL history is included, so this caps the positive-side row count."
        ),
        "seed": "int — random seed for clean-customer sampling in ibm_stratified mode (default 42).",
        "force_rebuild": (
            "bool — if True, ignore the parquet cache and re-stratify from the raw CSV, then "
            "overwrite the cache. Use after changing seed or target_size. Default False."
        ),
    },
    description=(
        "Load a dataset and convert it to the canonical transactions + customers schema "
        "(Contract 0, docs/CONTRACTS.md).  Returns ToolResult with df=transactions_df, "
        "artifacts['customers']=customers_df and artifacts['transactions_reference']="
        "the unfiltered transactions, which ml_detect ranks percentiles against."
    ),
)
def load_data(
    ctx: ToolContext,
    source: str = "synthetic_alt",
    nrows: Optional[int] = None,
    target_size: int = 200_000,
    max_pos_customers: int = 500,
    seed: int = 42,
    force_rebuild: bool = False,
    **kw,
) -> ToolResult:
    """Load and canonicalise a dataset.

    Parameters
    ----------
    ctx               : ToolContext — executor-managed context; df and customers will be set.
    source            : 'ibm', 'ibm_stratified', 'synthetic', or 'synthetic_alt'
    nrows             : optional row limit for 'ibm' source (smoke-testing, sequential rows)
    target_size       : approximate transaction count for 'ibm_stratified' (default 200,000)
    max_pos_customers : max positive customers for 'ibm_stratified' (default 500)
    seed              : random seed for clean-customer sampling in 'ibm_stratified' (default 42)
    force_rebuild     : if True, ignore the parquet cache and rebuild from the raw CSV (default False)
    """
    try:
        if source == "ibm":
            if not _IBM_TRANS_FILE.exists():
                return ToolResult(
                    ok=False,
                    error=(
                        f"IBM AML HI-Small dataset not found at {_IBM_TRANS_FILE}. "
                        "Run: import kagglehub; kagglehub.dataset_download("
                        "'ealtman2019/ibm-transactions-for-anti-money-laundering-aml')"
                    ),
                )
            tx_df, cust_df = _adapt_ibm(
                trans_path=str(_IBM_TRANS_FILE),
                accts_path=str(_IBM_ACCTS_FILE),
                nrows=nrows,
            )
            source_label = "IBM AML HI-Small"

        elif source == "ibm_stratified":
            use_cache = _IBM_STRAT_TX_CACHE.exists() and not force_rebuild

            if use_cache:
                # Fast path: read pre-built parquet (~2s vs ~153s)
                import json as _json
                tx_df   = pd.read_parquet(_IBM_STRAT_TX_CACHE)
                cust_df = pd.read_parquet(_IBM_STRAT_CUST_CACHE)
                tx_df["timestamp"] = pd.to_datetime(tx_df["timestamp"])
                cache_note = "(warm parquet cache)"
                if _IBM_STRAT_META_CACHE.exists():
                    with open(_IBM_STRAT_META_CACHE) as _f:
                        meta = _json.load(_f)
                    cache_note = (
                        f"(warm cache: target={meta.get('target_size','?')}, "
                        f"seed={meta.get('seed','?')}, "
                        f"built={meta.get('built_utc','?')[:10]})"
                    )
            else:
                # Cold path: build from raw CSV and save cache for next time
                if not _IBM_TRANS_FILE.exists():
                    return ToolResult(
                        ok=False,
                        error=(
                            f"IBM AML HI-Small dataset not found at {_IBM_TRANS_FILE}. "
                            "Run: import kagglehub; kagglehub.dataset_download("
                            "'ealtman2019/ibm-transactions-for-anti-money-laundering-aml')"
                        ),
                    )
                tx_df, cust_df = _stratified_sample_ibm(
                    trans_path=str(_IBM_TRANS_FILE),
                    accts_path=str(_IBM_ACCTS_FILE),
                    target_size=target_size,
                    max_pos_customers=max_pos_customers,
                    seed=seed,
                )
                # Write cache for next run
                import json as _json
                from datetime import datetime as _dt, timezone as _tz
                _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
                tx_df.to_parquet(_IBM_STRAT_TX_CACHE,   index=False)
                cust_df.to_parquet(_IBM_STRAT_CUST_CACHE, index=False)
                meta = {
                    "built_utc":         _dt.now(_tz.utc).isoformat(),
                    "source_csv":        str(_IBM_TRANS_FILE),
                    "target_size":       target_size,
                    "max_pos_customers": max_pos_customers,
                    "seed":              seed,
                    "row_count":         len(tx_df),
                    "customer_count":    int(tx_df["sender_id"].nunique()),
                    "pos_row_count":     int(tx_df["label_is_laundering"].sum()),
                }
                with open(_IBM_STRAT_META_CACHE, "w") as _f:
                    _json.dump(meta, _f, indent=2)
                cache_note = "(cache written for next run)"

            source_label = (
                f"IBM AML HI-Small stratified {cache_note}: "
                f"target={target_size:,}, max_pos={max_pos_customers}, seed={seed}"
            )

        elif source == "synthetic":
            if not _SYNTHETIC_FILE.exists():
                return ToolResult(
                    ok=False,
                    error=(
                        f"Synthetic dataset not found at {_SYNTHETIC_FILE}. "
                        "Run: python data/generate_synthetic.py"
                    ),
                )
            tx_df, cust_df = _adapt_synthetic()
            # Name the SOURCE KEY, not just the family. Both synthetic sources
            # used to render as "synthetic (<file>.csv)", differing only by the
            # filename in parentheses — and these two are exactly the pair that
            # must never be confused: every published metric is computed
            # against this one, while load_data's own default is the other. A
            # label that requires reading the filename to tell them apart is a
            # trap in a trace panel people skim.
            source_label = "synthetic — the labelled metrics set (aml_sample.csv)"

        elif source == "synthetic_alt":
            if not _SYNTHETIC_ALT_FILE.exists():
                return ToolResult(
                    ok=False,
                    error=(
                        f"Alt synthetic dataset not found at {_SYNTHETIC_ALT_FILE}. "
                        "Run: python data/generate_synthetic_alt.py"
                    ),
                )
            tx_df, cust_df = _adapt_synthetic_alt()
            source_label = "synthetic_alt — the alt-schema set, NOT the metrics set (aml_sample_alt.csv)"

        else:
            return ToolResult(
                ok=False,
                error=f"Unknown source '{source}'. Valid values: 'ibm', 'ibm_stratified', 'synthetic', 'synthetic_alt'.",
            )

        # Schema validation
        violations = _validate_canonical(tx_df, cust_df)
        notes = [
            f"load_data: loaded {len(tx_df):,} transactions, "
            f"{len(cust_df):,} customers from {source_label}"
        ]
        warnings: list[str] = []
        if violations:
            warnings = [f"Schema violation: {v}" for v in violations]
            notes.extend(warnings)

        # Inject customers into context so downstream tools can access them
        ctx.customers = cust_df

        return ToolResult(
            ok=True,
            df=tx_df,
            # transactions_reference is the population ml_detect ranks against. It is
            # captured here, before filter_data can narrow ctx.df, so an entity's
            # anomaly percentile is a property of the customer rather than of whatever
            # filters the analyst happened to type. Nothing may mutate it in place.
            artifacts={"customers": cust_df, "transactions_reference": tx_df},
            metrics={
                "txn_count": len(tx_df),
                "customer_count": len(cust_df),
                "source": source_label,
                "schema_violations": len(violations),
            },
            notes=notes,
        )

    except Exception as exc:  # noqa: BLE001
        return ToolResult(ok=False, error=f"load_data failed: {exc}")
