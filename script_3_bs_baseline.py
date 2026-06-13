"""Build the Black-Scholes feature table used by the BOLA/LSTM pipeline."""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH = "data/fobhav_nifty_nse_full.csv"
VIX_PATH    = "data/india_vix.csv"  # downloaded via fetch_vix.py
DATA_START  = pd.Timestamp("2010-07-19")
DATA_END    = pd.Timestamp("2026-04-26")
RISK_FREE   = 0.05          # annualised, ~RBI repo rate (Jul–Aug 2020)
TRANS_COST  = 0.001         # 0.1% per rebalance leg (one-way)
LOT_SIZE    = 75            # Nifty lot size (standard NSE lot)
ATM_BAND    = 0.03          # ±3% of spot = "At-The-Money" zone
OUTPUT_LEVEL = os.getenv("BS_OUTPUT", "summary").lower()
VERBOSE_OUTPUT = OUTPUT_LEVEL in {"verbose", "debug"}
QUIET_OUTPUT = OUTPUT_LEVEL in {"quiet", "silent"}


def log(message="", *, verbose=False):
    """Keep normal runs compact while allowing detailed diagnostics."""
    if QUIET_OUTPUT:
        return
    if verbose and not VERBOSE_OUTPUT:
        return
    print(message)


os.makedirs("results", exist_ok=True)

log("Black-Scholes feature build")
df_raw = pd.read_csv(DATA_PATH)
log(f"  Raw rows: {len(df_raw):,}")
log(f"  Columns: {df_raw.columns.tolist()}", verbose=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

log("\nPreprocessing")

# 2a. Filter NIFTY FIRST (before date parsing — saves time on 112M rows)
df_raw = df_raw[df_raw["SYMBOL"] == "NIFTY"].copy()
log(f"  NIFTY rows: {len(df_raw):,}")

# 2b. Parse dates — format='mixed' handles dd-Mon-yy and other formats
df_raw["TIMESTAMP"] = pd.to_datetime(df_raw["TIMESTAMP"], format='mixed', dayfirst=True)
df_raw["EXPIRY_DT"] = pd.to_datetime(df_raw["EXPIRY_DT"], format='mixed', dayfirst=True)

# 2b.1 Restrict to the requested project date window
df_raw = df_raw[
    df_raw["TIMESTAMP"].between(DATA_START, DATA_END)
].copy()
log(f"  Date-filtered rows: {len(df_raw):,}")

# 2c. Separate futures (spot proxy) and options
df_fut = df_raw[df_raw["INSTRUMENT"] == "FUTIDX"].copy()
df_fut["CLOSE"] = pd.to_numeric(df_fut["CLOSE"], errors="coerce")
df_opt = df_raw[df_raw["INSTRUMENT"] == "OPTIDX"].copy()

# 2d. Build spot series: use near-month futures CLOSE as spot proxy
df_spot = (
    df_fut
    .sort_values(["TIMESTAMP", "EXPIRY_DT"])
    .groupby("TIMESTAMP")
    .first()
    .reset_index()[["TIMESTAMP", "CLOSE"]]
    .rename(columns={"CLOSE": "SPOT"})
)

log(f"  Futures rows: {len(df_fut):,}; options rows: {len(df_opt):,}")
log(f"  Dates: {df_raw['TIMESTAMP'].min().date()} to {df_raw['TIMESTAMP'].max().date()} "
    f"({df_raw['TIMESTAMP'].nunique()} trading dates)")

# 2e. Merge spot into options
df_opt = df_opt.merge(df_spot, on="TIMESTAMP", how="left")

# 2f. Drop rows with zero or NaN close (illiquid strikes)
# Convert numeric columns (read as strings in large CSV)
numeric_cols = ["STRIKE_PR", "OPEN", "HIGH", "LOW", "CLOSE", "SETTLE_PR",
                "CONTRACTS", "VAL_INLAKH", "OPEN_INT", "CHG_IN_OI"]
for col in numeric_cols:
    df_opt[col] = pd.to_numeric(df_opt[col], errors="coerce")

# Drop rows with zero or NaN close (illiquid strikes)
df_opt = df_opt[df_opt["CLOSE"] > 0].copy()

# 2g. Calculate Time-to-Expiry (T)
df_opt["T_DAYS"] = (df_opt["EXPIRY_DT"] - df_opt["TIMESTAMP"]).dt.days
df_opt["T"]      = df_opt["T_DAYS"] / 365.0

# Remove already-expired or same-day expiry
df_opt = df_opt[df_opt["T"] > 0].copy()

log(f"  Options after quality filter: {len(df_opt):,}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2b — LOAD & MERGE NSE INDIA VIX
# ─────────────────────────────────────────────────────────────────────────────

log("\nVIX merge")

if not os.path.exists(VIX_PATH):
    raise FileNotFoundError(
        f"VIX file not found: '{VIX_PATH}'\n"
        "  Run:  python fetch_vix.py --nsit <YOUR_NSIT_COOKIE>"
    )
df_vix = pd.read_csv(VIX_PATH, parse_dates=["DATE"])
df_vix = df_vix[["DATE", "VIX_CLOSE"]].rename(columns={"VIX_CLOSE": "VIX"})
df_vix = df_vix[df_vix["DATE"].between(DATA_START, DATA_END)].copy()
df_vix["VIX_SIGMA"] = df_vix["VIX"] / 100.0  # convert % → decimal σ
log(f"  VIX rows: {len(df_vix):,}; range: {df_vix['DATE'].min().date()} to {df_vix['DATE'].max().date()}")
log(f"  VIX mean/std: {df_vix['VIX'].mean():.2f}/{df_vix['VIX'].std():.2f}", verbose=True)

# Restrict the option dataset to the overlapping VIX-supported period.
vix_start = df_vix["DATE"].min()
pre_vix_rows = (df_opt["TIMESTAMP"] < vix_start).sum()
if pre_vix_rows:
    log(f"  Dropping {pre_vix_rows:,} option rows before VIX start date ({vix_start.date()})")
    df_opt = df_opt[df_opt["TIMESTAMP"] >= vix_start].copy()
log(f"  Options in VIX-supported period: {len(df_opt):,}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — ATM SELECTION
# ─────────────────────────────────────────────────────────────────────────────

log("\nATM selection")

df_opt["MONEYNESS"] = abs(df_opt["STRIKE_PR"] - df_opt["SPOT"]) / df_opt["SPOT"]
df_atm = df_opt[df_opt["MONEYNESS"] <= ATM_BAND].copy()

log(f"  ATM rows: {len(df_atm):,}; types: {df_atm['OPTION_TYP'].value_counts().to_dict()}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — BLACK-SCHOLES FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def bs_delta(S, K, T, r, sigma, opt_type="CE"):
    """
    Black-Scholes delta — the hedge ratio.
    sigma is sourced from NSE India VIX (VIX_CLOSE / 100).
    For a CALL: delta ∈ (0, 1)
    For a PUT : delta ∈ (-1, 0)
    """
    if T <= 0 or sigma <= 0:
        if opt_type == "CE":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if opt_type == "CE":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1.0

# Quick sanity check — using a representative VIX level (σ = 16%)
_test_delta_ce = bs_delta(11400, 11400, 5/365, RISK_FREE, 0.16, "CE")
_test_delta_pe = bs_delta(11400, 11400, 5/365, RISK_FREE, 0.16, "PE")
log(f"  BS sanity delta: CE={_test_delta_ce:.4f}, PE={_test_delta_pe:.4f}", verbose=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — BUILD CLEAN FEATURE TABLE
# ─────────────────────────────────────────────────────────────────────────────

log("\nFeature table")

keep_cols = [
    "TIMESTAMP", "EXPIRY_DT", "SYMBOL",
    "STRIKE_PR", "OPTION_TYP",
    "SPOT", "CLOSE",          # CLOSE = option market price
    "T", "T_DAYS", "MONEYNESS"
]
df_features = df_atm[keep_cols].copy()
df_features = df_features.rename(columns={
    "TIMESTAMP":  "DATE",
    "CLOSE":      "OPTION_PRICE",
    "STRIKE_PR":  "STRIKE",
    "OPTION_TYP": "TYPE",
})
df_features = df_features.sort_values(["DATE", "EXPIRY_DT", "STRIKE"])

# ── Merge NSE India VIX by date ──────────────────────────────────────────────
df_features = df_features.merge(df_vix[["DATE", "VIX", "VIX_SIGMA"]],
                                on="DATE", how="left")
n_missing = df_features["VIX_SIGMA"].isna().sum()
if n_missing > 0:
    log(f"  Warning: {n_missing} rows with no VIX match; forward-filling")
    df_features["VIX_SIGMA"] = df_features["VIX_SIGMA"].ffill()
    df_features["VIX"]       = df_features["VIX"].ffill()
log(f"  VIX merged; mean VIX={df_features['VIX'].mean():.2f}", verbose=True)

# ── Compute BS Delta using VIX as σ ──────────────────────────────────────────
def compute_delta_row(row):
    return bs_delta(
        S        = row["SPOT"],
        K        = row["STRIKE"],
        T        = row["T"],
        r        = RISK_FREE,
        sigma    = row["VIX_SIGMA"],
        opt_type = row["TYPE"]
    )

df_features["DELTA"] = df_features.apply(compute_delta_row, axis=1)
log(f"  Feature rows: {len(df_features):,}; columns: {df_features.shape[1]}")
log(f"  Delta mean/min/max: {df_features['DELTA'].mean():.3f}/"
    f"{df_features['DELTA'].min():.3f}/{df_features['DELTA'].max():.3f}", verbose=True)
log("\nSample feature rows:", verbose=True)
log(df_features.head(8).to_string(index=False), verbose=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — DELTA-HEDGING SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
"""
Strategy: For each (option series = same expiry + strike + type),
  - Day 1 : write 1 lot of the option. Hedge with Delta_t lots of Nifty futures.
  - Day t+1: spot moves → new Delta. Rebalance hedge by (Delta_new - Delta_old).
  - Transaction cost = TRANS_COST * |rebalance| * Spot
  - P&L = change in option value − change in hedge value − transaction cost
  - Tracking Error = std(daily P&L)
"""

log("\nBlack-Scholes simulation")

results = []  # one record per hedging day

# Group by option series
groups = df_features.groupby(["EXPIRY_DT", "STRIKE", "TYPE"])
log(f"  Option series: {len(groups):,}", verbose=True)

for (expiry, strike, opt_type), grp in groups:
    grp = grp.sort_values("DATE").reset_index(drop=True)
    if len(grp) < 2:
        continue  # need at least 2 days to simulate

    prev_delta = None
    prev_option_price = None

    for i, row in grp.iterrows():
        curr_delta        = row["DELTA"]
        curr_option_price = row["OPTION_PRICE"]
        curr_spot         = row["SPOT"]
        date              = row["DATE"]

        if prev_delta is None:
            # Day 1: open position, record entry state
            prev_delta        = curr_delta
            prev_option_price = curr_option_price
            continue  # no P&L yet

        # ── Rebalancing
        delta_change   = curr_delta - prev_delta        # futures lots to trade
        trans_cost     = abs(delta_change) * curr_spot * TRANS_COST

        # ── P&L for this time-step
        #    Writer's P&L = (received premium yesterday - today's option price)
        #                   + hedge gain/loss
        #                   - transaction cost
        option_pnl  = prev_option_price - curr_option_price   # we wrote the option
        hedge_pnl   = prev_delta * (curr_spot - grp.loc[i-1, "SPOT"])
        daily_pnl   = option_pnl + hedge_pnl - trans_cost

        results.append({
            "DATE":          date,
            "EXPIRY_DT":     expiry,
            "STRIKE":        strike,
            "TYPE":          opt_type,
            "SPOT":          curr_spot,
            "OPTION_PRICE":  curr_option_price,
            "DELTA":         curr_delta,
            "DELTA_CHANGE":  delta_change,
            "OPTION_PNL":    option_pnl,
            "HEDGE_PNL":     hedge_pnl,
            "TRANS_COST":    trans_cost,
            "DAILY_PNL":     daily_pnl,
            "T":             row["T"],
            "VIX":           row["VIX"],
        })

        prev_delta        = curr_delta
        prev_option_price = curr_option_price

df_sim = pd.DataFrame(results)
log(f"  Simulation rows: {len(df_sim):,}", verbose=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — METRICS & BENCHMARKING
# ─────────────────────────────────────────────────────────────────────────────

log("\nBaseline metrics")

total_pnl          = df_sim["DAILY_PNL"].sum()
mean_daily_pnl     = df_sim["DAILY_PNL"].mean()
tracking_error     = df_sim["DAILY_PNL"].std()          # std of daily P&L
total_trans_cost   = df_sim["TRANS_COST"].sum()
avg_delta_change   = df_sim["DELTA_CHANGE"].abs().mean()
sharpe             = (mean_daily_pnl / tracking_error) * np.sqrt(252) if tracking_error > 0 else np.nan

# Per-type breakdown
ce_pnl = df_sim[df_sim["TYPE"]=="CE"]["DAILY_PNL"]
pe_pnl = df_sim[df_sim["TYPE"]=="PE"]["DAILY_PNL"]

log(f"  Total P&L: {total_pnl:.2f}")
log(f"  Mean daily P&L: {mean_daily_pnl:.2f}")
log(f"  Tracking error: {tracking_error:.2f}")
log(f"  Sharpe annualized: {sharpe:.4f}")
log(f"  Transaction cost: {total_trans_cost:.2f}")
log(f"  Avg |delta change|: {avg_delta_change:.5f}", verbose=True)
log(f"  CE mean P&L/TE: {ce_pnl.mean():.2f}/{ce_pnl.std():.2f}", verbose=True)
log(f"  PE mean P&L/TE: {pe_pnl.mean():.2f}/{pe_pnl.std():.2f}", verbose=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

df_features.to_csv("results/bs_features.csv", index=False)
log("\nSaved: results/bs_features.csv")
log("Black-Scholes baseline complete. Ready for BOLA pipeline.")
