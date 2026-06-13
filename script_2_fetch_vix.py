#!/usr/bin/env python3
"""
==============================================================================
fetch_vix.py  — Download NSE India VIX Historical Data
==============================================================================
NSE API (confirmed columns):
  EOD_TIMESTAMP, EOD_INDEX_NAME,
  EOD_OPEN_INDEX_VAL, EOD_HIGH_INDEX_VAL, EOD_LOW_INDEX_VAL,
  EOD_CLOSE_INDEX_VAL, EOD_PREV_CLOSE, VIX_PTS_CHG, VIX_PERC_CHG

HOW TO RUN
----------
Option A — Automatic (uses curl + cookies from your active browser session):
  1. Open https://www.nseindia.com in Chrome/Firefox (stay logged in)
  2. In Chrome: DevTools → Application → Cookies → copy the value of
     'nsit' and 'nseappid' cookies
  3. Run:  python fetch_vix.py --nsit <NSIT_VALUE> --nseappid <NSEAPPID_VALUE>

Option B — Manual download (guaranteed to work):
  1. Open https://www.nseindia.com/reports-indices-historical-vix
  2. Select Custom, enter date range (max 365 days), click GO, click Download (.csv)
  3. Repeat for each year, save files to data/vix_raw/ folder
  4. Run:  python fetch_vix.py --from_csv data/vix_raw/

Option C — Direct requests (may fail with 403 due to NSE bot protection):
  python fetch_vix.py --method requests
==============================================================================
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
import glob

import pandas as pd

DEFAULT_FROM  = "2001-01-01"
DEFAULT_TO    = "2026-04-26"
OUTPUT_PATH   = "data/india_vix.csv"
NSE_API       = "https://www.nseindia.com/api/historicalOR/vixhistory"

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN CLEANING  (handles both API and web-table CSV formats)
# ─────────────────────────────────────────────────────────────────────────────

# Confirmed NSE API → our standard name
API_RENAME = {
    "EOD_TIMESTAMP":       "DATE",
    "EOD_OPEN_INDEX_VAL":  "VIX_OPEN",
    "EOD_HIGH_INDEX_VAL":  "VIX_HIGH",
    "EOD_LOW_INDEX_VAL":   "VIX_LOW",
    "EOD_CLOSE_INDEX_VAL": "VIX_CLOSE",   # ← key column used in BS
}

# Web-table CSV → our standard name
WEB_RENAME = {
    "DATE":  "DATE",
    "OPEN":  "VIX_OPEN",
    "HIGH":  "VIX_HIGH",
    "LOW":   "VIX_LOW",
    "CLOSE": "VIX_CLOSE",
}


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a VIX dataframe to: DATE | VIX_OPEN | VIX_HIGH | VIX_LOW | VIX_CLOSE"""
    df.columns = [c.strip().upper() for c in df.columns]
    rename = {**API_RENAME, **WEB_RENAME}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = [c for c in ["DATE", "VIX_OPEN", "VIX_HIGH", "VIX_LOW", "VIX_CLOSE"]
            if c in df.columns]
    df = df[keep].copy()
    df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["DATE"])
    for col in ["VIX_OPEN", "VIX_HIGH", "VIX_LOW", "VIX_CLOSE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["VIX_CLOSE"])
    df = df.sort_values("DATE").drop_duplicates(subset=["DATE"]).reset_index(drop=True)
    return df


def _report(df: pd.DataFrame):
    print(f"\n  ✓ VIX data ready: {len(df)} rows")
    print(f"    Date range : {df['DATE'].min().date()} → {df['DATE'].max().date()}")
    print(f"    VIX CLOSE  : mean={df['VIX_CLOSE'].mean():.2f}  "
          f"min={df['VIX_CLOSE'].min():.2f}  max={df['VIX_CLOSE'].max():.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# OPTION A — curl with manual cookies
# ─────────────────────────────────────────────────────────────────────────────

def fmt(dt: datetime) -> str:
    return dt.strftime("%d-%m-%Y")


def fetch_with_curl(from_date: str, to_date: str,
                    nsit: str,
                    nseappid: str = "",
                    extra_cookies: str = "") -> pd.DataFrame:
    """
    Use system curl with user-supplied cookies.
    Only `nsit` is required; `nseappid` and `extra_cookies` are optional extras.
    Fetches in 1-year chunks (NSE API limit = 365 days per request).
    """
    import subprocess, json

    from_dt = datetime.strptime(from_date, "%Y-%m-%d")
    to_dt   = datetime.strptime(to_date,   "%Y-%m-%d")

    # Build cookie string
    cookie_parts = [f"nsit={nsit}"]
    if nseappid:
        cookie_parts.append(f"nseappid={nseappid}")
    if extra_cookies:
        cookie_parts.append(extra_cookies.strip(";"))
    cookie_str = "; ".join(cookie_parts)

    print(f"  Cookie string: {cookie_str[:80]}…")

    all_records = []
    start = from_dt
    while start <= to_dt:
        # 90-day chunks: NSE API returns max ~70 rows/request (~63 trading days/quarter)
        end = min(start + timedelta(days=89), to_dt)
        url = f"{NSE_API}?from={fmt(start)}&to={fmt(end)}"

        # List-form args — handles spaces in User-Agent correctly without shell=True
        cmd = [
            "curl", "-s", "--http1.1", "--compressed",
            "-H", f"User-Agent: {UA}",
            "-H", "Accept: application/json, text/plain, */*",
            "-H", "Accept-Language: en-US,en;q=0.9",
            "-H", "Referer: https://www.nseindia.com/reports-indices-historical-vix",
            "-H", "X-Requested-With: XMLHttpRequest",
            "-H", f"Cookie: {cookie_str}",
            url,
        ]

        print(f"    {fmt(start)} → {fmt(end)} …", end=" ", flush=True)
        try:
            result = subprocess.run(cmd, capture_output=True,
                                    text=True, timeout=30)
            raw = result.stdout.strip()
            if not raw:
                print(f"empty response (stderr: {result.stderr[:80]})")
                start = end + timedelta(days=1)
                time.sleep(0.5)
                continue
            data = json.loads(raw)
            records = data.get("data", data) if isinstance(data, dict) else data
            if isinstance(records, list) and records:
                all_records.extend(records)
                print(f"✓ {len(records)} rows")
            else:
                print(f"no data — response: {str(raw)[:120]}")
        except json.JSONDecodeError:
            print(f"JSON error — raw response: {result.stdout[:120]}")
        except Exception as e:
            print(f"ERROR: {e}")

        start = end + timedelta(days=1)
        time.sleep(0.5)

    if not all_records:
        return pd.DataFrame()
    return _clean_df(pd.DataFrame(all_records))


# ─────────────────────────────────────────────────────────────────────────────
# OPTION B — Manual CSV concatenation
# ─────────────────────────────────────────────────────────────────────────────

def load_from_csv_folder(folder: str) -> pd.DataFrame:
    """
    Concatenate all *.csv files from a folder (manually downloaded per year).
    """
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {folder}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df = _clean_df(df)
            print(f"    Loaded {f}: {len(df)} rows")
            dfs.append(df)
        except Exception as e:
            print(f"    Warning — could not parse {f}: {e}")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values("DATE").drop_duplicates(subset=["DATE"]).reset_index(drop=True)
    return combined


def load_from_single_csv(path: str) -> pd.DataFrame:
    """Load and clean a single CSV file."""
    df = pd.read_csv(path)
    return _clean_df(df)


# ─────────────────────────────────────────────────────────────────────────────
# OPTION C — requests (likely 403, but try anyway)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_with_requests(from_date: str, to_date: str) -> pd.DataFrame:
    import requests

    session = requests.Session()
    session.headers.update({
        "User-Agent":      UA,
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer":         "https://www.nseindia.com/reports-indices-historical-vix",
    })

    try:
        session.get("https://www.nseindia.com", timeout=15)
        time.sleep(2)
        session.get("https://www.nseindia.com/reports-indices-historical-vix", timeout=15)
        time.sleep(2)
    except Exception as e:
        print(f"  Session setup: {e}")

    from_dt = datetime.strptime(from_date, "%Y-%m-%d")
    to_dt   = datetime.strptime(to_date,   "%Y-%m-%d")

    all_records = []
    start = from_dt
    while start <= to_dt:
        # NSE's VIX endpoint often truncates long ranges to ~70 trading rows,
        # so use ~90-day chunks for the requests path too.
        end = min(start + timedelta(days=89), to_dt)
        url = f"{NSE_API}?from={fmt(start)}&to={fmt(end)}"
        print(f"    {fmt(start)} → {fmt(end)} …", end=" ", flush=True)
        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            records = data.get("data", data) if isinstance(data, dict) else data
            if records:
                all_records.extend(records)
                print(f"✓ {len(records)} rows")
            else:
                print("empty")
        except Exception as e:
            print(f"FAILED: {e}")
        start = end + timedelta(days=1)
        time.sleep(1.5)

    if not all_records:
        return pd.DataFrame()
    return _clean_df(pd.DataFrame(all_records))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch NSE India VIX historical data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--from_date",   default=DEFAULT_FROM, help="YYYY-MM-DD")
    parser.add_argument("--to_date",     default=DEFAULT_TO,   help="YYYY-MM-DD")
    parser.add_argument("--nsit",         default=None,
                        help="Value of 'nsit' cookie from your browser (required for Option A)")
    parser.add_argument("--nseappid",     default=None,
                        help="Value of 'nseappid' cookie (optional, use if you have it)")
    parser.add_argument("--extra_cookies",default=None,
                        help="Any other cookies as a string, e.g. 'bm_sv=...; RT=...'")
    parser.add_argument("--from_csv",     default=None,
                        help="Folder containing manually downloaded CSVs (Option B)")
    parser.add_argument("--single_csv",   default=None,
                        help="Single pre-downloaded CSV file (Option B)")
    parser.add_argument("--method",       default="curl",
                        choices=["curl", "requests"],
                        help="API method when no CSV provided")
    args = parser.parse_args()

    print("=" * 60)
    print("NSE India VIX Downloader")
    print("=" * 60)
    os.makedirs("data", exist_ok=True)

    df_vix = pd.DataFrame()

    # ── Option B: load existing CSVs ─────────────────────────────────────────
    if args.single_csv:
        print(f"  Loading single CSV: {args.single_csv}")
        df_vix = load_from_single_csv(args.single_csv)

    elif args.from_csv:
        print(f"  Loading all CSVs from: {args.from_csv}")
        df_vix = load_from_csv_folder(args.from_csv)

    # ── Option A: curl with cookies ──────────────────────────────────────────
    elif args.nsit:
        print(f"  Fetching via curl with browser cookies ({args.from_date} → {args.to_date})")
        if not args.nseappid:
            print("  (nseappid not provided — using nsit only)")
        df_vix = fetch_with_curl(
            args.from_date, args.to_date,
            nsit=args.nsit,
            nseappid=args.nseappid or "",
            extra_cookies=args.extra_cookies or "",
        )

    # ── Option C: direct requests ─────────────────────────────────────────────
    else:
        print(f"  No cookies provided — trying requests method …")
        print("  NOTE: May fail with 403.")
        df_vix = fetch_with_requests(args.from_date, args.to_date)

    # ── Save ─────────────────────────────────────────────────────────────────
    if df_vix.empty:
        print("""
❌  No VIX data obtained.

MANUAL STEPS TO GET VIX DATA:
─────────────────────────────
1. Open: https://www.nseindia.com/reports-indices-historical-vix
2. Select 'Custom', pick a 1-year range, click GO, then Download (.csv)
3. Save each year's file to: data/vix_raw/vix_<year>.csv
4. Run: python fetch_vix.py --from_csv data/vix_raw/

─── OR ──────────────────────────────────────────────────────────────────
To use your browser cookies (Chrome):
  1. Open NSE VIX page in Chrome
  2. Press F12 → Application tab → Cookies → nseindia.com
  3. Copy the values of 'nsit' and 'nseappid'
  4. Run:
     python fetch_vix.py --nsit <NSIT_COOKIE> --nseappid <NSEAPPID_COOKIE>
        """)
        sys.exit(1)

    _report(df_vix)
    df_vix.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅  Saved {len(df_vix)} rows → {OUTPUT_PATH}")
    print("\nSample:")
    print(df_vix.head(5).to_string(index=False))
