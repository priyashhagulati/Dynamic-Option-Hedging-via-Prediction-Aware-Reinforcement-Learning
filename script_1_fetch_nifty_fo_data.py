#!/usr/bin/env python3
"""
Fetch/prepare NIFTY futures and options contract-wise historical data from NSE.

The output schema matches data/fobhav_nifty_nse_full.csv, which is the raw input
used by bs_baseline.py before feature generation.

Examples
--------
Direct NSE API, appending NIFTY options data after 2020:
  python fetch_nifty_fo_data.py --from 2020-09-01 --to 2024-12-31 --append

Include futures too, only if you need the existing bs_baseline.py futures-based
spot proxy:
  python fetch_nifty_fo_data.py --from 2020-09-01 --to 2024-12-31 --append --include-futures

If NSE blocks requests, use browser cookies:
  python fetch_nifty_fo_data.py --from 2020-09-01 --to 2024-12-31 --append \
    --cookie "nsit=...; nseappid=..."

Manual CSV fallback:
  1. Download CSV files from https://www.nseindia.com/report-detail/fo_eq_security
  2. Save them under data/nse_fo_raw/
  3. Run:
     python fetch_nifty_fo_data.py --from-csv data/nse_fo_raw --append
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode

import pandas as pd
import requests


NSE_HOME = "https://www.nseindia.com"
NSE_REPORT_PAGE = "https://www.nseindia.com/report-detail/fo_eq_security"
NSE_API = "https://www.nseindia.com/api/historicalOR/foCPV"
DEFAULT_OUTPUT = Path("data/fobhav_nifty_nse_full.csv")
PROJECT_RAW = Path("data/fobhav_nifty.csv")
DEFAULT_RAW_DIR = Path("data/nse_fo_chunks")

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

TARGET_COLUMNS = [
    "INSTRUMENT",
    "SYMBOL",
    "EXPIRY_DT",
    "STRIKE_PR",
    "OPTION_TYP",
    "OPEN",
    "HIGH",
    "LOW",
    "CLOSE",
    "SETTLE_PR",
    "CONTRACTS",
    "VAL_INLAKH",
    "OPEN_INT",
    "CHG_IN_OI",
    "TIMESTAMP",
]

RENAME = {
    # Existing project/raw bhavcopy names.
    "INSTRUMENT": "INSTRUMENT",
    "SYMBOL": "SYMBOL",
    "EXPIRY_DT": "EXPIRY_DT",
    "STRIKE_PR": "STRIKE_PR",
    "OPTION_TYP": "OPTION_TYP",
    "OPEN": "OPEN",
    "HIGH": "HIGH",
    "LOW": "LOW",
    "CLOSE": "CLOSE",
    "SETTLE_PR": "SETTLE_PR",
    "CONTRACTS": "CONTRACTS",
    "VAL_INLAKH": "VAL_INLAKH",
    "OPEN_INT": "OPEN_INT",
    "CHG_IN_OI": "CHG_IN_OI",
    "TIMESTAMP": "TIMESTAMP",
    # NSE web-table names.
    "OPTION TYPE": "OPTION_TYP",
    "STRIKE PRICE": "STRIKE_PR",
    "SETTLE PRICE": "SETTLE_PR",
    "NO. OF CONTRACTS": "CONTRACTS",
    "TURNOVER * IN ₹ LAKHS": "VAL_INLAKH",
    "TURNOVER * IN RS. LAKHS": "VAL_INLAKH",
    "TURNOVER IN ₹ LAKHS": "VAL_INLAKH",
    "PREMIUM TURNOVER ** IN ₹ LAKHS": "PREMIUM_VAL_INLAKH",
    "OPEN INT": "OPEN_INT",
    "CHANGE IN OI": "CHG_IN_OI",
    "UNDERLYING VALUE": "UNDERLYING_VALUE",
    "DATE": "TIMESTAMP",
    "EXPIRY": "EXPIRY_DT",
    "LTP": "LTP",
    # NSE API names commonly returned by /api/historical/fo/derivatives.
    "FH_INSTRUMENT": "INSTRUMENT",
    "FH_SYMBOL": "SYMBOL",
    "FH_EXPIRY_DT": "EXPIRY_DT",
    "FH_STRIKE_PRICE": "STRIKE_PR",
    "FH_OPTION_TYPE": "OPTION_TYP",
    "FH_OPENING_PRICE": "OPEN",
    "FH_TRADE_HIGH_PRICE": "HIGH",
    "FH_TRADE_LOW_PRICE": "LOW",
    "FH_CLOSING_PRICE": "CLOSE",
    "FH_LAST_TRADED_PRICE": "LTP",
    "FH_SETTLE_PRICE": "SETTLE_PR",
    "FH_MARKET_LOT": "MARKET_LOT",
    "FH_TOT_TRADED_QTY": "CONTRACTS",
    "FH_TOT_TRADED_VAL": "VAL_INLAKH",
    "FH_OPEN_INT": "OPEN_INT",
    "FH_CHANGE_IN_OI": "CHG_IN_OI",
    "FH_TIMESTAMP": "TIMESTAMP",
    "FH_UNDERLYING_VALUE": "UNDERLYING_VALUE",
}


def parse_yyyy_mm_dd(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def nse_date(value: datetime) -> str:
    return value.strftime("%d-%m-%Y")


def chunks(start: datetime, end: datetime, days: int) -> Iterable[tuple[datetime, datetime]]:
    current = start
    while current <= end:
        year_end = datetime(current.year, 12, 31)
        chunk_end = min(current + timedelta(days=days - 1), year_end, end)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def clean_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.replace("-", "", regex=False),
        errors="coerce",
    )


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        " ".join(str(col).replace("\n", " ").replace("**", "").split()).strip().upper()
        for col in df.columns
    ]
    df = df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns})

    if "VAL_INLAKH" in df.columns:
        df["VAL_INLAKH"] = clean_number(df["VAL_INLAKH"])
        # API often returns rupees. Old bhavcopy files use Rs. lakhs.
        if df["VAL_INLAKH"].dropna().median() > 10_000_000:
            df["VAL_INLAKH"] = df["VAL_INLAKH"] / 100_000.0

    if "CONTRACTS" in df.columns:
        df["CONTRACTS"] = clean_number(df["CONTRACTS"])
    if "MARKET_LOT" in df.columns:
        df["MARKET_LOT"] = clean_number(df["MARKET_LOT"])
    # The current NSE API returns total traded quantity; the old bhavcopy column
    # used by this project is number of contracts. Convert when market lot exists.
    if "CONTRACTS" in df.columns and "MARKET_LOT" in df.columns:
        mask = df["MARKET_LOT"].notna() & df["MARKET_LOT"].gt(0)
        df.loc[mask, "CONTRACTS"] = df.loc[mask, "CONTRACTS"] / df.loc[mask, "MARKET_LOT"]

    for col in ["STRIKE_PR", "OPEN", "HIGH", "LOW", "CLOSE", "SETTLE_PR", "OPEN_INT", "CHG_IN_OI"]:
        if col in df.columns:
            df[col] = clean_number(df[col])

    if "OPTION_TYP" not in df.columns:
        df["OPTION_TYP"] = "XX"
    df["OPTION_TYP"] = df["OPTION_TYP"].fillna("XX").replace({"-": "XX", "": "XX"})

    if "STRIKE_PR" not in df.columns:
        df["STRIKE_PR"] = 0.0
    df["STRIKE_PR"] = df["STRIKE_PR"].fillna(0.0)

    if "INSTRUMENT" in df.columns:
        df["INSTRUMENT"] = df["INSTRUMENT"].astype(str).str.upper().str.strip()
    if "SYMBOL" in df.columns:
        df["SYMBOL"] = df["SYMBOL"].astype(str).str.upper().str.strip()

    for col in ["EXPIRY_DT", "TIMESTAMP"]:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
            df[col] = parsed.dt.strftime("%d-%b-%Y")

    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[TARGET_COLUMNS].copy()
    df = df[df["SYMBOL"].eq("NIFTY")]
    df = df[df["INSTRUMENT"].isin(["FUTIDX", "OPTIDX"])]
    df = df.dropna(subset=["TIMESTAMP", "EXPIRY_DT", "CLOSE"])
    return df


def make_session(cookie: str = "") -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": UA,
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": NSE_REPORT_PAGE,
            "X-Requested-With": "XMLHttpRequest",
        }
    )
    if cookie:
        session.headers.update({"Cookie": cookie})

    # Warm up cookies. NSE frequently requires this before API calls.
    session.get(NSE_HOME, timeout=20)
    session.get(NSE_REPORT_PAGE, timeout=20)
    return session


def records_from_response(resp: requests.Response) -> list[dict]:
    text = resp.text.strip()
    if not text:
        return []
    try:
        payload = resp.json()
    except json.JSONDecodeError:
        # Some NSE responses are direct CSV downloads.
        if "," in text and "\n" in text:
            return pd.read_csv(io.StringIO(text)).to_dict("records")
        raise RuntimeError(f"Non-JSON response {resp.status_code}: {text[:200]}")

    if isinstance(payload, dict):
        for key in ["data", "records", "result"]:
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return []
    if isinstance(payload, list):
        return payload
    return []


def get_with_retries(session: requests.Session, url: str, attempts: int = 5) -> requests.Response:
    last_resp = None
    for attempt in range(1, attempts + 1):
        try:
            resp = session.get(url, timeout=60)
            last_resp = resp
            if resp.status_code < 500:
                return resp
            print(f"server {resp.status_code}, retry {attempt}/{attempts}", end=" ... ", flush=True)
        except requests.RequestException as exc:
            if attempt == attempts:
                raise
            print(f"{exc.__class__.__name__}, retry {attempt}/{attempts}", end=" ... ", flush=True)
        time.sleep(min(2 ** attempt, 30))
    return last_resp


def fetch_api(
    start: datetime,
    end: datetime,
    cookie: str,
    chunk_days: int,
    sleep: float,
    include_futures: bool,
) -> pd.DataFrame:
    session = make_session(cookie)
    all_records: list[dict] = []

    instruments = ["OPTIDX"]
    if include_futures:
        # Futures are useful only for the current bs_baseline.py spot proxy.
        instruments.append("FUTIDX")
    for instrument in instruments:
        for left, right in chunks(start, end, chunk_days):
            params = {
                "from": nse_date(left),
                "to": nse_date(right),
                "instrumentType": instrument,
                "symbol": "NIFTY",
                "year": str(left.year),
                "csv": "true",
            }
            url = f"{NSE_API}?{urlencode(params)}"
            print(f"  {instrument}: {nse_date(left)} -> {nse_date(right)}", end=" ... ", flush=True)

            resp = get_with_retries(session, url)
            if resp.status_code in {401, 403}:
                raise RuntimeError(
                    f"NSE returned {resp.status_code}. Re-run with --cookie copied from your browser."
                )
            if resp.status_code >= 400:
                raise RuntimeError(f"NSE returned {resp.status_code}: {resp.text[:200]}")

            records = records_from_response(resp)
            print(f"{len(records)} rows")
            all_records.extend(records)
            time.sleep(sleep)

    if not all_records:
        return pd.DataFrame(columns=TARGET_COLUMNS)
    return standardize_columns(pd.DataFrame(all_records))


def fetch_api_to_chunks(
    start: datetime,
    end: datetime,
    cookie: str,
    chunk_days: int,
    sleep: float,
    include_futures: bool,
    raw_dir: Path,
) -> list[Path]:
    session = make_session(cookie)
    raw_dir.mkdir(parents=True, exist_ok=True)
    instruments = ["OPTIDX"]
    if include_futures:
        instruments.append("FUTIDX")

    written: list[Path] = []
    for instrument in instruments:
        for left, right in chunks(start, end, chunk_days):
            chunk_name = f"{instrument}_{left:%Y%m%d}_{right:%Y%m%d}.csv"
            chunk_path = raw_dir / chunk_name
            if chunk_path.exists() and chunk_path.stat().st_size > 0:
                print(f"  {instrument}: {nse_date(left)} -> {nse_date(right)} ... cached")
                written.append(chunk_path)
                continue

            params = {
                "from": nse_date(left),
                "to": nse_date(right),
                "instrumentType": instrument,
                "symbol": "NIFTY",
                "year": str(left.year),
                "csv": "true",
            }
            url = f"{NSE_API}?{urlencode(params)}"
            print(f"  {instrument}: {nse_date(left)} -> {nse_date(right)}", end=" ... ", flush=True)
            resp = get_with_retries(session, url)
            if resp.status_code in {401, 403}:
                raise RuntimeError(
                    f"NSE returned {resp.status_code}. Re-run with --cookie copied from your browser."
                )
            if resp.status_code >= 400:
                raise RuntimeError(f"NSE returned {resp.status_code}: {resp.text[:200]}")

            records = records_from_response(resp)
            frame = standardize_columns(pd.DataFrame(records)) if records else pd.DataFrame(columns=TARGET_COLUMNS)
            frame = dedupe_sort(frame)
            frame.to_csv(chunk_path, index=False)
            print(f"{len(frame)} rows -> {chunk_path}")
            written.append(chunk_path)
            time.sleep(sleep)

    return written


def load_csvs(path: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(path) / "*.csv"))) if Path(path).is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No CSV files found under {path}")

    frames = []
    for file in files:
        raw = pd.read_csv(file)
        frame = standardize_columns(raw)
        print(f"  loaded {file}: {len(frame):,} NIFTY rows")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=TARGET_COLUMNS)


def combine_chunk_files(files: list[Path]) -> pd.DataFrame:
    frames = []
    for file in files:
        frame = pd.read_csv(file, low_memory=False)
        frames.append(standardize_columns(frame))
    if not frames:
        return pd.DataFrame(columns=TARGET_COLUMNS)
    return dedupe_sort(pd.concat(frames, ignore_index=True))


def combine_existing(new_df: pd.DataFrame, existing_path: Path) -> pd.DataFrame:
    if not existing_path.exists():
        return new_df
    old_df = standardize_columns(pd.read_csv(existing_path))
    combined = pd.concat([old_df, new_df], ignore_index=True)
    return dedupe_sort(combined)


def dedupe_sort(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    sort_keys = ["TIMESTAMP", "EXPIRY_DT", "INSTRUMENT", "STRIKE_PR", "OPTION_TYP"]
    parsed_ts = pd.to_datetime(df["TIMESTAMP"], dayfirst=True, errors="coerce")
    parsed_exp = pd.to_datetime(df["EXPIRY_DT"], dayfirst=True, errors="coerce")
    df = df.assign(_TS=parsed_ts, _EXP=parsed_exp)
    df = df.sort_values(["_TS", "_EXP", "INSTRUMENT", "STRIKE_PR", "OPTION_TYP"])
    df = df.drop_duplicates(
        subset=["TIMESTAMP", "EXPIRY_DT", "INSTRUMENT", "SYMBOL", "STRIKE_PR", "OPTION_TYP"],
        keep="last",
    )
    df = df.drop(columns=["_TS", "_EXP"]).reset_index(drop=True)
    return df[TARGET_COLUMNS]


def report(df: pd.DataFrame, output: Path) -> None:
    if df.empty:
        print("No rows were written.")
        return
    dates = pd.to_datetime(df["TIMESTAMP"], dayfirst=True, errors="coerce")
    print(f"\nSaved: {output}")
    print(f"Rows : {len(df):,}")
    print(f"Dates: {dates.min().date()} -> {dates.max().date()}")
    print(df.groupby("INSTRUMENT").size().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NIFTY F&O contract-wise data from NSE.")
    parser.add_argument("--from", dest="from_date", default="2020-09-01", help="YYYY-MM-DD")
    parser.add_argument("--to", dest="to_date", default=datetime.today().strftime("%Y-%m-%d"), help="YYYY-MM-DD")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Directory for resumable chunk CSVs")
    parser.add_argument("--stream-chunks", action="store_true", help="Write each NSE request to a chunk CSV before combining")
    parser.add_argument("--append", action="store_true", help=f"Append/merge with {PROJECT_RAW}")
    parser.add_argument("--replace-project-raw", action="store_true", help=f"Write merged output to {PROJECT_RAW}")
    parser.add_argument("--from-csv", help="Folder or CSV downloaded manually from NSE")
    parser.add_argument("--cookie", default="", help='Browser cookie string, e.g. "nsit=...; nseappid=..."')
    parser.add_argument("--chunk-days", type=int, default=7, help="NSE request chunk size")
    parser.add_argument("--sleep", type=float, default=0.8, help="Seconds between NSE API requests")
    parser.add_argument(
        "--include-futures",
        action="store_true",
        help="Also fetch FUTIDX rows. By default only OPTIDX option rows are fetched.",
    )
    args = parser.parse_args()

    output = PROJECT_RAW if args.replace_project_raw else Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        new_df = load_csvs(args.from_csv)
    elif args.stream_chunks:
        chunk_files = fetch_api_to_chunks(
            start=parse_yyyy_mm_dd(args.from_date),
            end=parse_yyyy_mm_dd(args.to_date),
            cookie=args.cookie,
            chunk_days=args.chunk_days,
            sleep=args.sleep,
            include_futures=args.include_futures,
            raw_dir=Path(args.raw_dir),
        )
        new_df = combine_chunk_files(chunk_files)
    else:
        new_df = fetch_api(
            start=parse_yyyy_mm_dd(args.from_date),
            end=parse_yyyy_mm_dd(args.to_date),
            cookie=args.cookie,
            chunk_days=args.chunk_days,
            sleep=args.sleep,
            include_futures=args.include_futures,
        )

    new_df = dedupe_sort(new_df)
    final_df = combine_existing(new_df, PROJECT_RAW) if (args.append or args.replace_project_raw) else new_df
    final_df.to_csv(output, index=False)
    report(final_df, output)

    if output != PROJECT_RAW:
        print("\nNext step:")
        print(f"  Review {output}, then merge into {PROJECT_RAW} with:")
        print(f"  python fetch_nifty_fo_data.py --from-csv {output} --replace-project-raw")


if __name__ == "__main__":
    main()
