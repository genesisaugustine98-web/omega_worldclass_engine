from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.errors import ProviderError
from omega.local_import import ImportSchema, import_history_file
from omega.runtime import RuntimePaths

# Stooq intraday interval codes map directly to minutes.
SUPPORTED_INTERVALS = {1, 5, 15, 30, 60}

# Common FX pair symbols on Stooq are lowercase without separators.
INSTRUMENT_SYMBOLS = {
    "EUR_USD": "eurusd",
    "GBP_USD": "gbpusd",
    "USD_JPY": "usdjpy",
    "USD_CHF": "usdchf",
    "AUD_USD": "audusd",
    "NZD_USD": "nzdusd",
    "USD_CAD": "usdcad",
}

HTML_SNIFF = b"<!DOCTYPE html"


def stooq_url(symbol: str, interval: int, start: str, end: str) -> str:
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"stooq interval {interval} unsupported; choose one of {sorted(SUPPORTED_INTERVALS)}")
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    query = urllib.parse.urlencode(
        {
            "s": symbol,
            "i": interval,
            "d1": start_dt.strftime("%Y%m%d"),
            "d2": end_dt.strftime("%Y%m%d"),
        }
    )
    return f"https://stooq.com/q/d/l/?{query}"


def fetch_stooq_csv(url: str, timeout: float = 30.0) -> str:
    """Download the stooq CSV response with a bounded timeout.

    Stooq currently answers automated requests with a JavaScript bot-wall
    (an HTML challenge page, not CSV). This is detected and surfaced as a
    ProviderError so the caller can switch to the browser-download path
    instead of silently importing HTML.
    """
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    if raw.lstrip().startswith(HTML_SNIFF):
        raise ProviderError(
            "stooq.com answered with a JavaScript bot-wall instead of data. "
            "Automated bulk download is blocked; download the CSV manually in a "
            "browser (this script's --url-only mode prints the exact URL) and "
            "re-run with the file path argument."
        )
    return raw.decode("utf-8")


def parse_stooq_csv(text: str) -> pd.DataFrame:
    """Convert a stooq intraday CSV into the canonical OHLCV import frame.

    Stooq emits a Date column (YYYY-MM-DD) and a separate Time column
    (HH:MM:SS) in exchange-local time. They are combined into a single naive
    timestamp; callers must pass an explicit IANA timezone when importing so
    validation can verify the FX calendar. Volume is retained even when zero so
    the all-NaN feature guard can drop it deterministically.
    """
    frame = pd.read_csv(pd.io.common.StringIO(text))
    required = {"Date", "Time", "Open", "High", "Low", "Close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"stooq CSV missing columns: {missing}")
    timestamp = frame["Date"].astype(str) + " " + frame["Time"].astype(str)
    out = pd.DataFrame(
        {
            "timestamp": timestamp,
            "open": pd.to_numeric(frame["Open"], errors="raise"),
            "high": pd.to_numeric(frame["High"], errors="raise"),
            "low": pd.to_numeric(frame["Low"], errors="raise"),
            "close": pd.to_numeric(frame["Close"], errors="raise"),
        }
    )
    if "Volume" in frame.columns:
        out["volume"] = pd.to_numeric(frame["Volume"], errors="raise")
    return out


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description=(
            "Import Stooq M30 FX history through the provenance pipeline. "
            "Stooq blocks automated downloads with a bot-wall, so the flow is: "
            "print the URL (--url-only), download it in a browser, then pass the "
            "saved CSV file as the positional argument to import it."
        )
    )
    command.add_argument("path", nargs="?", type=Path,
                         help="Local stooq CSV already downloaded in a browser")
    command.add_argument("--instrument", default="EUR_USD",
                         help="Canonical instrument; maps to a stooq symbol unless --symbol is given")
    command.add_argument("--symbol", help="Explicit stooq symbol (lowercase, no separator), e.g. eurusd")
    command.add_argument("--interval", type=int, default=30, choices=sorted(SUPPORTED_INTERVALS),
                         help="Stooq intraday interval in minutes")
    command.add_argument("--start", required=True, help="Inclusive start date, e.g. 2024-01-01")
    command.add_argument("--end", required=True, help="Inclusive end date, e.g. 2024-06-01")
    command.add_argument("--timezone", required=True,
                         help="IANA timezone of stooq timestamps (exchange-local), e.g. UTC or Europe/Warsaw")
    command.add_argument("--source", default="stooq_v1",
                         help="Stable source identifier; change it to keep distinct raw versions immutable")
    command.add_argument("--data-root", type=Path)
    command.add_argument("--url-only", action="store_true",
                         help="Print the download URL and exit without fetching")
    command.add_argument("--execute", action="store_true",
                         help="Attempt the automated download; the browser-download path is recommended")
    command.add_argument("--accept-provider-terms", action="store_true",
                         help="Confirm you have reviewed stooq.com's terms of use for this download")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)

    symbol = args.symbol or INSTRUMENT_SYMBOLS.get(args.instrument)
    if not symbol:
        parser().error(
            f"no default stooq symbol for {args.instrument}; provide --symbol "
            f"(known: {sorted(INSTRUMENT_SYMBOLS)})"
        )

    url = stooq_url(symbol, args.interval, args.start, args.end)
    report = {"url": url, "symbol": symbol, "instrument": args.instrument, "interval_minutes": args.interval}

    if args.url_only:
        print(url)
        return 0

    if args.path is None:
        if not args.execute:
            print(json.dumps(report, indent=2))
            print("Open the URL above in a browser to download the CSV, then re-run with that file path.")
            return 0
        if not args.accept_provider_terms:
            parser().error("--execute requires --accept-provider-terms (review stooq.com terms first)")
        text = fetch_stooq_csv(url)
    else:
        path = Path(args.path).resolve()
        if not path.is_file():
            parser().error(f"CSV file not found: {path}")
        text = path.read_text(encoding="utf-8")
        report["mode"] = "local_file"

    frame = parse_stooq_csv(text)
    report["rows_downloaded"] = len(frame)
    if frame.empty:
        print(json.dumps(report, indent=2))
        print("No rows returned by stooq for the requested window.")
        return 1

    data_root = args.data_root or RuntimePaths.detect(ROOT).data_root
    result = import_history_file(
        args.path if args.path is not None else _write_temp_csv(frame),
        data_root=data_root,
        source=args.source,
        instrument=args.instrument,
        schema=ImportSchema(
            timestamp="timestamp",
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume" if "volume" in frame.columns else None,
            timezone=args.timezone,
            timestamp_format="%Y-%m-%d %H:%M:%S",
        ),
        require_spread=False,
    )
    report.update(result)
    print(json.dumps(report, indent=2, default=str))
    return 0


def _write_temp_csv(frame: pd.DataFrame) -> Path:
    import tempfile
    temporary = tempfile.NamedTemporaryFile(prefix="omega-stooq-", suffix=".csv", delete=False)
    path = Path(temporary.name)
    temporary.close()
    frame.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
