from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.local_import import ImportSchema, import_history_file
from omega.runtime import RuntimePaths


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Validate and import user-supplied M30 FX history")
    command.add_argument("path", type=Path)
    command.add_argument("--source", required=True, help="Stable source/version identifier, e.g. broker_export_v1")
    command.add_argument("--instrument", default="EUR_USD")
    command.add_argument("--data-root", type=Path)
    command.add_argument("--timestamp-column", default="timestamp")
    command.add_argument("--open-column", default="open")
    command.add_argument("--high-column", default="high")
    command.add_argument("--low-column", default="low")
    command.add_argument("--close-column", default="close")
    command.add_argument("--spread-column")
    command.add_argument("--volume-column")
    command.add_argument("--timezone", help="Required IANA timezone for naive timestamps, e.g. UTC")
    command.add_argument("--timestamp-format", help="Explicit strptime format when automatic parsing is unsuitable")
    command.add_argument("--delimiter", default=",")
    command.add_argument("--require-spread", action="store_true")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    data_root = args.data_root or RuntimePaths.detect(ROOT).data_root
    schema = ImportSchema(
        timestamp=args.timestamp_column,
        open=args.open_column,
        high=args.high_column,
        low=args.low_column,
        close=args.close_column,
        spread=args.spread_column,
        volume=args.volume_column,
        timezone=args.timezone,
        timestamp_format=args.timestamp_format,
        delimiter=args.delimiter,
    )
    result = import_history_file(
        args.path,
        data_root=data_root,
        source=args.source,
        instrument=args.instrument,
        schema=schema,
        require_spread=args.require_spread,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())