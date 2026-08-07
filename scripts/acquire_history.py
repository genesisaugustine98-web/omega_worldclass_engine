from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.acquisition import run_acquisition
from omega.cloud_config import load_cloud_config
from omega.secrets import load_platform_secrets


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Plan or execute bounded M30 history acquisition")
    command.add_argument("--config", type=Path, default=ROOT / "config" / "cloud_free.yaml")
    command.add_argument("--start", required=True, help="Inclusive UTC month boundary, e.g. 2024-01-01T00:00:00Z")
    command.add_argument("--end", required=True, help="Exclusive UTC month boundary, e.g. 2024-02-01T00:00:00Z")
    command.add_argument("--max-partitions", type=int, default=1)
    command.add_argument("--execute", action="store_true", help="Perform network and storage operations; default is dry-run")
    command.add_argument("--accept-provider-terms", action="store_true", help="Confirm terms review for this execution")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    load_platform_secrets()
    config = load_cloud_config(args.config)
    result = run_acquisition(
        config=config,
        project_root=ROOT,
        start=args.start,
        end=args.end,
        execute=args.execute,
        accept_provider_terms=args.accept_provider_terms,
        max_partitions=args.max_partitions,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())