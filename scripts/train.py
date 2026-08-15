from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.acquisition import load_dataset, refresh_dataset
from omega.cloud_config import load_cloud_config
from omega.config import load_config
from omega.pipeline import run_pipeline
from omega.secrets import load_platform_secrets
import pandas as pd


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Auto-refresh free data, load a validated panel, and run the training pipeline"
    )
    command.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "cloud_twelvedata.yaml",
        help="Cloud config that supplies data_source (provider/instrument/terms)",
    )
    command.add_argument(
        "--research-config",
        type=Path,
        default=ROOT / "config.yaml",
        help="Full research schema (features, labels, evaluation, models)",
    )
    command.add_argument("--start", help="Inclusive UTC month boundary; defaults to data_source.start")
    command.add_argument("--end", help="Exclusive UTC month boundary; defaults to data_source.end")
    command.add_argument("--dry-run", action="store_true", help="Report what a refresh would fetch without executing")
    command.add_argument("--max-partitions", type=int, default=None, help="Cap partitions fetched per refresh (default: all missing)")
    command.add_argument("--accept-provider-terms", action="store_true", help="Confirm terms review for this execution")
    command.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts" / "run")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    load_platform_secrets()
    cloud_config = load_cloud_config(args.config)
    research_config = load_config(args.research_config)
    research_config["data_source"] = cloud_config["data_source"]

    start = args.start or cloud_config["data_source"].get("start")
    end = args.end or cloud_config["data_source"].get("end")
    if not start or not end:
        raise SystemExit("Provide --start/--end or data_source.start/end in the config")

    if args.dry_run:
        refresh = {"dry_run": True, "config": str(args.config), "start": start, "end": end}
        summary = {"config": str(args.config), "research_config": str(args.research_config), "refresh": refresh}
        print(json.dumps(summary, indent=2, default=str))
        return 0

    refresh = refresh_dataset(
        config=cloud_config,
        project_root=ROOT,
        start=start,
        end=end,
        accept_provider_terms=args.accept_provider_terms,
        max_partitions=args.max_partitions,
    )

    panel = load_dataset(config=cloud_config, project_root=ROOT, start=start, end=end)
    result = run_pipeline(panel, research_config, artifact_dir=str(args.artifact_dir))

    pipeline_summary = (
        result.astype(object).where(result.notna(), None).to_dict(orient="records")
        if isinstance(result, pd.DataFrame)
        else result
    )
    summary = {
        "config": str(args.config),
        "refresh": refresh,
        "panel_rows": int(len(panel)),
        "panel_first": panel["timestamp"].min().isoformat(),
        "panel_last": panel["timestamp"].max().isoformat(),
        "artifact_dir": str(args.artifact_dir),
        "pipeline": pipeline_summary,
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
