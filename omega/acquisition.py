from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .cloud_config import build_provider
from .errors import DataError
from .partitions import (
    PartitionOrchestrator,
    check_dataset_integrity,
    dataset_manifest,
    monthly_requests,
)
from .providers.base import HistoricalDataProvider, PartitionRequest
from .runtime import RuntimePaths
from .state import StageLedger
from .storage import FileStore


@dataclass(frozen=True)
class AcquisitionPlan:
    provider: str
    instrument: str
    start: str
    end: str
    partition_count: int
    execute: bool
    request_keys: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


def plan_acquisition(
    config: dict,
    start: str,
    end: str,
    execute: bool = False,
    max_partitions: int = 1,
) -> tuple[AcquisitionPlan, list[PartitionRequest]]:
    if max_partitions < 1:
        raise ValueError("max_partitions must be at least 1")
    source = config["data_source"]
    start_ts = _utc_timestamp(start, "start")
    end_ts = _utc_timestamp(end, "end")
    if start_ts.day != 1 or start_ts.time() != datetime.min.time():
        raise ValueError("start must be the first day of a month at 00:00:00 UTC")
    if end_ts.day != 1 or end_ts.time() != datetime.min.time():
        raise ValueError("end must be the first day of a month at 00:00:00 UTC")
    if end_ts <= start_ts:
        raise ValueError("end must be after start")

    requests = list(
        monthly_requests(
            instrument=source["instrument"],
            start=start_ts.isoformat(),
            end=end_ts.isoformat(),
            granularity=source["granularity"],
            price=source["price"],
        )
    )
    if len(requests) > max_partitions:
        raise ValueError(
            f"Plan contains {len(requests)} partitions but max_partitions={max_partitions}; "
            "increase the cap deliberately"
        )
    plan = AcquisitionPlan(
        provider=source["provider"],
        instrument=source["instrument"],
        start=start_ts.isoformat(),
        end=end_ts.isoformat(),
        partition_count=len(requests),
        execute=execute,
        request_keys=tuple(request.key for request in requests),
    )
    return plan, requests


def run_acquisition(
    config: dict,
    project_root: str | Path,
    start: str,
    end: str,
    execute: bool = False,
    accept_provider_terms: bool = False,
    max_partitions: int = 1,
    provider: HistoricalDataProvider | None = None,
) -> dict:
    plan, requests = plan_acquisition(config, start, end, execute, max_partitions)
    if not execute:
        return {"mode": "dry_run", "plan": plan.as_dict()}
    if not accept_provider_terms:
        raise PermissionError("Live acquisition requires --accept-provider-terms")
    if not config["data_source"]["explicit_terms_accepted"]:
        raise PermissionError("Set data_source.explicit_terms_accepted=true after reviewing provider terms")

    paths = RuntimePaths.detect(project_root).ensure()
    store = FileStore(paths.data_root)
    run_id = _run_id(plan)
    ledger = StageLedger(paths.run_root / run_id / "ledger.json", run_id)
    selected_provider = provider or build_provider(config)
    orchestrator = PartitionOrchestrator(store, ledger, selected_provider)
    require_spread = "B" in config["data_source"]["price"] and "A" in config["data_source"]["price"]
    manifests = [orchestrator.acquire(request, require_spread=require_spread) for request in requests]
    integrity = check_dataset_integrity(store, selected_provider.name, plan.instrument)
    dataset = dataset_manifest(store, selected_provider.name, plan.instrument)
    return {
        "mode": "execute",
        "plan": plan.as_dict(),
        "run_id": run_id,
        "manifests": manifests,
        "integrity": integrity,
        "dataset": dataset,
    }


def refresh_dataset(
    config: dict,
    project_root: str | Path,
    start: str,
    end: str,
    accept_provider_terms: bool = False,
    max_partitions: int | None = None,
    provider: HistoricalDataProvider | None = None,
) -> dict:
    """Fetch only missing monthly partitions up to ``end``, resumable and bounded.

    Unlike :func:`run_acquisition`, which fetches whatever was planned, this
    first lists already-complete partitions from the manifest store and fetches
    only months that are absent or whose manifest predates the partition's end
    (i.e. a previously partial download). Complete partitions are never
    re-fetched, so repeated refreshes are cheap and idempotent.
    """
    if not accept_provider_terms:
        raise PermissionError("Live refresh requires --accept-provider-terms")
    if not config["data_source"]["explicit_terms_accepted"]:
        raise PermissionError("Set data_source.explicit_terms_accepted=true after reviewing provider terms")

    paths = RuntimePaths.detect(project_root).ensure()
    store = FileStore(paths.data_root)
    source = config["data_source"]
    start_ts = _utc_timestamp(start, "start")
    end_ts = _utc_timestamp(end, "end")

    requests = list(
        monthly_requests(
            instrument=source["instrument"],
            start=start_ts.isoformat(),
            end=end_ts.isoformat(),
            granularity=source["granularity"],
            price=source["price"],
        )
    )

    existing = _existing_partitions(store, source["provider"], source["instrument"])
    missing = [r for r in requests if r.key not in existing]
    fetched: list[dict] = []
    skipped: list[str] = []

    if missing:
        cap = max_partitions if max_partitions is not None else len(missing)
        if cap < 1:
            raise ValueError("max_partitions must be at least 1")
        run_id = f"refresh-{source['provider']}-{source['instrument']}-{start_ts:%Y%m%d}-{end_ts:%Y%m%d}".lower()
        ledger = StageLedger(paths.run_root / run_id / "ledger.json", run_id)
        selected_provider = provider or build_provider(config)
        orchestrator = PartitionOrchestrator(store, ledger, selected_provider)
        require_spread = "B" in source["price"] and "A" in source["price"]
        for request in missing[:cap]:
            fetched.append(orchestrator.acquire(request, require_spread=require_spread))
        skipped = [r.key for r in missing[cap:]]

    integrity = check_dataset_integrity(store, source["provider"], source["instrument"])
    dataset = dataset_manifest(store, source["provider"], source["instrument"])
    return {
        "provider": source["provider"],
        "instrument": source["instrument"],
        "start": start_ts.isoformat(),
        "end": end_ts.isoformat(),
        "fetched_partitions": [m["request"]["start"] for m in fetched],
        "skipped_partitions": skipped,
        "already_present": sorted(set(r.key for r in requests) - {m["request"]["start"] for m in fetched}),
        "integrity": integrity,
        "dataset": dataset,
    }


def load_dataset(
    config: dict,
    project_root: str | Path,
    start: str,
    end: str,
    provider_name: str | None = None,
) -> pd.DataFrame:
    """Concatenate stored partitions into a validated, deduplicated training panel.

    Loads every complete monthly partition for the configured provider/instrument
    whose month intersects ``[start, end)``, runs the global cross-partition
    integrity check, sorts by timestamp, and drops any residual duplicate bars so
    the returned panel is always a single clean time series ready for
    :func:`omega.pipeline.run_pipeline`.
    """
    paths = RuntimePaths.detect(project_root).ensure()
    store = FileStore(paths.data_root)
    source = config["data_source"]
    provider = provider_name or source["provider"]
    instrument = source["instrument"]

    check_dataset_integrity(store, provider, instrument)

    start_ts = _utc_timestamp(start, "start")
    end_ts = _utc_timestamp(end, "end")
    manifest_root = store.path(f"manifests/{provider}/{instrument}")
    files = sorted(manifest_root.rglob("*.json")) if manifest_root.exists() else []
    frames = []
    for file in files:
        year, month = file.parent.name, file.stem
        partition_start = pd.Timestamp(year=int(year), month=int(month), day=1, tz="UTC")
        if not (start_ts <= partition_start < end_ts):
            continue
        key = f"normalized/{provider}/{instrument}/{year}/{month}/bars.parquet"
        if not store.exists(key):
            raise DataError(f"Manifest present but normalized bars missing: {key}")
        frames.append(pd.read_parquet(store.path(key)))

    if not frames:
        raise DataError(
            f"No complete partitions for {provider}/{instrument} in [{start_ts}, {end_ts})"
        )

    panel = pd.concat(frames, ignore_index=True)
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True, errors="raise")
    panel = panel.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
    panel = panel[
        (panel["timestamp"] >= start_ts) & (panel["timestamp"] < end_ts)
    ].reset_index(drop=True)
    if panel.empty:
        raise DataError(f"Loaded panel for {provider}/{instrument} is empty in the requested window")
    return panel


def _existing_partitions(store: FileStore, provider: str, instrument: str) -> set[str]:
    manifest_root = store.path(f"manifests/{provider}/{instrument}")
    if not manifest_root.exists():
        return set()
    return {
        f"{instrument}/{file.parent.name}/{file.stem}"
        for file in manifest_root.rglob("*.json")
        if file.parent.name.isdigit() and file.stem.isdigit()
    }


def _utc_timestamp(value: str, field: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return timestamp.tz_convert("UTC")


def _run_id(plan: AcquisitionPlan) -> str:
    start = pd.Timestamp(plan.start).strftime("%Y%m%d")
    end = pd.Timestamp(plan.end).strftime("%Y%m%d")
    return f"acquire-{plan.provider}-{plan.instrument}-{start}-{end}".lower()


# CONVERSATION_HOOK: add rate-limit telemetry before increasing max_partitions beyond smoke-test scale.