from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .cloud_config import build_provider
from .partitions import PartitionOrchestrator, dataset_manifest, monthly_requests
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
    dataset = dataset_manifest(store, selected_provider.name, plan.instrument)
    return {
        "mode": "execute",
        "plan": plan.as_dict(),
        "run_id": run_id,
        "manifests": manifests,
        "dataset": dataset,
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