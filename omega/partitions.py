from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .providers.base import HistoricalDataProvider, PartitionRequest
from .state import StageLedger, utc_now
from .storage import FileStore
from .utils import sha256_file
from .validation import validate


def monthly_requests(instrument: str, start: str, end: str, granularity="M30", price="MBA"):
    boundaries = pd.date_range(start=start, end=end, freq="MS", inclusive="left", tz="UTC")
    requested_end = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end).tz_convert("UTC")
    for boundary in boundaries:
        next_boundary = boundary + pd.offsets.MonthBegin(1)
        yield PartitionRequest(
            instrument=instrument,
            start=boundary.to_pydatetime(),
            end=min(next_boundary, requested_end).to_pydatetime(),
            granularity=granularity,
            price=price,
        )


class PartitionOrchestrator:
    def __init__(self, store: FileStore, ledger: StageLedger, provider: HistoricalDataProvider):
        self.store = store
        self.ledger = ledger
        self.provider = provider

    def acquire(self, request: PartitionRequest, require_spread: bool = True) -> dict:
        stage_key = f"download/{self.provider.name}/{request.key}"
        manifest_key = f"manifests/{self.provider.name}/{request.key}.json"
        with self.ledger.stage(stage_key, request=asdict(request)) as should_run:
            if not should_run:
                if not self.store.exists(manifest_key):
                    raise RuntimeError(f"Ledger says complete but manifest is missing: {manifest_key}")
                return self.store.read_json(manifest_key)

            raw, frame, provider_metadata = self.provider.fetch(request)
            checks = validate(frame, timeframe_minutes=30, require_spread=require_spread)
            raw_digest = hashlib.sha256(raw).hexdigest()
            raw_key = f"raw/{self.provider.name}/{request.key}/{raw_digest}.json"
            normalized_key = f"normalized/{self.provider.name}/{request.key}/bars.parquet"

            with tempfile.TemporaryDirectory() as temporary_dir:
                temporary = Path(temporary_dir)
                raw_path = temporary / "response.json"
                normalized_path = temporary / "bars.parquet"
                raw_path.write_bytes(raw)
                frame.to_parquet(normalized_path, index=False)
                self.store.put_immutable(raw_path, raw_key)
                destination = self.store.path(normalized_key)
                destination.parent.mkdir(parents=True, exist_ok=True)
                normalized_path.replace(destination)

            manifest = {
                "schema_version": 1,
                "created_at": utc_now(),
                "request": {
                    "instrument": request.instrument,
                    "start": request.start.isoformat(),
                    "end": request.end.isoformat(),
                    "granularity": request.granularity,
                    "price": request.price,
                },
                "provider": provider_metadata,
                "raw": {"key": raw_key, "sha256": raw_digest, "bytes": len(raw)},
                "normalized": {
                    "key": normalized_key,
                    "sha256": sha256_file(self.store.path(normalized_key)),
                    "rows": len(frame),
                    "first_timestamp": frame.timestamp.min().isoformat(),
                    "last_timestamp": frame.timestamp.max().isoformat(),
                },
                "validation": checks,
            }
            self.store.write_json_atomic(manifest_key, manifest)
            return manifest


def dataset_manifest(store: FileStore, provider: str, instrument: str) -> dict:
    manifest_root = store.path(f"manifests/{provider}/{instrument}")
    files = sorted(manifest_root.rglob("*.json")) if manifest_root.exists() else []
    partitions = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    leaves = [partition["normalized"]["sha256"] for partition in partitions]
    root_hash = hashlib.sha256("".join(leaves).encode("ascii")).hexdigest()
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "instrument": instrument,
        "partition_count": len(partitions),
        "partition_hashes": leaves,
        "dataset_root_sha256": root_hash,
    }
