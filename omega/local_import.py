from __future__ import annotations

import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .partitions import dataset_manifest
from .storage import FileStore
from .utils import sha256_file
from .validation import validate


SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class ImportSchema:
    timestamp: str = "timestamp"
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    spread: str | None = None
    volume: str | None = None
    timezone: str | None = None
    timestamp_format: str | None = None
    delimiter: str = ","

    def column_map(self) -> dict[str, str]:
        pairs = [
            (self.timestamp, "timestamp"),
            (self.open, "open"),
            (self.high, "high"),
            (self.low, "low"),
            (self.close, "close"),
        ]
        if self.spread:
            pairs.append((self.spread, "spread"))
        if self.volume:
            pairs.append((self.volume, "volume"))
        source_columns = [source for source, _ in pairs]
        if len(source_columns) != len(set(source_columns)):
            raise ValueError("Each source column may map to only one canonical column")
        return dict(pairs)


def import_history_file(
    source_path: str | Path,
    data_root: str | Path,
    source: str,
    instrument: str,
    schema: ImportSchema,
    require_spread: bool = False,
) -> dict:
    source_path = Path(source_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    _validate_identifier(source, "source")
    _validate_identifier(instrument, "instrument")

    store = FileStore(data_root)
    raw_digest = sha256_file(source_path)
    raw_key = f"raw/local/{source}/{instrument}/{raw_digest}{source_path.suffix.lower()}"
    frame = read_local_history(source_path, schema)
    if frame.empty:
        raise ValueError("Source file contains no rows")

    provider_name = f"local-{source}"
    with tempfile.TemporaryDirectory(prefix="omega-local-import-") as temporary_dir:
        prepared = _prepare_partitions(
            frame, Path(temporary_dir), provider_name, instrument, raw_digest, require_spread
        )
        _preflight_conflicts(store, prepared)
        store.put_immutable(source_path, raw_key)

        raw_manifest_key = f"raw/local/{source}/{instrument}/{raw_digest}.manifest.json"
        if not store.exists(raw_manifest_key):
            store.write_json_atomic(
                raw_manifest_key,
                {
                    "schema_version": 1,
                    "source": source,
                    "instrument": instrument,
                    "original_filename": source_path.name,
                    "imported_at": datetime.now(timezone.utc).isoformat(),
                    "raw": {"key": raw_key, "sha256": raw_digest, "bytes": source_path.stat().st_size},
                    "schema": asdict(schema),
                },
            )

        manifests = []
        for item in prepared:
            store.put_immutable(item["temporary_path"], item["normalized_key"])
            if store.exists(item["manifest_key"]):
                manifest = store.read_json(item["manifest_key"])
            else:
                manifest = item["manifest"]
                manifest["raw"] = {"key": raw_key, "sha256": raw_digest}
                store.write_json_atomic(item["manifest_key"], manifest)
            manifests.append(manifest)

    return {
        "mode": "local_import",
        "source": source,
        "instrument": instrument,
        "raw_sha256": raw_digest,
        "rows": len(frame),
        "partition_count": len(manifests),
        "manifests": manifests,
        "dataset": dataset_manifest(store, provider_name, instrument),
    }


def read_local_history(path: str | Path, schema: ImportSchema) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        source_frame = pd.read_parquet(path)
    elif suffix in {".csv", ".txt"}:
        source_frame = pd.read_csv(path, sep=schema.delimiter)
    else:
        raise ValueError("Supported local formats are CSV, TXT, Parquet, and PQ")

    mapping = schema.column_map()
    missing = sorted(set(mapping) - set(source_frame.columns))
    if missing:
        raise ValueError(f"Missing mapped source columns: {missing}")
    frame = source_frame[list(mapping)].rename(columns=mapping).copy()
    frame["timestamp"] = _parse_timestamps(frame["timestamp"], schema)
    for column in ("open", "high", "low", "close", "spread", "volume"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame.sort_values("timestamp", kind="stable").reset_index(drop=True)


def _parse_timestamps(values: pd.Series, schema: ImportSchema) -> pd.Series:
    parsed = pd.to_datetime(values, format=schema.timestamp_format, errors="raise")
    timezone_name = getattr(parsed.dt, "tz", None)
    if timezone_name is None:
        if not schema.timezone:
            raise ValueError("Naive source timestamps require an explicit --timezone")
        parsed = parsed.dt.tz_localize(schema.timezone, ambiguous="raise", nonexistent="raise")
    elif schema.timezone:
        raise ValueError("Do not provide --timezone when source timestamps already include offsets")
    return parsed.dt.tz_convert("UTC")


def _prepare_partitions(frame, temporary_root, provider_name, instrument, raw_digest, require_spread):
    prepared = []
    period_key = frame["timestamp"].dt.strftime("%Y/%m")
    for key, partition in frame.groupby(period_key, sort=True):
        partition = partition.reset_index(drop=True)
        checks = validate(partition, timeframe_minutes=30, require_spread=require_spread)
        normalized_key = f"normalized/{provider_name}/{instrument}/{key}/bars.parquet"
        manifest_key = f"manifests/{provider_name}/{instrument}/{key}.json"
        temporary_path = temporary_root / key.replace("/", "-")
        temporary_path = temporary_path.with_suffix(".parquet")
        partition.to_parquet(temporary_path, index=False)
        normalized_digest = sha256_file(temporary_path)
        prepared.append(
            {
                "temporary_path": temporary_path,
                "normalized_key": normalized_key,
                "manifest_key": manifest_key,
                "manifest": {
                    "schema_version": 1,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "request": {
                        "instrument": instrument,
                        "month": key,
                        "granularity": "M30",
                    },
                    "provider": {"name": provider_name, "kind": "user_supplied_local_file"},
                    "normalized": {
                        "key": normalized_key,
                        "sha256": normalized_digest,
                        "rows": len(partition),
                        "first_timestamp": partition.timestamp.min().isoformat(),
                        "last_timestamp": partition.timestamp.max().isoformat(),
                    },
                    "validation": checks,
                    "source_file_sha256": raw_digest,
                },
            }
        )
    return prepared


def _preflight_conflicts(store: FileStore, prepared: list[dict]) -> None:
    for item in prepared:
        destination = store.path(item["normalized_key"])
        if destination.exists() and sha256_file(destination) != item["manifest"]["normalized"]["sha256"]:
            raise FileExistsError(
                f"Partition already exists with different content: {item['normalized_key']}. "
                "Use a new source identifier to preserve both versions."
            )
        if store.exists(item["manifest_key"]):
            existing = store.read_json(item["manifest_key"])
            if existing.get("source_file_sha256") != item["manifest"]["source_file_sha256"]:
                raise FileExistsError(f"Partition manifest belongs to a different raw file: {item['manifest_key']}")


def _validate_identifier(value: str, field: str) -> None:
    if not SAFE_IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field} must match {SAFE_IDENTIFIER.pattern}")


# CONVERSATION_HOOK: add audited presets only after a specific vendor export format is selected.