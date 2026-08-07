from __future__ import annotations

import json
import os
import socket
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .utils import atomic_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StageLedger:
    """Durable stage state stored beside cloud artifacts for restartable runs."""

    def __init__(self, path: str | Path, run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def read(self) -> dict:
        if not self.path.exists():
            return {"schema_version": 1, "run_id": self.run_id, "stages": {}}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("run_id") != self.run_id:
            raise ValueError("Ledger run_id does not match requested run")
        return payload

    def status(self, key: str) -> str | None:
        return self.read()["stages"].get(key, {}).get("status")

    def is_complete(self, key: str) -> bool:
        return self.status(key) == "complete"

    def update(self, key: str, status: str, **metadata) -> None:
        if status not in {"running", "complete", "failed", "skipped"}:
            raise ValueError(f"Unsupported stage status: {status}")
        with self._lock():
            payload = self.read()
            previous = payload["stages"].get(key, {})
            payload["stages"][key] = {
                **previous,
                **metadata,
                "status": status,
                "updated_at": utc_now(),
                "host": socket.gethostname(),
            }
            payload["updated_at"] = utc_now()
            atomic_json(self.path, payload)

    @contextmanager
    def stage(self, key: str, **metadata) -> Iterator[bool]:
        if self.is_complete(key):
            yield False
            return
        self.update(key, "running", **metadata)
        try:
            yield True
        except Exception as exc:
            self.update(key, "failed", error_type=type(exc).__name__, error=str(exc)[:2000])
            raise
        else:
            self.update(key, "complete")

    @contextmanager
    def _lock(self, timeout_seconds: float = 20.0) -> Iterator[None]:
        import time

        deadline = time.monotonic() + timeout_seconds
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Could not acquire ledger lock {self.lock_path}")
                time.sleep(0.1)
        try:
            yield
        finally:
            self.lock_path.unlink(missing_ok=True)
