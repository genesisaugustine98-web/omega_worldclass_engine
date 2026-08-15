from __future__ import annotations

import json
import os
import socket
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .errors import IntegrityError, OperationalError
from .utils import atomic_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StageLedger:
    """Durable stage state stored beside cloud artifacts for restartable runs.

    Locks carry ownership metadata (host, pid, acquired-at) and a stale-lock
    budget: if a lock is older than ``stale_after_seconds`` it is presumed
    abandoned by a crashed process and reclaimed, so a dead run can never
    permanently deadlock the ledger.
    """

    def __init__(self, path: str | Path, run_id: str, stale_after_seconds: float = 120.0):
        self.path = Path(path)
        self.run_id = run_id
        self.stale_after_seconds = stale_after_seconds
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def read(self) -> dict:
        if not self.path.exists():
            return {"schema_version": 1, "run_id": self.run_id, "stages": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise IntegrityError(f"Ledger file is corrupt (not valid JSON): {self.path}") from exc
        if payload.get("run_id") != self.run_id:
            raise IntegrityError("Ledger run_id does not match requested run")
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
        deadline = time.monotonic() + timeout_seconds
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        owner = {"host": socket.gethostname(), "pid": os.getpid(), "acquired_at": utc_now()}
        while True:
            self._reclaim_stale_lock()
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise OperationalError(
                        f"Could not acquire ledger lock {self.lock_path} within {timeout_seconds}s"
                    )
                time.sleep(0.1)
                continue
            try:
                with os.fdopen(fd, "w", encoding="ascii") as handle:
                    json.dump(owner, handle)
            except BaseException:
                self.lock_path.unlink(missing_ok=True)
                raise
            break
        try:
            yield
        finally:
            self._release_lock(owner)

    def _reclaim_stale_lock(self) -> None:
        if not self.lock_path.exists():
            return
        try:
            age_seconds = time.time() - self.lock_path.stat().st_mtime
        except OSError:
            return
        if age_seconds > self.stale_after_seconds:
            self.lock_path.unlink(missing_ok=True)

    def _release_lock(self, owner: dict) -> None:
        try:
            if not self.lock_path.exists():
                return
            try:
                recorded = json.loads(self.lock_path.read_text(encoding="ascii"))
            except (json.JSONDecodeError, OSError):
                self.lock_path.unlink(missing_ok=True)
                return
            if recorded.get("pid") == owner["pid"]:
                self.lock_path.unlink(missing_ok=True)
        except OSError:
            pass
