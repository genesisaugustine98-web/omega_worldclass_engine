from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from .utils import sha256_file


class FileStore:
    """Filesystem store usable on local disk, mounted Drive, or Kaggle working storage."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        clean = key.replace("\\", "/").lstrip("/")
        path = (self.root / clean).resolve()
        if self.root.resolve() not in path.parents and path != self.root.resolve():
            raise ValueError("Storage key escapes the configured root")
        return path

    def exists(self, key: str) -> bool:
        return self.path(key).exists()

    def read_json(self, key: str) -> dict:
        return json.loads(self.path(key).read_text(encoding="utf-8"))

    def write_json_atomic(self, key: str, payload: dict) -> Path:
        destination = self.path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as handle:
            json.dump(payload, handle, indent=2, default=str)
            temporary = Path(handle.name)
        temporary.replace(destination)
        return destination

    def put_immutable(self, source: str | Path, key: str) -> Path:
        source = Path(source)
        destination = self.path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256_file(source) != sha256_file(destination):
                raise FileExistsError(f"Immutable object exists with different content: {destination}")
            return destination
        shutil.copy2(source, destination)
        return destination
