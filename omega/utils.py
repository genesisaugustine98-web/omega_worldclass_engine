from __future__ import annotations
import hashlib, json, logging, os, random, tempfile
from pathlib import Path
import numpy as np

def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def atomic_json(path: str | Path, payload: dict) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        json.dump(payload, f, indent=2, default=str); temp = Path(f.name)
    temp.replace(path)

def get_logger(name="omega"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger(name)

# CONVERSATION_HOOK: Add structured JSON logs and optional redacted Telegram notifications.
