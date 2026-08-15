from __future__ import annotations
import hashlib, json, logging, os, random, tempfile, time
from functools import wraps
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
    """Atomically write JSON, fsyncing the file and its directory so a crash
    never leaves a truncated manifest behind."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise

def get_logger(name="omega"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return logging.getLogger(name)

def retry(
    attempts: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    exceptions: tuple[type[BaseException], ...] = (OSError,),
    on_retry=None,
):
    """Exponential backoff with jitter for transient failures.

    The wrapped function is retried only on the given exception types; any
    other exception propagates immediately. The final failure is re-raised.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    attempt += 1
                    if attempt >= attempts:
                        raise
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    delay = delay * (0.5 + random.random() * 0.5)
                    if on_retry is not None:
                        on_retry(attempt, delay, exc)
                    time.sleep(delay)
        return wrapper

    return decorator
