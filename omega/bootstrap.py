from __future__ import annotations

import importlib
import os
import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from .runtime import RuntimePaths
from .secrets import load_platform_secrets


@dataclass(frozen=True)
class BootstrapReport:
    platform: str
    python: str
    project_root: str
    data_root: str
    run_root: str
    storage_free_bytes: int
    required_free_bytes: int
    telegram_configured: bool
    dependencies_ok: bool
    twelvedata_key_present: bool
    polygon_key_present: bool

    def as_dict(self) -> dict:
        return asdict(self)


def check_dependencies(packages: tuple[str, ...] = ("numpy", "pandas", "pyarrow", "yaml", "sklearn")) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for package in packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
    return not missing, missing


def check_storage(root: str | Path, required_free_bytes: int = 1_000_000_000) -> int:
    path = Path(root).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < required_free_bytes:
        raise RuntimeError(
            f"Insufficient storage at {path}: {free} free bytes, "
            f"{required_free_bytes} required"
        )
    return free


def bootstrap(project_root: str | Path | None = None, required_free_bytes: int = 1_000_000_000) -> BootstrapReport:
    paths = RuntimePaths.detect(project_root).ensure()
    secret_status = load_platform_secrets()
    dependencies_ok, missing = check_dependencies()
    if not dependencies_ok:
        raise RuntimeError(f"Missing dependencies: {', '.join(missing)}")
    free = check_storage(paths.data_root, required_free_bytes)
    return BootstrapReport(
        platform=paths.platform,
        python=platform.python_version(),
        project_root=str(paths.project_root),
        data_root=str(paths.data_root),
        run_root=str(paths.run_root),
        storage_free_bytes=free,
        required_free_bytes=required_free_bytes,
        telegram_configured=bool(secret_status["OMEGA_TELEGRAM_BOT_TOKEN"] and secret_status["OMEGA_TELEGRAM_CHAT_ID"]),
        dependencies_ok=dependencies_ok,
        twelvedata_key_present=bool(secret_status.get("OMEGA_TWELVEDATA_API_KEY")),
        polygon_key_present=bool(secret_status.get("OMEGA_POLYGON_API_KEY")),
    )


# CONVERSATION_HOOK: add provider-specific secret checks here when a licensed adapter is selected.
# TODO: add a Drive quota API check when the notebook is running with a mounted Drive.