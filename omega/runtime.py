from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROFILES = {
    "SMOKE": {"max_workers": 1, "memory_budget_mb": 512, "use_gpu": False},
    "FREE_CPU": {"max_workers": 1, "memory_budget_mb": 1800, "use_gpu": False},
    "FREE_GPU": {"max_workers": 1, "memory_budget_mb": 2400, "use_gpu": True},
    "FULL_RESEARCH": {"max_workers": 2, "memory_budget_mb": 6000, "use_gpu": True},
}


@dataclass(frozen=True)
class RuntimePaths:
    platform: str
    project_root: Path
    data_root: Path
    run_root: Path

    @classmethod
    def detect(cls, project_root: str | Path | None = None) -> "RuntimePaths":
        root = Path(project_root or os.getenv("OMEGA_PROJECT_ROOT", Path.cwd())).resolve()
        if "COLAB_RELEASE_TAG" in os.environ or Path("/content").exists():
            platform = "colab"
            default_data = Path("/content/drive/MyDrive/OMEGA_DATA_v2")
            default_runs = Path("/content/drive/MyDrive/OMEGA_RUNS_v2")
        elif "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/working").exists():
            platform = "kaggle"
            default_data = Path("/kaggle/working/omega_data")
            default_runs = Path("/kaggle/working/omega_runs")
        else:
            platform = "local"
            default_data = root / "artifacts" / "cloud_data"
            default_runs = root / "artifacts" / "cloud_runs"
        return cls(
            platform=platform,
            project_root=root,
            data_root=Path(os.getenv("OMEGA_DATA_ROOT", default_data)).expanduser(),
            run_root=Path(os.getenv("OMEGA_RUN_ROOT", default_runs)).expanduser(),
        )

    def ensure(self) -> "RuntimePaths":
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.run_root.mkdir(parents=True, exist_ok=True)
        probe = self.run_root / ".omega_write_probe"
        probe.write_text("ok", encoding="ascii")
        probe.unlink()
        return self


def get_profile(name: str) -> dict:
    key = name.upper()
    if key not in PROFILES:
        raise ValueError(f"Unknown runtime profile {name!r}; choose from {sorted(PROFILES)}")
    return dict(PROFILES[key])
