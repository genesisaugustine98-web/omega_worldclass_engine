#!/usr/bin/env python3
"""Repair a broken scientific stack (numpy/scipy/sklearn) on managed hosts.

Kaggle's base image sometimes ships a numpy whose .py files and compiled
extensions come from different versions. Plain ``pip install`` sees the pinned
versions as already satisfied and leaves the mismatch in place, so later
scipy/sklearn imports fail with errors like::

    ImportError: cannot import name '_center' from 'numpy._core.umath'

This script detects the broken state in a subprocess and force-reinstalls the
pinned numpy/scipy pair to restore consistency. Run it after installing the
requirements, and restart the kernel if a broken stack was already imported.
"""

from __future__ import annotations

import subprocess
import sys

PINNED = {
    "numpy": "2.1.3",
    "scipy": "1.17.1",
}

HEALTH_PROBE = (
    "import numpy; import scipy; import sklearn; "
    "from sklearn.metrics import brier_score_loss; "
    "import numpy._core.strings"
)


def _healthy() -> bool:
    probe = subprocess.run(
        [sys.executable, "-c", HEALTH_PROBE],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def main() -> int:
    if _healthy():
        print("scientific stack healthy")
        return 0
    print("scientific stack broken - force-reinstalling pinned versions")
    install = [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-cache-dir"]
    for name, version in PINNED.items():
        install.append(f"{name}=={version}")
    subprocess.run(install, check=True)
    if _healthy():
        print("scientific stack repaired")
        return 0
    print("repair failed - restart the kernel and rerun setup")
    return 1


if __name__ == "__main__":
    sys.exit(main())
