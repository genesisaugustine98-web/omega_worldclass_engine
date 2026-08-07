from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.bootstrap import bootstrap
from omega.capacity import inspect_capacity


if __name__ == "__main__":
    report = bootstrap(ROOT)
    print(json.dumps({"bootstrap": report.as_dict(), "capacity": inspect_capacity(report.data_root).as_dict()}, indent=2))