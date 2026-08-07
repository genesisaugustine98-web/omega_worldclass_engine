from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd


@dataclass(frozen=True)
class PartitionRequest:
    instrument: str
    start: datetime
    end: datetime
    granularity: str = "M30"
    price: str = "MBA"

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("Partition boundaries must be timezone-aware")
        if self.end <= self.start:
            raise ValueError("Partition end must be after start")
        if self.granularity != "M30":
            raise ValueError("V1 supports M30 partitions only")

    @property
    def key(self) -> str:
        start = self.start.astimezone(timezone.utc)
        return f"{self.instrument}/{start:%Y/%m}"


class HistoricalDataProvider(ABC):
    name: str

    @abstractmethod
    def fetch(self, request: PartitionRequest) -> tuple[bytes, pd.DataFrame, dict]:
        """Return immutable raw bytes, normalized bars, and provenance metadata."""