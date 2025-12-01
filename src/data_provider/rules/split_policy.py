from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class SplitManifest:
    train_start: str
    train_end_excl: str
    val_start: str
    val_end_excl: str
    test_start: str
    test_end_excl: str
    last_date: str
    gap: int
    seq_len: int
    stride: int
    train_ratio: float
    val_ratio: float
    test_ratio: Optional[float]
    universe_asof: Optional[str] = None
    fingerprint: Optional[str] = None
    adjust: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

class SplitPolicy:
    """The ONLY date split logic."""

    @staticmethod
    def compute(unique_dates: np.ndarray, *, seq_len: int, stride: int, gap: int, train_ratio: float, val_ratio: float, test_ratio: Optional[float] = None,
                universe_asof: Optional[str] = None, fingerprint: Optional[str] = None, adjust: Optional[str] = None) -> SplitManifest:
        if unique_dates.size == 0:
            raise ValueError("Empty unique_dates")
        u = np.sort(pd.to_datetime(unique_dates))
        n = int(len(u))
        if n < 5:
            raise ValueError("Not enough dates")

        if test_ratio is not None:
            s = float(train_ratio) + float(val_ratio) + float(test_ratio)
            if s > 0:
                train_ratio, val_ratio, test_ratio = float(train_ratio)/s, float(val_ratio)/s, float(test_ratio)/s

        train_end_idx = int(n * float(train_ratio))
        val_end_idx = int(n * float(train_ratio + val_ratio))

        train_end_excl = pd.to_datetime(u[min(max(train_end_idx, 1)-1, n-1)])
        val_start = pd.to_datetime(u[min(train_end_idx + gap, n-1)])
        val_end_excl = pd.to_datetime(u[min(max(val_end_idx, train_end_idx+1)-1, n-1)])
        test_start = pd.to_datetime(u[min(val_end_idx + gap, n-1)])
        last_date = pd.to_datetime(u[-1])
        test_end_excl = last_date + pd.Timedelta(days=1)

        return SplitManifest(
            train_start=str(pd.to_datetime(u[0]).date()),
            train_end_excl=str(train_end_excl.date()),
            val_start=str(val_start.date()),
            val_end_excl=str(val_end_excl.date()),
            test_start=str(test_start.date()),
            test_end_excl=str(pd.to_datetime(test_end_excl).date()),
            last_date=str(last_date.date()),
            gap=int(gap),
            seq_len=int(seq_len),
            stride=int(stride),
            train_ratio=float(train_ratio),
            val_ratio=float(val_ratio),
            test_ratio=float(test_ratio) if test_ratio is not None else None,
            universe_asof=universe_asof,
            fingerprint=fingerprint,
            adjust=adjust,
        )
