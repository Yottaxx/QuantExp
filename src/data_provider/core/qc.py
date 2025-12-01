from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

def validate_price_frame(df: pd.DataFrame) -> Tuple[bool, str]:
    need = ["code", "date", "open", "high", "low", "close"]
    for c in need:
        if c not in df.columns:
            return False, f"MissingCol({c})"
    if df["code"].isna().any():
        return False, "BadCode(NaN)"
    if df["date"].isna().any():
        return False, "BadDate(NaT)"
    g = df.groupby("code", sort=False)
    if bool(g["date"].apply(lambda x: x.duplicated().any()).any()):
        return False, "DupDate"
    for c in ["open","high","low","close"]:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.isna().mean() > 0.1:
            return False, f"TooManyNaN({c})"
        if (s <= 0).mean() > 0.01:
            return False, f"NonPositive({c})"
    oh = np.maximum(df["open"].astype(float), df["close"].astype(float))
    ol = np.minimum(df["open"].astype(float), df["close"].astype(float))
    if ((df["high"].astype(float) + 1e-9) < oh).mean() > 0.001:
        return False, "OHLCHighInconsistent"
    if ((df["low"].astype(float) - 1e-9) > ol).mean() > 0.001:
        return False, "OHLCLowInconsistent"
    return True, "OK"

@dataclass
class RejectReport:
    counts: Dict[str, int] = field(default_factory=dict)
    samples: Dict[str, List[str]] = field(default_factory=dict)
    max_samples_per_reason: int = 30

    def add(self, reason: str, code: str) -> None:
        self.counts[reason] = self.counts.get(reason, 0) + 1
        lst = self.samples.setdefault(reason, [])
        if len(lst) < self.max_samples_per_reason:
            lst.append(code)

    def to_dict(self) -> Dict:
        return {"counts": dict(sorted(self.counts.items(), key=lambda x: (-x[1], x[0]))), "samples": self.samples}
