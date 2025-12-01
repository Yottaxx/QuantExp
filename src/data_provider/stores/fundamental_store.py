from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..core.config import DPConfig
from ..utils.code import normalize_code
from .paths import fundamental_path


@dataclass
class FundamentalFrame:
    df: pd.DataFrame
    schema_ver: int = 1


class FundamentalStore:
    """Per-code fundamental parquet store (written by FundamentalPipeline)."""

    def __init__(self, cfg: DPConfig, logger=None):
        self.cfg = cfg
        self.logger = logger

    def load_one(self, code: str) -> Optional[FundamentalFrame]:
        c = normalize_code(code)
        if not c:
            return None
        p = fundamental_path(self.cfg, c)
        if not os.path.exists(p) or os.path.getsize(p) < 256:
            return None
        try:
            df = pd.read_parquet(p)
        except Exception:
            return None
        if df is None or df.empty:
            return None
        df = df.copy()
        # require date columns
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "pub_date" in df.columns:
            df["pub_date"] = pd.to_datetime(df["pub_date"], errors="coerce")
        return FundamentalFrame(df=df, schema_ver=int(df.attrs.get("schema_ver", 1) if hasattr(df, "attrs") else 1))

    def to_pit(self, f: FundamentalFrame) -> pd.DataFrame:
        """Return PIT-aligned frame with merge_date (= pub_date or date+lag)."""
        df = f.df.copy()
        lag = int(self.cfg.get("FUND_FALLBACK_LAG_DAYS", 90) or 90)
        if "date" not in df.columns:
            return pd.DataFrame()
        df = df.dropna(subset=["date"]).sort_values("date")

        if "pub_date" in df.columns:
            merge_date = df["pub_date"].where(df["pub_date"].notna(), df["date"] + pd.Timedelta(days=lag))
        else:
            merge_date = df["date"] + pd.Timedelta(days=lag)

        out = df.drop(columns=["date", "pub_date"], errors="ignore").copy()
        out.insert(0, "date", pd.to_datetime(merge_date, errors="coerce"))
        out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return out
