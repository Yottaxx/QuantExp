from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

_AK_RENAME = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover",
}

def normalize_ak_hist_df(raw: pd.DataFrame) -> pd.DataFrame:
    """AkShare raw -> canonical price schema. This MAY drop non-core columns by design."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.rename(columns=_AK_RENAME).copy()
    if "date" not in df.columns:
        df = df.reset_index().rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    keep = [c for c in ["date", "open", "high", "low", "close", "volume", "amount", "turnover"] if c in df.columns]
    df = df[keep].copy()

    for c in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32, copy=False)

    if "close" in df.columns:
        df = df[df["close"] > 0].copy()
    return df.reset_index(drop=True)

def coerce_cached_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """Cached parquet -> schema coercion. DOES NOT drop extra columns."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "date" not in out.columns:
        out = out.reset_index().rename(columns={"index": "date"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Ensure core cols exist; do not prune.
    for c in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float32, copy=False)
        else:
            out[c] = np.nan

    return out
