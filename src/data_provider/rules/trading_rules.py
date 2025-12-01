from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class TradingParams:
    min_list_days: int = 60
    min_dollar_vol: float = 1e6
    min_price: float = 1.0
    limit_eps: float = 0.002

def add_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["code", "date"]).copy()
    if "amount" in df.columns:
        dv = df["amount"].fillna(0.0)
    else:
        dv = df["close"].fillna(0.0) * df.get("volume", 0.0).fillna(0.0)
    df["dollar_vol"] = dv.astype(np.float32)

    g = df.groupby("code", sort=False)
    df["adv20"] = g["dollar_vol"].transform(lambda x: x.rolling(20, min_periods=1).mean()).astype(np.float32)
    return df

def add_trade_masks(df: pd.DataFrame, params: TradingParams) -> pd.DataFrame:
    """Compute tradable/buyable/sellable masks and a robust limit-up/down heuristic."""
    df = df.sort_values(["code", "date"]).copy()
    g = df.groupby("code", sort=False)

    list_days = g.cumcount() + 1
    prev_close = g["close"].shift(1)
    pct_chg = df["close"] / (prev_close + 1e-9) - 1.0

    lr = pd.to_numeric(df.get("limit_rate", np.nan), errors="coerce")
    lr = lr.fillna(lr.median() if np.isfinite(lr).any() else 0.10).astype(float)

    cond_new = list_days < int(params.min_list_days)
    cond_suspended = df.get("volume", 0.0).fillna(0.0) <= 0
    cond_illiquid = df.get("dollar_vol", 0.0).fillna(0.0) < float(params.min_dollar_vol)
    cond_tiny_price = df["close"].fillna(0.0) < float(params.min_price)

    base_noise = cond_new | cond_suspended | cond_illiquid | cond_tiny_price
    df["tradable_mask"] = np.where(base_noise, np.nan, 1.0).astype(np.float32)

    # One-word board approximation: open==high==low==close (more robust than high==low only)
    o = df.get("open", np.nan).astype(float)
    h = df.get("high", np.nan).astype(float)
    l = df.get("low", np.nan).astype(float)
    c = df.get("close", np.nan).astype(float)
    oneword = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c) & np.isclose(o, h, atol=1e-6) & np.isclose(h, l, atol=1e-6) & np.isclose(l, c, atol=1e-6)

    limit_up = pct_chg >= (lr - float(params.limit_eps))
    limit_dn = pct_chg <= -(lr - float(params.limit_eps))

    df["buyable_mask"] = df["tradable_mask"].copy()
    df.loc[oneword & limit_up, "buyable_mask"] = np.nan

    df["sellable_mask"] = df["tradable_mask"].copy()
    df.loc[oneword & limit_dn, "sellable_mask"] = np.nan

    # sanity on entry/exit prices
    df.loc[~(df["open"].astype(float) > 0), "buyable_mask"] = np.nan
    df.loc[~(df["close"].astype(float) > 0), "sellable_mask"] = np.nan

    return df

def compute_hold_all_days_mask(df: pd.DataFrame, pred_len: int) -> pd.Series:
    """For each t, require tradable for all holding days [t+1, t+pred_len]."""
    if pred_len <= 0:
        return pd.Series(True, index=df.index)
    g = df.groupby("code", sort=False)
    trad = g["tradable_mask"].shift(-1).notna().astype(np.int8)
    # rolling window on shifted series: require all ones for length=pred_len
    hold_ok = trad.groupby(df["code"]).transform(lambda x: x.rolling(pred_len, min_periods=pred_len).min())
    return hold_ok.fillna(0).astype(bool)
