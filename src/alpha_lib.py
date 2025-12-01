# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import factor_ops as ops
from .config import Config
from utils.logging_utils import get_logger

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = get_logger()


class AlphaFactory:
    """
    SOTA Alpha Engine (Panel-safe, Execution-aware)

    ✅ Designed for full-market panel input: multiple codes x dates.
    - TS operators: groupby(code) (no cross-asset leakage)
    - CS pipeline: groupby(date), mask-aware (tradable-only stats)
    - No aggressive local ffill by default (avoid smoothing suspensions)
    """

    VERSION = "v18.1-panel-fund"  # Updated version
    EPS = 1e-9

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        # ---- Ensure code/date ----
        if "date" not in self.df.columns:
            if isinstance(self.df.index, pd.DatetimeIndex):
                self.df = self.df.reset_index().rename(columns={"index": "date"})
            else:
                raise KeyError("AlphaFactory(panel) requires column 'date'")

        if "code" not in self.df.columns:
            raise KeyError("AlphaFactory(panel) requires column 'code'")

        # ---- Parse date + FIX: drop NaT to avoid groupby pollution ----
        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
        self.df = self.df.dropna(subset=["date"]).copy()

        # ---- Required OHLCV ----
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in self.df.columns:
                raise KeyError(f"AlphaFactory requires column '{col}'")

        # ---- Sort to lock TS operations ----
        self.df = self.df.sort_values(["code", "date"], kind="mergesort").reset_index(drop=True)
        self.code = self.df["code"]

        # ---- Cast core numeric (best-effort) ----
        self._cast_float32(
            [
                "open", "high", "low", "close", "volume",
                "turnover", "amount",
                "dollar_vol", "adv20",
                "pe_ttm", "pb", "roe", "profit_growth", "rev_growth", "debt_ratio",
                "tradable_mask", "buyable_mask", "sellable_mask",
                "limit_rate",
                "eps", "bps",  # Added: support basic per-share metrics
            ]
        )

        self.open = self.df["open"]
        self.high = self.df["high"]
        self.low = self.df["low"]
        self.close = self.df["close"]
        self.volume = self.df["volume"]

        self.turnover = self.df["turnover"] if "turnover" in self.df.columns else pd.Series(np.nan, index=self.df.index, dtype=np.float32)
        self.amount = self.df["amount"] if "amount" in self.df.columns else pd.Series(np.nan, index=self.df.index, dtype=np.float32)

        # ---- Prefer upstream dollar_vol/adv20 ----
        if "dollar_vol" in self.df.columns:
            self.dollar_vol = self.df["dollar_vol"].astype(np.float32)
        else:
            fallback = self.amount.where(self.amount.notna(), (self.close * self.volume).astype(np.float32))
            self.dollar_vol = fallback.astype(np.float32)

        if "adv20" in self.df.columns:
            self.adv20 = self.df["adv20"].astype(np.float32)
        else:
            self.adv20 = ops.ts_mean(self.dollar_vol, 20, by=self.code, min_periods=1).astype(np.float32)

        # ---- Trade masks ----
        self.tradable_mask = self._mask_series(self.df["tradable_mask"]) if "tradable_mask" in self.df.columns else pd.Series(True, index=self.df.index)
        self.buyable_mask = self._mask_series(self.df["buyable_mask"]) if "buyable_mask" in self.df.columns else self.tradable_mask.copy()
        self.sellable_mask = self._mask_series(self.df["sellable_mask"]) if "sellable_mask" in self.df.columns else self.tradable_mask.copy()

        # Meta
        self.df["meta_tradable"] = self.tradable_mask.astype(np.float32)
        self.df["meta_buyable"] = self.buyable_mask.astype(np.float32)
        self.df["meta_sellable"] = self.sellable_mask.astype(np.float32)
        self.df["meta_log_dv"] = ops.log1p_pos(self.dollar_vol).astype(np.float32)

        # ---- Returns (panel-safe) ----
        prev_close = self.close.groupby(self.code, sort=False).shift(1)
        prev_close = prev_close.fillna(self.open)
        prev_close = (prev_close + self.EPS).astype(np.float32)

        self.returns = (self.close / prev_close - 1.0).astype(np.float32)
        self.log_ret = np.log((self.close + self.EPS) / (prev_close + self.EPS)).astype(np.float32)

        # Typical price (not true vwap)
        self.vwap = ((self.high + self.low + self.close) / 3.0).astype(np.float32)

    # =============================================================================
    # Public API
    # =============================================================================
    def make_factors(self) -> pd.DataFrame:
        self._build_raw_factors()
        self._build_style_factors()
        self._build_microstructure()
        self._build_ccf_a_factors()
        self._build_fundamental_factors()
        self._build_advanced_stats()
        self._build_interactions()
        self._build_calendar_factors()
        self._clean_factors_locally()
        return self.df

    # =============================================================================
    # 0) Raw
    # =============================================================================
    def _build_raw_factors(self):
        prev_close = self.close.groupby(self.code, sort=False).shift(1)
        prev_close = prev_close.fillna(self.open) + self.EPS

        self.df["raw_ret"] = self.log_ret
        self.df["raw_open_n"] = (self.open / prev_close - 1.0).astype(np.float32)
        self.df["raw_high_n"] = (self.high / prev_close - 1.0).astype(np.float32)
        self.df["raw_low_n"] = (self.low / prev_close - 1.0).astype(np.float32)
        self.df["raw_close_n"] = (self.close / prev_close - 1.0).astype(np.float32)

        vol_ma = ops.ts_mean(self.volume, 20, by=self.code, min_periods=1) + self.EPS
        self.df["raw_volume_n"] = np.log((self.volume + self.EPS) / vol_ma).astype(np.float32)

        self.df["raw_turnover"] = self.turnover.astype(np.float32)

        if "amount" in self.df.columns:
            amt_ma = ops.ts_mean(self.amount, 20, by=self.code, min_periods=1) + self.EPS
            self.df["raw_amount_n"] = np.log((self.amount + self.EPS) / amt_ma).astype(np.float32)

        self.df["raw_dv_log"] = ops.log1p_pos(self.dollar_vol).astype(np.float32)
        self.df["raw_adv20_log"] = ops.log1p_pos(self.adv20).astype(np.float32)

    # =============================================================================
    # 1) Style
    # =============================================================================
    def _build_style_factors(self):
        self.df["style_mom_1m"] = ops.ts_sum(self.log_ret, 20, by=self.code, min_periods=5).astype(np.float32)
        self.df["style_mom_3m"] = ops.ts_sum(self.log_ret, 60, by=self.code, min_periods=10).astype(np.float32)

        self.df["style_vol_1m"] = ops.ts_std(self.returns, 20, by=self.code, min_periods=5).astype(np.float32)
        self.df["style_risk_adj_mom"] = (self.df["style_mom_1m"] / (self.df["style_vol_1m"] + self.EPS)).astype(np.float32)

        self.df["style_liq_adv20"] = ops.log1p_pos(self.adv20).astype(np.float32)

    # =============================================================================
    # 2) Microstructure (execution-aware)
    # =============================================================================
    def _build_microstructure(self):
        raw_amihud = (self.returns.abs() / (self.dollar_vol + self.EPS)).astype(np.float32)
        self.df["ind_amihud"] = ops.ewm_mean(raw_amihud, span=3, by=self.code).astype(np.float32)

        range_pct = ((self.high - self.low) / (self.close + self.EPS)).astype(np.float32)
        raw_smart = (range_pct / (ops.log1p_pos(self.volume) + self.EPS)).astype(np.float32)
        self.df["ind_smart_money"] = ops.ewm_mean(raw_smart, span=3, by=self.code).astype(np.float32)

        delta_p = self.close.groupby(self.code, sort=False).diff().astype(np.float32)
        cov_delta = ops.ts_cov(delta_p, delta_p.groupby(self.code, sort=False).shift(1), 20, by=self.code, min_periods=10)
        cov_delta = cov_delta.where(cov_delta < 0, 0.0)
        self.df["ind_roll_spread"] = (2.0 * np.sqrt(-cov_delta + self.EPS)).astype(np.float32)

        self.df["ind_participation"] = (self.dollar_vol / (self.adv20 + self.EPS)).astype(np.float32)

        tmask = self.tradable_mask
        for c in ["ind_amihud", "ind_smart_money", "ind_roll_spread", "ind_participation"]:
            self.df.loc[~tmask, c] = np.nan

    # =============================================================================
    # 3) CCF-A-ish
    # =============================================================================
    def _build_ccf_a_factors(self):
        close_lag20 = self.close.groupby(self.code, sort=False).shift(20)
        change = (self.close - close_lag20).abs()
        path_len = ops.ts_sum(self.close.groupby(self.code, sort=False).diff().abs(), 20, by=self.code, min_periods=10)
        self.df["ccf_efficiency_ratio"] = (change / (path_len + self.EPS)).astype(np.float32)

        vol_short = ops.ts_std(self.returns, 5, by=self.code, min_periods=3)
        vol_long = ops.ts_std(self.returns, 20, by=self.code, min_periods=5)
        self.df["ccf_vol_regime"] = (vol_short / (vol_long + self.EPS)).astype(np.float32)

        ret_up = self.returns.clip(lower=0)
        ret_dn = (-self.returns.clip(upper=0))
        self.df["ccf_asymmetry"] = (
            ops.ts_std(ret_up, 20, by=self.code, min_periods=5)
            - ops.ts_std(ret_dn, 20, by=self.code, min_periods=5)
        ).astype(np.float32)

    # =============================================================================
    # 4) Fundamentals (keep NaN; CS layer handles fill)
    # =============================================================================
    def _build_fundamental_factors(self):
        # ---- 1. Derive Valuation from EPS/BPS if missing ----
        # If upstream didn't provide pre-calculated PE/PB, we compute using current price.
        # NOTE: If 'eps' is raw quarterly EPS, this PE will be Quarterly PE, not TTM.
        # Ensure your input 'eps' is aligned with your expectations (e.g. TTM).
        if "eps" in self.df.columns and "pe_ttm" not in self.df.columns:
            self.df["pe_ttm"] = (self.close / (self.df["eps"].replace(0, np.nan))).astype(np.float32)

        if "bps" in self.df.columns and "pb" not in self.df.columns:
            self.df["pb"] = (self.close / (self.df["bps"].replace(0, np.nan))).astype(np.float32)

        # ---- 2. Build EP/BP Factors (Inverse Valuation) ----
        if "pe_ttm" in self.df.columns:
            pe = self.df["pe_ttm"].replace(0, np.nan)
            self.df["fund_ep"] = (1.0 / (pe + self.EPS)).astype(np.float32)
        if "pb" in self.df.columns:
            pb = self.df["pb"].replace(0, np.nan)
            self.df["fund_bp"] = (1.0 / (pb + self.EPS)).astype(np.float32)

        # ---- 3. Pass-through other fundamental ratios ----
        for col in ["roe", "profit_growth", "rev_growth", "debt_ratio"]:
            if col in self.df.columns:
                self.df[f"fund_{col}"] = self.df[col].astype(np.float32)

    # =============================================================================
    # 5) Advanced stats & interactions
    # =============================================================================
    def _build_advanced_stats(self):
        self.df["adv_skew_20"] = ops.ts_skew(self.returns, 20, by=self.code, min_periods=10).astype(np.float32)
        self.df["adv_kurt_20"] = ops.ts_kurt(self.returns, 20, by=self.code, min_periods=10).astype(np.float32)
        self.df["adv_pv_corr"] = ops.ts_corr(self.close, self.volume, 10, by=self.code, min_periods=5).astype(np.float32)

    def _build_interactions(self):
        self.df["int_ret_div_vol"] = (self.returns / (self.df["style_vol_1m"] + self.EPS)).astype(np.float32)

    # =============================================================================
    # 6) Calendar
    # =============================================================================
    def _build_calendar_factors(self):
        dt = self.df["date"].dt
        self.df["time_dow"] = (dt.dayofweek / 6.0 - 0.5).astype(np.float32)
        self.df["time_dom"] = ((dt.day - 1) / 30.0 - 0.5).astype(np.float32)
        self.df["time_moy"] = ((dt.month - 1) / 11.0 - 0.5).astype(np.float32)

    # =============================================================================
    # Local cleaning (NO aggressive ffill by default)
    # =============================================================================
    def _clean_factors_locally(self):
        prefixes = ["raw_", "style_", "ind_", "ccf_", "adv_", "int_", "time_", "meta_", "fund_"]
        fac_cols = [c for c in self.df.columns if any(str(c).startswith(p) for p in prefixes)]
        if not fac_cols:
            return

        self.df[fac_cols] = self.df[fac_cols].replace([np.inf, -np.inf], np.nan)

        if bool(getattr(Config, "ALPHA_LOCAL_FFILL", False)):
            self.df[fac_cols] = self.df.groupby("code", sort=False)[fac_cols].ffill()

        self.df.loc[~self.tradable_mask, fac_cols] = np.nan
        self.df[fac_cols] = self.df[fac_cols].astype(np.float32, copy=False)

    # =============================================================================
    # Cross-sectional processing (mask-aware; no look-ahead)
    # =============================================================================
    @staticmethod
    def add_cross_sectional_factors(panel_df: pd.DataFrame) -> pd.DataFrame:
        """
        Mask-aware CS pipeline (per date):
          A) Fill NaNs with median (tradable-only)
          B) Winsorize (1%/99%) (tradable-only)
          C) Neutralize (industry mean or market mean)
          D) Residualize on risk (liq + vol) (tradable-only)
          E) GaussRank -> N(0,1) (tradable-only)
        """
        df = panel_df.copy()

        if "date" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={"index": "date"})
            else:
                raise KeyError("add_cross_sectional_factors requires 'date' column")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).copy()
        dates = df["date"]

        target_prefixes = ["style_", "ind_", "fund_", "adv_", "int_", "ccf_"]
        cols = [c for c in df.columns if any(str(c).startswith(p) for p in target_prefixes)]
        if not cols:
            return df

        if "tradable_mask" in df.columns:
            mask = pd.to_numeric(df["tradable_mask"], errors="coerce").fillna(0.0) > 0.5
        elif "meta_tradable" in df.columns:
            mask = pd.to_numeric(df["meta_tradable"], errors="coerce").fillna(0.0) > 0.5
        else:
            mask = pd.Series(True, index=df.index)

        X = df[cols].replace([np.inf, -np.inf], np.nan)
        Xm = X.where(mask)

        # A) fill with median per date (tradable-only)
        med = Xm.groupby(dates)[cols].transform("median")
        Xf = Xm.fillna(med).fillna(0.0).where(mask)

        # B) winsor 1%/99% per date
        date_codes, date_uniques = pd.factorize(dates, sort=False)

        q = Xf.groupby(dates)[cols].quantile([0.01, 0.99])
        lo_df = q.xs(0.01, level=1).reindex(date_uniques)
        hi_df = q.xs(0.99, level=1).reindex(date_uniques)

        lo = np.nan_to_num(lo_df.to_numpy(dtype=np.float64), nan=-np.inf)[date_codes]
        hi = np.nan_to_num(hi_df.to_numpy(dtype=np.float64), nan=np.inf)[date_codes]

        val = Xf.to_numpy(dtype=np.float64, copy=False)
        val = np.minimum(np.maximum(val, lo), hi)
        Xw = pd.DataFrame(val, index=df.index, columns=cols).where(mask)

        # C) neutralize
        if "industry" in df.columns:
            ind_mean = Xw.groupby([dates, df["industry"]])[cols].transform("mean")
            Xn = (Xw - ind_mean).where(mask)
        else:
            mkt_mean = Xw.groupby(dates)[cols].transform("mean")
            Xn = (Xw - mkt_mean).where(mask)

        # D) residualize on risk factors
        risk_specs: List[Tuple[str, pd.Series]] = []
        if "meta_log_dv" in df.columns:
            risk_specs.append(("meta_log_dv", pd.to_numeric(df["meta_log_dv"], errors="coerce")))
        if "style_vol_1m" in df.columns:
            risk_specs.append(("style_vol_1m", pd.to_numeric(df["style_vol_1m"], errors="coerce")))

        Xr = Xn.copy()
        for _, r in risk_specs:
            r = r.where(mask)
            r_mean = r.groupby(dates).transform("mean")
            r_std = r.groupby(dates).transform("std")
            r_std = r_std.replace(0, np.nan).fillna(1.0)
            rz = (r - r_mean) / r_std

            prod = Xr.mul(rz, axis=0)
            beta = prod.groupby(dates)[cols].transform("mean")
            Xr = (Xr - beta.mul(rz, axis=0)).where(mask)

        # E) GaussRank
        ranks = Xr.groupby(dates)[cols].rank(pct=True, method="average")
        p = (ranks * 0.99 + 0.005).clip(1e-4, 1 - 1e-4)
        gauss = norm.ppf(p)

        out = pd.DataFrame(gauss, index=df.index, columns=[f"cs_{c}" for c in cols]).where(mask)
        df = pd.concat([df, out.astype(np.float32)], axis=1)

        if "target" in df.columns:
            t = pd.to_numeric(df["target"], errors="coerce").where(mask)
            pr = t.groupby(dates).rank(pct=True, method="average")
            pr = (pr * 0.99 + 0.005).clip(1e-4, 1 - 1e-4)
            df["rank_label"] = norm.ppf(pr).astype(np.float32)

            mkt = t.groupby(dates).transform("mean")
            df["excess_label"] = (t - mkt).astype(np.float32)

        return df

    # =============================================================================
    # Utils
    # =============================================================================
    def _cast_float32(self, cols: List[str]):
        for c in cols:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors="coerce").astype(np.float32)

    @staticmethod
    def _mask_series(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce").fillna(0.0)
        return s > 0.5