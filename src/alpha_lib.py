import logging
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import factor_ops as ops
from .config import Config

# Jeff Dean style: narrow ignores; no blanket ignore.
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)


class AlphaFactory:
    """
    【SOTA Alpha Engine v17.0 - "Phantom" Pro, Execution-Aware】

    Key updates:
      1) Prefer upstream fields: dollar_vol / adv20 / (trade masks) if present.
      2) No aggressive local ffill by default (avoid smoothing across suspensions / missing).
      3) Cross-sectional pipeline computes stats only on tradable rows (mask-aware).
      4) Add VERSION for DataProvider cache fingerprinting.
    """

    VERSION = "v17.0"
    EPS = 1e-9

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        # ---- Ensure date column exists ----
        if "date" not in self.df.columns:
            if isinstance(self.df.index, pd.DatetimeIndex):
                self.df = self.df.reset_index().rename(columns={"index": "date"})
            else:
                self.df["date"] = pd.NaT
        self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")

        # ---- Cast core numeric columns (best-effort) ----
        self._cast_float32(
            [
                "open", "high", "low", "close", "volume", "turnover", "amount",
                "dollar_vol", "adv20",
                "pe_ttm", "pb", "roe", "profit_growth", "rev_growth", "debt_ratio",
                "tradable_mask", "buyable_mask", "sellable_mask",
                "limit_rate",
            ]
        )

        # ---- Required OHLCV ----
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in self.df.columns:
                raise KeyError(f"AlphaFactory requires column '{col}'")

        self.open = self.df["open"]
        self.high = self.df["high"]
        self.low = self.df["low"]
        self.close = self.df["close"]
        self.volume = self.df["volume"]

        self.turnover = (
            self.df["turnover"] if "turnover" in self.df.columns
            else pd.Series(np.nan, index=self.df.index, dtype=np.float32)
        )
        self.amount = (
            self.df["amount"] if "amount" in self.df.columns
            else pd.Series(np.nan, index=self.df.index, dtype=np.float32)
        )

        # ---- Prefer upstream dollar_vol/adv20 (DataProvider v23 provides them) ----
        if "dollar_vol" in self.df.columns:
            self.dollar_vol = self.df["dollar_vol"].astype(np.float32)
        else:
            self.dollar_vol = self.amount.where(self.amount.notna(), self.close * self.volume).astype(np.float32)

        if "adv20" in self.df.columns:
            self.adv20 = self.df["adv20"].astype(np.float32)
        else:
            self.adv20 = self.dollar_vol.rolling(20, min_periods=1).mean().astype(np.float32)

        # ---- Trade masks (upstream from DataProvider) ----
        self.tradable_mask = (
            self.df["tradable_mask"].astype(np.float32)
            if "tradable_mask" in self.df.columns
            else pd.Series(1.0, index=self.df.index, dtype=np.float32)
        )
        self.buyable_mask = (
            self.df["buyable_mask"].astype(np.float32)
            if "buyable_mask" in self.df.columns
            else self.tradable_mask.copy()
        )
        self.sellable_mask = (
            self.df["sellable_mask"].astype(np.float32)
            if "sellable_mask" in self.df.columns
            else self.tradable_mask.copy()
        )

        # Meta flags for model / CS masking
        self.df["meta_tradable"] = self.tradable_mask
        self.df["meta_buyable"] = self.buyable_mask
        self.df["meta_sellable"] = self.sellable_mask
        self.df["meta_log_dv"] = self._safe_log(self.dollar_vol)

        # ---- Precomputed basics (NO look-ahead) ----
        prev_close = self.close.shift(1) + self.EPS
        self.returns = self.close / prev_close - 1.0
        self.log_ret = np.log((self.close + self.EPS) / (prev_close + self.EPS))

        # Typical price
        self.vwap = (self.high + self.low + self.close) / 3.0

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
    # 0) Raw Factors
    # =============================================================================
    def _build_raw_factors(self):
        prev_close = self.close.shift(1) + self.EPS

        self.df["raw_ret"] = self.log_ret
        self.df["raw_open_n"] = self.open / prev_close - 1.0
        self.df["raw_high_n"] = self.high / prev_close - 1.0
        self.df["raw_low_n"] = self.low / prev_close - 1.0
        self.df["raw_close_n"] = self.close / prev_close - 1.0

        vol_ma = self.volume.rolling(20, min_periods=1).mean() + self.EPS
        self.df["raw_volume_n"] = np.log((self.volume + self.EPS) / vol_ma)

        self.df["raw_turnover"] = self.turnover

        # amount & dollar volume features
        if "amount" in self.df.columns:
            amt_ma = self.amount.rolling(20, min_periods=1).mean() + self.EPS
            self.df["raw_amount_n"] = np.log((self.amount + self.EPS) / amt_ma)

        self.df["raw_dv_log"] = self._safe_log(self.dollar_vol)
        self.df["raw_adv20_log"] = self._safe_log(self.adv20)

    # =============================================================================
    # 1) Style Factors
    # =============================================================================
    def _build_style_factors(self):
        self.df["style_mom_1m"] = ops.ts_sum(self.log_ret, 20)
        self.df["style_mom_3m"] = ops.ts_sum(self.log_ret, 60)

        self.df["style_vol_1m"] = ops.ts_std(self.returns, 20)
        self.df["style_risk_adj_mom"] = self.df["style_mom_1m"] / (self.df["style_vol_1m"] + self.EPS)

        # Liquidity scale for downstream neutralization (already set in meta_log_dv)
        self.df["style_liq_adv20"] = self._safe_log(self.adv20)

    # =============================================================================
    # 2) Microstructure
    # =============================================================================
    def _build_microstructure(self):
        # Amihud: |ret| / dollar_vol (smoothed)
        raw_amihud = self.returns.abs() / (self.dollar_vol + self.EPS)
        self.df["ind_amihud"] = raw_amihud.ewm(span=3, adjust=False).mean() * self.tradable_mask

        # Smart money: range / log(vol)
        range_pct = (self.high - self.low) / (self.close + self.EPS)
        raw_smart = range_pct / (self._safe_log(self.volume) + self.EPS)
        self.df["ind_smart_money"] = raw_smart.ewm(span=3, adjust=False).mean() * self.tradable_mask

        # Roll spread proxy
        delta_p = self.close.diff()
        cov_delta = ops.ts_cov(delta_p, delta_p.shift(1), 20)
        cov_delta = cov_delta.where(cov_delta < 0, 0)
        self.df["ind_roll_spread"] = 2.0 * np.sqrt(-cov_delta + self.EPS) * self.tradable_mask

        # Participation proxy: dv / adv20 (helps execution realism)
        self.df["ind_participation"] = (self.dollar_vol / (self.adv20 + self.EPS)).astype(np.float32) * self.tradable_mask

    # =============================================================================
    # 3) CCF-A / SOTA-ish Factors
    # =============================================================================
    def _build_ccf_a_factors(self):
        change = (self.close - self.close.shift(20)).abs()
        path_len = ops.ts_sum(self.close.diff().abs(), 20)
        self.df["ccf_efficiency_ratio"] = change / (path_len + self.EPS)

        vol_short = ops.ts_std(self.returns, 5)
        vol_long = ops.ts_std(self.returns, 20)
        self.df["ccf_vol_regime"] = vol_short / (vol_long + self.EPS)

        ret_up = self.returns.clip(lower=0)
        ret_down = (-self.returns.clip(upper=0))
        self.df["ccf_asymmetry"] = ops.ts_std(ret_up, 20) - ops.ts_std(ret_down, 20)

    # =============================================================================
    # 4) Fundamentals (keep NaN; CS layer handles fill)
    # =============================================================================
    def _build_fundamental_factors(self):
        if "pe_ttm" in self.df.columns:
            self.df["fund_ep"] = 1.0 / (self.df["pe_ttm"].replace(0, np.nan) + self.EPS)
        if "pb" in self.df.columns:
            self.df["fund_bp"] = 1.0 / (self.df["pb"].replace(0, np.nan) + self.EPS)

        for col in ["roe", "profit_growth", "rev_growth", "debt_ratio"]:
            if col in self.df.columns:
                self.df[f"fund_{col}"] = self.df[col]

    # =============================================================================
    # 5) Advanced stats & interactions
    # =============================================================================
    def _build_advanced_stats(self):
        self.df["adv_skew_20"] = ops.ts_skew(self.returns, 20)
        self.df["adv_kurt_20"] = ops.ts_kurt(self.returns, 20)
        self.df["adv_pv_corr"] = ops.ts_corr(self.close, self.volume, 10)

    def _build_interactions(self):
        self.df["int_ret_div_vol"] = self.returns / (self.df["style_vol_1m"] + self.EPS)

    # =============================================================================
    # 6) Calendar factors
    # =============================================================================
    def _build_calendar_factors(self):
        if "date" not in self.df.columns:
            return
        dt = self.df["date"].dt
        self.df["time_dow"] = dt.dayofweek / 6.0 - 0.5
        self.df["time_dom"] = (dt.day - 1) / 30.0 - 0.5
        self.df["time_moy"] = (dt.month - 1) / 11.0 - 0.5

    # =============================================================================
    # Local cleaning (NO aggressive ffill by default)
    # =============================================================================
    def _clean_factors_locally(self):
        factor_prefixes = ["raw_", "style_", "ind_", "ccf_", "adv_", "int_", "time_", "meta_", "fund_"]
        fac_cols = [c for c in self.df.columns if any(str(c).startswith(p) for p in factor_prefixes)]

        if not fac_cols:
            return

        self.df[fac_cols] = self.df[fac_cols].replace([np.inf, -np.inf], np.nan)

        # Optional local fill (off by default)
        # Recommendation: keep False; let downstream (panel/cs) handle fill to avoid smoothing missingness.
        if bool(getattr(Config, "ALPHA_LOCAL_FFILL", False)):
            self.df[fac_cols] = self.df[fac_cols].ffill()

        # enforce mask: untradable -> NaN
        tradable = self.df["meta_tradable"].notna()
        self.df.loc[~tradable, fac_cols] = np.nan

    # =============================================================================
    # Cross-sectional processing (mask-aware; no look-ahead)
    # =============================================================================
    @staticmethod
    def add_cross_sectional_factors(panel_df: pd.DataFrame) -> pd.DataFrame:
        """
        Institutional-ish CS pipeline (mask-aware):
          1) Fill NaNs with daily median (on tradable rows only)
          2) Winsorize within each date (1%/99%) on tradable rows only
          3) Sector/market neutralize (mean subtraction) on tradable rows only
          4) Residualize on risk factors (liquidity + vol) on tradable rows only
          5) GaussRank to N(0,1) on tradable rows only
        """
        if "date" not in panel_df.columns:
            if isinstance(panel_df.index, pd.DatetimeIndex):
                panel_df = panel_df.reset_index().rename(columns={"index": "date"})
            else:
                panel_df = panel_df.reset_index()

        panel_df["date"] = pd.to_datetime(panel_df["date"], errors="coerce")

        target_prefixes = ["style_", "ind_", "fund_", "adv_", "int_", "ccf_"]
        target_cols = [c for c in panel_df.columns if any(str(c).startswith(p) for p in target_prefixes)]
        if not target_cols:
            return panel_df

        # mask: only tradable rows participate in CS stats
        if "tradable_mask" in panel_df.columns:
            mask = panel_df["tradable_mask"].notna()
        elif "meta_tradable" in panel_df.columns:
            mask = panel_df["meta_tradable"].notna()
        else:
            mask = pd.Series(True, index=panel_df.index)

        dates = panel_df["date"]
        X = panel_df[target_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan)

        # A) Fill NaNs per date with median (computed on tradable rows only)
        Xm = X.where(mask)
        med = Xm.groupby(dates)[target_cols].transform("median")
        med = med.where(mask)  # prevent filling non-tradable rows
        Xf = Xm.fillna(med).fillna(0.0).where(mask)  # non-tradable back to NaN

        # B) Winsorize per date (1%/99%) on tradable rows
        g = Xf.groupby(dates)[target_cols]
        q = g.quantile([0.01, 0.99])
        try:
            lower = q.xs(0.01, level=1)
            upper = q.xs(0.99, level=1)
        except Exception:
            # fallback if quantile shape is unexpected
            lower = Xf.groupby(dates)[target_cols].transform(lambda s: s.quantile(0.01))
            upper = Xf.groupby(dates)[target_cols].transform(lambda s: s.quantile(0.99))
            Xw = Xf.clip(lower, upper, axis=1)
        else:
            idx = pd.Index(dates.values)
            lo = lower.reindex(idx).to_numpy()
            hi = upper.reindex(idx).to_numpy()
            lo = np.nan_to_num(lo, nan=-np.inf)
            hi = np.nan_to_num(hi, nan=np.inf)
            val = Xf.to_numpy(dtype=np.float64, copy=False)
            val = np.minimum(np.maximum(val, lo), hi)
            Xw = pd.DataFrame(val, index=panel_df.index, columns=target_cols).where(mask)

        # C) Sector/market neutralization (mean subtraction) on tradable rows
        if "industry" in panel_df.columns:
            ind_mean = Xw.groupby([dates, panel_df["industry"]])[target_cols].transform("mean")
            Xn = Xw - ind_mean
        else:
            mkt_mean = Xw.groupby(dates)[target_cols].transform("mean")
            Xn = Xw - mkt_mean
        Xn = Xn.where(mask)

        # D) Style residualization
        risk_specs: List[Tuple[str, pd.Series]] = []
        if "meta_log_dv" in panel_df.columns:
            risk_specs.append(("meta_log_dv", pd.to_numeric(panel_df["meta_log_dv"], errors="coerce")))
        if "style_vol_1m" in panel_df.columns:
            risk_specs.append(("style_vol_1m", pd.to_numeric(panel_df["style_vol_1m"], errors="coerce")))

        Xr = Xn.copy()
        for risk_name, risk_series in risk_specs:
            r = risk_series.where(mask)
            r_mean = r.groupby(dates).transform("mean")
            r_std = r.groupby(dates).transform("std") + 1e-9
            rz = (r - r_mean) / r_std

            resid_cols = [c for c in Xr.columns if c != risk_name]
            if not resid_cols:
                continue

            prod = Xr[resid_cols].multiply(rz, axis=0)
            betas = prod.groupby(dates).transform("mean")
            Xr.loc[:, resid_cols] = Xr[resid_cols] - betas.multiply(rz, axis=0)
            Xr = Xr.where(mask)

        # E) Gauss rank (tradable-only)
        ranks = Xr.groupby(dates)[target_cols].rank(pct=True)
        ranks = ranks * 0.99 + 0.005
        gauss = norm.ppf(ranks)

        out = pd.DataFrame(gauss, index=panel_df.index, columns=[f"cs_{c}" for c in target_cols]).where(mask)
        panel_df = pd.concat([panel_df, out], axis=1)

        # Labels (if present): rank among available (non-NaN) target; target already gated upstream.
        if "target" in panel_df.columns:
            lbl = panel_df.groupby("date")["target"].rank(pct=True) * 0.99 + 0.005
            panel_df["rank_label"] = norm.ppf(lbl)
            mkt = panel_df.groupby("date")["target"].transform("mean")
            panel_df["excess_label"] = panel_df["target"] - mkt

        return panel_df

    # =============================================================================
    # Utils
    # =============================================================================
    def _cast_float32(self, cols: List[str]):
        for c in cols:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors="coerce").astype(np.float32)

    @staticmethod
    def _safe_log(x: pd.Series) -> pd.Series:
        x = pd.to_numeric(x, errors="coerce")
        return np.log(x.clip(lower=0) + AlphaFactory.EPS).astype(np.float32)
