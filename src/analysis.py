# -*- coding: utf-8 -*-
"""
analysis.py / backtest_analyzer.py (v4 final, reuse SignalEngine, unified masks)

External API unchanged:
- BacktestAnalyzer(target_set='test'|'eval'|'train'|'custom', start_date=None, end_date=None, adjust='qfq')
- generate_historical_predictions()
- analyze_performance()

Core原则：
1) 历史推理不再手写滑窗：复用 SignalEngine.score_date_range（SSOT）
2) masks 语义统一：以 DataProvider._add_trade_masks 为准（NaN=不可交易，notna=可交易）
3) split 使用 end-exclusive，且 end_excl -> end_incl 用“上一交易日”而不是 -1 day
4) 严格回测：信号T -> 最早可买 d>=T+1 buyable；到期/止损卖出卖不掉可延迟

Assumptions:
- DataProvider.load_and_process_panel(mode='train', adjust=...) returns panel_df + feature_cols
- panel_df 尽量包含 OHLCV 与 masks；如果不含 masks，会在 price_df 上补 masks（调用 _add_trade_masks）

"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.logging_utils import get_logger

from .config import Config
from .data_provider import DataProvider

from .core.signal_engine import SignalEngine  # type: ignore

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

logger = get_logger()


class BacktestAnalyzer:
    def __init__(
        self,
        target_set: str = "test",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: str = "qfq",
    ):
        self.model_path = f"{Config.OUTPUT_DIR}/final_model"

        self.target_set = str(target_set).lower().strip()
        self.user_start_date = start_date
        self.user_end_date = end_date
        self.adjust = adjust

        # end-exclusive
        self.analysis_start_date: Optional[pd.Timestamp] = None
        self.analysis_end_date_excl: Optional[pd.Timestamp] = None

        # outputs
        self.results_df: Optional[pd.DataFrame] = None
        self._price_df: Optional[pd.DataFrame] = None
        self._price_cache: Optional[dict] = None
        self._bt_daily: Optional[pd.DataFrame] = None

    # =============================================================================
    # Split logic (end-exclusive, aligned with DataProvider.make_dataset)
    # =============================================================================

    @staticmethod
    def _fallback_date_splits(panel_df: pd.DataFrame, seq_len: int) -> Dict[str, pd.Timestamp]:
        if "date" not in panel_df.columns:
            raise ValueError("panel_df missing 'date'")

        unique_dates = np.sort(pd.to_datetime(panel_df["date"], errors="coerce").dropna().unique())
        n_dates = len(unique_dates)
        if n_dates == 0:
            raise ValueError("panel_df has no valid dates")

        train_ratio = float(getattr(Config, "TRAIN_RATIO", 0.7))
        val_ratio = float(getattr(Config, "VAL_RATIO", 0.15))
        gap = int(seq_len)

        train_end_idx = int(n_dates * train_ratio)
        val_end_idx = int(n_dates * (train_ratio + val_ratio))

        train_limit = pd.to_datetime(unique_dates[min(train_end_idx, n_dates - 1)])
        val_start = pd.to_datetime(unique_dates[min(train_end_idx + gap, n_dates - 1)])
        val_limit = pd.to_datetime(unique_dates[min(val_end_idx, n_dates - 1)])
        test_start = pd.to_datetime(unique_dates[min(val_end_idx + gap, n_dates - 1)])
        last_date = pd.to_datetime(unique_dates[-1])

        return {
            "train_start": pd.to_datetime(unique_dates[0]),
            "train_end_excl": train_limit,
            "val_start": val_start,
            "val_end_excl": val_limit,
            "test_start": test_start,
            "test_end_excl": last_date + pd.Timedelta(days=1),
            "last_date": last_date,
        }

    def _get_date_splits(self, panel_df: pd.DataFrame) -> Dict[str, pd.Timestamp]:
        seq_len = int(getattr(Config, "CONTEXT_LEN", 64))
        if hasattr(DataProvider, "_get_date_splits"):
            try:
                return DataProvider._get_date_splits(panel_df, seq_len=seq_len)  # type: ignore
            except Exception:
                return self._fallback_date_splits(panel_df, seq_len=seq_len)
        return self._fallback_date_splits(panel_df, seq_len=seq_len)

    def _resolve_analysis_range(self, panel_df: pd.DataFrame) -> None:
        splits = self._get_date_splits(panel_df)

        if self.target_set == "test":
            self.analysis_start_date = splits["test_start"]
            self.analysis_end_date_excl = splits["test_end_excl"]
            logger.info("🔒 [Target: TEST SET] 样本外测试集 (Strict Split, end-exclusive)")
            logger.info(f"   配置比例: Train({Config.TRAIN_RATIO:.0%}) + Val({Config.VAL_RATIO:.0%}) -> Test")

        elif self.target_set in ["validation", "eval", "val"]:
            self.analysis_start_date = splits["val_start"]
            self.analysis_end_date_excl = splits["val_end_excl"]
            logger.info("🔓 [Target: VALIDATION SET] 验证集 (Strict Split, end-exclusive)")
            logger.info(f"   配置比例: Train({Config.TRAIN_RATIO:.0%}) -> Val({Config.VAL_RATIO:.0%})")

        elif self.target_set == "train":
            self.analysis_start_date = splits["train_start"]
            self.analysis_end_date_excl = splits["train_end_excl"]
            logger.info("📈 [Target: TRAIN SET] 训练集 (In-Sample, end-exclusive)")

        else:
            s_date = self.user_start_date or getattr(Config, "START_DATE", "20000101")
            e_date = self.user_end_date or "2099-12-31"
            self.analysis_start_date = pd.to_datetime(s_date)
            self.analysis_end_date_excl = pd.to_datetime(e_date) + pd.Timedelta(days=1)
            logger.info("🛠️ [Target: CUSTOM] 自定义时间范围 (end-exclusive)")

        assert self.analysis_start_date is not None and self.analysis_end_date_excl is not None
        logger.info(
            f"   分析区间: {self.analysis_start_date.date()} ~ {(self.analysis_end_date_excl - pd.Timedelta(days=1)).date()}"
        )

    @staticmethod
    def _prev_trading_date(panel_df: pd.DataFrame, end_excl: pd.Timestamp) -> pd.Timestamp:
        ud = np.sort(pd.to_datetime(panel_df["date"], errors="coerce").dropna().unique())
        if len(ud) == 0:
            return end_excl - pd.Timedelta(days=1)
        pos = int(np.searchsorted(ud, np.datetime64(end_excl), side="left")) - 1
        pos = max(pos, 0)
        return pd.to_datetime(ud[pos]).normalize()

    # =============================================================================
    # Inference (REUSE SignalEngine) + attach labels + build price cache
    # =============================================================================

    def generate_historical_predictions(self) -> None:
        logger.info("=" * 72)
        logger.info(f">>> [Analysis] v4 (reuse SignalEngine) (Target: {self.target_set}, Adjust: {self.adjust})")
        logger.info("=" * 72)

        if not os.path.exists(self.model_path):
            logger.error(f"❌ 模型未找到: {self.model_path}")
            return

        # 1) Load model/panel via SignalEngine (SSOT)
        try:
            model = SignalEngine.load_model(self.model_path)
        except Exception as e:
            logger.error(f"❌ Load model failed: {e}")
            return

        try:
            panel_df, feature_cols = SignalEngine.load_panel(adjust=self.adjust, mode="train")
        except Exception as e:
            logger.error(f"❌ Load panel failed: {e}")
            return

        if panel_df.empty:
            logger.error("❌ panel_df 为空")
            return

        panel_df = panel_df.copy()
        panel_df["date"] = pd.to_datetime(panel_df["date"], errors="coerce").dt.normalize()
        panel_df["code"] = panel_df["code"].astype(str)
        panel_df = panel_df.dropna(subset=["date", "code"]).sort_values(["code", "date"]).reset_index(drop=True)

        # 2) Resolve analysis window (end-exclusive)
        self._resolve_analysis_range(panel_df)
        assert self.analysis_start_date is not None and self.analysis_end_date_excl is not None

        start_incl = pd.to_datetime(self.analysis_start_date).normalize()
        end_incl = self._prev_trading_date(panel_df, self.analysis_end_date_excl)

        # 3) Prepare price slice for strict backtest (need extra tail for delayed exits)
        hold_days = int(getattr(Config, "PRED_LEN", 5))
        max_sell_delay = int(getattr(Config, "BACKTEST_MAX_SELL_DELAY", 5))
        extra_calendar_days = int(max(90, (hold_days + max_sell_delay + 10) * 4))
        price_end_excl = self.analysis_end_date_excl + pd.Timedelta(days=extra_calendar_days)

        lookback_buffer = int(getattr(Config, "CONTEXT_LEN", 64) * 2 + 60)
        read_start_date = start_incl - pd.Timedelta(days=lookback_buffer)

        # Price columns
        needed_price_cols = ["date", "code", "open", "high", "low", "close", "volume"]
        optional_price_cols = ["amount", "dollar_vol", "limit_rate", "tradable_mask", "buyable_mask", "sellable_mask"]
        keep_price_cols = [c for c in (needed_price_cols + optional_price_cols) if c in panel_df.columns]

        mask_price = (panel_df["date"] >= read_start_date) & (panel_df["date"] < price_end_excl)
        price_df = panel_df.loc[mask_price, keep_price_cols].copy()

        # ensure dollar_vol for masks
        if "dollar_vol" not in price_df.columns:
            if "amount" in price_df.columns:
                dv = pd.to_numeric(price_df["amount"], errors="coerce")
                price_df["dollar_vol"] = dv.where(
                    dv.notna(),
                    pd.to_numeric(price_df["close"], errors="coerce") * pd.to_numeric(price_df["volume"], errors="coerce"),
                )
            else:
                price_df["dollar_vol"] = pd.to_numeric(price_df["close"], errors="coerce") * pd.to_numeric(price_df["volume"], errors="coerce")

        # if masks missing, add via DataProvider (unified semantics)
        if not {"tradable_mask", "buyable_mask", "sellable_mask"}.issubset(price_df.columns) and hasattr(DataProvider, "_add_trade_masks"):
            try:
                price_df = price_df.sort_values(["code", "date"]).reset_index(drop=True)
                price_df = DataProvider._add_trade_masks(price_df)  # type: ignore
            except Exception:
                price_df["tradable_mask"] = 1.0
                price_df["buyable_mask"] = 1.0
                price_df["sellable_mask"] = 1.0

        self._price_df = price_df

        # 4) Score range (REUSE SignalEngine.score_date_range)
        df_scoring = panel_df[(panel_df["date"] >= read_start_date) & (panel_df["date"] <= end_incl)].copy()
        if df_scoring.empty:
            logger.error("❌ scoring window 为空")
            return

        scores_df = SignalEngine.score_date_range(
            model=model,
            panel_df=df_scoring,
            feature_cols=feature_cols,
            start_date=start_incl,
            end_date=end_incl,
            seq_len=int(getattr(Config, "CONTEXT_LEN", 64)),
            batch_size=int(getattr(Config, "ANALYSIS_BATCH_SIZE", 2048)),
            desc="Analysis Scoring",
        )

        if scores_df is None or scores_df.empty:
            logger.error("❌ 未生成 score")
            return

        scores_df = scores_df.copy()
        scores_df["date"] = pd.to_datetime(scores_df["date"], errors="coerce").dt.normalize()
        scores_df["code"] = scores_df["code"].astype(str)
        scores_df["score"] = pd.to_numeric(scores_df["score"], errors="coerce")
        scores_df = scores_df.dropna(subset=["date", "code", "score"]).reset_index(drop=True)

        # 5) Attach labels (rank_label/excess_label/target) by merge(date,code)
        label_cols = [c for c in ["rank_label", "excess_label", "target"] if c in panel_df.columns]
        if not label_cols:
            merged = scores_df.copy()
            merged["rank_label"] = np.nan
            merged["excess_label"] = np.nan
        else:
            lab = panel_df[["date", "code"] + label_cols].copy()
            lab["date"] = pd.to_datetime(lab["date"], errors="coerce").dt.normalize()
            lab["code"] = lab["code"].astype(str)
            lab = lab.drop_duplicates(subset=["date", "code"], keep="last")

            merged = scores_df.merge(lab, on=["date", "code"], how="left")

            # rank_label fallback to target
            if "rank_label" in merged.columns:
                rl = pd.to_numeric(merged["rank_label"], errors="coerce")
            else:
                rl = pd.Series(np.nan, index=merged.index)
            tgt = pd.to_numeric(merged["target"], errors="coerce") if "target" in merged.columns else pd.Series(np.nan, index=merged.index)
            merged["rank_label"] = rl.where(rl.notna(), tgt)

            # excess_label fallback
            if "excess_label" in merged.columns:
                el = pd.to_numeric(merged["excess_label"], errors="coerce")
            else:
                el = pd.Series(np.nan, index=merged.index)
            merged["excess_label"] = el.where(el.notna(), tgt)

        self.results_df = merged[["date", "code", "score", "rank_label", "excess_label"]].copy()
        self.results_df = self.results_df.dropna(subset=["date", "code", "score"]).reset_index(drop=True)

        logger.info(f"✅ 推理完成：{len(self.results_df)} 条预测记录。")

        self._price_cache = self._build_price_cache(self._price_df)

    # =============================================================================
    # Strict backtest helpers (unified masks)
    # =============================================================================

    @staticmethod
    def _build_price_cache(price_df: Optional[pd.DataFrame]) -> dict:
        if price_df is None or price_df.empty:
            return {"codes": {}, "calendar": np.array([], dtype="datetime64[ns]")}

        df = price_df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df["code"] = df["code"].astype(str)
        df = df.dropna(subset=["date", "code"]).sort_values(["code", "date"]).reset_index(drop=True)

        cal = np.unique(np.sort(df["date"].values.astype("datetime64[ns]")))
        codes_cache: Dict[str, Dict[str, Any]] = {}

        for code, g in df.groupby("code", sort=False):
            gg = g.sort_values("date")
            codes_cache[str(code)] = {
                "date": gg["date"].values.astype("datetime64[ns]"),
                "open": pd.to_numeric(gg["open"], errors="coerce").values.astype(np.float64),
                "high": pd.to_numeric(gg["high"], errors="coerce").values.astype(np.float64) if "high" in gg.columns else None,
                "low": pd.to_numeric(gg["low"], errors="coerce").values.astype(np.float64) if "low" in gg.columns else None,
                "close": pd.to_numeric(gg["close"], errors="coerce").values.astype(np.float64),
                # 关键：mask 用 notna()，避免 bool(np.nan)==True
                "buyable": gg["buyable_mask"].notna().values if "buyable_mask" in gg.columns else None,
                "sellable": gg["sellable_mask"].notna().values if "sellable_mask" in gg.columns else None,
            }
        return {"codes": codes_cache, "calendar": cal}

    @staticmethod
    def _calc_drawdown(equity: np.ndarray) -> Tuple[float, np.ndarray]:
        peak = np.maximum.accumulate(equity)
        dd = equity / (peak + 1e-12) - 1.0
        return float(dd.min()), dd

    @staticmethod
    def _calc_stats(daily_ret: pd.Series, rf_annual: float = 0.0) -> Dict[str, Any]:
        r = daily_ret.fillna(0.0).values.astype(np.float64)
        n = len(r)
        if n == 0:
            return {}

        rf_daily = rf_annual / 252.0
        equity = np.cumprod(1.0 + r)
        total_ret = float(equity[-1] - 1.0)
        ann_ret = float((equity[-1] ** (252.0 / max(n, 1))) - 1.0)
        vol = float(np.std(r, ddof=1) * np.sqrt(252.0)) if n > 1 else 0.0
        sharpe = float((np.mean(r) - rf_daily) / (np.std(r, ddof=1) + 1e-12) * np.sqrt(252.0)) if n > 1 else 0.0
        mdd, dd_series = BacktestAnalyzer._calc_drawdown(equity)
        win_rate = float((r > 0).mean())
        return {
            "total_return": total_ret,
            "annual_return": ann_ret,
            "annual_vol": vol,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "win_rate": win_rate,
            "equity": equity,
            "drawdown": dd_series,
        }

    def _compute_trade_path_unified(
        self,
        code_cache: Dict[str, Any],
        signal_date: pd.Timestamp,
        hold_days: int,
        slippage: float,
        commission: float,
        stamp: float,
        stop_loss: float,
        max_entry_delay: int,
        max_sell_delay: int,
    ) -> Optional[List[Tuple[np.datetime64, float]]]:
        code_dates = code_cache["date"]
        op = code_cache["open"]
        cl = code_cache["close"]
        lo = code_cache.get("low", None)

        buyable = code_cache.get("buyable", None)
        sellable = code_cache.get("sellable", None)

        t = np.datetime64(pd.to_datetime(signal_date).normalize(), "ns")
        i = int(np.searchsorted(code_dates, t))
        if i >= len(code_dates) or code_dates[i] != t:
            return None

        # entry: first buyable day >= T+1
        entry_i0 = i + 1
        if entry_i0 >= len(code_dates):
            return None

        entry_i = None
        end_search = min(len(code_dates), entry_i0 + max_entry_delay + 1)
        for j in range(entry_i0, end_search):
            if not (np.isfinite(op[j]) and op[j] > 0):
                continue
            if buyable is not None and (not bool(buyable[j])):
                continue
            entry_i = j
            break
        if entry_i is None:
            return None

        eff_entry = op[entry_i] * (1.0 + slippage + commission)
        if not (np.isfinite(eff_entry) and eff_entry > 0):
            return None

        exit_i = entry_i + max(int(hold_days) - 1, 0)
        if exit_i >= len(code_dates):
            return None

        # stop-loss
        if stop_loss > 0 and lo is not None:
            sl = abs(float(stop_loss))
            for j in range(entry_i, exit_i + 1):
                if not np.isfinite(lo[j]):
                    continue
                dd = lo[j] / (eff_entry + 1e-12) - 1.0
                if dd <= -sl:
                    req = j
                    cand = req
                    if sellable is not None:
                        while cand < len(code_dates) and (not bool(sellable[cand])) and (cand - req) <= max_sell_delay:
                            cand += 1
                        if cand >= len(code_dates) or (not bool(sellable[cand])):
                            return None
                    exit_i = cand
                    break

        # sell delay
        if sellable is not None and (not bool(sellable[exit_i])):
            req = exit_i
            cand = req
            while cand < len(code_dates) and (not bool(sellable[cand])) and (cand - req) < max_sell_delay:
                cand += 1
            if cand >= len(code_dates) or (not bool(sellable[cand])):
                return None
            exit_i = cand

        sell_cost = slippage + commission + stamp
        path: List[Tuple[np.datetime64, float]] = []

        for j in range(entry_i, exit_i + 1):
            d = code_dates[j]
            if j == entry_i:
                if not (np.isfinite(cl[j]) and cl[j] > 0):
                    return None
                r = cl[j] / (eff_entry + 1e-12) - 1.0
            elif j < exit_i:
                if not (np.isfinite(cl[j]) and np.isfinite(cl[j - 1]) and cl[j - 1] > 0):
                    return None
                r = cl[j] / (cl[j - 1] + 1e-12) - 1.0
            else:
                if not (np.isfinite(cl[j]) and np.isfinite(cl[j - 1]) and cl[j - 1] > 0):
                    return None
                adj_exit = cl[j] * (1.0 - sell_cost)
                r = adj_exit / (cl[j - 1] + 1e-12) - 1.0
            path.append((d, float(r)))

        return path

    def _simulate_overlap_topk(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        assert self._price_cache is not None
        codes_cache = self._price_cache["codes"]
        calendar = self._price_cache["calendar"]
        if len(calendar) == 0:
            raise ValueError("Empty price calendar for backtest")

        hold_days = int(getattr(Config, "PRED_LEN", 5))
        top_k = int(getattr(Config, "TOP_K", 5))

        enable_ls = bool(getattr(Config, "BACKTEST_LONG_SHORT", True))
        short_k = int(getattr(Config, "BACKTEST_SHORT_K", top_k))

        cash_buffer = float(getattr(Config, "CASH_BUFFER", 0.95))
        rf_annual = float(getattr(Config, "RISK_FREE_RATE", 0.0))

        slippage = float(getattr(Config, "SLIPPAGE", 0.0))
        commission = float(getattr(Config, "COMMISSION_RATE", 0.0))
        stamp = float(getattr(Config, "STAMP_DUTY", 0.0))
        stop_loss = float(getattr(Config, "STOP_LOSS_PCT", 0.0))

        max_entry_delay = int(getattr(Config, "BACKTEST_MAX_ENTRY_DELAY", 3))
        max_sell_delay = int(getattr(Config, "BACKTEST_MAX_SELL_DELAY", 5))

        cap_slice = cash_buffer / max(hold_days, 1)

        assert self.analysis_start_date is not None and self.analysis_end_date_excl is not None
        sig_start = np.datetime64(self.analysis_start_date.normalize(), "ns")
        sig_end = np.datetime64(self.analysis_end_date_excl.normalize(), "ns")

        date_to_idx = {d: i for i, d in enumerate(calendar)}
        ret_long = np.zeros(len(calendar), dtype=np.float64)
        exp_long = np.zeros(len(calendar), dtype=np.float64)
        ret_ls = np.zeros(len(calendar), dtype=np.float64)
        exp_ls = np.zeros(len(calendar), dtype=np.float64)

        pred_df = pred_df.copy()
        pred_df["date"] = pd.to_datetime(pred_df["date"], errors="coerce").dt.normalize()
        pred_df["code"] = pred_df["code"].astype(str)
        pred_df["score"] = pd.to_numeric(pred_df["score"], errors="coerce")
        pred_df = pred_df.dropna(subset=["date", "code", "score"]).copy()

        signal_dates = np.sort(pred_df["date"].unique())
        for d in tqdm(signal_dates, desc="StrictBacktest(signal dates)"):
            dn = np.datetime64(pd.to_datetime(d), "ns")
            if not (sig_start <= dn < sig_end):
                continue

            day = pred_df[pred_df["date"] == d]
            if day.empty:
                continue

            day_sorted_desc = day.sort_values("score", ascending=False)
            long_codes = day_sorted_desc["code"].head(top_k).tolist()

            short_codes: List[str] = []
            if enable_ls:
                day_sorted_asc = day.sort_values("score", ascending=True)
                short_codes = day_sorted_asc["code"].head(short_k).tolist()

            long_paths: List[List[Tuple[np.datetime64, float]]] = []
            for code in long_codes:
                cc = codes_cache.get(code)
                if cc is None:
                    continue
                p = self._compute_trade_path_unified(
                    cc, pd.to_datetime(d), hold_days, slippage, commission, stamp, stop_loss, max_entry_delay, max_sell_delay
                )
                if p:
                    long_paths.append(p)

            short_paths: List[List[Tuple[np.datetime64, float]]] = []
            for code in short_codes:
                cc = codes_cache.get(code)
                if cc is None:
                    continue
                p = self._compute_trade_path_unified(
                    cc, pd.to_datetime(d), hold_days, slippage, commission, stamp, stop_loss, max_entry_delay, max_sell_delay
                )
                if p:
                    short_paths.append(p)

            if long_paths:
                w = cap_slice / len(long_paths)
                for path in long_paths:
                    for dt64, r in path:
                        idx = date_to_idx.get(dt64)
                        if idx is None:
                            continue
                        ret_long[idx] += w * r
                        exp_long[idx] += abs(w)

            if enable_ls:
                if long_paths:
                    wl = (cap_slice * 0.5) / len(long_paths)
                    for path in long_paths:
                        for dt64, r in path:
                            idx = date_to_idx.get(dt64)
                            if idx is None:
                                continue
                            ret_ls[idx] += wl * r
                            exp_ls[idx] += abs(wl)
                if short_paths:
                    ws = (cap_slice * 0.5) / len(short_paths)
                    for path in short_paths:
                        for dt64, r in path:
                            idx = date_to_idx.get(dt64)
                            if idx is None:
                                continue
                            ret_ls[idx] -= ws * r
                            exp_ls[idx] += abs(ws)

        rf_daily = rf_annual / 252.0
        out = pd.DataFrame(
            {"date": pd.to_datetime(calendar), "ret_long_raw": ret_long, "exp_long": exp_long, "ret_ls_raw": ret_ls, "exp_ls": exp_ls}
        ).sort_values("date")

        clip_start = self.analysis_start_date - pd.Timedelta(days=5)
        clip_end = self.analysis_end_date_excl + pd.Timedelta(days=180)
        out = out[(out["date"] >= clip_start) & (out["date"] < clip_end)].reset_index(drop=True)

        # leverage guard：exp > CASH_BUFFER 则缩放当日收益 & 截断 exp
        def _cap_exposure(ret_raw: pd.Series, exp: pd.Series, cap: float) -> Tuple[pd.Series, pd.Series]:
            expv = exp.values.astype(np.float64)
            rrv = ret_raw.values.astype(np.float64)
            scale = np.ones_like(expv)
            over = expv > cap
            scale[over] = cap / (expv[over] + 1e-12)
            return pd.Series(rrv * scale, index=ret_raw.index), pd.Series(np.minimum(expv, cap), index=exp.index)

        out["ret_long_raw"], out["exp_long"] = _cap_exposure(out["ret_long_raw"], out["exp_long"], cash_buffer)
        out["ret_ls_raw"], out["exp_ls"] = _cap_exposure(out["ret_ls_raw"], out["exp_ls"], cash_buffer)

        out["cash_w_long"] = (1.0 - out["exp_long"]).clip(0.0, 1.0)
        out["cash_w_ls"] = (1.0 - out["exp_ls"]).clip(0.0, 1.0)

        out["ret_long"] = out["ret_long_raw"] + out["cash_w_long"] * rf_daily
        out["ret_ls"] = out["ret_ls_raw"] + out["cash_w_ls"] * rf_daily

        out["equity_long"] = (1.0 + out["ret_long"].fillna(0.0)).cumprod()
        out["equity_ls"] = (1.0 + out["ret_ls"].fillna(0.0)).cumprod()
        return out

    # =============================================================================
    # Public performance: IC + strict backtest
    # =============================================================================

    def analyze_performance(self) -> None:
        if self.results_df is None or self.results_df.empty:
            logger.warning("⚠️ 结果集为空（先运行 generate_historical_predictions）")
            return

        df = self.results_df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df["rank_label"] = pd.to_numeric(df["rank_label"], errors="coerce")
        df = df.dropna(subset=["date", "score", "rank_label"]).copy()
        if df.empty:
            logger.warning("⚠️ 清洗后结果集为空（score/rank_label 全 NaN）")
            return

        # ---- IC ----
        min_cs = int(getattr(Config, "ANALYSIS_MIN_CROSS_SECTION", 50))
        cnt = df.groupby("date")["score"].transform("size")
        df_ic = df[cnt >= min_cs].copy()

        daily_ic = None
        ic_mean = icir = ic_win_rate = 0.0

        if not df_ic.empty:
            df_ic["score_rank"] = df_ic.groupby("date")["score"].rank(pct=True)
            df_ic["label_rank"] = df_ic.groupby("date")["rank_label"].rank(pct=True)
            daily_ic = df_ic.groupby("date").apply(lambda x: x["score_rank"].corr(x["label_rank"]))

            ic_mean = float(daily_ic.mean())
            ic_std = float(daily_ic.std())
            icir = ic_mean / (ic_std + 1e-9) * math.sqrt(252.0)
            ic_win_rate = float((daily_ic > 0).mean())

            logger.info("-" * 60)
            logger.info(f"📊 【因子绩效(IC)】 (Set: {self.target_set.upper()}, Adjust: {self.adjust})")
            logger.info("-" * 60)
            logger.info(f"Rank IC (Mean) : {ic_mean:.4f}")
            logger.info(f"ICIR (Annual)  : {icir:.4f}")
            logger.info(f"IC Win Rate    : {ic_win_rate:.2%}")
            logger.info(f"Days Evaluated : {len(daily_ic)}")
            logger.info("-" * 60)
        else:
            logger.warning(f"⚠️ IC：每日截面样本数不足（<{min_cs}），跳过 IC 统计。")

        # ---- strict backtest ----
        if self._price_cache is None:
            self._price_cache = self._build_price_cache(self._price_df)

        bt = self._simulate_overlap_topk(df)
        self._bt_daily = bt

        rf_annual = float(getattr(Config, "RISK_FREE_RATE", 0.0))
        stats_long = self._calc_stats(bt["ret_long"], rf_annual=rf_annual)
        stats_ls = self._calc_stats(bt["ret_ls"], rf_annual=rf_annual)

        logger.info("-" * 60)
        logger.info("💼 【严格回测(可交易, 重叠持仓, masks 口径统一)】")
        logger.info("-" * 60)
        logger.info(
            f"Long-only:  Total={stats_long['total_return']:.2%}  Ann={stats_long['annual_return']:.2%}  "
            f"Vol={stats_long['annual_vol']:.2%}  Sharpe={stats_long['sharpe']:.2f}  "
            f"MDD={stats_long['max_drawdown']:.2%}  Win={stats_long['win_rate']:.2%}"
        )
        logger.info(
            f"Long-Short: Total={stats_ls['total_return']:.2%}  Ann={stats_ls['annual_return']:.2%}  "
            f"Vol={stats_ls['annual_vol']:.2%}  Sharpe={stats_ls['sharpe']:.2f}  "
            f"MDD={stats_ls['max_drawdown']:.2%}  Win={stats_ls['win_rate']:.2%}"
        )
        logger.info("-" * 60)

        self._plot_results(daily_ic, ic_mean, icir, ic_win_rate, bt, stats_long, stats_ls)

    def _plot_results(self, daily_ic, ic_mean, icir, ic_win_rate, bt_df, stats_long, stats_ls) -> None:
        plt.figure(figsize=(16, 12))

        ax1 = plt.subplot(3, 1, 1)
        if daily_ic is not None and len(daily_ic) > 0:
            ic_curve = daily_ic.fillna(0.0).cumsum()
            ax1.plot(ic_curve.index, ic_curve.values, label="Cumulative Rank IC", linewidth=1.5)
            ax1.set_title(f"Cumulative Rank IC (ICIR={icir:.2f}) - {self.target_set.upper()}", fontsize=12, fontweight="bold")
            ax1.legend(loc="upper left")
        else:
            ax1.text(0.02, 0.5, "IC not available", transform=ax1.transAxes, fontsize=12)
            ax1.set_title("Cumulative Rank IC", fontsize=12, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.4)

        ax2 = plt.subplot(3, 1, 2)
        if daily_ic is not None and len(daily_ic) > 0:
            vals = daily_ic.fillna(0.0).values
            colors = ["#d32f2f" if v < 0 else "#388e3c" for v in vals]
            ax2.bar(daily_ic.index, vals, color=colors, alpha=0.6, width=1.0, label="Daily IC")
            ax2.axhline(ic_mean, linestyle="--", linewidth=1.5, label=f"Mean IC: {ic_mean:.3f}")
            ax2.axhline(0, color="black", linewidth=0.8)
            ax2.set_title(f"Daily IC (Win Rate={ic_win_rate:.1%})", fontsize=12, fontweight="bold")
            ax2.legend(loc="upper right")
        else:
            ax2.text(0.02, 0.5, "IC not available", transform=ax2.transAxes, fontsize=12)
            ax2.set_title("Daily IC", fontsize=12, fontweight="bold")
        ax2.grid(True, axis="y", linestyle="--", alpha=0.4)

        ax3 = plt.subplot(3, 1, 3)
        ax3.plot(bt_df["date"], bt_df["equity_long"], label=f"Long-only (Sharpe={stats_long['sharpe']:.2f}, MDD={stats_long['max_drawdown']:.2%})", linewidth=1.6)
        ax3.plot(bt_df["date"], bt_df["equity_ls"], label=f"Long-Short (Sharpe={stats_ls['sharpe']:.2f}, MDD={stats_ls['max_drawdown']:.2%})", linestyle="--", linewidth=1.6)
        ax3.set_title(f"Strict Backtest (Overlap Holding, H={int(getattr(Config,'PRED_LEN',5))})", fontsize=12, fontweight="bold")
        ax3.legend(loc="upper left")
        ax3.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(Config.OUTPUT_DIR, f"report_{self.target_set}.png")
        plt.savefig(save_path, dpi=150)
        logger.info(f"📈 图表已保存至: {save_path}")


if __name__ == "__main__":
    logger.info(">>> Mode: Eval Set (QFQ)")
    analyzer = BacktestAnalyzer(target_set="eval", adjust="qfq")
    analyzer.generate_historical_predictions()
    analyzer.analyze_performance()
