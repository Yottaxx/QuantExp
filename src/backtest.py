# -*- coding: utf-8 -*-
"""
backtest.py (v4, Jeff-Dean-grade, unified with DataProvider + SignalEngine)

目标：
  - 不重复造轮子：路径/复权/可交易性/一字涨跌停全部复用 DataProvider 口径
  - 模型推理与信号矩阵生成：复用 SignalEngine（与 analysis 同一套推理/阈值/熊市日逻辑）
  - Backtrader 只做：读同一份价格数据 -> 挂统一 trade masks -> 按信号交易
  - 对外接口保持不变：
      - run_walk_forward_backtest(start_date, end_date, initial_cash, top_k=..., adjust=...)
      - run_backtest(top_stocks_list, initial_cash=..., top_k=..., adjust=...)
      - run_single_backtest(...)

依赖：
  - .config.Config
  - .data_provider.DataProvider  (需包含 _norm_adjust, _price_dir, _add_trade_masks)
  - .signal_engine.SignalEngine  (需包含 load_model/load_panel/score_date_range/scores_to_signal_matrix)
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import akshare as ak
import backtrader as bt
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tqdm import tqdm  # noqa: E402

from .config import Config
from .data_provider import DataProvider
from .core.signal_engine import SignalEngine

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# =============================================================================
# Helpers: 路径 & DataFrame 预处理（统一 DataProvider 口径）
# =============================================================================

def _get_price_dir(adjust: str) -> str:
    """使用 DataProvider 的内部规范，确保全项目用同一目录结构。"""
    adj_norm = DataProvider._norm_adjust(adjust)
    return DataProvider._price_dir(adj_norm)


def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """强健地得到 date 列（datetime64[ns]），并做排序。"""
    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif isinstance(df.index, pd.DatetimeIndex):
        df = df.rename_axis("date").reset_index()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df = df.reset_index().rename(columns={"index": "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def _ensure_ohlcv_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _prepare_bt_price_df(
    code: str,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
    adjust: str = "qfq",
) -> Optional[pd.DataFrame]:
    """
    单股票 Parquet -> 带 trade masks 的 DataFrame（供 Backtrader 使用）

    统一逻辑：
      - 路径: DataProvider._price_dir(DataProvider._norm_adjust(adjust))
      - 列: 至少包含 date/open/high/low/close/volume
      - 计算: 若无 dollar_vol 列，生成 (close * volume)
      - mask: DataProvider._add_trade_masks(df) -> tradable/buyable/sellable (用 NaN 表示不可交易)
      - 索引: DatetimeIndex(date)
    """
    price_dir = _get_price_dir(adjust)
    fpath = os.path.join(price_dir, f"{code}.parquet")
    if not os.path.exists(fpath):
        return None

    try:
        df = pd.read_parquet(fpath)
    except Exception:
        return None

    try:
        df = _ensure_date_column(df)
        df["code"] = str(code)
        df = _ensure_ohlcv_numeric(df)
    except Exception:
        return None

    if start_date is not None:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["date"] <= pd.to_datetime(end_date)]
    if df.empty:
        return None

    if "dollar_vol" not in df.columns:
        df["dollar_vol"] = df["close"].fillna(0.0) * df["volume"].fillna(0.0)

    # 统一 trade masks（同 DataProvider 口径）
    if hasattr(DataProvider, "_add_trade_masks"):
        try:
            df = DataProvider._add_trade_masks(df)
        except Exception:
            # 极端情况下回退
            df["tradable_mask"] = 1.0
            df["buyable_mask"] = 1.0
            df["sellable_mask"] = 1.0
    else:
        df["tradable_mask"] = 1.0
        df["buyable_mask"] = 1.0
        df["sellable_mask"] = 1.0

    # Backtrader feed：要求索引为 datetime
    df_bt = df.set_index("date").sort_index()

    # 避免 Backtrader 因空列出问题
    for m in ["tradable_mask", "buyable_mask", "sellable_mask"]:
        if m not in df_bt.columns:
            df_bt[m] = 1.0

    return df_bt


# =============================================================================
# Backtrader 数据源：挂上 buyable/sellable/tradable 三个 line
# =============================================================================

class AShareDataFeed(bt.feeds.PandasData):
    """扩展 PandasData：增加 tradable/buyable/sellable 三条 line。"""
    lines = ("tradable", "buyable", "sellable")
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", -1),
        ("tradable", "tradable_mask"),
        ("buyable", "buyable_mask"),
        ("sellable", "sellable_mask"),
    )


# =============================================================================
# 手续费模型（参数从 Config 里拿，避免硬编码）
# =============================================================================

class AShareCommission(bt.CommInfoBase):
    params = (
        ("stocklike", True),
        ("commtype", bt.CommInfoBase.COMM_PERC),
        ("perc", getattr(Config, "COMMISSION_RATE", 0.0003)),
        ("stamp_duty", getattr(Config, "STAMP_DUTY", 0.0005)),
        ("min_comm", 5.0),
    )

    def _getcommission(self, size, price, pseudoexec):
        if size > 0:  # 买
            return max(abs(size) * price * self.p.perc, self.p.min_comm)
        elif size < 0:  # 卖
            commission = max(abs(size) * price * self.p.perc, self.p.min_comm)
            stamp_duty = abs(size) * price * self.p.stamp_duty
            return commission + stamp_duty
        return 0.0


# =============================================================================
# 模型驱动 TopK 策略（使用 buyable/sellable masks；注意 NaN 的布尔陷阱）
# =============================================================================

class ModelDrivenStrategy(bt.Strategy):
    params = (
        ("signals", None),  # DataFrame: index=date, columns=code, values=score or -1
        ("top_k", Config.TOP_K),
        ("hold_days", Config.PRED_LEN),
        ("min_volume_percent", Config.MIN_VOLUME_PERCENT),
        ("cash_buffer", Config.CASH_BUFFER),
    )

    def __init__(self):
        self.hold_time: Dict[str, int] = {}
        self.signal_dict: Dict[pd.Timestamp, List[str]] = {}

        sig = self.p.signals
        if sig is None:
            return

        clean = sig.copy()
        clean = clean.apply(pd.to_numeric, errors="coerce").fillna(-1.0)
        if isinstance(clean.index, pd.DatetimeIndex):
            clean.index = clean.index.normalize()
        else:
            clean.index = pd.to_datetime(clean.index).normalize()

        for d, row in clean.iterrows():
            vr = row[row > -1]
            if not vr.empty:
                self.signal_dict[pd.to_datetime(d).normalize()] = vr.nlargest(self.p.top_k).index.astype(str).tolist()

    @staticmethod
    def _mask_is_true(v) -> bool:
        """DataProvider 用 NaN 表示不可交易；bool(np.nan) == True 是大坑，必须显式处理。"""
        try:
            fv = float(v)
            return np.isfinite(fv) and fv > 0.0
        except Exception:
            return False

    def _is_buyable(self, data) -> bool:
        try:
            return self._mask_is_true(data.tradable[0]) and self._mask_is_true(data.buyable[0])
        except Exception:
            return True  # 兜底：不因为字段缺失直接拒单

    def _is_sellable(self, data) -> bool:
        try:
            return self._mask_is_true(data.tradable[0]) and self._mask_is_true(data.sellable[0])
        except Exception:
            return True

    def next(self):
        # 当前作为信号日 T，订单默认在 T+1 Open 执行（Backtrader 默认行为）
        current_date = pd.to_datetime(self.data.datetime.date(0)).normalize()
        target_codes = self.signal_dict.get(current_date, [])
        target_set = set(target_codes)

        # --- Sell ---
        holding_datas = [d for d in self.datas if self.getposition(d).size > 0]
        current_pos_count = len(holding_datas)

        freed_slots = 0
        estimated_cash_release = 0.0

        for data in holding_datas:
            name = str(data._name)
            pos = self.getposition(data).size

            # 无法卖出（停牌/一字跌停等）-> 延后卖；持有天数仍累计
            self.hold_time[name] = self.hold_time.get(name, 0) + 1
            if not self._is_sellable(data):
                continue

            should_sell = (name not in target_set) or (self.hold_time[name] >= int(self.p.hold_days))
            if should_sell:
                self.close(data=data)
                self.hold_time[name] = 0
                freed_slots += 1
                estimated_cash_release += abs(pos) * float(data.close[0])

        # --- Buy ---
        if not target_codes:
            return

        current_cash = float(self.broker.get_cash())
        total_available_cash = current_cash + estimated_cash_release
        if total_available_cash < 1000:
            return

        real_slots_gap = int(self.p.top_k) - current_pos_count
        effective_slots = real_slots_gap + freed_slots
        if effective_slots <= 0:
            return

        target_val_per_slot = total_available_cash / effective_slots * float(self.p.cash_buffer)

        buy_count = 0
        for code in target_codes:
            if buy_count >= effective_slots:
                break

            data = self.getdatabyname(code)
            if data is None:
                continue

            if self.getposition(data).size > 0:
                continue

            # 当前日不可买（停牌/一字涨停等）
            if not self._is_buyable(data):
                continue

            price = float(data.close[0])
            vol = float(data.volume[0])
            if price <= 1e-6 or vol <= 1e-6:
                continue

            # 资金 -> 股数（100 股一手）
            raw_size = target_val_per_slot / price
            size = int(raw_size // 100) * 100
            if size < 100:
                continue

            # 单日成交量占比
            limit_size = int((vol * float(self.p.min_volume_percent)) // 100) * 100
            final_size = min(size, limit_size)

            if final_size >= 100:
                self.buy(data=data, size=final_size)
                self.hold_time[code] = 0
                buy_count += 1


# =============================================================================
# Walk-Forward Backtester：信号生成复用 SignalEngine（与 analysis 同口径）
# =============================================================================

class WalkForwardBacktester:
    def __init__(self, start_date: str, end_date: str, initial_cash: float = 1_000_000.0, adjust: str = "qfq"):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = float(initial_cash)
        self.adjust = adjust
        self.model_path = f"{Config.OUTPUT_DIR}/final_model"

    def generate_signal_matrix(self) -> Optional[pd.DataFrame]:
        """
        关键点：
          - 不再手写 batch/滑窗推理逻辑
          - 复用 SignalEngine.score_date_range + scores_to_signal_matrix
          - 与 analysis 同一套：
              - panel 加载口径
              - lookback buffer
              - MIN_SCORE_THRESHOLD
              - bear_mean_th=0.45（可改成 Config.MARKET_BEAR_MEAN_TH 之类）
        """
        print(f"⏳ [Signal Gen] {self.start_date} ~ {self.end_date} (Adjust={self.adjust})")

        # 1) Model
        try:
            model = SignalEngine.load_model(self.model_path)
        except Exception as e:
            print(f"❌ Model load failed: {e}")
            return None

        # 2) Panel (Same as analysis)
        try:
            panel_df, feature_cols = SignalEngine.load_panel(adjust=self.adjust, mode="train")
        except Exception as e:
            print(f"❌ Panel load failed: {e}")
            return None

        # 3) Cut window with lookback (ensure first day can be predicted)
        lookback_days = int(getattr(Config, "CONTEXT_LEN", 64)) * 2 + 60
        read_start = pd.to_datetime(self.start_date) - pd.Timedelta(days=lookback_days)
        read_end = pd.to_datetime(self.end_date)

        sub = panel_df[(panel_df["date"] >= read_start.normalize()) & (panel_df["date"] <= read_end.normalize())].copy()
        if sub.empty:
            print("❌ No panel data for given window.")
            return None

        # 4) Score range (single source of truth)
        scores_df = SignalEngine.score_date_range(
            model=model,
            panel_df=sub,
            feature_cols=feature_cols,
            start_date=pd.to_datetime(self.start_date),
            end_date=pd.to_datetime(self.end_date),
            seq_len=int(getattr(Config, "CONTEXT_LEN", 64)),
            batch_size=int(getattr(Config, "ANALYSIS_BATCH_SIZE", 2048)),
            desc="WalkForward Scoring",
        )
        if scores_df.empty:
            print("❌ No scores generated.")
            return None

        # 5) Build signal matrix (threshold + bear days)
        signals = SignalEngine.scores_to_signal_matrix(
            scores_df,
            min_score_threshold=float(getattr(Config, "MIN_SCORE_THRESHOLD", 0.6)),
            bear_mean_th=float(getattr(Config, "MARKET_BEAR_MEAN_TH", 0.45)),
        )

        # Optional：确保索引覆盖回测期（缺失日会导致策略无信号）
        if not signals.empty:
            idx = pd.date_range(pd.to_datetime(self.start_date), pd.to_datetime(self.end_date), freq="B")
            signals = signals.reindex(idx, copy=False)  # 交易日对齐（如你有交易日历可改用交易日历）
        return signals

    def run(self, top_k: int = Config.TOP_K) -> None:
        signals = self.generate_signal_matrix()
        if signals is None or signals.empty:
            return

        print("🔍 Filtering Active Universe...")
        valid_signals = signals.replace(-1, np.nan)
        daily_ranks = valid_signals.rank(axis=1, ascending=False)
        active_mask = (daily_ranks <= int(top_k) * 2).any(axis=0)
        active_codes = signals.columns[active_mask].astype(str).tolist()

        print(f"Active Universe Size: {len(active_codes)}")
        if not active_codes:
            return

        cerebro = bt.Cerebro()
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.set_checksubmit(False)

        cerebro.broker.addcommissioninfo(AShareCommission())
        cerebro.broker.set_slippage_perc(float(getattr(Config, "SLIPPAGE", 0.0)))

        print(f"📂 Loading Market Data ({self.adjust.upper()})...")
        loaded_cnt = 0
        for code in tqdm(active_codes, desc="LoadData"):
            df_bt = _prepare_bt_price_df(code, self.start_date, self.end_date, self.adjust)
            if df_bt is None or df_bt.empty:
                continue
            data = AShareDataFeed(dataname=df_bt, name=str(code), plot=False)
            cerebro.adddata(data)
            loaded_cnt += 1

        if loaded_cnt == 0:
            print("❌ No valid market data to backtest.")
            return

        print(f"🚀 Launching Walk-Forward Backtest (Top {top_k})...")
        cerebro.addstrategy(
            ModelDrivenStrategy,
            signals=signals,
            top_k=int(top_k),
            hold_days=int(getattr(Config, "PRED_LEN", 5)),
        )

        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="returns")
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=float(getattr(Config, "RISK_FREE_RATE", 0.0)))
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

        results = cerebro.run()
        strat = results[0]
        self._generate_report(strat, cerebro)

    def _generate_report(self, strat, cerebro) -> None:
        final_val = float(cerebro.broker.getvalue())
        ret = (final_val - self.initial_cash) / self.initial_cash

        sharpe_an = strat.analyzers.sharpe.get_analysis()
        sharpe = sharpe_an.get("sharperatio") or 0.0

        dd_an = strat.analyzers.drawdown.get_analysis()
        max_dd = dd_an.get("max", {}).get("drawdown", 0.0)

        print("\n" + "=" * 40)
        print("📊 [Backtest Report]")
        print(f"Range: {self.start_date} ~ {self.end_date}")
        print(f"Equity: {self.initial_cash:,.0f} -> {final_val:,.2f}")
        print(f"Return: {ret:.2%}")
        print(f"Sharpe: {sharpe:.2f}")
        print(f"Max DD: {max_dd:.2f}%")
        print("=" * 40)

        ret_series = pd.Series(strat.analyzers.returns.get_analysis())
        ret_series.index = pd.to_datetime(ret_series.index)
        cumulative = (1 + ret_series).cumprod()

        # 对比基准（CSI300）
        try:
            bench = ak.stock_zh_index_daily(symbol=Config.BENCHMARK_SYMBOL)
            bench["date"] = pd.to_datetime(bench["date"])
            bench.set_index("date", inplace=True)
            mask = (bench.index >= ret_series.index.min()) & (bench.index <= ret_series.index.max())
            bench = bench.loc[mask]
            bench_ret = bench["close"].pct_change().fillna(0.0)
            bench_cum = (1 + bench_ret).cumprod()

            plt.figure(figsize=(12, 6))
            plt.plot(cumulative.index, cumulative, label="Strategy (Net)", linewidth=1.5)
            plt.plot(bench_cum.index, bench_cum, label="CSI 300", linestyle="--", alpha=0.7)
        except Exception:
            cumulative.plot(figsize=(12, 6), label="Strategy (Net)")

        plt.title(f"Walk-Forward Equity Curve (Slippage={float(getattr(Config, 'SLIPPAGE', 0.0)) * 10000:.0f} bp)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        out_path = os.path.join(Config.OUTPUT_DIR, "walk_forward_result.png")
        plt.savefig(out_path)
        print(f"📈 Chart saved to {out_path}")


# =============================================================================
# External API: keep unchanged
# =============================================================================

def run_walk_forward_backtest(
    start_date: str,
    end_date: str,
    initial_cash: float,
    top_k: int = Config.TOP_K,
    adjust: str = "qfq",
):
    engine = WalkForwardBacktester(start_date, end_date, initial_cash, adjust=adjust)
    engine.run(top_k=top_k)


# =============================================================================
# 简化版 TopK 策略：sanity check / 验证价格 + masks
# =============================================================================

class TopKStrategy(bt.Strategy):
    params = (
        ("top_k", Config.TOP_K),
        ("hold_days", Config.PRED_LEN),
        ("min_volume_percent", Config.MIN_VOLUME_PERCENT),
        ("cash_buffer", Config.CASH_BUFFER),
    )

    def __init__(self):
        self.hold_time: Dict[str, int] = {}

    @staticmethod
    def _mask_is_true(v) -> bool:
        try:
            fv = float(v)
            return np.isfinite(fv) and fv > 0.0
        except Exception:
            return False

    def next(self):
        holding_datas = [d for d in self.datas if self.getposition(d).size > 0]
        current_pos_count = len(holding_datas)

        freed_slots = 0
        estimated_cash_release = 0.0

        for data in holding_datas:
            name = str(data._name)
            self.hold_time[name] = self.hold_time.get(name, 0) + 1

            sellable = True
            try:
                sellable = self._mask_is_true(data.sellable[0])
            except Exception:
                pass

            if self.hold_time[name] >= int(self.p.hold_days) and sellable:
                pos = self.getposition(data).size
                self.close(data=data)
                self.hold_time[name] = 0
                freed_slots += 1
                estimated_cash_release += abs(pos) * float(data.close[0])

        current_cash = float(self.broker.get_cash())
        total_available_cash = current_cash + estimated_cash_release
        if total_available_cash < 1000:
            return

        real_slots_gap = int(self.p.top_k) - current_pos_count
        effective_slots = real_slots_gap + freed_slots
        if effective_slots <= 0:
            return

        target = total_available_cash / effective_slots * float(self.p.cash_buffer)

        buy_cnt = 0
        for data in self.datas:
            if buy_cnt >= effective_slots:
                break
            if self.getposition(data).size != 0:
                continue

            price = float(data.close[0])
            vol = float(data.volume[0])
            if price <= 1e-6 or vol <= 1e-6:
                continue

            buyable = True
            try:
                buyable = self._mask_is_true(data.buyable[0])
            except Exception:
                pass
            if not buyable:
                continue

            size = int(target / price / 100) * 100
            if size < 100:
                continue

            limit_size = int((vol * float(self.p.min_volume_percent)) // 100) * 100
            final_size = min(size, limit_size)

            if final_size >= 100:
                self.buy(data=data, size=final_size)
                self.hold_time[str(data._name)] = 0
                buy_cnt += 1


# =============================================================================
# 单次回测 & 性能归因（支持 with/without fees）
# =============================================================================

class PerformanceAnalyzer:
    @staticmethod
    def get_benchmark(start_date, end_date):
        try:
            df = ak.stock_zh_index_daily(symbol=Config.BENCHMARK_SYMBOL)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
            return df.loc[mask, "close"].pct_change().fillna(0.0)
        except Exception:
            return None

    @staticmethod
    def calculate_metrics(strategy_returns: pd.Series, benchmark_returns: Optional[pd.Series]):
        if benchmark_returns is None:
            return None
        df = pd.concat([strategy_returns, benchmark_returns], axis=1, join="inner")
        df.columns = ["Strategy", "Benchmark"]
        if len(df) < 10:
            return None

        Rp, Rm = df["Strategy"], df["Benchmark"]
        days = len(df)

        ann_p = (1 + Rp).prod() ** (252 / days) - 1
        ann_m = (1 + Rm).prod() ** (252 / days) - 1

        vol = Rp.std() * np.sqrt(252)
        sharpe = (ann_p - float(getattr(Config, "RISK_FREE_RATE", 0.0))) / (vol + 1e-9)

        cum = (1 + Rp).cumprod()
        dd = ((cum.cummax() - cum) / cum.cummax()).max()

        cov = np.cov(Rp, Rm)
        beta = cov[0, 1] / (cov[1, 1] + 1e-9)
        alpha = ann_p - (float(getattr(Config, "RISK_FREE_RATE", 0.0)) + beta * (ann_m - float(getattr(Config, "RISK_FREE_RATE", 0.0))))

        return {
            "Ann. Return": ann_p,
            "Benchmark Ret": ann_m,
            "Alpha": alpha,
            "Beta": beta,
            "Sharpe": sharpe,
            "Max Drawdown": dd,
            "Win Rate": (Rp > 0).mean(),
        }

    @staticmethod
    def plot_curve(strategy_returns: pd.Series, benchmark_returns: pd.Series):
        df = pd.concat([strategy_returns, benchmark_returns], axis=1, join="inner")
        df.columns = ["Strategy", "CSI 300"]
        (1 + df).cumprod().plot(figsize=(12, 6), grid=True)
        plt.savefig(os.path.join(Config.OUTPUT_DIR, "backtest_result.png"))


def run_single_backtest(
    codes: List[str],
    with_fees: bool = True,
    initial_cash: float = 1_000_000.0,
    top_k: int = Config.TOP_K,
    adjust: str = "qfq",
):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(float(initial_cash))
    cerebro.broker.set_checksubmit(False)

    if with_fees:
        cerebro.broker.addcommissioninfo(AShareCommission())
        cerebro.broker.set_slippage_perc(float(getattr(Config, "SLIPPAGE", 0.0)))
    else:
        cerebro.broker.setcommission(commission=0.0)

    loaded = False
    for code in codes:
        df_bt = _prepare_bt_price_df(str(code), None, None, adjust)
        if df_bt is None or df_bt.empty:
            continue
        if len(df_bt) > 250:
            df_bt = df_bt.iloc[-250:]
        data = AShareDataFeed(dataname=df_bt, name=str(code), plot=False)
        cerebro.adddata(data)
        loaded = True

    if not loaded:
        return None

    cerebro.addstrategy(TopKStrategy, top_k=int(top_k), hold_days=int(getattr(Config, "PRED_LEN", 5)))
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="ret")

    res = cerebro.run()
    strat = res[0]
    final_val = float(cerebro.broker.getvalue())
    prof = (final_val - float(initial_cash)) / float(initial_cash)

    ret_s = pd.Series(strat.analyzers.ret.get_analysis())
    ret_s.index = pd.to_datetime(ret_s.index)

    tr = strat.analyzers.trade.get_analysis()
    tot = tr.total.closed if "total" in tr else 0
    won = tr.won.total if "won" in tr else 0
    win_rate = won / tot if tot > 0 else 0.0

    return {
        "final_value": final_val,
        "profit_rate": prof,
        "win_rate": win_rate,
        "total_trades": tot,
        "returns": ret_s,
        "start_date": ret_s.index.min(),
        "end_date": ret_s.index.max(),
    }


def run_backtest(
    top_stocks_list,
    initial_cash: float = 1_000_000.0,
    top_k: int = Config.TOP_K,
    adjust: str = "qfq",
):
    """
    用一个简单 TopK 策略，对若干“代表性个股组合”做有/无费用对比回测。
    top_stocks_list: [(code, score, ...), ...]
    """
    print(f"\n>>> Launching Validation Backtest (Cash: {float(initial_cash):,.0f}, TopK: {int(top_k)}, Adjust: {adjust})")
    codes = [str(x[0]) for x in top_stocks_list[: int(top_k)]]
    if not codes:
        print("❌ Empty stock list.")
        return

    print("Comparing: Fees & Slippage vs. Frictionless...")
    res_fees = run_single_backtest(codes, True, float(initial_cash), int(top_k), adjust)
    res_no = run_single_backtest(codes, False, float(initial_cash), int(top_k), adjust)

    if not res_fees:
        print("❌ Backtest Failed (No Data)")
        return

    print(f"{'Metric':<15} | {'Production':<15} | {'Ideal':<15}")
    print("-" * 50)
    print(f"{'Equity':<15} | {res_fees['final_value']:<15,.2f} | {res_no['final_value']:<15,.2f}")
    print(f"{'Return':<15} | {res_fees['profit_rate']:<15.2%} | {res_no['profit_rate']:<15.2%}")
    print(f"{'Win Rate':<15} | {res_fees['win_rate']:<15.2%} | {res_no['win_rate']:<15.2%}")
    print("=" * 50)

    bench = PerformanceAnalyzer.get_benchmark(res_fees["start_date"], res_fees["end_date"])
    if bench is not None:
        m = PerformanceAnalyzer.calculate_metrics(res_fees["returns"], bench)
        if m:
            print(f"📊 Attribution: Alpha {m['Alpha']:.4f} | Sharpe {m['Sharpe']:.4f} | Excess {(m['Ann. Return'] - m['Benchmark Ret']):.2%}")
            PerformanceAnalyzer.plot_curve(res_fees["returns"], bench)
