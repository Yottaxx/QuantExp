import backtrader as bt
import pandas as pd
import numpy as np
import os
import torch
import akshare as ak
import matplotlib.pyplot as plt
from tqdm import tqdm
from .config import Config
from .model import PatchTSTForStock
from .data_provider import DataProvider

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
#  1. 交易环境模型 (Transaction Cost Analysis - TCA)
# ==============================================================================
class AShareCommission(bt.CommInfoBase):
    """
    A股印花税与佣金模型
    Buy: Commission only
    Sell: Commission + Stamp Duty
    """
    params = (('stocklike', True), ('commtype', bt.CommInfoBase.COMM_PERC),
              ('perc', 0.0003), ('stamp_duty', 0.0005), ('min_comm', 5.0))

    def _getcommission(self, size, price, pseudoexec):
        if size > 0:  # Buy
            return max(abs(size) * price * self.p.perc, self.p.min_comm)
        elif size < 0:  # Sell
            commission = max(abs(size) * price * self.p.perc, self.p.min_comm)
            stamp_duty = abs(size) * price * self.p.stamp_duty
            return commission + stamp_duty
        return 0.0


# [Jeff Dean Fix] Removed redundant StockSlippage class.
# Reason: It duplicated the built-in bt.SlippagePerc logic and caused import errors.
# We will use cerebro.broker.set_slippage_perc() directly in the engine.


# ==============================================================================
#  2. 核心策略 (Core Strategy)
# ==============================================================================
class ModelDrivenStrategy(bt.Strategy):
    """
    【Production Strategy】
    1. Dynamic Ranking Exit
    2. Sector-Specific Risk Control (STAR/ChiNext 20%)
    3. Strict Liquidity Constraints
    """
    params = (
        ('signals', None),
        ('top_k', Config.TOP_K),
        ('hold_days', Config.PRED_LEN),
        ('min_volume_percent', Config.MIN_VOLUME_PERCENT),
    )

    def __init__(self):
        self.hold_time = {}
        self.signal_dict = {}
        # Pre-process signals into a hash map for O(1) lookup
        if self.p.signals is not None:
            # [Jeff Dean Fix] Ensure signals are numeric to prevent 'nlargest' TypeError
            # Convert to numeric, coercing errors to NaN, then fill with -1 (ignore signal)
            clean_signals = self.p.signals.apply(pd.to_numeric, errors='coerce').fillna(-1.0)

            for date, row in clean_signals.iterrows():
                # vectorized filtering
                valid_row = row[row > -1]
                if not valid_row.empty:
                    top_codes = valid_row.nlargest(self.p.top_k).index.tolist()
                    self.signal_dict[date.date()] = top_codes

    def get_limit_threshold(self, code):
        """区分板块涨跌停幅度"""
        if code.startswith('688') or code.startswith('300'): return 0.20
        return 0.10

    def check_limit_status(self, data, code):
        """T日涨跌停检测 - Robust Implementation"""
        try:
            prev_close = data.close[-1]
            # 数据完整性检查
            if prev_close <= 1e-6: return False, False

            curr_close = data.close[0]
            curr_high = data.high[0]
            curr_low = data.low[0]

            limit = self.get_limit_threshold(code)
            # 使用 epsilon 防止浮点数精度问题
            epsilon = 1e-4

            is_limit_up = (curr_close >= prev_close * (1 + limit) - epsilon) and (abs(curr_close - curr_high) < epsilon)
            is_limit_down = (curr_close <= prev_close * (1 - limit) + epsilon) and (
                        abs(curr_close - curr_low) < epsilon)
            return is_limit_up, is_limit_down
        except IndexError:
            return False, False

    def next(self):
        current_date = self.data.datetime.date(0)
        target_codes = self.signal_dict.get(current_date, [])
        target_set = set(target_codes)  # O(1) lookup

        # --- Sell Logic ---
        # 遍历当前持仓，使用 items() 避免重复调用
        for data in self.datas:
            pos = self.getposition(data).size
            if pos > 0:
                name = data._name
                is_limit_up, is_limit_down = self.check_limit_status(data, name)

                # 跌停无法卖出
                if is_limit_down: continue

                should_sell = False
                # 逻辑：不在 TopK 池中 OR 持仓时间超限
                if name not in target_set: should_sell = True

                self.hold_time[name] = self.hold_time.get(name, 0) + 1
                if self.hold_time[name] >= self.p.hold_days: should_sell = True

                if should_sell:
                    self.close(data=data)
                    self.hold_time[name] = 0

        # --- Buy Logic ---
        if not target_codes: return
        cash = self.broker.get_cash()
        if cash < 5000: return

        # 计算可用槽位
        current_pos_count = sum(1 for d in self.datas if self.getposition(d).size > 0)
        slots_available = self.p.top_k - current_pos_count
        if slots_available <= 0: return

        # 预留 2% 资金防止滑点导致拒单
        target_val_per_slot = cash / slots_available * 0.98

        buy_count = 0
        for code in target_codes:
            if buy_count >= slots_available: break

            data = self.getdatabyname(code)
            if data is None: continue
            if self.getposition(data).size > 0: continue

            # 涨停无法买入
            is_limit_up, is_limit_down = self.check_limit_status(data, code)
            if is_limit_up: continue

            price = data.close[0]
            vol = data.volume[0]
            if price <= 1e-6 or vol <= 1e-6: continue

            # 向下取整到 100 股 (hand)
            raw_size = target_val_per_slot / price
            size = int(raw_size // 100) * 100
            if size < 100: continue

            # 流动性风控：不超过当日成交量的 X%
            limit_size = int((vol * self.p.min_volume_percent) // 100) * 100
            final_size = min(size, limit_size)

            if final_size >= 100:
                self.buy(data=data, size=final_size)
                self.hold_time[code] = 0
                buy_count += 1


# ==============================================================================
#  3. 滚动回测引擎 (Walk-Forward Engine)
# ==============================================================================
class WalkForwardBacktester:
    def __init__(self, start_date, end_date, initial_cash=1000000.0):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.device = Config.DEVICE
        self.model_path = f"{Config.OUTPUT_DIR}/final_model"

    def generate_signal_matrix(self):
        print(f"⏳ [Signal Gen] Generating historical signals ({self.start_date} ~ {self.end_date})...")

        if not os.path.exists(self.model_path):
            print("❌ Model artifact not found.")
            return None

        # Load Model
        model = PatchTSTForStock.from_pretrained(self.model_path).to(self.device)
        model.eval()

        # Load Data
        panel_df, feature_cols = DataProvider.load_and_process_panel(mode='predict')

        # Filter Date Range (Broadened for context)
        s_dt = pd.to_datetime(self.start_date) - pd.Timedelta(days=Config.CONTEXT_LEN * 2)
        e_dt = pd.to_datetime(self.end_date)
        mask = (panel_df['date'] >= s_dt) & (panel_df['date'] <= e_dt)
        df_sub = panel_df[mask].copy()

        results = []
        batch_inputs, batch_meta = [], []
        BATCH_SIZE = 2048  # VRAM friendly batch size

        print("🚀 Batch Inference Initiated...")
        grouped = df_sub.groupby('code')

        # Inference Loop
        for code, group in tqdm(grouped, desc="Inference"):
            if len(group) < Config.CONTEXT_LEN: continue

            feats = group[feature_cols].values.astype(np.float32)
            dates = group['date'].values

            # Sliding Window
            # Optimization: Use stride_tricks if speed is critical, loop is fine for daily freq
            valid_indices = range(len(group) - Config.CONTEXT_LEN + 1)

            for i in valid_indices:
                pred_date = pd.to_datetime(dates[i + Config.CONTEXT_LEN - 1])
                if pred_date < pd.to_datetime(self.start_date): continue

                batch_inputs.append(feats[i: i + Config.CONTEXT_LEN])
                batch_meta.append((pred_date, code))

                if len(batch_inputs) >= BATCH_SIZE:
                    self._flush_batch(model, batch_inputs, batch_meta, results)
                    batch_inputs, batch_meta = [], []

        if batch_inputs:
            self._flush_batch(model, batch_inputs, batch_meta, results)

        if not results:
            print("❌ No signals generated.")
            return None

        print("🏗️ Reconstructing Signal Matrix...")
        res_df = pd.DataFrame(results, columns=['date', 'code', 'score'])

        # Global Risk Control (Market Timing)
        daily_mean = res_df.groupby('date')['score'].mean()
        bear_days = daily_mean[daily_mean < 0.45].index
        res_df.loc[res_df['date'].isin(bear_days), 'score'] = -1
        res_df.loc[res_df['score'] < Config.MIN_SCORE_THRESHOLD, 'score'] = -1

        signal_matrix = res_df.pivot(index='date', columns='code', values='score')
        signal_matrix = signal_matrix.sort_index()

        return signal_matrix

    def _flush_batch(self, model, inputs, meta, res):
        t = torch.tensor(np.array(inputs), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            # Handle output shape robustness
            out = model(past_values=t).logits
            if out.dim() > 1: out = out.squeeze()
            s = out.cpu().numpy()

        if s.ndim == 0: s = [s]

        # Extend is faster than append in loop
        res.extend([(meta[i][0], meta[i][1], float(score)) for i, score in enumerate(s)])

    def run(self, top_k=Config.TOP_K):
        signals = self.generate_signal_matrix()
        if signals is None: return

        print("🔍 Filtering Active Universe...")
        # Reduce memory footprint by only loading active stocks
        valid_signals = signals.replace(-1, np.nan)
        daily_ranks = valid_signals.rank(axis=1, ascending=False)
        active_mask = (daily_ranks <= top_k * 2).any(axis=0)
        active_codes = signals.columns[active_mask].tolist()

        print(f"Active Universe Size: {len(active_codes)}")
        if not active_codes: return

        cerebro = bt.Cerebro()
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.addcommissioninfo(AShareCommission())

        # [Jeff Dean Fix] Use Built-in Slippage Configuration
        # Direct method call avoids import issues with SlippageBase and SlippagePerc
        cerebro.broker.set_slippage_perc(Config.SLIPPAGE)

        print("📂 Loading Market Data...")
        loaded_cnt = 0
        for code in tqdm(active_codes):
            fpath = os.path.join(Config.DATA_DIR, f"{code}.parquet")
            if not os.path.exists(fpath): continue
            try:
                df = pd.read_parquet(fpath)
                mask = (df.index >= pd.to_datetime(self.start_date)) & (df.index <= pd.to_datetime(self.end_date))
                df = df[mask]
                if df.empty: continue
                data = bt.feeds.PandasData(dataname=df, name=code, plot=False)
                cerebro.adddata(data)
                loaded_cnt += 1
            except Exception as e:
                # Log error silently or debug
                continue

        if loaded_cnt == 0:
            print("❌ No valid market data found.")
            return

        print(f"🚀 Launching Walk-Forward Backtest (Top {top_k})...")
        cerebro.addstrategy(
            ModelDrivenStrategy,
            signals=signals,
            top_k=top_k,
            hold_days=Config.PRED_LEN
        )

        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=Config.RISK_FREE_RATE)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

        results = cerebro.run()
        strat = results[0]
        self._generate_report(strat, cerebro)

    def _generate_report(self, strat, cerebro):
        final_val = cerebro.broker.getvalue()
        ret = (final_val - self.initial_cash) / self.initial_cash

        # Safely get analyzer results
        sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
        if sharpe is None: sharpe = 0

        max_dd = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)

        print("\n" + "=" * 40)
        print(f"📊 [Backtest Report]")
        print(f"Range: {self.start_date} ~ {self.end_date}")
        print(f"Equity: {self.initial_cash:,.0f} -> {final_val:,.2f}")
        print(f"Return: {ret:.2%}")
        print(f"Sharpe: {sharpe:.2f}")
        print(f"Max DD: {max_dd:.2%}")
        print("=" * 40)

        ret_series = pd.Series(strat.analyzers.returns.get_analysis())
        cumulative = (1 + ret_series).cumprod()

        # Benchmarking
        try:
            bench = ak.stock_zh_index_daily(symbol=Config.BENCHMARK_SYMBOL)
            bench['date'] = pd.to_datetime(bench['date'])
            bench.set_index('date', inplace=True)
            bench_ret = bench['close'].pct_change().reindex(ret_series.index).fillna(0)
            bench_cum = (1 + bench_ret).cumprod()

            plt.figure(figsize=(12, 6))
            plt.plot(cumulative.index, cumulative, label='Strategy (Net)', color='red', linewidth=1.5)
            plt.plot(bench_cum.index, bench_cum, label='CSI 300', color='gray', linestyle='--', alpha=0.7)
        except:
            cumulative.plot(figsize=(12, 6), label='Strategy (Net)')

        plt.title(f'Walk-Forward Equity Curve (Slippage={Config.SLIPPAGE * 1000}bp)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        out_path = os.path.join(Config.OUTPUT_DIR, "walk_forward_result.png")
        plt.savefig(out_path)
        print(f"📈 Chart saved to {out_path}")


def run_walk_forward_backtest(start_date, end_date, initial_cash, top_k=Config.TOP_K):
    engine = WalkForwardBacktester(start_date, end_date, initial_cash)
    engine.run(top_k=top_k)


# ==============================================================================
#  4. 验证回测 (Simple Validation Strategy)
# ==============================================================================

class TopKStrategy(bt.Strategy):
    """
    Simple TopK Validation without signal rotation
    """
    params = (
        ('top_k', Config.TOP_K),
        ('hold_days', Config.PRED_LEN),
        ('min_volume_percent', Config.MIN_VOLUME_PERCENT)
    )

    def __init__(self):
        self.hold_time = {}

    def next(self):
        # 1. Sell Logic
        for data in self.datas:
            if self.getposition(data).size > 0:
                self.hold_time[data._name] = self.hold_time.get(data._name, 0) + 1
                if self.hold_time[data._name] >= self.p.hold_days:
                    self.close(data=data)
                    self.hold_time[data._name] = 0

        # 2. Buy Logic
        cash = self.broker.get_cash()
        if cash < 5000: return
        current_pos = len([d for d in self.datas if self.getposition(d).size > 0])
        slots = self.p.top_k - current_pos
        if slots <= 0: return
        target = cash / slots * 0.98

        buy_cnt = 0
        for data in self.datas:
            if buy_cnt >= slots: break
            if self.getposition(data).size == 0:
                price = data.close[0];
                vol = data.volume[0]
                if price <= 1e-6 or vol <= 1e-6: continue

                # Simple Limit-Up check (approx)
                prev = data.close[-1]
                if prev > 0 and data.close[0] >= prev * 1.095: continue

                size = int(target / price / 100) * 100
                if size < 100: continue

                limit_size = int(vol * self.p.min_volume_percent) // 100 * 100
                final_size = min(size, limit_size)

                if final_size >= 100:
                    self.buy(data=data, size=final_size)
                    self.hold_time[data._name] = 0
                    buy_cnt += 1


class PerformanceAnalyzer:
    @staticmethod
    def get_benchmark(start_date, end_date):
        try:
            df = ak.stock_zh_index_daily(symbol=Config.BENCHMARK_SYMBOL)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
            return df.loc[mask, 'close'].pct_change().fillna(0)
        except Exception:
            return None

    @staticmethod
    def calculate_metrics(strategy_returns, benchmark_returns):
        df = pd.concat([strategy_returns, benchmark_returns], axis=1, join='inner')
        df.columns = ['Strategy', 'Benchmark']
        if len(df) < 10: return None

        Rp, Rm = df['Strategy'], df['Benchmark']
        days = len(df)

        # Annualized Return
        ann_p = (1 + Rp).prod() ** (252 / days) - 1
        ann_m = (1 + Rm).prod() ** (252 / days) - 1

        # Volatility & Sharpe
        vol = Rp.std() * np.sqrt(252)
        sharpe = (ann_p - Config.RISK_FREE_RATE) / (vol + 1e-9)

        # Max Drawdown
        cum = (1 + Rp).cumprod()
        dd = ((cum.cummax() - cum) / cum.cummax()).max()

        # Alpha / Beta
        cov = np.cov(Rp, Rm)
        beta = cov[0, 1] / (cov[1, 1] + 1e-9)
        alpha = ann_p - (Config.RISK_FREE_RATE + beta * (ann_m - Config.RISK_FREE_RATE))

        return {
            "Ann. Return": ann_p,
            "Benchmark Ret": ann_m,
            "Alpha": alpha,
            "Beta": beta,
            "Sharpe": sharpe,
            "Max Drawdown": dd,
            "Win Rate": (Rp > 0).mean()
        }

    @staticmethod
    def plot_curve(strategy_returns, benchmark_returns):
        df = pd.concat([strategy_returns, benchmark_returns], axis=1, join='inner')
        df.columns = ['Strategy', 'CSI 300']
        (1 + df).cumprod().plot(figsize=(12, 6), grid=True)
        plt.savefig(os.path.join(Config.OUTPUT_DIR, "backtest_result.png"))


def run_single_backtest(codes, with_fees=True, initial_cash=1000000.0, top_k=Config.TOP_K):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)

    if with_fees:
        cerebro.broker.addcommissioninfo(AShareCommission())
        # [Jeff Dean Fix] Direct API call
        # Avoids class import issues entirely.
        cerebro.broker.set_slippage_perc(Config.SLIPPAGE)
    else:
        cerebro.broker.setcommission(commission=0.0)

    loaded = False
    for code in codes:
        fpath = os.path.join(Config.DATA_DIR, f"{code}.parquet")
        if not os.path.exists(fpath): continue
        try:
            df = pd.read_parquet(fpath)
            if len(df) > 250: df = df.iloc[-250:]
            data = bt.feeds.PandasData(dataname=df, fromdate=df.index[0], plot=False)
            cerebro.adddata(data, name=code)
            loaded = True
        except:
            continue
    if not loaded: return None

    cerebro.addstrategy(TopKStrategy, top_k=top_k, hold_days=Config.PRED_LEN)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='ret')

    res = cerebro.run()
    strat = res[0]
    final = cerebro.broker.getvalue()
    prof = (final - initial_cash) / initial_cash

    ret_s = pd.Series(strat.analyzers.ret.get_analysis())
    ret_s.index = pd.to_datetime(ret_s.index)

    tr = strat.analyzers.trade.get_analysis()
    tot = tr.total.closed if 'total' in tr else 0
    won = tr.won.total if 'won' in tr else 0
    win_rate = won / tot if tot > 0 else 0

    return {
        "final_value": final,
        "profit_rate": prof,
        "win_rate": win_rate,
        "total_trades": tot,
        "returns": ret_s,
        "start_date": strat.data.datetime.date(0),
        "end_date": strat.data.datetime.date(-1)
    }


def run_backtest(top_stocks_list, initial_cash=1000000.0, top_k=Config.TOP_K):
    """
    Entry point for validation
    """
    print(f"\n>>> Launching Validation Backtest (Cash: {initial_cash:,.0f}, TopK: {top_k})")
    codes = [x[0] for x in top_stocks_list[:top_k]]
    if not codes:
        print("❌ Empty stock list.")
        return

    print("Comparing: Fees & Slippage vs. Frictionless...")
    res_fees = run_single_backtest(codes, True, initial_cash, top_k)
    res_no = run_single_backtest(codes, False, initial_cash, top_k)

    if not res_fees:
        print("❌ Backtest Failed (No Data)")
        return

    print(f"{'Metric':<15} | {'Production':<15} | {'Ideal':<15}")
    print("-" * 50)
    print(f"{'Equity':<15} | {res_fees['final_value']:<15,.2f} | {res_no['final_value']:<15,.2f}")
    print(f"{'Return':<15} | {res_fees['profit_rate']:<15.2%} | {res_no['profit_rate']:<15.2%}")
    print(f"{'Win Rate':<15} | {res_fees['win_rate']:<15.2%} | {res_no['win_rate']:<15.2%}")
    print("=" * 50)

    bench = PerformanceAnalyzer.get_benchmark(res_fees['start_date'], res_fees['end_date'])
    if bench is not None:
        m = PerformanceAnalyzer.calculate_metrics(res_fees['returns'], bench)
        if m:
            print(
                f"📊 Attribution: Alpha {m['Alpha']:.4f} | Sharpe {m['Sharpe']:.4f} | Excess {m['Ann. Return'] - m['Benchmark Ret']:.2%}")
            PerformanceAnalyzer.plot_curve(res_fees['returns'], bench)