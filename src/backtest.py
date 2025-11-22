import backtrader as bt
import pandas as pd
import numpy as np
import os
import akshare as ak
import matplotlib.pyplot as plt
from tqdm import tqdm
from .config import Config
from .model import PatchTSTForStock
from .data_provider import DataProvider
import torch

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
#  1. 费率模型
# ==============================================================================
class AShareCommission(bt.CommInfoBase):
    params = (('stocklike', True), ('commtype', bt.CommInfoBase.COMM_PERC),
              ('perc', 0.0003), ('stamp_duty', 0.0005), ('min_comm', 5.0))

    def _getcommission(self, size, price, pseudoexec):
        if size > 0:
            return max(abs(size) * price * self.p.perc, self.p.min_comm)
        elif size < 0:
            return max(abs(size) * price * self.p.perc, self.p.min_comm) + abs(size) * price * self.p.stamp_duty
        return 0.0


# ==============================================================================
#  2. 绩效分析引擎 (补全缺失部分)
# ==============================================================================
class PerformanceAnalyzer:
    @staticmethod
    def get_benchmark(start_date, end_date):
        """获取沪深300基准数据"""
        try:
            df = ak.stock_zh_index_daily(symbol=Config.BENCHMARK_SYMBOL)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
            return df.loc[mask, 'close'].pct_change().fillna(0)
        except Exception as e:
            print(f"⚠️ 无法获取基准数据: {e}")
            return None

    @staticmethod
    def calculate_metrics(strategy_returns, benchmark_returns):
        """计算 Alpha, Beta, Sharpe, MaxDD 等核心指标"""
        # 对齐日期索引
        df = pd.concat([strategy_returns, benchmark_returns], axis=1, join='inner')
        df.columns = ['Strategy', 'Benchmark']
        if len(df) < 10: return None

        R_p = df['Strategy']
        R_m = df['Benchmark']

        # 1. 年化收益率
        days = len(df)
        total_ret_p = (1 + R_p).prod() - 1
        ann_ret_p = (1 + total_ret_p) ** (252 / days) - 1

        total_ret_m = (1 + R_m).prod() - 1
        ann_ret_m = (1 + total_ret_m) ** (252 / days) - 1

        # 2. 波动率
        vol_p = R_p.std() * np.sqrt(252)

        # 3. 夏普比率
        sharpe = (ann_ret_p - Config.RISK_FREE_RATE) / (vol_p + 1e-9)

        # 4. 最大回撤
        cum_returns = (1 + R_p).cumprod()
        drawdown = (cum_returns.cummax() - cum_returns) / cum_returns.cummax()
        max_dd = drawdown.max()

        # 5. Beta & Alpha
        cov_matrix = np.cov(R_p, R_m)
        beta = cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-9)
        alpha = ann_ret_p - (Config.RISK_FREE_RATE + beta * (ann_ret_m - Config.RISK_FREE_RATE))

        return {
            "Ann. Return": ann_ret_p,
            "Benchmark Ret": ann_ret_m,
            "Alpha": alpha,
            "Beta": beta,
            "Sharpe": sharpe,
            "Max Drawdown": max_dd,
            "Win Rate": (R_p > 0).mean()
        }

    @staticmethod
    def plot_curve(strategy_returns, benchmark_returns):
        """绘制资金曲线对比图"""
        df = pd.concat([strategy_returns, benchmark_returns], axis=1, join='inner')
        df.columns = ['Strategy', 'CSI 300']
        cumulative = (1 + df).cumprod()

        plt.figure(figsize=(12, 6))
        plt.plot(cumulative.index, cumulative['Strategy'], label='Strategy', color='red', linewidth=2)
        plt.plot(cumulative.index, cumulative['CSI 300'], label='Benchmark', color='gray', linestyle='--', alpha=0.7)

        plt.title('Strategy Equity Curve vs Benchmark')
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(Config.OUTPUT_DIR, "backtest_result.png")
        plt.savefig(save_path)
        print(f"📈 资金曲线图已保存至: {save_path}")


# ==============================================================================
#  3. 策略与回测引擎
# ==============================================================================

class ModelDrivenStrategy(bt.Strategy):
    """
    Walk-Forward 专用策略：每日根据信号动态换仓
    """
    params = (
        ('signals', None),
        ('top_k', 5),
        ('hold_days', 5),
        ('min_volume_percent', Config.MIN_VOLUME_PERCENT),
    )

    def __init__(self):
        self.hold_time = {}
        self.signal_dict = {}
        if self.p.signals is not None:
            for date, row in self.p.signals.iterrows():
                # 过滤掉无效分数(-1或NaN)
                valid_row = row[row > 0]
                if not valid_row.empty:
                    top_codes = valid_row.nlargest(self.p.top_k).index.tolist()
                    self.signal_dict[date.date()] = top_codes

    def next(self):
        current_date = self.data.datetime.date(0)

        # 1. 卖出
        for data in self.datas:
            if self.getposition(data).size > 0:
                name = data._name
                self.hold_time[name] = self.hold_time.get(name, 0) + 1
                if self.hold_time[name] >= self.p.hold_days:
                    self.close(data=data)
                    self.hold_time[name] = 0

        # 2. 买入
        target_codes = self.signal_dict.get(current_date, [])
        if not target_codes: return

        cash = self.broker.get_cash()
        if cash < 5000: return

        current_pos = len([d for d in self.datas if self.getposition(d).size > 0])
        slots = self.p.top_k - current_pos
        if slots <= 0: return

        target_val = cash / slots * 0.98

        buy_cnt = 0
        for code in target_codes:
            if buy_cnt >= slots: break
            data = self.getdatabyname(code)
            if data is None: continue

            if self.getposition(data).size == 0:
                price = data.close[0]
                vol = data.volume[0]
                if price <= 0 or vol <= 0: continue

                size = int(target_val / price / 100) * 100

                # 风控
                if size < 100: continue
                limit_size = int(vol * 100 * self.p.min_volume_percent / 100) * 100
                if size > limit_size: size = limit_size

                if size >= 100:
                    self.buy(data=data, size=size)
                    self.hold_time[code] = 0
                    buy_cnt += 1


class WalkForwardBacktester:
    def __init__(self, start_date, end_date, initial_cash=1000000.0):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.device = Config.DEVICE
        self.model_path = f"{Config.OUTPUT_DIR}/final_model"

    def generate_signal_matrix(self):
        print(f"⏳ [Signal Gen] 正在生成历史信号 ({self.start_date} ~ {self.end_date})...")

        if not os.path.exists(self.model_path):
            print("❌ 模型文件不存在，请先训练")
            return None

        model = PatchTSTForStock.from_pretrained(self.model_path).to(self.device)
        model.eval()

        panel_df, feature_cols = DataProvider.load_and_process_panel(mode='predict')

        s_dt = pd.to_datetime(self.start_date) - pd.Timedelta(days=Config.CONTEXT_LEN * 2)
        e_dt = pd.to_datetime(self.end_date)
        mask = (panel_df['date'] >= s_dt) & (panel_df['date'] <= e_dt)
        df_sub = panel_df[mask].copy()

        results = []
        batch_inputs, batch_meta = [], []

        print("正在批量推理历史数据...")
        grouped = df_sub.groupby('code')

        for code, group in tqdm(grouped, desc="Inference"):
            if len(group) < Config.CONTEXT_LEN: continue

            feats = group[feature_cols].values.astype(np.float32)
            dates = group['date'].values

            for i in range(len(group) - Config.CONTEXT_LEN + 1):
                pred_date = pd.to_datetime(dates[i + Config.CONTEXT_LEN - 1])
                if pred_date < pd.to_datetime(self.start_date): continue

                batch_inputs.append(feats[i: i + Config.CONTEXT_LEN])
                batch_meta.append((pred_date, code))

                if len(batch_inputs) >= 2048:
                    self._flush_batch(model, batch_inputs, batch_meta, results)
                    batch_inputs, batch_meta = [], []

        if batch_inputs:
            self._flush_batch(model, batch_inputs, batch_meta, results)

        if not results:
            print("❌ 未生成任何信号")
            return None

        print("正在重构信号矩阵...")
        res_df = pd.DataFrame(results, columns=['date', 'code', 'score'])

        # 回测风控：熊市/低分 熔断
        daily_mean = res_df.groupby('date')['score'].mean()
        bear_days = daily_mean[daily_mean < 0.45].index  # 假设阈值
        res_df.loc[res_df['date'].isin(bear_days), 'score'] = -1
        res_df.loc[res_df['score'] < Config.MIN_SCORE_THRESHOLD, 'score'] = -1

        signal_matrix = res_df.pivot(index='date', columns='code', values='score').sort_index()

        return signal_matrix

    def _flush_batch(self, model, inputs, meta, res):
        t = torch.tensor(np.array(inputs), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            s = model(past_values=t).logits.squeeze().cpu().numpy()
        if s.ndim == 0: s = [s]
        for i, score in enumerate(s):
            res.append((meta[i][0], meta[i][1], float(score)))

    def run(self, top_k=5):
        signals = self.generate_signal_matrix()
        if signals is None: return

        print("正在筛选活跃股票池...")
        valid_signals = signals.replace(-1, np.nan)
        daily_ranks = valid_signals.rank(axis=1, ascending=False)
        active_mask = (daily_ranks <= top_k * 2).any(axis=0)
        active_codes = signals.columns[active_mask].tolist()

        if not active_codes: return

        cerebro = bt.Cerebro()
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.addcommissioninfo(AShareCommission())

        print("正在加载回测行情数据...")
        loaded_cnt = 0
        for code in tqdm(active_codes):
            fpath = os.path.join(Config.DATA_DIR, f"{code}.parquet")
            if not os.path.exists(fpath): continue
            try:
                df = pd.read_parquet(fpath)
                df = df[(df.index >= pd.to_datetime(self.start_date)) & (df.index <= pd.to_datetime(self.end_date))]
                if df.empty: continue
                data = bt.feeds.PandasData(dataname=df, name=code, plot=False)
                cerebro.adddata(data)
                loaded_cnt += 1
            except:
                continue

        if loaded_cnt == 0: return

        print(f"🚀 开始 Walk-Forward 回测 (Top {top_k})...")
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
        sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
        max_dd = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)

        print("\n" + "=" * 40)
        print(f"📊 [Walk-Forward 真实回测报告]")
        print(f"区间: {self.start_date} ~ {self.end_date}")
        print(f"资金: {self.initial_cash:,.0f} -> {final_val:,.2f}")
        print(f"收益: {ret:.2%}")
        print(f"夏普: {sharpe:.2f}")
        print(f"回撤: {max_dd:.2%}")
        print("=" * 40)

        ret_series = pd.Series(strat.analyzers.returns.get_analysis())

        # 调用 PerformanceAnalyzer 绘图 (现在此类已存在)
        try:
            bench = ak.stock_zh_index_daily(symbol=Config.BENCHMARK_SYMBOL)
            bench['date'] = pd.to_datetime(bench['date'])
            bench.set_index('date', inplace=True)
            bench_ret = bench['close'].pct_change().reindex(ret_series.index).fillna(0)

            PerformanceAnalyzer.plot_curve(ret_series, bench_ret)
        except:
            print("⚠️ 基准数据获取失败，仅保存策略曲线")
            (1 + ret_series).cumprod().plot(figsize=(12, 6), title='Equity Curve')
            plt.savefig(os.path.join(Config.OUTPUT_DIR, "walk_forward_result.png"))


def run_walk_forward_backtest(start_date, end_date, initial_cash, top_k):
    engine = WalkForwardBacktester(start_date, end_date, initial_cash)
    engine.run(top_k=top_k)


# --- 简单的 TopKStrategy (用于 predict 后的验证性回测) ---
class TopKStrategy(bt.Strategy):
    params = (('top_k', 5), ('hold_days', 5), ('min_volume_percent', Config.MIN_VOLUME_PERCENT))

    def __init__(self):
        self.hold_time = {}

    def next(self):
        for data in self.datas:
            if self.getposition(data).size > 0:
                self.hold_time[data._name] = self.hold_time.get(data._name, 0) + 1
                if self.hold_time[data._name] >= self.p.hold_days:
                    self.close(data=data);
                    self.hold_time[data._name] = 0
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
                if price <= 0 or vol <= 0: continue
                size = int(target / price / 100) * 100
                if size < 100: continue
                limit_size = int(vol * 100 * self.p.min_volume_percent / 100) * 100
                if size > limit_size: size = limit_size
                if size >= 100: self.buy(data=data, size=size); self.hold_time[data._name] = 0; buy_cnt += 1


def run_single_backtest(codes, with_fees=True, initial_cash=1000000.0, top_k=5):
    cerebro = bt.Cerebro();
    cerebro.broker.setcash(initial_cash)
    if with_fees:
        cerebro.broker.addcommissioninfo(AShareCommission())
    else:
        cerebro.broker.setcommission(commission=0.0)
    loaded = False
    for code in codes:
        fpath = os.path.join(Config.DATA_DIR, f"{code}.parquet")
        if not os.path.exists(fpath): continue
        try:
            df = pd.read_parquet(fpath);
            start = pd.to_datetime(Config.START_DATE)
            if len(df) > 250: df = df.iloc[-250:]
            data = bt.feeds.PandasData(dataname=df, fromdate=df.index[0], plot=False)
            cerebro.adddata(data, name=code);
            loaded = True
        except:
            continue
    if not loaded: return None

    cerebro.addstrategy(TopKStrategy, top_k=top_k, hold_days=Config.PRED_LEN)

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='ret')
    res = cerebro.run();
    strat = res[0]
    final = cerebro.broker.getvalue();
    prof = (final - initial_cash) / initial_cash
    ret_s = pd.Series(strat.analyzers.ret.get_analysis());
    ret_s.index = pd.to_datetime(ret_s.index)
    tr = strat.analyzers.trade.get_analysis();
    tot = tr.total.closed if 'total' in tr else 0
    won = tr.won.total if 'won' in tr else 0;
    win_rate = won / tot if tot > 0 else 0
    return {"final_value": final, "profit_rate": prof, "win_rate": win_rate, "total_trades": tot, "returns": ret_s,
            "start_date": strat.data.datetime.date(0), "end_date": strat.data.datetime.date(-1)}


def run_backtest(top_stocks_list, initial_cash=1000000.0, top_k=5):
    print(f"\n>>> 启动验证性回测 (资金: {initial_cash:,.0f}, TopK: {top_k})")
    codes = [x[0] for x in top_stocks_list[:top_k]]
    if not codes: return
    res_fees = run_single_backtest(codes, True, initial_cash, top_k)
    res_no = run_single_backtest(codes, False, initial_cash, top_k)
    if not res_fees: return
    print(f"{'指标':<15} | {'含费':<15} | {'无费':<15}")
    print("-" * 50)
    print(f"{'最终市值':<15} | {res_fees['final_value']:<15,.2f} | {res_no['final_value']:<15,.2f}")
    print(f"{'收益率':<15} | {res_fees['profit_rate']:<15.2%} | {res_no['profit_rate']:<15.2%}")
    print(f"{'胜率':<15} | {res_fees['win_rate']:<15.2%} | {res_no['win_rate']:<15.2%}")
    print("=" * 50)
    bench = PerformanceAnalyzer.get_benchmark(res_fees['start_date'], res_fees['end_date'])
    if bench is not None:
        m = PerformanceAnalyzer.calculate_metrics(res_fees['returns'], bench)
        if m:
            print(
                f"📊 绩效归因: Alpha {m['Alpha']:.4f} | Sharpe {m['Sharpe']:.4f} | Excess {m['Ann. Return'] - m['Benchmark Ret']:.2%}")
            PerformanceAnalyzer.plot_curve(res_fees['returns'], bench)