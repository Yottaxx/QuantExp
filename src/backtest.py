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

# 设置 Matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
#  1. 交易环境模型 (费率 + 滑点)
# ==============================================================================
class AShareCommission(bt.CommInfoBase):
    """A股费率：佣金万三，印花税万五(卖出)，最低5元"""
    params = (('stocklike', True), ('commtype', bt.CommInfoBase.COMM_PERC),
              ('perc', 0.0003), ('stamp_duty', 0.0005), ('min_comm', 5.0))

    def _getcommission(self, size, price, pseudoexec):
        if size > 0:  # 买入
            return max(abs(size) * price * self.p.perc, self.p.min_comm)
        elif size < 0:  # 卖出
            commission = max(abs(size) * price * self.p.perc, self.p.min_comm)
            stamp_duty = abs(size) * price * self.p.stamp_duty
            return commission + stamp_duty
        return 0.0


class StockSlippage(bt.SlippageBase):
    """
    【新增】模拟冲击成本
    固定百分比滑点：无论买卖，成交价都会向不利方向偏移
    """
    params = (('perc', Config.SLIPPAGE),)

    def _calculate(self, order, price, parent=None):
        if order.isbuy():
            return price * (1 + self.p.perc)
        elif order.issell():
            return price * (1 - self.p.perc)
        return price


# ==============================================================================
#  2. 核心策略：信号驱动型 (Walk-Forward)
# ==============================================================================
class ModelDrivenStrategy(bt.Strategy):
    """
    【Walk-Forward 专用策略 - 增强版】
    包含：
    1. 动态止盈/止损 (优胜劣汰)
    2. 严格的涨跌停风控 (区分 10% 和 20%)
    3. 流动性限制
    """
    params = (
        ('signals', None),  # 信号矩阵 (DataFrame)
        ('top_k', Config.TOP_K),
        ('hold_days', Config.PRED_LEN),
        ('min_volume_percent', Config.MIN_VOLUME_PERCENT),
    )

    def __init__(self):
        self.hold_time = {}
        self.signal_dict = {}
        if self.p.signals is not None:
            # 预处理每日 TopK，加速查找
            for date, row in self.p.signals.iterrows():
                valid_row = row[row > -1]
                if not valid_row.empty:
                    top_codes = valid_row.nlargest(self.p.top_k).index.tolist()
                    self.signal_dict[date.date()] = top_codes

    def get_limit_threshold(self, code):
        """获取该股票的涨跌停幅度 (20% 或 10%)"""
        if code.startswith('688') or code.startswith('300'):
            return 0.20
        else:
            return 0.10

    def check_limit_status(self, data, code):
        """
        检查今日(T)是否涨跌停
        返回: (is_limit_up, is_limit_down)
        """
        try:
            prev_close = data.close[-1]
            curr_close = data.close[0]
            curr_high = data.high[0]
            curr_low = data.low[0]

            if prev_close == 0: return False, False

            limit = self.get_limit_threshold(code)

            # 判定涨停：收盘价涨幅接近 limit 且 收盘价等于最高价 (封板)
            is_limit_up = (curr_close >= prev_close * (1 + limit - 0.005)) and (curr_close == curr_high)

            # 判定跌停
            is_limit_down = (curr_close <= prev_close * (1 - limit + 0.005)) and (curr_close == curr_low)

            return is_limit_up, is_limit_down
        except:
            return False, False

    def next(self):
        current_date = self.data.datetime.date(0)
        target_codes = self.signal_dict.get(current_date, [])

        # --- 1. 动态持仓管理 (卖出) ---
        for data in self.datas:
            if self.getposition(data).size > 0:
                name = data._name  # stock code

                # A. 涨跌停检查 (如果跌停，无法卖出，跳过)
                is_limit_up, is_limit_down = self.check_limit_status(data, name)
                if is_limit_down:
                    continue

                    # B. 动态优胜劣汰
                # 规则：如果该股票不再属于今日的 Top K，则清仓
                should_sell = False

                if name not in target_codes:
                    should_sell = True

                self.hold_time[name] = self.hold_time.get(name, 0) + 1
                if self.hold_time[name] >= self.p.hold_days:
                    should_sell = True

                if should_sell:
                    self.close(data=data)
                    self.hold_time[name] = 0

        # --- 2. 信号开仓 (买入) ---
        if not target_codes: return

        cash = self.broker.get_cash()
        if cash < 5000: return

        current_pos = len([d for d in self.datas if self.getposition(d).size > 0])
        slots_available = self.p.top_k - current_pos
        if slots_available <= 0: return

        target_val = cash / slots_available * 0.98

        buy_count = 0
        for code in target_codes:
            if buy_count >= slots_available: break

            data = self.getdatabyname(code)
            if data is None: continue

            if self.getposition(data).size > 0: continue

            # A. 涨跌停风控 (如果今日涨停，明日大概率买不进，跳过)
            is_limit_up, is_limit_down = self.check_limit_status(data, code)
            if is_limit_up:
                continue

            price = data.close[0]
            vol = data.volume[0]

            if price <= 0 or vol <= 0: continue

            size = int(target_val / price / 100) * 100
            if size < 100: continue

            # B. 流动性限制
            limit_size = int(vol * self.p.min_volume_percent) // 100 * 100
            if size > limit_size: size = limit_size

            if size >= 100:
                self.buy(data=data, size=size)
                self.hold_time[code] = 0
                buy_count += 1


# ==============================================================================
#  3. 滚动回测引擎
# ==============================================================================
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

        # 回测风控 (Filter Bear Days and Low Scores)
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
            s = model(past_values=t).logits.squeeze().cpu().numpy()
        if s.ndim == 0: s = [s]
        for i, score in enumerate(s):
            res.append((meta[i][0], meta[i][1], float(score)))

    def run(self, top_k=Config.TOP_K):
        signals = self.generate_signal_matrix()
        if signals is None: return

        print("正在筛选活跃股票池...")
        valid_signals = signals.replace(-1, np.nan)
        daily_ranks = valid_signals.rank(axis=1, ascending=False)
        active_mask = (daily_ranks <= top_k * 2).any(axis=0)
        active_codes = signals.columns[active_mask].tolist()

        print(f"回测涉及股票数量: {len(active_codes)}")
        if not active_codes: return

        cerebro = bt.Cerebro()
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.addcommissioninfo(AShareCommission())
        # 【新增】滑点注入
        cerebro.broker.add_slippage_perc(StockSlippage, perc=Config.SLIPPAGE)

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

        if loaded_cnt == 0:
            print("❌ 无有效行情数据")
            return

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
        cumulative = (1 + ret_series).cumprod()

        try:
            bench = ak.stock_zh_index_daily(symbol=Config.BENCHMARK_SYMBOL)
            bench['date'] = pd.to_datetime(bench['date'])
            bench.set_index('date', inplace=True)
            bench_ret = bench['close'].pct_change().reindex(ret_series.index).fillna(0)
            bench_cum = (1 + bench_ret).cumprod()

            plt.figure(figsize=(12, 6))
            plt.plot(cumulative.index, cumulative, label='Strategy (Net)', color='red')
            plt.plot(bench_cum.index, bench_cum, label='CSI 300', color='gray', linestyle='--')
        except:
            cumulative.plot(figsize=(12, 6), label='Strategy (Net)')

        plt.title('Walk-Forward Equity Curve (w/ Slippage & Fees)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(Config.OUTPUT_DIR, "walk_forward_result.png"))
        print("📈 曲线已保存。")


def run_walk_forward_backtest(start_date, end_date, initial_cash, top_k=Config.TOP_K):
    engine = WalkForwardBacktester(start_date, end_date, initial_cash)
    engine.run(top_k=top_k)


# --- TopKStrategy & PerformanceAnalyzer (用于 predict 后的验证性回测) ---

class TopKStrategy(bt.Strategy):
    """
    【修复】TopK 验证策略
    用于 predict 模式后的简单验证，不含信号轮动，但包含风控
    """
    params = (
        ('top_k', Config.TOP_K),
        ('hold_days', Config.PRED_LEN),
        ('min_volume_percent', Config.MIN_VOLUME_PERCENT)
    )

    def __init__(self):
        self.hold_time = {}

    def next(self):
        # 1. 卖出逻辑
        for data in self.datas:
            if self.getposition(data).size > 0:
                self.hold_time[data._name] = self.hold_time.get(data._name, 0) + 1

                # 时间止盈/止损
                if self.hold_time[data._name] >= self.p.hold_days:
                    self.close(data=data)
                    self.hold_time[data._name] = 0

        # 2. 买入逻辑
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

                # 简单涨停不买
                prev = data.close[-1]
                if prev > 0 and data.close[0] >= prev * 1.095: continue

                size = int(target / price / 100) * 100
                if size < 100: continue

                limit_size = int(vol * self.p.min_volume_percent) // 100 * 100
                if size > limit_size: size = limit_size

                if size >= 100:
                    self.buy(data=data, size=size)
                    self.hold_time[data._name] = 0
                    buy_cnt += 1


class PerformanceAnalyzer:
    @staticmethod
    def get_benchmark(start_date, end_date):
        try:
            df = ak.stock_zh_index_daily(symbol=Config.BENCHMARK_SYMBOL)
            df['date'] = pd.to_datetime(df['date']);
            df.set_index('date', inplace=True)
            mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
            return df.loc[mask, 'close'].pct_change().fillna(0)
        except:
            return None

    @staticmethod
    def calculate_metrics(strategy_returns, benchmark_returns):
        df = pd.concat([strategy_returns, benchmark_returns], axis=1, join='inner')
        df.columns = ['Strategy', 'Benchmark']
        if len(df) < 10: return None
        Rp, Rm = df['Strategy'], df['Benchmark']
        days = len(df)
        ann_p = (1 + Rp).prod() ** (252 / days) - 1
        ann_m = (1 + Rm).prod() ** (252 / days) - 1
        vol = Rp.std() * np.sqrt(252)
        sharpe = (ann_p - Config.RISK_FREE_RATE) / (vol + 1e-9)
        cum = (1 + Rp).cumprod()
        dd = ((cum.cummax() - cum) / cum.cummax()).max()
        cov = np.cov(Rp, Rm);
        beta = cov[0, 1] / (cov[1, 1] + 1e-9)
        alpha = ann_p - (Config.RISK_FREE_RATE + beta * (ann_m - Config.RISK_FREE_RATE))
        return {"Ann. Return": ann_p, "Benchmark Ret": ann_m, "Alpha": alpha, "Beta": beta, "Sharpe": sharpe,
                "Max Drawdown": dd, "Win Rate": (Rp > 0).mean()}

    @staticmethod
    def plot_curve(strategy_returns, benchmark_returns):
        df = pd.concat([strategy_returns, benchmark_returns], axis=1, join='inner')
        df.columns = ['Strategy', 'CSI 300']
        (1 + df).cumprod().plot(figsize=(12, 6), grid=True)
        plt.savefig(os.path.join(Config.OUTPUT_DIR, "backtest_result.png"))


def run_single_backtest(codes, with_fees=True, initial_cash=1000000.0, top_k=Config.TOP_K):
    cerebro = bt.Cerebro();
    cerebro.broker.setcash(initial_cash)

    # 配置费率
    if with_fees:
        cerebro.broker.addcommissioninfo(AShareCommission())
        # 【新增】加入滑点 (如果是含费模式，通常也意味着要含滑点)
        cerebro.broker.add_slippage_perc(StockSlippage, perc=Config.SLIPPAGE)
    else:
        cerebro.broker.setcommission(commission=0.0)

    loaded = False
    for code in codes:
        fpath = os.path.join(Config.DATA_DIR, f"{code}.parquet")
        if not os.path.exists(fpath): continue
        try:
            df = pd.read_parquet(fpath);
            # start = pd.to_datetime(Config.START_DATE) # Unused
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


def run_backtest(top_stocks_list, initial_cash=1000000.0, top_k=Config.TOP_K):
    """
    【恢复】完整的验证性回测入口
    """
    print(f"\n>>> 启动验证性回测 (资金: {initial_cash:,.0f}, TopK: {top_k})")
    codes = [x[0] for x in top_stocks_list[:top_k]]
    if not codes:
        print("❌ 股票列表为空，跳过回测。")
        return

    print("对比回测中: 含费+滑点 vs 无摩擦成本...")
    res_fees = run_single_backtest(codes, True, initial_cash, top_k)
    res_no = run_single_backtest(codes, False, initial_cash, top_k)

    if not res_fees:
        print("❌ 回测失败 (可能无行情数据)")
        return

    print(f"{'指标':<15} | {'含费+滑点':<15} | {'理想情况':<15}")
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