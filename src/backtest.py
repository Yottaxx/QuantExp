import backtrader as bt
import pandas as pd
import numpy as np
import os
import akshare as ak
import datetime
import matplotlib.pyplot as plt
from .config import Config

# 设置 Matplotlib 中文字体 (避免乱码)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
#  1. A 股专用费率模型
# ==============================================================================
class AShareCommission(bt.CommInfoBase):
    """
    A股费率：佣金万三，印花税万五(卖出)，最低5元
    """
    params = (
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
        ('perc', 0.0003),
        ('stamp_duty', 0.0005),
        ('min_comm', 5.0),
    )

    def _getcommission(self, size, price, pseudoexec):
        if size > 0:  # 买入
            commission = abs(size) * price * self.p.perc
            return max(commission, self.p.min_comm)
        elif size < 0:  # 卖出
            commission = abs(size) * price * self.p.perc
            commission = max(commission, self.p.min_comm)
            stamp_duty = abs(size) * price * self.p.stamp_duty
            return commission + stamp_duty
        return 0.0


# ==============================================================================
#  2. 策略实现 (增加净值记录)
# ==============================================================================
class TopKStrategy(bt.Strategy):
    params = (
        ('top_k', 5),
        ('hold_days', 5),
    )

    def __init__(self):
        self.hold_time = {}

    def next(self):
        # --- 卖出逻辑 ---
        for data in self.datas:
            pos = self.getposition(data).size
            if pos > 0:
                name = data._name
                self.hold_time[name] = self.hold_time.get(name, 0) + 1
                if self.hold_time[name] >= self.p.hold_days:
                    self.close(data=data)
                    self.hold_time[name] = 0

        # --- 买入逻辑 ---
        cash = self.broker.get_cash()
        if cash < 5000: return

        target_val = self.broker.get_value() * 0.95 / self.p.top_k
        buy_count = 0

        # 假设 datas 已经按预测分排序传入
        for data in self.datas:
            if buy_count >= self.p.top_k: break

            pos = self.getposition(data).size
            if pos == 0:
                price = data.close[0]
                if price <= 0: continue

                # A股 100 股一手
                size = int(target_val / price / 100) * 100
                if size >= 100:
                    self.buy(data=data, size=size)
                    self.hold_time[data._name] = 0
                    buy_count += 1


# ==============================================================================
#  3. 绩效分析引擎 (Metrics Engine)
# ==============================================================================
class PerformanceAnalyzer:
    @staticmethod
    def get_benchmark(start_date, end_date):
        """获取沪深300基准数据"""
        print(f"⏳ 正在获取沪深300基准数据 ({start_date} - {end_date})...")
        try:
            # 使用 AkShare 接口
            df = ak.stock_zh_index_daily(symbol="sh000300")
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            # 截取对应时间段
            mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
            bench_series = df.loc[mask, 'close']

            # 计算日收益率
            return bench_series.pct_change().fillna(0)
        except Exception as e:
            print(f"⚠️ 无法获取基准数据: {e}")
            return None

    @staticmethod
    def calculate_metrics(strategy_returns, benchmark_returns):
        """
        计算 Alpha, Beta, Sharpe, MaxDD 等核心指标
        """
        # 对齐日期索引
        df = pd.concat([strategy_returns, benchmark_returns], axis=1, join='inner')
        df.columns = ['Strategy', 'Benchmark']

        if len(df) < 10:
            return None

        R_p = df['Strategy']
        R_m = df['Benchmark']
        risk_free = 0.03 / 252  # 假设年化无风险利率 3%

        # 1. 年化收益率 (Simple Annualized)
        days = len(df)
        total_ret_p = (1 + R_p).prod() - 1
        ann_ret_p = (1 + total_ret_p) ** (252 / days) - 1

        total_ret_m = (1 + R_m).prod() - 1
        ann_ret_m = (1 + total_ret_m) ** (252 / days) - 1

        # 2. 波动率 (Annualized Volatility)
        vol_p = R_p.std() * np.sqrt(252)

        # 3. 夏普比率 (Sharpe Ratio)
        sharpe = (ann_ret_p - 0.03) / (vol_p + 1e-9)

        # 4. 最大回撤 (Max Drawdown)
        cum_returns = (1 + R_p).cumprod()
        drawdown = (cum_returns.cummax() - cum_returns) / cum_returns.cummax()
        max_dd = drawdown.max()

        # 5. Beta & Alpha
        # Cov(Rp, Rm) / Var(Rm)
        cov_matrix = np.cov(R_p, R_m)
        beta = cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-9)

        # Alpha = Rp - [Rf + Beta * (Rm - Rf)]
        alpha = ann_ret_p - (0.03 + beta * (ann_ret_m - 0.03))

        # 6. 信息比率 (Information Ratio)
        # (Rp - Rm) / Std(Rp - Rm)
        active_ret = R_p - R_m
        ir = (active_ret.mean() * 252) / (active_ret.std() * np.sqrt(252) + 1e-9)

        return {
            "Ann. Return": ann_ret_p,
            "Benchmark Ret": ann_ret_m,
            "Alpha": alpha,
            "Beta": beta,
            "Sharpe": sharpe,
            "Max Drawdown": max_dd,
            "Info Ratio": ir,
            "Win Rate": (R_p > 0).mean()  # 简单的日胜率
        }

    @staticmethod
    def plot_curve(strategy_returns, benchmark_returns):
        """绘制净值对比图"""
        df = pd.concat([strategy_returns, benchmark_returns], axis=1, join='inner')
        df.columns = ['Strategy', 'CSI 300']

        # 归一化净值 (从1.0开始)
        equity = (1 + df).cumprod()

        plt.figure(figsize=(12, 6))
        plt.plot(equity.index, equity['Strategy'], label='Our Strategy', color='#d62728', linewidth=2)
        plt.plot(equity.index, equity['CSI 300'], label='Benchmark (CSI300)', color='gray', linestyle='--', alpha=0.8)

        # 标记最大回撤区域
        # (这里简化处理，只画曲线)

        plt.title('Strategy Equity Curve vs Benchmark')
        plt.grid(True, alpha=0.3)
        plt.legend()

        save_path = os.path.join(Config.OUTPUT_DIR, "backtest_result.png")
        plt.savefig(save_path)
        print(f"📈 资金曲线图已保存至: {save_path}")


# ==============================================================================
#  4. 回测主入口
# ==============================================================================
def run_backtest(top_stocks_list):
    print("\n" + "=" * 60)
    print(">>> 启动第三阶段：专业绩效归因 (Professional Attribution)")
    print("=" * 60)

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(1000000.0)
    cerebro.broker.addcommissioninfo(AShareCommission())

    # 添加 TimeReturn 分析器，用于提取每日收益序列
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns')

    # 提取 Top 5 股票
    target_codes = [x[0] for x in top_stocks_list[:5]]
    if not target_codes: return

    print(f"持仓组合: {target_codes}")

    # 加载数据
    start_date = None
    end_date = None

    for code in target_codes:
        fpath = os.path.join(Config.DATA_DIR, f"{code}.parquet")
        if not os.path.exists(fpath): continue
        try:
            df = pd.read_parquet(fpath)
            # 自动确定回测区间
            if start_date is None: start_date = df.index[0]
            if end_date is None: end_date = df.index[-1]

            data = bt.feeds.PandasData(dataname=df, fromdate=start_date, plot=False)
            cerebro.adddata(data, name=code)
        except:
            continue

    cerebro.addstrategy(TopKStrategy, top_k=5, hold_days=5)

    print("⏳ 策略回测运行中...")
    results = cerebro.run()
    strat = results[0]

    # --- 绩效分析 ---

    # 1. 获取策略每日收益率 Series
    ret_dict = strat.analyzers.returns.get_analysis()
    strategy_ret = pd.Series(ret_dict, name='Strategy')
    strategy_ret.index = pd.to_datetime(strategy_ret.index)

    # 2. 获取基准每日收益率 Series
    bench_ret = PerformanceAnalyzer.get_benchmark(start_date, end_date)

    if bench_ret is None:
        print("❌ 缺少基准数据，无法计算 Alpha/Beta。")
        return

    # 3. 计算指标
    metrics = PerformanceAnalyzer.calculate_metrics(strategy_ret, bench_ret)

    if metrics:
        print("\n" + "-" * 40)
        print(f"📊 【基金经理级绩效报告】")
        print("-" * 40)
        print(f"{'年化收益率 (Ann. Return)':<25} : {metrics['Ann. Return']:>8.2%}")
        print(f"{'基准收益率 (Benchmark)':<25} : {metrics['Benchmark Ret']:>8.2%}")
        print(f"{'超额收益 (Excess)':<25} : {metrics['Ann. Return'] - metrics['Benchmark Ret']:>8.2%}")
        print("-" * 40)
        print(f"{'Alpha (阿尔法)':<25} : {metrics['Alpha']:>8.4f} (核心能力)")
        print(f"{'Beta (贝塔)':<25} : {metrics['Beta']:>8.4f} (市场敞口)")
        print(f"{'Sharpe Ratio (夏普)':<25} : {metrics['Sharpe']:>8.4f} (>1.0 优秀)")
        print(f"{'Info Ratio (信息比)':<25} : {metrics['Info Ratio']:>8.4f}")
        print("-" * 40)
        print(f"{'最大回撤 (Max Drawdown)':<25} : {metrics['Max Drawdown']:>8.2%}")
        print(f"{'日胜率 (Win Rate)':<25} : {metrics['Win Rate']:>8.2%}")
        print("=" * 60)

        # 4. 绘图
        PerformanceAnalyzer.plot_curve(strategy_ret, bench_ret)
    else:
        print("❌ 数据长度不足，无法计算指标")