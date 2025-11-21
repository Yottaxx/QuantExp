import backtrader as bt
import pandas as pd
import os
import math
import numpy as np
import akshare as ak
import matplotlib.pyplot as plt
from .config import Config

# 设置 Matplotlib 中文字体 (避免乱码)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
#  1. A 股专用费率模型 (Commission Scheme)
# ==============================================================================
class AShareCommission(bt.CommInfoBase):
    """
    A股费率：佣金万三，印花税万五(卖出)，最低5元
    """
    params = (
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
        ('perc', 0.0003),  # 佣金
        ('stamp_duty', 0.0005),  # 印花税
        ('min_comm', 5.0),  # 最低佣金
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
#  2. 策略实现 (增加资金风控逻辑)
# ==============================================================================
class TopKStrategy(bt.Strategy):
    params = (
        ('top_k', 5),
        ('hold_days', 5),
        ('min_volume_percent', 0.02),  # 风控：持仓不能超过该股票日成交量的 2%
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

        # --- 智能买入逻辑 ---
        cash = self.broker.get_cash()
        # [风控 1] 资金太少，甚至不够付最低佣金，停止交易
        if cash < 5000: return

        # 计算当前持仓数量
        current_positions = len([d for d in self.datas if self.getposition(d).size > 0])

        # 还能买几只？
        slots_available = self.p.top_k - current_positions
        if slots_available <= 0: return

        # 每只股票分配资金 (预留 2% 现金防止滑点)
        target_val = cash / slots_available * 0.98

        buy_count = 0

        # 假设 datas 已经按预测分排序传入
        for data in self.datas:
            if buy_count >= slots_available: break

            pos = self.getposition(data).size
            if pos == 0:
                price = data.close[0]
                volume = data.volume[0]  # 单位通常是手

                if price <= 0 or volume <= 0: continue

                # 计算理论买入股数 (向下取整到 100 股)
                size = int(target_val / price / 100) * 100

                # [风控 2: 小资金保护]
                # 如果连一手都买不起，跳过
                if size < 100:
                    continue

                # [风控 3: 大资金保护 - 流动性上限]
                # 防止资金量过大对盘面造成冲击
                # volume * 100 是当日总成交股数
                max_liquid_size = volume * 100 * self.p.min_volume_percent

                if size > max_liquid_size:
                    # 强制缩减仓位至流动性允许范围
                    size = int(max_liquid_size / 100) * 100

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

            mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
            bench_series = df.loc[mask, 'close']

            return bench_series.pct_change().fillna(0)
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
        sharpe = (ann_ret_p - 0.03) / (vol_p + 1e-9)

        # 4. 最大回撤
        cum_returns = (1 + R_p).cumprod()
        drawdown = (cum_returns.cummax() - cum_returns) / cum_returns.cummax()
        max_dd = drawdown.max()

        # 5. Beta & Alpha
        cov_matrix = np.cov(R_p, R_m)
        beta = cov_matrix[0, 1] / (cov_matrix[1, 1] + 1e-9)
        alpha = ann_ret_p - (0.03 + beta * (ann_ret_m - 0.03))

        # 6. 信息比率
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
            "Win Rate": (R_p > 0).mean()
        }

    @staticmethod
    def plot_curve(strategy_returns, benchmark_returns):
        df = pd.concat([strategy_returns, benchmark_returns], axis=1, join='inner')
        df.columns = ['Strategy', 'CSI 300']
        equity = (1 + df).cumprod()

        plt.figure(figsize=(12, 6))
        plt.plot(equity.index, equity['Strategy'], label='Our Strategy', color='#d62728', linewidth=2)
        plt.plot(equity.index, equity['CSI 300'], label='Benchmark (CSI300)', color='gray', linestyle='--', alpha=0.8)
        plt.title('Strategy Equity Curve vs Benchmark')
        plt.grid(True, alpha=0.3)
        plt.legend()
        save_path = os.path.join(Config.OUTPUT_DIR, "backtest_result.png")
        plt.savefig(save_path)
        print(f"📈 资金曲线图已保存至: {save_path}")


# ==============================================================================
#  回测执行核心 (支持 有费/无费 对比)
# ==============================================================================
def run_single_backtest(codes, with_fees=True, initial_cash=1000000.0):
    """
    执行单次特定配置的回测
    """
    cerebro = bt.Cerebro()

    # 1. 资金设置
    cerebro.broker.setcash(initial_cash)

    # 2. 费率设置
    if with_fees:
        cerebro.broker.addcommissioninfo(AShareCommission())
    else:
        cerebro.broker.setcommission(commission=0.0)

    # 3. 数据加载
    data_loaded = False
    for code in codes:
        fpath = os.path.join(Config.DATA_DIR, f"{code}.parquet")
        if not os.path.exists(fpath): continue

        try:
            df = pd.read_parquet(fpath)
            start_date = pd.to_datetime(Config.START_DATE)
            if len(df) > 250:
                df = df.iloc[-250:]
                start_date = df.index[0]

            data = bt.feeds.PandasData(
                dataname=df,
                fromdate=start_date,
                plot=False
            )
            cerebro.adddata(data, name=code)
            data_loaded = True
        except:
            continue

    if not data_loaded: return None

    # 4. 策略与分析器
    cerebro.addstrategy(TopKStrategy, top_k=5, hold_days=5)
    # 添加交易分析器，用于计算胜率
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    # 添加时间收益分析器，用于计算 Alpha/Beta
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns')

    # 5. 运行
    results = cerebro.run()
    strat = results[0]

    # 6. 提取指标
    final_value = cerebro.broker.getvalue()
    profit_rate = (final_value - initial_cash) / initial_cash

    # 提取胜率
    trade_analysis = strat.analyzers.trade_analyzer.get_analysis()
    total_trades = trade_analysis.total.closed if 'total' in trade_analysis else 0
    won_trades = trade_analysis.won.total if 'won' in trade_analysis else 0
    win_rate = (won_trades / total_trades) if total_trades > 0 else 0.0

    # 提取收益序列
    ret_dict = strat.analyzers.returns.get_analysis()
    strategy_ret = pd.Series(ret_dict, name='Strategy')
    strategy_ret.index = pd.to_datetime(strategy_ret.index)

    return {
        "final_value": final_value,
        "profit_rate": profit_rate,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "returns": strategy_ret,
        "start_date": strat.data.datetime.date(0),  # 记录开始时间方便获取基准
        "end_date": strat.data.datetime.date(-1)
    }


def run_backtest(top_stocks_list, initial_cash=50000.0):
    """
    主入口：执行两次回测并生成对比报告
    """
    print("\n" + "=" * 50)
    print(f">>> 启动 SOTA 策略回测分析 (初始资金: {initial_cash:,.0f})")
    print("=" * 50)

    # 提取股票代码 (取 Top 5 进行演示)
    target_codes = [x[0] for x in top_stocks_list[:5]]

    if not target_codes:
        print("❌ 没有可用的股票列表")
        return

    # 1. 运行含手续费回测 (真实模拟)
    print("⏳ 正在进行 [真实环境] 回测 (含印花税/佣金)...")
    res_fees = run_single_backtest(target_codes, with_fees=True, initial_cash=initial_cash)

    # 2. 运行无手续费回测 (理论上限)
    print("⏳ 正在进行 [理论环境] 回测 (无摩擦成本)...")
    res_no_fees = run_single_backtest(target_codes, with_fees=False, initial_cash=initial_cash)

    if not res_fees or not res_no_fees:
        print("❌ 回测失败：无法加载数据")
        return

    # 3. 生成对比报表
    print("\n" + "=" * 50)
    print(f"{'指标 (Metric)':<15} | {'含手续费 (Real)':<15} | {'无手续费 (Ideal)':<15}")
    print("-" * 50)

    # 市值对比
    print(f"{'最终市值':<15} | {res_fees['final_value']:<15,.2f} | {res_no_fees['final_value']:<15,.2f}")

    # 收益率对比
    p_real = res_fees['profit_rate']
    p_ideal = res_no_fees['profit_rate']
    print(f"{'累计收益率':<15} | {p_real:<15.2%} | {p_ideal:<15.2%}")

    # 胜率对比
    w_real = res_fees['win_rate']
    w_ideal = res_no_fees['win_rate']
    print(f"{'交易胜率':<15} | {w_real:<15.2%} | {w_ideal:<15.2%}")

    # 交易次数
    print(f"{'交易总次数':<15} | {res_fees['total_trades']:<15} | {res_no_fees['total_trades']:<15}")

    print("=" * 50)

    # 简评
    cost_impact = p_ideal - p_real
    print(f"💡 费率损耗分析: 交易摩擦成本共吞噬了 {cost_impact:.2%} 的利润。")
    if w_real < 0.5:
        print("⚠️ 警告: 真实胜率不足 50%，策略在费率压力下可能失效。")
    elif cost_impact > 0.1:
        print("⚠️ 警告: 费率损耗过高，建议降低换仓频率 (增加 hold_days)。")
    else:
        print("✅ 评价: 策略对交易成本不敏感，鲁棒性较好。")

    # --- 4. 绩效归因与绘图 (仅基于真实含费结果) ---
    # 获取基准数据
    bench_ret = PerformanceAnalyzer.get_benchmark(res_fees['start_date'], res_fees['end_date'])

    if bench_ret is not None:
        metrics = PerformanceAnalyzer.calculate_metrics(res_fees['returns'], bench_ret)
        if metrics:
            print("\n" + "-" * 40)
            print(f"📊 【基金经理级绩效报告 (基于真实净值)】")
            print("-" * 40)
            print(f"{'年化收益率':<15} : {metrics['Ann. Return']:>8.2%}")
            print(f"{'基准收益率':<15} : {metrics['Benchmark Ret']:>8.2%}")
            # 【核心新增】超额收益率展示
            print(f"{'超额收益 (Excess)':<15} : {metrics['Ann. Return'] - metrics['Benchmark Ret']:>8.2%}")
            print(f"{'Alpha (阿尔法)':<15} : {metrics['Alpha']:>8.4f}")
            print(f"{'Beta (贝塔)':<15} : {metrics['Beta']:>8.4f}")
            print(f"{'Sharpe (夏普)':<15} : {metrics['Sharpe']:>8.4f}")
            print(f"{'最大回撤':<15} : {metrics['Max Drawdown']:>8.2%}")
            print("=" * 60)

            # 绘图
            PerformanceAnalyzer.plot_curve(res_fees['returns'], bench_ret)