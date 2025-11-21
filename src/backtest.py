import backtrader as bt
import pandas as pd
import os
import math
from .config import Config


# ==============================================================================
#  A 股专用费率模型 (Commission Scheme)
# ==============================================================================
class AShareCommission(bt.CommInfoBase):
    """
    模拟 A 股交易费用：
    1. 佣金: 万分之三 (0.0003)，双向收取
    2. 印花税: 万分之五 (0.0005)，仅卖出收取
    3. 最低佣金: 5 元
    """
    params = (
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
        ('perc', 0.0003),  # 佣金费率
        ('stamp_duty', 0.0005),  # 印花税费率
        ('min_comm', 5.0),  # 最低佣金
    )

    def _getcommission(self, size, price, pseudoexec):
        """
        计算具体的佣金和税费
        """
        if size > 0:  # 买入
            # 仅计算佣金，有最低门槛
            commission = abs(size) * price * self.p.perc
            if commission < self.p.min_comm:
                commission = self.p.min_comm
            return commission

        elif size < 0:  # 卖出
            # 1. 佣金 (含最低 5 元)
            commission = abs(size) * price * self.p.perc
            if commission < self.p.min_comm:
                commission = self.p.min_comm

            # 2. 印花税 (无最低限制，仅按比例，目前A股是卖出收)
            stamp_duty = abs(size) * price * self.p.stamp_duty

            return commission + stamp_duty

        return 0.0


# ==============================================================================
#  策略实现
# ==============================================================================
class TopKStrategy(bt.Strategy):
    params = (
        ('top_k', 5),  # 每日持仓只数
        ('hold_days', 5),  # 换仓周期 (轮动)
    )

    def __init__(self):
        self.orders = {}
        self.hold_time = {}

    def next(self):
        # 简单的轮动逻辑：
        # 持有期满则卖出，有现金且在目标列表里则买入

        # 1. 卖出检查
        for data in self.datas:
            pos = self.getposition(data).size
            if pos > 0:
                name = data._name
                self.hold_time[name] = self.hold_time.get(name, 0) + 1

                # 持有满 N 天，强制卖出
                if self.hold_time[name] >= self.p.hold_days:
                    self.close(data=data)
                    self.hold_time[name] = 0

        # 2. 买入检查
        cash = self.broker.get_cash()
        if cash < 5000: return  # 资金过小不交易

        # 简单的等权分配资金
        target_value_per_stock = self.broker.get_value() * 0.98 / self.p.top_k

        buy_count = 0
        for data in self.datas:
            # 限制持仓数量
            if buy_count >= self.p.top_k: break

            pos = self.getposition(data).size
            if pos == 0:
                # A 股必须买 100 的整数倍
                price = data.close[0]
                if price <= 0: continue

                size = int(target_value_per_stock / price / 100) * 100

                if size >= 100:
                    self.buy(data=data, size=size)
                    self.hold_time[data._name] = 0
                    buy_count += 1


# ==============================================================================
#  回测执行核心 (支持 有费/无费 对比)
# ==============================================================================
def run_single_backtest(codes, with_fees=True):
    """
    执行单次特定配置的回测
    """
    cerebro = bt.Cerebro()

    # 1. 资金设置
    INITIAL_CASH = 1000000.0
    cerebro.broker.setcash(INITIAL_CASH)

    # 2. 费率设置
    if with_fees:
        cerebro.broker.addcommissioninfo(AShareCommission())
    else:
        # 无费率模式 (Commission = 0)
        cerebro.broker.setcommission(commission=0.0)

    # 3. 数据加载
    data_loaded = False
    for code in codes:
        fpath = os.path.join(Config.DATA_DIR, f"{code}.parquet")
        if not os.path.exists(fpath): continue

        try:
            df = pd.read_parquet(fpath)
            # 截取最近一年的数据进行回测演示，避免回测太久
            start_date = pd.to_datetime(Config.START_DATE)
            if len(df) > 250:
                # 如果数据很长，取最近250个交易日，更有代表性
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

    if not data_loaded:
        return None

    # 4. 策略与分析器
    cerebro.addstrategy(TopKStrategy, top_k=5, hold_days=5)
    # 添加交易分析器，用于计算胜率
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')

    # 5. 运行
    results = cerebro.run()
    strat = results[0]

    # 6. 提取指标
    final_value = cerebro.broker.getvalue()
    profit_rate = (final_value - INITIAL_CASH) / INITIAL_CASH

    # 提取胜率
    trade_analysis = strat.analyzers.trade_analyzer.get_analysis()
    total_trades = trade_analysis.total.closed if 'total' in trade_analysis else 0
    won_trades = trade_analysis.won.total if 'won' in trade_analysis else 0

    win_rate = (won_trades / total_trades) if total_trades > 0 else 0.0

    return {
        "final_value": final_value,
        "profit_rate": profit_rate,
        "win_rate": win_rate,
        "total_trades": total_trades
    }


def run_backtest(top_stocks_list):
    """
    主入口：执行两次回测并生成对比报告
    """
    print("\n" + "=" * 50)
    print(">>> 启动 SOTA 策略回测分析 (含费率压力测试)")
    print("=" * 50)

    # 提取股票代码 (取 Top 5 进行演示)
    target_codes = [x[0] for x in top_stocks_list[:5]]

    if not target_codes:
        print("❌ 没有可用的股票列表")
        return

    # 1. 运行含手续费回测 (真实模拟)
    print("⏳ 正在进行 [真实环境] 回测 (含印花税/佣金)...")
    res_fees = run_single_backtest(target_codes, with_fees=True)

    # 2. 运行无手续费回测 (理论上限)
    print("⏳ 正在进行 [理论环境] 回测 (无摩擦成本)...")
    res_no_fees = run_single_backtest(target_codes, with_fees=False)

    if not res_fees or not res_no_fees:
        print("❌ 回测失败：无法加载数据")
        return

    # 3. 生成对比报表
    print("\n" + "=" * 50)
    print(f"{'指标 (Metric)':<15} | {'含手续费 (Real)':<15} | {'无手续费 (Ideal)':<15}")
    print("-" * 50)

    # 市值对比
    print(f"{'最终市值':<15} | {res_fees['final_value']:<15.2f} | {res_no_fees['final_value']:<15.2f}")

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