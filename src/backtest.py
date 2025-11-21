import backtrader as bt
import pandas as pd
import os
from .config import Config


class TopKStrategy(bt.Strategy):
    def __init__(self):
        self.order = None

    def next(self):
        # 简单示例：如果你在 Top K 列表里，就买入
        # 实际需要传入 Top K 列表参数进行判断
        if not self.position:
            self.buy()
        elif len(self) % 5 == 0:  # 持有5天卖出
            self.close()


def run_backtest(stock_list):
    cerebro = bt.Cerebro()

    # 只回测选出来的第一只，作为演示
    if not stock_list: return
    code = stock_list[0][0]

    fpath = os.path.join(Config.DATA_DIR, f"{code}.parquet")
    df = pd.read_parquet(fpath)

    data = bt.feeds.PandasData(
        dataname=df,
        fromdate=pd.to_datetime("2024-01-01"),
        plot=False
    )

    cerebro.adddata(data)
    cerebro.addstrategy(TopKStrategy)
    cerebro.broker.setcash(100000.0)

    print(f"回测 {code} | 初始资金: {cerebro.broker.getvalue()}")
    cerebro.run()
    print(f"回测 {code} | 最终资金: {cerebro.broker.getvalue()}")