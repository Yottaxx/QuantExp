import unittest
import backtrader as bt
import pandas as pd
import datetime
import numpy as np
from src.backtest import AShareCommission, ModelDrivenStrategy


def get_dummy_dataframe():
    """生成 10 天的虚拟行情数据"""
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    df = pd.DataFrame({
        'open': [10.0] * 10,
        'high': [11.0] * 10,
        'low': [9.0] * 10,
        'close': [10.0] * 10,
        'volume': [100000] * 10,
    }, index=dates)
    return df


# ==============================================================================
#  测试用例
# ==============================================================================
class TestBacktestLogic(unittest.TestCase):

    def test_ashare_commission_buy(self):
        """测试 A 股买入费率 (佣金万三，最低5元)"""
        comm_info = AShareCommission()

        # 场景 1: 大额买入 (1000股, 100元) -> 金额 100,000
        # 佣金 = 100,000 * 0.0003 = 30元 (>5元)
        size = 1000
        price = 100.0
        comm = comm_info._getcommission(size, price, pseudoexec=False)
        self.assertAlmostEqual(comm, 30.0, places=2)

        # 场景 2: 小额买入 (100股, 10元) -> 金额 1,000
        # 佣金 = 1,000 * 0.0003 = 0.3元 (<5元) -> 应收 5元
        size = 100
        price = 10.0
        comm = comm_info._getcommission(size, price, pseudoexec=False)
        self.assertEqual(comm, 5.0)

    def test_ashare_commission_sell(self):
        """测试 A 股卖出费率 (佣金 + 印花税万五)"""
        comm_info = AShareCommission()

        # 场景 1: 大额卖出 (1000股, 100元) -> 金额 100,000
        # 佣金 = 30元
        # 印花税 = 100,000 * 0.0005 = 50元
        # 总计 = 80元
        size = -1000
        price = 100.0
        comm = comm_info._getcommission(size, price, pseudoexec=False)
        self.assertAlmostEqual(comm, 80.0, places=2)

        # 场景 2: 小额卖出 (100股, 10元) -> 金额 1,000
        # 佣金 = 5元 (最低)
        # 印花税 = 1,000 * 0.0005 = 0.5元
        # 总计 = 5.5元
        size = -100
        price = 10.0
        comm = comm_info._getcommission(size, price, pseudoexec=False)
        self.assertAlmostEqual(comm, 5.5, places=2)

    def test_strategy_signal_execution(self):
        """测试策略能否正确响应信号"""
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100000.0)

        # 1. 准备数据
        df = get_dummy_dataframe()

        # [Fix] 直接使用 PandasData 并显式指定列名映射
        # 这避免了自定义类 params 隐式映射可能导致的默认值问题（如读取不到列导致价格为 -1/NaN）
        data = bt.feeds.PandasData(
            dataname=df,
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1  # -1 表示该列不存在
        )
        cerebro.adddata(data, name="600000")

        # 2. 构造信号: 第3天买入 (index=2, date=2024-01-03)

        # [Fix 1] 显式指定 dtype=float
        signals = pd.DataFrame(index=df.index, columns=["600000"], dtype=float)

        # [Fix 2] 初始化为 -100.0 (无效信号)，而不是 0.0
        # 因为策略中 valid_row = row[row > -1]，如果设为 0.0，策略每天都会尝试买入
        signals[:] = -100.0

        # [Fix 3] 使用 df.index[2] (Timestamp) 确保索引类型一致
        target_ts = df.index[2]
        signals.loc[target_ts, "600000"] = 0.99  # 高分买入

        # 3. 添加策略
        cerebro.addstrategy(
            ModelDrivenStrategy,
            signals=signals,
            top_k=1,
            hold_days=2,  # 持有2天后卖出
            min_volume_percent=1.0  # 放宽限制以便测试
        )

        # 4. 运行
        results = cerebro.run()
        # 确保策略运行成功
        self.assertTrue(len(results) > 0)

        # 5. 验证
        # 预期发生了交易，因此最终资金应该少于初始资金（因为扣除了手续费）
        final_value = cerebro.broker.getvalue()

        # 如果 final_value == 100000.0，说明没有交易发生
        self.assertLess(final_value, 100000.0, f"Expected trades to occur, but final value is {final_value}")

        # 验证费率模型是否生效 (至少一次买卖，费用 > 10元)
        self.assertTrue(100000.0 - final_value > 10.0)


if __name__ == '__main__':
    unittest.main()