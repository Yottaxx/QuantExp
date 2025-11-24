import pytest
import backtrader as bt
from src.backtest import AShareCommission, ModelDrivenStrategy


# ==============================================================================
#  1. 测试交易费用模型 (AShareCommission)
# ==============================================================================
def test_commission_model():
    """
    测试 AShareCommission 的计算逻辑：
    1. 买入：仅佣金，且有最低消费 (5.0)
    2. 卖出：佣金 + 印花税
    """
    comm = AShareCommission()

    # 场景 1: 小额买入，触发最低佣金 5.0
    # 100股 * 10元 = 1000元。按 0.0003 算佣金为 0.3元，应收取 5.0元
    buy_cost_min = comm._getcommission(size=100, price=10.0, pseudoexec=False)
    assert buy_cost_min == 5.0

    # 场景 2: 大额买入，按比例收取
    # 100000股 * 10元 = 1,000,000元。佣金 300元 > 5元
    buy_cost_large = comm._getcommission(size=10000, price=100.0, pseudoexec=False)
    assert buy_cost_large == 10000 * 100.0 * 0.0003

    # 场景 3: 卖出，佣金 + 印花税 (0.0005)
    # 1000股 * 20元 = 20,000元
    # 佣金: 20000 * 0.0003 = 6.0元
    # 印花税: 20000 * 0.0005 = 10.0元
    # 总计: 16.0元
    sell_cost = comm._getcommission(size=-1000, price=20.0, pseudoexec=False)
    expected_comm = max(20000 * 0.0003, 5.0)
    expected_tax = 20000 * 0.0005
    assert abs(sell_cost - (expected_comm + expected_tax)) < 1e-6


# ==============================================================================
#  2. 测试涨跌停检测逻辑 (ModelDrivenStrategy)
# ==============================================================================

class MockLine:
    """模拟 Backtrader 的 Line 对象 (支持 [0] 和 [-1] 索引)"""

    def __init__(self, current, prev=None):
        self.current = current
        self.prev = prev if prev is not None else current

    def __getitem__(self, idx):
        if idx == 0:
            return self.current
        elif idx == -1:
            return self.prev
        raise IndexError("MockLine only supports index 0 (current) and -1 (prev)")


class MockData:
    """模拟 Backtrader 的 Data Feed 对象"""

    def __init__(self, prev_close, open_, high, low, close):
        self.close = MockLine(close, prev_close)
        self.high = MockLine(high)  # high[-1] 在逻辑中未被使用，给默认即可
        self.low = MockLine(low)  # low[-1] 同上
        self.open = MockLine(open_)


def test_limit_detection_logic():
    # [Fix] 使用 __new__ 绕过 Backtrader 的 __init__ 逻辑
    # 避免 "AttributeError: 'NoneType' object has no attribute '_next_stid'"
    strat = ModelDrivenStrategy.__new__(ModelDrivenStrategy)

    # 手动初始化必要的变量（如果需要），这里 check_limit_status 是纯函数式的，
    # 只需要类方法支持即可，不需要 cerebro 环境。
    # 如果 strat 内部访问了 self.p，我们需要手动 mock params，但此函数未访问。

    # --------------------------------------------------------------------------
    # Case A: 主板股票 (600xxx), 10% 涨跌幅
    # 昨收 10.0 -> 涨停价 11.0, 跌停价 9.0
    # --------------------------------------------------------------------------

    # A1. 正常涨停 (Close = 11.0, High = 11.0)
    data_up = MockData(prev_close=10.0, open_=10.5, high=11.0, low=10.5, close=11.0)
    is_up, is_down = strat.check_limit_status(data_up, "600001")
    assert is_up is True
    assert is_down is False

    # A2. 正常跌停 (Close = 9.0, Low = 9.0)
    data_down = MockData(prev_close=10.0, open_=9.5, high=9.5, low=9.0, close=9.0)
    is_up, is_down = strat.check_limit_status(data_down, "600001")
    assert is_up is False
    assert is_down is True

    # A3. 盘中触及涨停但未封住 (High=11.0, Close=10.8) -> 不算涨停
    data_touch = MockData(prev_close=10.0, open_=10.0, high=11.0, low=10.0, close=10.8)
    is_up, is_down = strat.check_limit_status(data_touch, "600001")
    assert is_up is False
    assert is_down is False

    # --------------------------------------------------------------------------
    # Case B: 科创板/创业板 (688xxx/300xxx), 20% 涨跌幅
    # 昨收 20.0 -> 涨停价 24.0
    # --------------------------------------------------------------------------

    # B1. 创业板涨停 (Code 300xxx)
    data_300 = MockData(prev_close=20.0, open_=21.0, high=24.0, low=21.0, close=24.0)
    is_up, is_down = strat.check_limit_status(data_300, "300001")
    assert is_up is True

    # B2. 科创板涨停 (Code 688xxx)
    data_688 = MockData(prev_close=50.0, open_=55.0, high=60.0, low=55.0, close=60.0)
    is_up, is_down = strat.check_limit_status(data_688, "688001")
    assert is_up is True

    # B3. 普通涨幅超过 10% 但未达 20% (例如涨 15%) -> 对于 688 不是涨停
    # 昨收 100, 收盘 115 (涨15%)
    data_run = MockData(prev_close=100.0, open_=100.0, high=115.0, low=100.0, close=115.0)
    is_up, is_down = strat.check_limit_status(data_run, "688001")
    assert is_up is False
    assert is_down is False

    # --------------------------------------------------------------------------
    # Case C: 脏数据处理
    # --------------------------------------------------------------------------
    # 昨收为 0 的情况 (新股或数据错误)
    data_zero = MockData(prev_close=0.0, open_=10.0, high=10.0, low=10.0, close=10.0)
    is_up, is_down = strat.check_limit_status(data_zero, "600001")
    assert is_up is False
    assert is_down is False