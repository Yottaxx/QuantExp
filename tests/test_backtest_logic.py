import types

import pytest

pytest.importorskip("backtrader")

from src.backtest import AShareCommission, StockSlippage, ModelDrivenStrategy


def test_commission_and_slippage():
    comm = AShareCommission()
    buy_cost = comm._getcommission(size=100, price=10, pseudoexec=False)
    sell_cost = comm._getcommission(size=-100, price=10, pseudoexec=False)

    assert buy_cost >= 5.0  # minimum commission enforced
    assert sell_cost > buy_cost  # stamp duty on sell

    slip = StockSlippage(perc=0.002)
    assert slip._calculate(types.SimpleNamespace(isbuy=lambda: True, issell=lambda: False), 10) > 10
    assert slip._calculate(types.SimpleNamespace(isbuy=lambda: False, issell=lambda: True), 10) < 10


def test_limit_detection_logic():
    class DummySeries:
        def __init__(self, values):
            self.values = values

        def __getitem__(self, idx):
            return self.values[idx]

    class DummyData:
        def __init__(self, prev_close, close, high, low):
            self.close = DummySeries([close, prev_close])
            self.high = DummySeries([high, prev_close])
            self.low = DummySeries([low, prev_close])

    # Avoid invoking bt.Strategy.__init__ by constructing a bare instance
    strat = ModelDrivenStrategy.__new__(ModelDrivenStrategy)
    # 科创/创业板 20% 阈值
    is_up, is_down = strat.check_limit_status(DummyData(10, 12, 12, 11.8), "300001")
    assert is_up is True
    assert is_down is False

    # 主板 10% 阈值
    is_up, is_down = strat.check_limit_status(DummyData(10, 10.5, 10.5, 9.5), "600001")
    assert is_up is True
    assert is_down is False

    # 跌停
    is_up, is_down = strat.check_limit_status(DummyData(10, 9, 9.2, 9), "600001")
    assert is_up is False
    assert is_down is True
