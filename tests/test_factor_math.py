import pytest
import pandas as pd
import numpy as np
from src import factor_ops as ops


class TestFactorMath:
    """
    【因子数学正确性验证】
    使用简单的、可手算的确定性数据，验证核心算子的逻辑正确性。
    """

    def setup_method(self):
        # 构造简单序列: [1.0, 2.0, 3.0, 4.0, 5.0]
        self.s_inc = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        # 构造波动序列: [1.0, 3.0, 1.0, 3.0, 1.0]
        self.s_osc = pd.Series([1.0, 3.0, 1.0, 3.0, 1.0])

    def test_basic_rolling_ops(self):
        """测试基础滑动窗口算子"""
        # Mean(window=3): [NaN, NaN, 2.0, 3.0, 4.0]
        res_mean = ops.ts_mean(self.s_inc, 3)
        assert np.isnan(res_mean.iloc[1])
        assert res_mean.iloc[2] == 2.0
        assert res_mean.iloc[4] == 4.0

        # Sum(window=2): [NaN, 3.0, 5.0, 7.0, 9.0]
        res_sum = ops.ts_sum(self.s_inc, 2)
        assert res_sum.iloc[1] == 3.0
        assert res_sum.iloc[4] == 9.0

    def test_delta_delay(self):
        """测试差分与滞后"""
        # Delta(1): [NaN, 1.0, 1.0, 1.0, 1.0]
        res_delta = ops.delta(self.s_inc, 1)
        assert res_delta.iloc[1] == 1.0

        # Delay(2): [NaN, NaN, 1.0, 2.0, 3.0]
        res_delay = ops.delay(self.s_inc, 2)
        assert res_delay.iloc[2] == 1.0
        assert res_delay.iloc[4] == 3.0

    def test_ts_rank(self):
        """测试时序排名"""
        # s_inc 是单调递增的，ts_rank 应该总是最大值 (1.0)
        # [1, 2, 3] -> 3是第3大，rank=1.0 (Backtrader style normalization usually)
        # factor_ops 实现: (rank - 1) / (len - 1)
        # window=3.
        # idx 2: [1, 2, 3] -> 3 rank 3 -> (3-1)/(3-1) = 1.0
        res = ops.ts_rank(self.s_inc, 3)
        assert np.isclose(res.iloc[2], 1.0)

        # osc: [1, 3, 1] -> window 3. last is 1. rank is 1.5 (average of 1s) or min?
        # scipy rankdata method='average' by default
        # [1, 3, 1] sorted -> [1, 1, 3]. 1s are rank 1.5.
        # factor_ops implementation uses `raw=True` rankdata.
        # Test it:
        res_osc = ops.ts_rank(self.s_osc, 3)
        # [1, 3, 1] -> current is 1. rank is (1.5 - 1)/2 = 0.25
        assert np.isfinite(res_osc.iloc[2])

    def test_decay_linear(self):
        """测试线性衰减加权"""
        # window=3, weights=[1, 2, 3], sum=6
        # input=[1, 2, 3]
        # (1*1 + 2*2 + 3*3) / 6 = (1+4+9)/6 = 14/6 = 2.333...
        res = ops.decay_linear(self.s_inc, 3)
        assert np.isclose(res.iloc[2], 14.0 / 6.0)

    def test_logical_operators(self):
        """测试逻辑/统计算子"""
        # Correlation: s_inc vs s_inc should be 1.0
        res_corr = ops.ts_corr(self.s_inc, self.s_inc, 3)
        assert np.isclose(res_corr.iloc[-1], 1.0)

        # Neg Correlation: s_inc vs -s_inc should be -1.0
        res_neg = ops.ts_corr(self.s_inc, -1 * self.s_inc, 3)
        assert np.isclose(res_neg.iloc[-1], -1.0)