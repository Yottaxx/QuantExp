import pandas as pd
import numpy as np


class OpLib:
    """基础数学算子库 (Vectorized)"""

    @staticmethod
    def ts_delay(series, d): return series.shift(d)

    @staticmethod
    def ts_delta(series, d): return series.diff(d)

    @staticmethod
    def ts_corr(s1, s2, d): return s1.rolling(d).corr(s2)

    @staticmethod
    def ts_std(series, d): return series.rolling(d).std()

    @staticmethod
    def ts_sum(series, d): return series.rolling(d).sum()

    @staticmethod
    def decay_linear(series, d):
        weights = np.arange(1, d + 1)
        return series.rolling(d).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


class AlphaFactory:
    """因子工厂：生产 SOTA 级别因子"""

    def __init__(self, df):
        self.df = df.copy()
        self.open = df['open']
        self.high = df['high']
        self.low = df['low']
        self.close = df['close']
        self.volume = df['volume']
        self.returns = self.close.pct_change()
        # VWAP 近似
        self.vwap = (self.volume * (self.high + self.low + self.close) / 3).cumsum() / (self.volume.cumsum() + 1e-9)

    def make_factors(self):
        # 1. 动量类
        self.df['alpha_006'] = -1 * OpLib.ts_corr(self.open, self.volume, 10)
        self.df['alpha_mom_10'] = self.close / self.close.shift(10) - 1

        # 2. 波动率类
        self.df['alpha_vol_std'] = OpLib.ts_std(self.returns, 20)
        self.df['alpha_high_low'] = (self.high - self.low) / self.close

        # 3. 资金流类
        clv = (self.close - self.low) / (self.high - self.low + 1e-9)
        self.df['alpha_flow'] = OpLib.decay_linear(clv * self.volume, 5) / (OpLib.ts_sum(self.volume, 5) + 1e-9)

        # 4. 乖离率
        self.df['alpha_bias'] = (self.close - self.vwap) / (self.vwap + 1e-9)

        # 清洗无穷大和空值 (填充0或前值)
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.fillna(method='ffill', inplace=True)
        self.df.fillna(0, inplace=True)

        return self.df