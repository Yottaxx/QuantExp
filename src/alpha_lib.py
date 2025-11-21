import pandas as pd
import numpy as np
from . import factor_ops as ops


class AlphaFactory:
    """
    【SOTA 多因子构建工厂 v4.0 - 无未来函数版】

    v4.0 核心修复：
    1. 移除时序去极值带来的未来函数泄露。
    2. 将去极值操作移动到截面处理阶段。
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # 基础字段映射
        self.open = df['open']
        self.high = df['high']
        self.low = df['low']
        self.close = df['close']
        self.volume = df['volume']

        self.returns = self.close.pct_change()
        self.vwap = (self.volume * (self.high + self.low + self.close) / 3).cumsum() / (self.volume.cumsum() + 1e-9)
        self.log_ret = np.log(self.close / self.close.shift(1))

    def make_factors(self) -> pd.DataFrame:
        """构建时间序列因子"""
        self._build_style_factors()
        self._build_technical_factors()
        self._build_sota_alphas()
        self._build_advanced_factors()
        self._preprocess_factors()
        return self.df

    @staticmethod
    def add_cross_sectional_factors(panel_df: pd.DataFrame) -> pd.DataFrame:
        """
        【截面因子增强 + 安全清洗】
        """

        # 1. 定义截面算子
        def apply_cs_clean_and_rank(x):
            # 【核心修复】在这里进行去极值 (Winsorize)
            # 因为是对同一天的数据操作，不涉及未来信息
            x = ops.winsorize(x, method='mad')

            # 然后再标准化和排名
            return x.rank(pct=True)

        target_cols = [
            'style_mom_1m', 'style_vol_1m', 'style_liquidity',
            'alpha_money_flow', 'adv_skew_20', 'alpha_006', 'tech_rsi_14'
        ]

        valid_cols = [c for c in target_cols if c in panel_df.columns]

        # 2. 计算截面因子
        for col in valid_cols:
            new_col = f"cs_rank_{col}"
            # GroupBy Date -> Winsorize -> Rank
            panel_df[new_col] = panel_df.groupby('date')[col].transform(apply_cs_clean_and_rank)

        # 3. 计算超额收益 Label
        if 'target' in panel_df.columns:
            market_ret = panel_df.groupby('date')['target'].transform('mean')
            panel_df['excess_label'] = panel_df['target'] - market_ret

        return panel_df

    # ... [中间的因子计算函数 _build_style_factors 等保持不变，省略以节省篇幅] ...
    def _build_style_factors(self):
        self.df['style_mom_1m'] = ops.ts_sum(self.log_ret, 20)
        self.df['style_mom_3m'] = ops.ts_sum(self.log_ret, 60)
        self.df['style_mom_6m'] = ops.ts_sum(self.log_ret, 120)
        self.df['style_vol_1m'] = ops.ts_std(self.returns, 20)
        self.df['style_vol_3m'] = ops.ts_std(self.returns, 60)
        self.df['style_liquidity'] = np.log(ops.ts_sum(self.volume, 20) + 1)
        self.df['style_size_proxy'] = np.log(self.close * self.volume + 1)

    def _build_technical_factors(self):
        ema_12 = self.close.ewm(span=12, adjust=False).mean()
        ema_26 = self.close.ewm(span=26, adjust=False).mean()
        diff = ema_12 - ema_26
        dea = diff.ewm(span=9, adjust=False).mean()
        self.df['tech_macd_hist'] = (diff - dea) * 2

        delta = self.close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        self.df['tech_rsi_14'] = 100 - (100 / (1 + rs))

        ma_20 = ops.ts_mean(self.close, 20)
        self.df['tech_bias_20'] = (self.close - ma_20) / ma_20
        is_up = (self.returns > 0).astype(int)
        self.df['tech_psy_12'] = ops.ts_sum(is_up, 12) / 12

    def _build_sota_alphas(self):
        self.df['alpha_006'] = -1 * ops.ts_corr(self.open, self.volume, 10)
        self.df['alpha_012'] = np.sign(ops.delta(self.volume, 1)) * (-1 * ops.delta(self.close, 1))
        self.df['alpha_trend_strength'] = ops.ts_mean((self.high - self.low) / self.close, 14)
        self.df['alpha_pv_corr'] = ops.ts_corr(ops.ts_rank(self.close, 5), ops.ts_rank(self.volume, 5), 5)
        clv = ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low + 1e-9)
        self.df['alpha_money_flow'] = ops.decay_linear(clv * self.volume, 20)

    def _build_advanced_factors(self):
        self.df['adv_skew_20'] = ops.ts_skew(self.returns, 20)
        self.df['adv_kurt_20'] = ops.ts_kurt(self.returns, 20)
        downside_ret = self.returns.copy()
        downside_ret[downside_ret > 0] = 0
        self.df['adv_downside_vol'] = ops.ts_std(downside_ret, 20)
        change = (self.close - self.close.shift(10)).abs()
        volatility = ops.ts_sum(self.close.diff().abs(), 10)
        self.df['adv_ker_10'] = change / (volatility + 1e-9)
        dollar_volume = self.close * self.volume
        self.df['adv_amihud'] = self.returns.abs() / (dollar_volume + 1e-9)
        hl_ratio = np.log(self.high / self.low)
        self.df['adv_spread_proxy'] = ops.ts_mean(hl_ratio, 5)
        vol_window = 5
        rolling_vol = ops.ts_std(self.returns, vol_window)
        self.df['adv_vol_of_vol'] = ops.ts_std(rolling_vol, 20)

    def _preprocess_factors(self):
        """
        时序预处理 (Time-Series Preprocessing)
        注意：这里只做滚动标准化，移除了全局去极值，防止未来函数泄露。
        """
        factor_cols = [c for c in self.df.columns
                       if any(c.startswith(p) for p in ['style_', 'tech_', 'alpha_', 'adv_'])]

        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df[factor_cols] = self.df[factor_cols].fillna(method='ffill').fillna(0)

        for col in factor_cols:
            series = self.df[col]

            # 【移除】 series = ops.winsorize(series, method='mad')
            # 原因是单只股票的时序去极值需要用到全局统计量，存在 Look-ahead Bias。
            # 我们已将其移至 add_cross_sectional_factors 中。

            # 滚动标准化 (Rolling Z-Score) 是安全的，因为它只看过去 Window 天
            series = ops.zscore(series, window=60)

            self.df[col] = series

        self.df[factor_cols] = self.df[factor_cols].fillna(0)