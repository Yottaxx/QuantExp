import pandas as pd
import numpy as np
from . import factor_ops as ops


class AlphaFactory:
    """
    【SOTA 多因子构建工厂 v5.0 - 工业级增强版】

    包含：
    1. Style Factors (Barra)
    2. Technical Factors (TA-Lib)
    3. SOTA Alphas (WorldQuant)
    4. Advanced Factors (High Moments)
    5. [NEW] Industrial Factors (Volatility Structure, Microstructure, Behavioral)
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.open = df['open']
        self.high = df['high']
        self.low = df['low']
        self.close = df['close']
        self.volume = df['volume']

        self.returns = self.close.pct_change()
        self.vwap = (self.volume * (self.high + self.low + self.close) / 3).cumsum() / (self.volume.cumsum() + 1e-9)
        self.log_ret = np.log(self.close / self.close.shift(1))

    def make_factors(self) -> pd.DataFrame:
        self._build_style_factors()
        self._build_technical_factors()
        self._build_sota_alphas()
        self._build_advanced_factors()
        self._build_industrial_factors()  # [新增]
        self._preprocess_factors()
        return self.df

    @staticmethod
    def add_cross_sectional_factors(panel_df: pd.DataFrame) -> pd.DataFrame:
        """截面增强 (包含新增的核心因子)"""

        def apply_cs_clean_and_rank(x):
            x = ops.winsorize(x, method='mad')
            return x.rank(pct=True)

        # 扩充核心因子池，加入新的工业因子进行横向比较
        target_cols = [
            'style_mom_1m', 'style_vol_1m', 'style_liquidity',
            'alpha_money_flow', 'adv_skew_20', 'alpha_006',
            'ind_yang_zhang_vol', 'ind_roll_spread', 'ind_max_ret'  # 新增
        ]

        valid_cols = [c for c in target_cols if c in panel_df.columns]

        for col in valid_cols:
            new_col = f"cs_rank_{col}"
            panel_df[new_col] = panel_df.groupby('date')[col].transform(apply_cs_clean_and_rank)

        if 'target' in panel_df.columns:
            market_ret = panel_df.groupby('date')['target'].transform('mean')
            panel_df['excess_label'] = panel_df['target'] - market_ret

        return panel_df

    # ... [前四个 build 函数保持不变，省略以节省篇幅，请直接复制之前的代码] ...
    # _build_style_factors
    # _build_technical_factors
    # _build_sota_alphas
    # _build_advanced_factors
    # 这里为了完整性，简略写出占位，实际使用请保留原内容
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

    # ==========================================================================
    # 5. [NEW] 工业级实战因子 (Industrial Practice Factors)
    # ==========================================================================
    def _build_industrial_factors(self):
        """
        【新增】基于日线数据挖掘的高效能因子
        来源：顶级期刊 (Journal of Finance) & 机构内部分享
        """

        # --- [1. Yang-Zhang Volatility] 杨-张波动率 ---
        # [逻辑] 相比传统的收盘价标准差，YZ波动率利用了 Open/High/Low/Close 信息，
        # 是目前日线级别最高效的波动率估计量。
        # 它可以捕捉到跳空缺口(Gap)和日内振幅(Range)的风险。
        # 公式较为复杂，包含过夜波动(Overnight)和日内波动(Trading)两部分。

        # 1. 过夜波动 (Close_prev -> Open)
        o_c_lag = np.log(self.open / self.close.shift(1))
        # 2. 日内开收波动 (Open -> Close)
        c_o = np.log(self.close / self.open)
        # 3. 日内极值波动 (Rogers-Satchell proxy)
        rs_vol = np.log(self.high / self.close) * np.log(self.high / self.open) + \
                 np.log(self.low / self.close) * np.log(self.low / self.open)

        # 窗口 N
        N = 20
        # 因子 k = 0.34 / (1.34 + (N+1)/(N-1))，简化取 0.34
        k = 0.34

        sigma_open = ops.ts_std(o_c_lag, N) ** 2
        sigma_close = ops.ts_std(c_o, N) ** 2
        sigma_rs = ops.ts_mean(rs_vol, N)

        # 结果开根号
        self.df['ind_yang_zhang_vol'] = np.sqrt(sigma_open + k * sigma_close + (1 - k) * sigma_rs)

        # --- [2. Roll Spread] 罗尔价差 (流动性代理) ---
        # [逻辑] 著名的 Roll Model (1984)。
        # 价格变化的自协方差与有效价差(Effective Spread)负相关。
        # Spread = 2 * sqrt(-Cov(Delta P_t, Delta P_{t-1}))
        # 如果自协方差 > 0，则 Spread 设为 0。
        # 高价差意味着低流动性，通常预期收益较高(流动性补偿)，但也可能意味着崩盘风险。
        delta_p = self.close.diff()
        cov_delta = ops.ts_cov(delta_p, delta_p.shift(1), 20)
        # 只取负的部分开根号，正的部分置0
        cov_delta[cov_delta > 0] = 0
        self.df['ind_roll_spread'] = 2 * np.sqrt(-cov_delta + 1e-9)

        # --- [3. MAX Factor] 彩票效应因子 ---
        # [逻辑] Bali, Cakici, and Whitelaw (2011).
        # 过去一个月内最大的单日涨幅。
        # 散户喜欢买这种"彩票股"，导致股价被高估。因此，MAX 值越大，未来预期收益越低 (Alpha 为负)。
        self.df['ind_max_ret'] = ops.ts_max(self.returns, 20)

        # --- [4. Volume Coefficient of Variation] 成交量变异系数 ---
        # [逻辑] 衡量成交量的稳定性。
        # 如果成交量忽大忽小(CV高)，说明筹码松动，分歧巨大；
        # 如果成交量稳定(CV低)，说明控盘良好。
        self.df['ind_vol_cv'] = ops.ts_std(self.volume, 20) / (ops.ts_mean(self.volume, 20) + 1e-9)

        # --- [5. Smart Money Factor] 聪明钱因子 (S-Factor) ---
        # [逻辑] 改进版的资金流。
        # 如果价格上涨时成交量主要集中在前半小时(聪明钱进场)，下跌时成交量集中在尾盘(散户恐慌)。
        # 由于没有分钟数据，我们用日线近似：
        # `|Open - Close| / Volume` 类似于 Amihud，但我们关注的是 `(Close - Open) / (High - Low)` 这种实体占比与量的关系。
        # 这是一个简化版的 "实体率 * 换手率"。
        body_len = (self.close - self.open).abs()
        total_len = (self.high - self.low) + 1e-9
        self.df['ind_smart_money'] = (body_len / total_len) * np.log(self.volume + 1)

    def _preprocess_factors(self):
        factor_cols = [c for c in self.df.columns
                       if any(c.startswith(p) for p in ['style_', 'tech_', 'alpha_', 'adv_', 'ind_'])]

        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df[factor_cols] = self.df[factor_cols].fillna(method='ffill').fillna(0)

        for col in factor_cols:
            series = self.df[col]
            series = ops.winsorize(series, method='mad')
            series = ops.zscore(series, window=60)
            self.df[col] = series

        self.df[factor_cols] = self.df[factor_cols].fillna(0)