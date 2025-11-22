import pandas as pd
import numpy as np
from . import factor_ops as ops


class AlphaFactory:
    # ... [__init__ 保持不变] ...
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.open = df['open']
        self.high = df['high']
        self.low = df['low']
        self.close = df['close']
        self.volume = df['volume']

        # 确保 date 列存在且为 datetime 类型
        # 如果 date 在索引里，这就把它还原为列
        if 'date' not in self.df.columns and isinstance(self.df.index, pd.DatetimeIndex):
            self.df = self.df.reset_index()

        self.returns = self.close.pct_change()
        self.vwap = (self.volume * (self.high + self.low + self.close) / 3).cumsum() / (self.volume.cumsum() + 1e-9)
        self.log_ret = np.log(self.close / self.close.shift(1))

    def make_factors(self) -> pd.DataFrame:
        self._build_style_factors()
        self._build_technical_factors()
        self._build_sota_alphas()
        self._build_advanced_factors()
        self._build_industrial_factors()
        self._build_calendar_factors()  # 【新增】构建日历因子
        self._preprocess_factors()
        return self.df

    # ... [其他 build 函数保持不变，直接复制之前的] ...
    def _build_style_factors(self):
        # ... (略) ...
        self.df['style_mom_1m'] = ops.ts_sum(self.log_ret, 20)
        self.df['style_mom_3m'] = ops.ts_sum(self.log_ret, 60)
        self.df['style_mom_6m'] = ops.ts_sum(self.log_ret, 120)
        self.df['style_vol_1m'] = ops.ts_std(self.returns, 20)
        self.df['style_vol_3m'] = ops.ts_std(self.returns, 60)
        self.df['style_liquidity'] = np.log(ops.ts_sum(self.volume, 20) + 1)
        self.df['style_size_proxy'] = np.log(self.close * self.volume + 1)

    def _build_technical_factors(self):
        # ... (略) ...
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
        # ... (略) ...
        self.df['alpha_006'] = -1 * ops.ts_corr(self.open, self.volume, 10)
        self.df['alpha_012'] = np.sign(ops.delta(self.volume, 1)) * (-1 * ops.delta(self.close, 1))
        self.df['alpha_trend_strength'] = ops.ts_mean((self.high - self.low) / self.close, 14)
        self.df['alpha_pv_corr'] = ops.ts_corr(ops.ts_rank(self.close, 5), ops.ts_rank(self.volume, 5), 5)
        clv = ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low + 1e-9)
        self.df['alpha_money_flow'] = ops.decay_linear(clv * self.volume, 20)

    def _build_advanced_factors(self):
        # ... (略) ...
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

    def _build_industrial_factors(self):
        # ... (略) ...
        o_c_lag = np.log(self.open / self.close.shift(1))
        c_o = np.log(self.close / self.open)
        rs_vol = np.log(self.high / self.close) * np.log(self.high / self.open) + \
                 np.log(self.low / self.close) * np.log(self.low / self.open)
        N = 20
        k = 0.34
        sigma_open = ops.ts_std(o_c_lag, N) ** 2
        sigma_close = ops.ts_std(c_o, N) ** 2
        sigma_rs = ops.ts_mean(rs_vol, N)
        self.df['ind_yang_zhang_vol'] = np.sqrt(sigma_open + k * sigma_close + (1 - k) * sigma_rs)
        delta_p = self.close.diff()
        cov_delta = ops.ts_cov(delta_p, delta_p.shift(1), 20)
        cov_delta[cov_delta > 0] = 0
        self.df['ind_roll_spread'] = 2 * np.sqrt(-cov_delta + 1e-9)
        self.df['ind_max_ret'] = ops.ts_max(self.returns, 20)
        self.df['ind_vol_cv'] = ops.ts_std(self.volume, 20) / (ops.ts_mean(self.volume, 20) + 1e-9)
        body_len = (self.close - self.open).abs()
        total_len = (self.high - self.low) + 1e-9
        self.df['ind_smart_money'] = (body_len / total_len) * np.log(self.volume + 1)

    def _build_calendar_factors(self):
        """
        【新增】日历效应因子 (Calendar Effect)
        让模型感知"今天是周几"、"是否月末"。
        """
        # 确保有 date 列
        if 'date' not in self.df.columns:
            # 尝试从索引恢复
            self.df = self.df.reset_index()

        if 'date' not in self.df.columns:
            return  # 无法计算

        dates = self.df['date'].dt

        # 1. Day of Week (周几)
        # 原始: 0=周一 ... 4=周五
        # 归一化到 [-0.5, 0.5] 区间，方便模型理解
        # (x - 2) / 2.0 -> 周一=-1.0, 周三=0.0, 周五=1.0
        self.df['time_dow'] = (dates.dayofweek - 2) / 2.0

        # 2. Day of Month (月初/月末)
        # 归一化到 [-1, 1]
        self.df['time_dom'] = (dates.day - 15) / 15.0

        # 3. Month of Year (季节性)
        self.df['time_moy'] = (dates.month - 6.5) / 5.5

    def _preprocess_factors(self):
        # 【核心修改】把 time_ 也加入到特征列列表中
        factor_cols = [c for c in self.df.columns
                       if any(
                c.startswith(p) for p in ['style_', 'tech_', 'alpha_', 'adv_', 'ind_', 'time_'])]  # Added time_

        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df[factor_cols] = self.df[factor_cols].fillna(method='ffill').fillna(0)

        for col in factor_cols:
            # 日历因子本身就是归一化好的，不需要去极值和ZScore，防止被破坏结构
            if col.startswith('time_'):
                continue

            series = self.df[col]
            series = ops.zscore(series, window=60)
            series = series.clip(-4, 4)
            self.df[col] = series

        self.df[factor_cols] = self.df[factor_cols].fillna(0)

    # ... [orthogonalize 和 add_cross_sectional_factors 保持不变] ...
    @staticmethod
    def _orthogonalize_factors(df, factor_cols):
        M = df[factor_cols].fillna(0).values
        M = (M - np.mean(M, axis=0)) / (np.std(M, axis=0) + 1e-9)
        try:
            U, S, Vh = np.linalg.svd(M, full_matrices=False)
            M_orth = np.dot(U, Vh)
            M_orth = (M_orth - np.mean(M_orth, axis=0)) / (np.std(M_orth, axis=0) + 1e-9)
            return pd.DataFrame(M_orth, columns=factor_cols, index=df.index)
        except:
            return df[factor_cols]

    @staticmethod
    def add_cross_sectional_factors(panel_df: pd.DataFrame) -> pd.DataFrame:
        def apply_cs_clean_and_rank(x):
            x = ops.winsorize(x, method='mad')
            rank_val = x.rank(pct=True)
            return (rank_val - 0.5) * 2

        target_cols = [
            'style_mom_1m', 'style_vol_1m', 'style_liquidity',
            'alpha_money_flow', 'adv_skew_20', 'alpha_006',
            'ind_yang_zhang_vol', 'ind_roll_spread', 'ind_max_ret'
        ]
        valid_cols = [c for c in target_cols if c in panel_df.columns]
        for col in valid_cols:
            new_col = f"cs_rank_{col}"
            panel_df[new_col] = panel_df.groupby('date')[col].transform(apply_cs_clean_and_rank)

        mkt_features = ['style_mom_1m', 'style_vol_1m', 'style_liquidity', 'alpha_006']
        mkt_valid = [c for c in mkt_features if c in panel_df.columns]
        for col in mkt_valid:
            mkt_col_name = f"mkt_mean_{col}"
            panel_df[mkt_col_name] = panel_df.groupby('date')[col].transform('mean')
            panel_df[f"rel_{col}"] = panel_df[col] - panel_df[mkt_col_name]

        if 'target' in panel_df.columns:
            panel_df['rank_label'] = panel_df.groupby('date')['target'].transform(lambda x: x.rank(pct=True))
            market_ret = panel_df.groupby('date')['target'].transform('mean')
            panel_df['excess_label'] = panel_df['target'] - market_ret

        return panel_df