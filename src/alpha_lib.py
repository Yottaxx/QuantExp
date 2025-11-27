import pandas as pd
import numpy as np
from . import factor_ops as ops
from .config import Config
from pandarallel import pandarallel
import os
import warnings

# 忽略除零警告等
warnings.filterwarnings('ignore')

# 初始化并行计算环境 (Verbose=0 静默模式)
pandarallel.initialize(progress_bar=False, nb_workers=os.cpu_count(), verbose=0)


class AlphaFactory:
    """
    【SOTA Alpha Engine v10.0 - Full Spectrum】

    涵盖七大类因子：
    0. Raw (原始): 标准化的 OHLCV + Turnover (新增)
    1. Style (风格): 动量, 波动率, 流动性 (类 Barra)
    2. Technical (技术): MACD, RSI, KDJ, BOLL
    3. Fundamental (基本面): 估值, 成长 (需外部财务数据)
    4. SOTA Alphas (前沿): WorldQuant 风格量价组合
    5. Advanced (高阶): 偏度, 峰度, Amihud, VoV
    6. Industrial (微观): YZ波动率, Roll价差, 聪明钱
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        # 1. 基础字段提取 (带类型强制转换)
        self.open = df['open'].astype(np.float32)
        self.high = df['high'].astype(np.float32)
        self.low = df['low'].astype(np.float32)
        self.close = df['close'].astype(np.float32)
        self.volume = df['volume'].astype(np.float32)
        self.turnover = df['turnover'].astype(np.float32)

        # 确保索引包含日期
        if 'date' not in self.df.columns and isinstance(self.df.index, pd.DatetimeIndex):
            self.df = self.df.reset_index()

        # 2. 预计算基础中间变量
        # 基础收益率
        self.returns = self.close.pct_change()

        # [工具函数] 安全对数 (防止 log(0) -> -inf)
        def safe_log(series):
            return np.log(np.abs(series) + 1e-9)

        self.safe_log = safe_log

        # VWAP (成交量加权平均价)
        # 用累积量计算更准确，但在滑动窗口中常用 (High+Low+Close)/3 * Vol 近似
        self.vwap = (self.volume * (self.high + self.low + self.close) / 3).cumsum() / (self.volume.cumsum() + 1e-9)

        # 对数收益率 (具有时间可加性)
        self.log_ret = safe_log(self.close / (self.close.shift(1) + 1e-9))

    def make_factors(self) -> pd.DataFrame:
        """因子构建流水线"""
        self._build_raw_factors()  # [新增] 原始行情特征
        self._build_style_factors()  # 风格
        self._build_technical_factors()  # 技术
        self._build_fundamental_factors()  # 基本面
        self._build_sota_alphas()  # WQ Alphas
        self._build_advanced_factors()  # 高阶统计
        self._build_industrial_factors()  # 微观结构
        self._build_calendar_factors()  # 时间嵌入

        self._preprocess_factors()  # 清洗与标准化
        return self.df

    # ==========================================================================
    # 0. Raw Factors (新增：原始数据标准化)
    # ==========================================================================
    def _build_raw_factors(self):
        """
        [Raw Inputs] 将原始 OHLCV 归一化，使模型能感知 K 线形态。
        注意：直接输入价格是没有意义的（不平稳），必须输入"相对于昨日收盘价"的比率。
        """
        # 1. 收益率作为特征
        self.df['raw_ret'] = self.log_ret

        # 2. K线形态特征 (相对于昨日收盘价的涨跌幅)
        # 这种方式保留了 Open/High/Low 的相对位置信息
        prev_close = self.close.shift(1) + 1e-9
        self.df['raw_open_n'] = self.open / prev_close - 1.0
        self.df['raw_high_n'] = self.high / prev_close - 1.0
        self.df['raw_low_n'] = self.low / prev_close - 1.0
        self.df['raw_close_n'] = self.close / prev_close - 1.0

        # 3. 成交量特征 (对数化)
        # 成交量本身绝对值差异极大，必须 Log
        self.df['raw_volume_n'] = self.safe_log(self.volume)

        # 4. 换手率 (如果可用)
        # 换手率本身就是归一化的 (Vol / Shares)，直接使用
        self.df['raw_turnover'] = self.turnover

    # ==========================================================================
    # 1. Style Factors (风格因子)
    # ==========================================================================
    def _build_style_factors(self):
        # 动量 (Momentum)
        self.df['style_mom_1m'] = ops.ts_sum(self.log_ret, 20)
        self.df['style_mom_3m'] = ops.ts_sum(self.log_ret, 60)
        self.df['style_mom_6m'] = ops.ts_sum(self.log_ret, 120)

        # 波动率 (Volatility)
        self.df['style_vol_1m'] = ops.ts_std(self.returns, 20)
        self.df['style_vol_3m'] = ops.ts_std(self.returns, 60)

        # 流动性 (Liquidity)
        self.df['style_liquidity'] = self.safe_log(ops.ts_sum(self.volume, 20) + 1)

        # 市值代理 (Size Proxy)
        self.df['style_size_proxy'] = self.safe_log(self.close * self.volume + 1)

        # Beta代理 (自相关性)
        self.df['style_beta_proxy'] = ops.ts_corr(self.returns, self.returns.shift(1), 20)

    # ==========================================================================
    # 2. Technical Factors (技术指标)
    # ==========================================================================
    def _build_technical_factors(self):
        # MACD
        ema_12 = self.close.ewm(span=12, adjust=False).mean()
        ema_26 = self.close.ewm(span=26, adjust=False).mean()
        diff = ema_12 - ema_26
        dea = diff.ewm(span=9, adjust=False).mean()
        self.df['tech_macd_hist'] = (diff - dea) * 2

        # RSI
        delta = self.close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        self.df['tech_rsi_14'] = 100 - (100 / (1 + rs))

        # BIAS
        ma_20 = ops.ts_mean(self.close, 20)
        self.df['tech_bias_20'] = (self.close - ma_20) / (ma_20 + 1e-9)

        # PSY (心理线)
        is_up = (self.returns > 0).astype(int)
        self.df['tech_psy_12'] = ops.ts_sum(is_up, 12) / 12

        # KDJ
        low_9 = ops.ts_min(self.low, 9)
        high_9 = ops.ts_max(self.high, 9)
        rsv = (self.close - low_9) / (high_9 - low_9 + 1e-9) * 100
        self.df['tech_kdj_k'] = rsv.ewm(com=2).mean()
        self.df['tech_kdj_d'] = self.df['tech_kdj_k'].ewm(com=2).mean()
        self.df['tech_kdj_j'] = 3 * self.df['tech_kdj_k'] - 2 * self.df['tech_kdj_d']

        # Bollinger Bands
        std_20 = ops.ts_std(self.close, 20)
        upper = ma_20 + 2 * std_20
        lower = ma_20 - 2 * std_20
        self.df['tech_bb_width'] = (upper - lower) / (ma_20 + 1e-9)
        self.df['tech_bb_pctb'] = (self.close - lower) / (upper - lower + 1e-9)

    # ==========================================================================
    # 3. Fundamental Factors (基本面 - 需外部数据)
    # ==========================================================================
    def _build_fundamental_factors(self):
        # 使用倒数 (EP, BP) 处理负值并保持线性
        if 'pe_ttm' in self.df.columns:
            self.df['fund_ep'] = 1.0 / (self.df['pe_ttm'] + 1e-9)
        if 'pb' in self.df.columns:
            self.df['fund_bp'] = 1.0 / (self.df['pb'] + 1e-9)
        if 'roe' in self.df.columns:
            self.df['fund_roe'] = self.df['roe']
        if 'profit_growth' in self.df.columns:
            self.df['fund_growth'] = self.df['profit_growth']
        if 'debt_ratio' in self.df.columns:
            self.df['fund_safety'] = -1 * self.df['debt_ratio']

    # ==========================================================================
    # 4. SOTA Alphas (量价组合)
    # ==========================================================================
    def _build_sota_alphas(self):
        # Alpha 006: -1 * Correlation(Open, Volume, 10)
        self.df['alpha_006'] = -1 * ops.ts_corr(self.open, self.volume, 10)

        # Alpha 012: Sign(Delta Vol) * (-1 * Delta Close)
        self.df['alpha_012'] = np.sign(ops.delta(self.volume, 1)) * (-1 * ops.delta(self.close, 1))

        # 趋势强度
        self.df['alpha_trend_strength'] = ops.ts_mean((self.high - self.low) / (self.close + 1e-9), 14)

        # 量价相关性
        self.df['alpha_pv_corr'] = ops.ts_corr(ops.ts_rank(self.close, 5), ops.ts_rank(self.volume, 5), 5)

        # 资金流向 (Money Flow / Accumulation Distribution)
        clv = ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low + 1e-9)
        self.df['alpha_money_flow'] = ops.decay_linear(clv * self.volume, 20)

    # ==========================================================================
    # 5. Advanced Factors (高阶统计)
    # ==========================================================================
    def _build_advanced_factors(self):
        # 偏度与峰度
        self.df['adv_skew_20'] = ops.ts_skew(self.returns, 20)
        self.df['adv_kurt_20'] = ops.ts_kurt(self.returns, 20)

        # 下行波动率
        downside_ret = self.returns.copy()
        downside_ret[downside_ret > 0] = 0
        self.df['adv_downside_vol'] = ops.ts_std(downside_ret, 20)

        # 考夫曼效率系数 (ER)
        change = (self.close - self.close.shift(10)).abs()
        volatility = ops.ts_sum(self.close.diff().abs(), 10)
        self.df['adv_ker_10'] = change / (volatility + 1e-9)

        # Amihud Illiquidity
        dollar_volume = self.close * self.volume
        self.df['adv_amihud'] = self.returns.abs() / (dollar_volume + 1e-9)

        # Spread Proxy (High/Low)
        hl_ratio = self.safe_log(self.high / (self.low + 1e-9))
        self.df['adv_spread_proxy'] = ops.ts_mean(hl_ratio, 5)

        # Vol of Vol
        vol_window = 5
        rolling_vol = ops.ts_std(self.returns, vol_window)
        self.df['adv_vol_of_vol'] = ops.ts_std(rolling_vol, 20)

    # ==========================================================================
    # 6. Industrial Factors (微观结构)
    # ==========================================================================
    def _build_industrial_factors(self):
        # Yang-Zhang Volatility
        o_c_lag = self.safe_log(self.open / (self.close.shift(1) + 1e-9))
        c_o = self.safe_log(self.close / (self.open + 1e-9))
        rs_vol = self.safe_log(self.high / (self.close + 1e-9)) * self.safe_log(self.high / (self.open + 1e-9)) + \
                 self.safe_log(self.low / (self.close + 1e-9)) * self.safe_log(self.low / (self.open + 1e-9))
        N = 20
        k = 0.34
        sigma_open = ops.ts_std(o_c_lag, N) ** 2
        sigma_close = ops.ts_std(c_o, N) ** 2
        sigma_rs = ops.ts_mean(rs_vol, N)
        self.df['ind_yang_zhang_vol'] = np.sqrt(sigma_open + k * sigma_close + (1 - k) * sigma_rs)

        # Roll Spread
        delta_p = self.close.diff()
        cov_delta = ops.ts_cov(delta_p, delta_p.shift(1), 20)
        cov_delta[cov_delta > 0] = 0
        self.df['ind_roll_spread'] = 2 * np.sqrt(-cov_delta + 1e-9)

        # MAX Factor
        self.df['ind_max_ret'] = ops.ts_max(self.returns, 20)

        # Vol CV
        self.df['ind_vol_cv'] = ops.ts_std(self.volume, 20) / (ops.ts_mean(self.volume, 20) + 1e-9)

        # Smart Money
        body_len = (self.close - self.open).abs()
        total_len = (self.high - self.low) + 1e-9
        self.df['ind_smart_money'] = (body_len / total_len) * self.safe_log(self.volume + 1)

    # ==========================================================================
    # 7. Calendar Factors (时间嵌入)
    # ==========================================================================
    def _build_calendar_factors(self):
        if 'date' not in self.df.columns: return
        dates = self.df['date'].dt
        self.df['time_dow'] = (dates.dayofweek - 2) / 2.0  # -1~1
        self.df['time_dom'] = (dates.day - 15) / 15.0  # -1~1
        self.df['time_moy'] = (dates.month - 6.5) / 5.5  # -1~1

    # ==========================================================================
    # Post-Processing (正交化与标准化)
    # ==========================================================================

    @staticmethod
    def _orthogonalize_factors(df, factor_cols):
        """SVD 正交化"""
        M = df[factor_cols].fillna(0).values
        try:
            M_mean = np.mean(M, axis=0)
            M_std = np.std(M, axis=0) + 1e-9
            M = (M - M_mean) / M_std
            U, S, Vh = np.linalg.svd(M, full_matrices=False)
            M_orth = np.dot(U, Vh)
            M_orth = (M_orth - np.mean(M_orth, axis=0)) / (np.std(M_orth, axis=0) + 1e-9)
            return pd.DataFrame(M_orth, columns=factor_cols, index=df.index)
        except:
            return df[factor_cols]

    @staticmethod
    def add_cross_sectional_factors(panel_df: pd.DataFrame) -> pd.DataFrame:
        """
        [截面处理]
        1. 截面 Rank (Uniform Dist)
        2. 市场中性化 (Market Neutral)
        3. 风格正交化 (Style Orthogonalization)
        """
        if 'date' not in panel_df.columns: panel_df = panel_df.reset_index()

        def apply_cs_clean_and_rank(x):
            x = ops.winsorize(x, method='mad')
            rank_val = x.rank(pct=True)
            return (rank_val - 0.5) * 2  # -> [-1, 1]

        # 1. CS Rank Target (关键因子)
        target_cols = [
            'style_mom_1m', 'style_vol_1m', 'style_liquidity',
            'tech_rsi_14', 'tech_kdj_j', 'tech_bb_width',
            'alpha_money_flow', 'adv_skew_20', 'alpha_006',
            'ind_yang_zhang_vol', 'ind_roll_spread', 'ind_max_ret',
            'fund_ep', 'fund_roe', 'fund_growth'
        ]
        valid_cols = [c for c in target_cols if c in panel_df.columns]

        for col in valid_cols:
            panel_df[f"cs_rank_{col}"] = panel_df.groupby('date')[col].transform(apply_cs_clean_and_rank)

        # 2. Market Neutralization
        mkt_features = ['style_mom_1m', 'style_vol_1m', 'style_liquidity', 'alpha_006']
        mkt_valid = [c for c in mkt_features if c in panel_df.columns]

        for col in mkt_valid:
            mkt_mean = panel_df.groupby('date')[col].transform('mean')
            panel_df[f"rel_{col}"] = panel_df[col] - mkt_mean
            panel_df[f"mkt_mean_{col}"] = mkt_mean

        # 3. SVD Orthogonalization (Optional but SOTA)
        style_cols = [c for c in panel_df.columns if c.startswith('style_')]
        if style_cols:
            # 只有当包含多个风格因子时才做
            # 注意：这里使用 transform 或者 apply 都可以，group_keys=False 避免索引错乱
            orth_results = panel_df.groupby('date', group_keys=False)[style_cols].apply(
                lambda g: AlphaFactory._orthogonalize_factors(g, style_cols)
            )
            panel_df.update(orth_results)

        # 4. Labeling
        if 'target' in panel_df.columns:
            panel_df['rank_label'] = panel_df.groupby('date')['target'].transform(lambda x: x.rank(pct=True))
            market_ret = panel_df.groupby('date')['target'].transform('mean')
            panel_df['excess_label'] = panel_df['target'] - market_ret

        return panel_df

    def _preprocess_factors(self):
        """
        [预处理]
        对所有 Config 定义的特征前缀进行统一的:
        FillNa -> Z-Score -> Clip -> FillNa
        """
        # 扫描符合 Config 前缀的列
        factor_cols = [c for c in self.df.columns
                       if any(c.startswith(p) for p in Config.FEATURE_PREFIXES)]

        # 1. 替换 Inf
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # 2. 初次填充
        self.df[factor_cols] = self.df[factor_cols].fillna(0)

        for col in factor_cols:
            if col.startswith('time_'): continue  # 时间嵌入不需要标准化

            # 3. Rolling Z-Score (防止未来数据泄露)
            # 使用过去 60 天的统计量进行标准化
            series = self.df[col]
            series = ops.zscore(series, window=60)

            # 4. Clip (盖帽法)
            series = series.clip(-4, 4)
            self.df[col] = series

        # 5. 再次填充 (Z-Score 后开头会有 NaN)
        self.df[factor_cols] = self.df[factor_cols].fillna(0)