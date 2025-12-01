import pandas as pd
import numpy as np
from scipy.special import erfinv
from .config import Config


class AlphaFactory:
    """
    【SOTA AlphaFactory v11.2 - Full Factor Restoration】

    Architecture:
    1. Instance Method (make_factors):
       - Calculates ALL Raw Indicators (Restored all missing factors).
       - Normalizes 'State' features (basic_) using Time-Series Z-Score.

    2. Static Method (add_cross_sectional_factors):
       - Normalizes 'Alpha' features (fund_, tech_, etc.) using Cross-Sectional RankGauss.
       - Performs Sector Neutralization.
    """

    WINDOWS = {
        'short': 5,
        'mid': 20,
        'long': 60,
        'trend': 120,
        'rsi': 14
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # 确保按时间排序，方便 rolling 计算
        if 'date' in self.df.columns:
            self.df = self.df.sort_values('date')

        self._load_raw_data()
        self._precompute_globals()

    def _load_raw_data(self):
        """Robust Data Loading"""
        cols = ['open', 'high', 'low', 'close', 'volume', 'turnover', 'amount', 'vwap', 'pe_ttm', 'pb', 'eps']
        idx = self.df.index
        # 针对单只股票数据，使用 ffill 修复停牌
        for c in cols:
            # 使用 get 避免 key error (比如 eps 可能不在 daily data 里而在 merge 后才有)
            if c in self.df.columns:
                self.df[c] = self.df[c].ffill().astype(np.float32)

    def _precompute_globals(self):
        self.safe_log = lambda x: np.log(np.abs(x) + 1e-9)
        self.close = self.df['close']
        self.open = self.df['open']
        self.high = self.df['high']
        self.low = self.df['low']
        self.volume = self.df['volume']

        # Returns
        self.pre_close = self.close.shift(1).fillna(self.open)
        self.log_ret = self.safe_log(self.close / self.pre_close)
        self.returns = self.close.pct_change()

    # =========================================================================
    # Part 1: Per-Stock Generation (Instance Method)
    # =========================================================================
    def make_factors(self) -> pd.DataFrame:
        """
        Executed inside parallel_apply per stock code.
        """
        # 1. Feature Generation (Full List)
        self._build_basic_inputs()
        self._build_fundamental()
        self._build_liquidity()
        self._build_technical()
        self._build_microstructure()
        self._build_interactions()
        self._build_time_embeddings()

        # 2. Time-Series Normalization Only (For basic_ features)
        self._normalize_time_series_only()

        return self.df

    def _normalize_time_series_only(self):
        """仅对 basic_ 开头的状态特征进行时序标准化 (Rolling Z-Score)"""
        all_cols = [c for c in self.df.columns if any(c.startswith(p) for p in Config.FEATURE_PREFIXES)]
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df[all_cols] = self.df[all_cols].fillna(0)

        ts_cols = [c for c in all_cols if c.startswith('basic_')]
        for col in ts_cols:
            mu = self.df[col].rolling(60, min_periods=20).mean()
            sigma = self.df[col].rolling(60, min_periods=20).std() + 1e-9
            z = (self.df[col] - mu) / sigma
            self.df[col] = z.clip(-4, 4).fillna(0).astype(np.float32)

    # --- Feature Builders ---

    def _build_basic_inputs(self):
        """[basic_] 基础状态 (恢复 basic_amp)"""
        pre = self.pre_close
        self.df['basic_ret_log'] = self.log_ret
        self.df['basic_gap'] = self.open / pre - 1.0
        self.df['basic_body'] = (self.close - self.open) / pre
        self.df['basic_shadow_up'] = (self.high - np.maximum(self.open, self.close)) / pre
        self.df['basic_shadow_low'] = (np.minimum(self.open, self.close) - self.low) / pre
        # [Restored] 振幅
        self.df['basic_amp'] = (self.high - self.low) / pre

        self.df['basic_clv'] = (self.close - self.low) / (self.high - self.low + 1e-9)
        vol_ma = self.volume.rolling(self.WINDOWS['mid']).mean() + 1.0
        self.df['basic_vol_bias'] = self.safe_log(self.volume / vol_ma)

    def _build_fundamental(self):
        """[fund_] 基本面 (恢复 eps_acc, leverage_inv)"""
        # 1. Valuation
        if 'pe_ttm' in self.df.columns:
            self.df['fund_ep'] = np.where(self.df['pe_ttm'] > 0, 1.0 / (self.df['pe_ttm'] + 1e-9), 0)
        if 'pb' in self.df.columns:
            self.df['fund_bp'] = np.where(self.df['pb'] > 0, 1.0 / (self.df['pb'] + 1e-9), 0)

        # 2. Quality & Growth Mapping
        # DataProvider 下载的列: 'roe', 'profit_growth', 'debt_ratio', 'eps'
        mapping = {
            'fund_roe': 'roe',
            'fund_profit_growth': 'profit_growth',
        }
        for t, s in mapping.items():
            if s in self.df.columns: self.df[t] = self.df[s]

        # [Restored] 负债率取反 (越高越差 -> 越低越好)
        if 'debt_ratio' in self.df.columns:
            self.df['fund_leverage_inv'] = -1.0 * self.df['debt_ratio']

        # [Restored] EPS Momentum (EPS 加速度)
        if 'eps' in self.df.columns:
            # 季度数据填充后的 diff，反映业绩环比变化
            self.df['fund_eps_acc'] = self.df['eps'].diff(self.WINDOWS['mid']).fillna(0)

    def _build_liquidity(self):
        """[liq_] 流动性 (恢复 liq_turnover_bias)"""
        win = self.WINDOWS['mid']
        long_win = self.WINDOWS['long']
        to = self.df.get('turnover', self.volume)

        self.df['liq_turnover_log'] = self.safe_log(to)

        to_std = to.rolling(win).std()
        to_mean = to.rolling(win).mean() + 1e-9
        self.df['liq_turnover_cv'] = -1.0 * (to_std / to_mean)

        # [Restored] Abnormal Turnover (Bias)
        self.df['liq_turnover_bias'] = to / (to.rolling(long_win).mean() + 1e-9)

        amt = self.df.get('amount', self.close * self.volume + 1.0)
        self.df['liq_amihud'] = self.returns.abs() / (amt + 1e-9)

    def _build_technical(self):
        """[tech_] 技术指标 (恢复 tech_boll_width)"""
        # RSI
        delta = self.close.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.rolling(self.WINDOWS['rsi']).mean()
        ma_down = down.rolling(self.WINDOWS['rsi']).mean()
        rs = ma_up / (ma_down + 1e-9)
        self.df['tech_rsi'] = 100 - (100 / (1 + rs))

        # [Restored] Bollinger Width
        win = self.WINDOWS['mid']
        std = self.close.rolling(win).std()
        ma = self.close.rolling(win).mean() + 1e-9
        self.df['tech_boll_width'] = (4 * std) / ma

        # Momentum & Vol
        self.df['style_mom_1m'] = self.close.pct_change(self.WINDOWS['mid'])
        self.df['style_mom_6m'] = self.close.pct_change(self.WINDOWS['trend'])
        self.df['style_vol_1m'] = self.returns.rolling(self.WINDOWS['mid']).std()

    def _build_microstructure(self):
        """[alpha_] 微观结构 (恢复 gap_reversion)"""
        win = self.WINDOWS['mid']
        k = 0.34

        # Yang-Zhang Components
        ret_overnight = self.safe_log(self.open / self.pre_close)
        var_overnight = ret_overnight.rolling(win).var()

        ret_open_close = self.safe_log(self.close / self.open)
        var_open_close = ret_open_close.rolling(win).var()

        h_c = self.safe_log(self.high / self.close)
        h_o = self.safe_log(self.high / self.open)
        l_c = self.safe_log(self.low / self.close)
        l_o = self.safe_log(self.low / self.open)
        rs_var = (h_c * h_o + l_c * l_o).rolling(win).mean()

        self.df['alpha_vol_yz'] = np.sqrt(var_overnight + k * var_open_close + (1 - k) * rs_var)
        self.df['alpha_gap_contribution'] = np.sqrt(var_overnight) / (self.df['alpha_vol_yz'] + 1e-9)

        # [Restored] Overnight-Intraday Correlation (高开低走特性)
        self.df['alpha_gap_reversion_corr'] = ret_overnight.rolling(win).corr(ret_open_close)

        # Spread Proxy
        gamma = (self.safe_log(self.high.rolling(2).max() / self.low.rolling(2).min())) ** 2
        beta = (self.safe_log(self.high / self.low) ** 2).rolling(2).sum()
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
        self.df['alpha_spread_proxy'] = alpha

    def _build_interactions(self):
        """[alpha_] 交互 (恢复 vol_skew)"""
        win = 20  # Align with vol window

        # PV Corr
        r = self.returns
        v = self.volume.diff()
        self.df['alpha_pv_corr'] = r.rolling(10).corr(v)

        # Smart Money
        body_str = (self.close - self.open).abs() / (self.high - self.low + 1e-9)
        vol_str = self.volume / (self.volume.rolling(20).mean() + 1e-9)
        self.df['alpha_smart_money'] = body_str * np.sqrt(vol_str)

        # [Restored] Vol Skew (Tail Risk Ratio: Upside Vol / Downside Vol)
        # 这种不对称性对神经网络非常有价值
        ret_pos = r.clip(lower=0)
        ret_neg = r.clip(upper=0)
        std_up = ret_pos.rolling(win).std()
        std_down = ret_neg.rolling(win).std() + 1e-9
        self.df['alpha_vol_skew'] = std_up / std_down

    def _build_time_embeddings(self):
        if 'date' in self.df.columns:
            dt = self.df['date'].dt
            self.df['time_dow'] = np.sin(2 * np.pi * dt.dayofweek / 5.0)
            self.df['time_dom'] = np.sin(2 * np.pi * dt.day / 30.0)
            self.df['time_moy'] = np.sin(2 * np.pi * dt.month / 12.0)

    # =========================================================================
    # Part 2: Cross-Sectional Processing (Static Method)
    # =========================================================================
    @staticmethod
    def add_cross_sectional_factors(panel_df: pd.DataFrame) -> pd.DataFrame:
        """
        Executed on the FULL PANEL.
        Responsibility:
        1. Industry Neutralization (Alpha)
        2. RankGauss (Alpha)
        3. Target Ranking (Label Generation) -> [Critical for Training]
        """
        print(">>> [AlphaFactory] Executing Cross-Sectional Enhancement...")

        # 识别 Alpha 因子 (排除 basic_ 和 time_)
        all_cols = [c for c in panel_df.columns if any(c.startswith(p) for p in Config.FEATURE_PREFIXES)]
        cs_targets = [c for c in all_cols if not c.startswith(('basic_', 'time_'))]

        panel_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        panel_df[cs_targets] = panel_df[cs_targets].fillna(0)

        # 1. 行业中性化 (Industry Neutralization)
        if 'industry_cat' in panel_df.columns:
            print(f"   -> Neutralizing {len(cs_targets)} factors by Industry...")
            mean_df = panel_df.groupby(['date', 'industry_cat'])[cs_targets].mean().reset_index()
            temp_cols = [f"{c}_mean" for c in cs_targets]
            mean_df.columns = ['date', 'industry_cat'] + temp_cols

            panel_df = pd.merge(panel_df, mean_df, on=['date', 'industry_cat'], how='left')

            for col in cs_targets:
                panel_df[col] = panel_df[col] - panel_df[f"{col}_mean"]

            panel_df.drop(columns=temp_cols, inplace=True)

        # 2. Global RankGauss (Standardization)
        print(f"   -> Applying RankGauss on {len(cs_targets)} factors...")

        def rank_gauss_vectorized(series):
            n = len(series)
            if n < 10: return series
            epsilon = 1e-6
            ranks = (series.rank(method='average') - 0.5) / n
            ranks = ranks.clip(epsilon, 1 - epsilon)
            return erfinv(2 * ranks - 1)

        # 确保有 date 列用于 groupby
        if 'date' not in panel_df.columns and isinstance(panel_df.index, pd.DatetimeIndex):
            panel_df = panel_df.reset_index()

        for col in cs_targets:
            panel_df[col] = panel_df.groupby('date')[col].transform(rank_gauss_vectorized)

        panel_df[cs_targets] = panel_df[cs_targets].fillna(0).astype(np.float32)

        # =========================================================================
        # [Missing Part Restored] 3. Label Generation (Target Ranking)
        # =========================================================================
        # Jane Street/WorldQuant 通常预测收益率的排名，而不是绝对数值
        if 'target' in panel_df.columns:
            print("   -> Generating Rank Labels (0.0 ~ 1.0)...")

            def pct_rank(x):
                return x.rank(pct=True)

            # 将绝对收益率转换为每日的百分比排名 (Uniform Distribution)
            # 这样模型学到的是"这只股票今天排第几"，消除了大盘涨跌的影响
            panel_df['rank_label'] = panel_df.groupby('date')['target'].transform(pct_rank)

            # 填充 NaN (比如某天只有一只股票，rank可能出问题，虽然很少见)
            panel_df['rank_label'] = panel_df['rank_label'].fillna(0.5).astype(np.float32)

        return panel_df