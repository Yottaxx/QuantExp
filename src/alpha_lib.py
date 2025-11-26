import pandas as pd
import numpy as np
from . import factor_ops as ops
from .config import Config
from pandarallel import pandarallel
import os

# 初始化并行计算环境
pandarallel.initialize(progress_bar=True, nb_workers=os.cpu_count())


class AlphaFactory:
    """
    【SOTA 多因子构建工厂 v7.5 - 深度注释版】

    该工厂负责将原始行情数据 (OHLCV) 转化为可用于机器学习的 Alpha 因子集。
    涵盖六大类因子：
    1. Style (风格): 动量, 波动率, 流动性等 (类 Barra)
    2. Technical (技术): MACD, RSI, KDJ, BOLL 等
    3. Fundamental (基本面): 估值, 成长, 盈利 (需外部财务数据)
    4. SOTA Alphas (前沿): 模仿 WorldQuant Alpha101/191 的量价组合
    5. Advanced (高阶): 偏度, 峰度, Amihud非流动性, 波动率的波动率
    6. Industrial (工业级): Yang-Zhang波动率, Roll价差, 聪明钱因子
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # 提取基础行情字段
        self.open = df['open']
        self.high = df['high']
        self.low = df['low']
        self.close = df['close']
        self.volume = df['volume']

        # 确保索引包含日期，便于后续的时间序列计算
        if 'date' not in self.df.columns and isinstance(self.df.index, pd.DatetimeIndex):
            self.df = self.df.reset_index()

        # 基础收益率
        self.returns = self.close.pct_change()

        # [工具函数] 安全对数
        # 金融数据中常出现0或极小正数，直接log会导致-inf，加上epsilon(1e-9)保证数值稳定性
        def safe_log(series):
            return np.log(np.abs(series) + 1e-9)

        self.safe_log = safe_log

        # VWAP (成交量加权平均价): 机构交易员的基准价格，反映市场平均持仓成本
        self.vwap = (self.volume * (self.high + self.low + self.close) / 3).cumsum() / (self.volume.cumsum() + 1e-9)

        # 对数收益率: 相比简单收益率，具有时间可加性，适合统计建模
        self.log_ret = safe_log(self.close / (self.close.shift(1) + 1e-9))

    def make_factors(self) -> pd.DataFrame:
        """因子构建流水线"""
        self._build_style_factors()  # 风格因子 (动量/波动等)
        self._build_technical_factors()  # 技术指标
        self._build_fundamental_factors()  # 基本面 (PE/PB等)
        self._build_sota_alphas()  # 复杂量价Alpha
        self._build_advanced_factors()  # 高阶统计特征
        self._build_industrial_factors()  # 微观结构因子
        self._build_calendar_factors()  # 日历效应
        self._preprocess_factors()  # 预处理 (去极值/标准化)
        return self.df

    @staticmethod
    def _orthogonalize_factors(df, factor_cols):
        """
        [数学处理] 因子正交化 (SVD分解)
        作用: 消除因子之间的共线性。
        解释: 如果因子A和因子B相关性高达0.9，模型会混淆它们的贡献。
             正交化将它们转换为互不相关的特征，同时保留信息量。
        """
        M = df[factor_cols].fillna(0).values
        try:
            # 标准化
            M_mean = np.mean(M, axis=0)
            M_std = np.std(M, axis=0) + 1e-9
            M = (M - M_mean) / M_std

            # SVD 奇异值分解 -> U * S * Vh
            U, S, Vh = np.linalg.svd(M, full_matrices=False)
            # 这里的正交化处理实际上是取了主成分(PCA降维的一种变体)或白化处理
            M_orth = np.dot(U, Vh)

            # 再次标准化输出
            M_orth = (M_orth - np.mean(M_orth, axis=0)) / (np.std(M_orth, axis=0) + 1e-9)
            return pd.DataFrame(M_orth, columns=factor_cols, index=df.index)
        except Exception as e:
            return df[factor_cols]

    @staticmethod
    def add_cross_sectional_factors(panel_df: pd.DataFrame) -> pd.DataFrame:
        """
        [截面处理] 截面标准化与中性化
        这是多因子模型中最关键的一步。
        """
        if 'date' not in panel_df.columns:
            panel_df = panel_df.reset_index()

        # 辅助函数: 去极值(Winsorize) + 排序标准化(Rank Gauss)
        # 将因子值转换为均匀分布或正态分布，消除异常值影响
        def apply_cs_clean_and_rank(x):
            x = ops.winsorize(x, method='mad')  # MAD法去极值
            rank_val = x.rank(pct=True)  # 转为百分比排名 (0~1)
            return (rank_val - 0.5) * 2  # 映射到 (-1, 1)

        # 需要进行截面处理的重点因子列表
        target_cols = [
            'style_mom_1m', 'style_vol_1m', 'style_liquidity',
            'tech_rsi_14', 'tech_kdj_j', 'tech_bb_width',
            'alpha_money_flow', 'adv_skew_20', 'alpha_006',
            'ind_yang_zhang_vol', 'ind_roll_spread', 'ind_max_ret',
            'fund_ep', 'fund_roe', 'fund_growth'
        ]

        valid_cols = [c for c in target_cols if c in panel_df.columns]

        # 1. 计算截面排名因子 (Cross-Sectional Rank)
        # 逻辑: 我们不关心RSI是80还是90，我们关心这只股票的RSI是否比全市场90%的股票都高
        for col in valid_cols:
            new_col = f"cs_rank_{col}"
            panel_df[new_col] = panel_df.groupby('date')[col].transform(apply_cs_clean_and_rank)

        # 2. 市场中性化 (Market Neutralization)
        # 逻辑: 剔除市场整体波动的影响，提取超额部分 (Relative Value)
        mkt_features = ['style_mom_1m', 'style_vol_1m', 'style_liquidity', 'alpha_006']
        mkt_valid = [c for c in mkt_features if c in panel_df.columns]

        for col in mkt_valid:
            mkt_col_name = f"mkt_mean_{col}"
            # 计算当天的市场均值
            panel_df[mkt_col_name] = panel_df.groupby('date')[col].transform('mean')
            # 因子值减去市场均值 -> 相对强弱
            panel_df[f"rel_{col}"] = panel_df[col] - panel_df[mkt_col_name]

        # 3. 风格因子正交化
        print(">>> [AlphaFactory] 正在进行风格因子正交化 (SVD)...")
        style_cols = [c for c in panel_df.columns if c.startswith('style_')]
        if style_cols:
            def apply_orth_wrapper(g):
                return AlphaFactory._orthogonalize_factors(g, style_cols)

            # 并行计算每天的因子正交化
            print("开始按日期分组，并并行计算正交化结果...")
            orth_results = panel_df.groupby('date', group_keys=False)[style_cols].parallel_apply(apply_orth_wrapper)
            panel_df.update(orth_results)

        # 4. 标签处理 (如果存在)
        if 'target' in panel_df.columns:
            # Rank Label: 预测排名比预测绝对收益率更稳健
            panel_df['rank_label'] = panel_df.groupby('date')['target'].transform(lambda x: x.rank(pct=True))
            # Excess Label: 超额收益率
            market_ret = panel_df.groupby('date')['target'].transform('mean')
            panel_df['excess_label'] = panel_df['target'] - market_ret

        return panel_df

    def _build_style_factors(self):
        """
        [风格因子] 描述资产的基础风险特征 (Barra CNE5/6 风格)
        """
        # 动量 (Momentum): 过去一段时间的累积收益
        # 1个月动量 (常用于捕捉反转或短期趋势)
        self.df['style_mom_1m'] = ops.ts_sum(self.log_ret, 20)
        # 3个月、6个月动量 (中期趋势)
        self.df['style_mom_3m'] = ops.ts_sum(self.log_ret, 60)
        self.df['style_mom_6m'] = ops.ts_sum(self.log_ret, 120)

        # 波动率 (Volatility): 风险的度量
        self.df['style_vol_1m'] = ops.ts_std(self.returns, 20)
        self.df['style_vol_3m'] = ops.ts_std(self.returns, 60)

        # 流动性 (Liquidity): 20日成交量对数，反映股票交易的活跃程度
        self.df['style_liquidity'] = self.safe_log(ops.ts_sum(self.volume, 20) + 1)

        # 市值代理 (Size Proxy): 收盘价*成交量 近似替代成交额，用于捕捉小盘股效应 (Size Effect)
        self.df['style_size_proxy'] = self.safe_log(self.close * self.volume + 1)

        # Beta代理 (Beta Proxy):
        # 注意: 这里计算的是收益率的自相关性(Auto-correlation)，衡量价格惯性或均值回归特性
        # 并非CAPM模型中的市场Beta，但在无大盘指数输入时常作为替代特征
        self.df['style_beta_proxy'] = ops.ts_corr(self.returns, self.returns.shift(1), 20)

    def _build_technical_factors(self):
        """
        [技术指标] 经典的趋势与震荡指标
        """
        # MACD (异同移动平均线): 判断趋势的强弱与转折
        ema_12 = self.close.ewm(span=12, adjust=False).mean()
        ema_26 = self.close.ewm(span=26, adjust=False).mean()
        diff = ema_12 - ema_26
        dea = diff.ewm(span=9, adjust=False).mean()
        self.df['tech_macd_hist'] = (diff - dea) * 2  # MACD柱状图

        # RSI (相对强弱指标): 衡量超买超卖 (0-100)
        delta = self.close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        self.df['tech_rsi_14'] = 100 - (100 / (1 + rs))

        # BIAS (乖离率): 价格偏离均线的程度，用于均值回归策略
        ma_20 = ops.ts_mean(self.close, 20)
        self.df['tech_bias_20'] = (self.close - ma_20) / ma_20

        # PSY (心理线): 过去12天上涨天数的比例，反映市场热度
        is_up = (self.returns > 0).astype(int)
        self.df['tech_psy_12'] = ops.ts_sum(is_up, 12) / 12

        # KDJ (随机指标):
        low_9 = ops.ts_min(self.low, 9)
        high_9 = ops.ts_max(self.high, 9)
        # RSV: 未成熟随机值，反映当前价格在过去9天波动范围的位置
        rsv = (self.close - low_9) / (high_9 - low_9 + 1e-9) * 100
        self.df['tech_kdj_k'] = rsv.ewm(com=2).mean()
        self.df['tech_kdj_d'] = self.df['tech_kdj_k'].ewm(com=2).mean()
        self.df['tech_kdj_j'] = 3 * self.df['tech_kdj_k'] - 2 * self.df['tech_kdj_d']  # J线最敏感

        # Bollinger Bands (布林带): 衡量波动性和相对价格高低
        std_20 = ops.ts_std(self.close, 20)
        upper = ma_20 + 2 * std_20
        lower = ma_20 - 2 * std_20
        self.df['tech_bb_width'] = (upper - lower) / (ma_20 + 1e-9)  # 布林带宽 (波动率指标)
        self.df['tech_bb_pctb'] = (self.close - lower) / (upper - lower + 1e-9)  # %B指标 (相对位置)

    def _build_fundamental_factors(self):
        """
        [基本面因子] 估值与成长 (依赖外部财务数据)
        注意: 这里使用倒数(EP, BP)是为了处理负值PE/PB时的线性关系，且因子越大越好(便宜)。
        """
        if 'pe_ttm' in self.df.columns:
            self.df['fund_ep'] = 1.0 / (self.df['pe_ttm'] + 1e-9)  # 盈利收益率 (Earnings Yield)
        if 'pb' in self.df.columns:
            self.df['fund_bp'] = 1.0 / (self.df['pb'] + 1e-9)  # 账面市值比 (Book-to-Price)
        if 'roe' in self.df.columns:
            self.df['fund_roe'] = self.df['roe']  # 净资产收益率 (盈利能力)
        if 'profit_growth' in self.df.columns:
            self.df['fund_growth'] = self.df['profit_growth']  # 利润增长率 (成长性)
        if 'debt_ratio' in self.df.columns:
            self.df['fund_safety'] = -1 * self.df['debt_ratio']  # 安全性 (负债率取反，越高越安全)

    def _build_sota_alphas(self):
        """
        [SOTA Alphas] 模仿 WorldQuant Alpha101 / 国泰君安 Alpha191
        通过量价的非线性组合挖掘隐藏规律。
        """
        # Alpha 006: 开盘价与成交量的相关性负值
        # 逻辑: 如果放量上涨(相关性高)，取负值做空；如果放量下跌，取正值做多 (反转逻辑)
        self.df['alpha_006'] = -1 * ops.ts_corr(self.open, self.volume, 10)

        # Alpha 012: 符号函数(成交量变化) * (收盘价变化取反)
        # 逻辑: 缩量上涨 或 放量下跌 时给正分 (复杂的反转逻辑)
        self.df['alpha_012'] = np.sign(ops.delta(self.volume, 1)) * (-1 * ops.delta(self.close, 1))

        # 趋势强度: (最高-最低)/收盘价 的均值
        # 逻辑: 衡量K线实体的平均长度，数值越大说明波动越剧烈，趋势可能不稳定
        self.df['alpha_trend_strength'] = ops.ts_mean((self.high - self.low) / self.close, 14)

        # 量价相关性: 收盘价排名与成交量排名的相关性
        # 逻辑: 经典的 "量价齐升" (正相关) 或 "背离" (负相关) 指标
        self.df['alpha_pv_corr'] = ops.ts_corr(ops.ts_rank(self.close, 5), ops.ts_rank(self.volume, 5), 5)

        # 资金流向 (Money Flow): CLV (Close Location Value) * Volume
        # 逻辑: 如果收盘价接近当日最高价，视为机构吸筹(Accumulation)；接近最低价视为派发(Distribution)
        clv = ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low + 1e-9)
        self.df['alpha_money_flow'] = ops.decay_linear(clv * self.volume, 20)

    def _build_advanced_factors(self):
        """
        [高阶统计因子] 基于收益率分布的高阶矩
        """
        # 偏度 (Skewness): 衡量收益率分布的非对称性
        # 逻辑: 负偏度意味着崩盘风险大(左肥尾)，正偏度通常像彩票(右肥尾)
        self.df['adv_skew_20'] = ops.ts_skew(self.returns, 20)

        # 峰度 (Kurtosis): 衡量极端行情发生的概率 (尖峰肥尾)
        self.df['adv_kurt_20'] = ops.ts_kurt(self.returns, 20)

        # 下行波动率 (Downside Volatility): 只计算下跌时的波动
        # 逻辑: 投资者只讨厌下跌的波动，上涨的波动是好事 (Sortino Ratio的分母)
        downside_ret = self.returns.copy()
        downside_ret[downside_ret > 0] = 0
        self.df['adv_downside_vol'] = ops.ts_std(downside_ret, 20)

        # 考夫曼效率系数 (KER, Efficiency Ratio)
        # 逻辑: (位移 / 路程)。如果价格直线涨到终点，ER=1；如果震荡很多次才到，ER接近0。衡量趋势的平滑度。
        change = (self.close - self.close.shift(10)).abs()  # 位移
        volatility = ops.ts_sum(self.close.diff().abs(), 10)  # 路程
        self.df['adv_ker_10'] = change / (volatility + 1e-9)

        # Amihud 非流动性因子 (Illiquidity)
        # 逻辑: |收益率| / 成交金额。 也就是单位成交金额带来的价格冲击。
        # 数值越大，越缺乏流动性(一点点买单就把价格拉飞了)，通常有非流动性溢价。
        dollar_volume = self.close * self.volume
        self.df['adv_amihud'] = self.returns.abs() / (dollar_volume + 1e-9)

        # 买卖价差代理 (Spread Proxy - Corwin-Schultz 简化版)
        # 逻辑: 利用最高价和最低价的比率来估算买卖价差。Ratio越大，意味着High/Low差异大，通常Spread也大。
        hl_ratio = self.safe_log(self.high / self.low)
        self.df['adv_spread_proxy'] = ops.ts_mean(hl_ratio, 5)

        # 波动率的波动率 (Vol of Vol)
        # 逻辑: 衡量市场风险的不确定性。如果VoV很高，说明市场处于极度恐慌或极度不稳定的状态。
        vol_window = 5
        rolling_vol = ops.ts_std(self.returns, vol_window)
        self.df['adv_vol_of_vol'] = ops.ts_std(rolling_vol, 20)

    def _build_industrial_factors(self):
        """
        [工业级微观结构因子] 针对高频数据特征的低频化映射
        """
        # Yang-Zhang 波动率 (最经典的波动率估计器)
        # 逻辑: 结合了隔夜跳空(Open-Close_prev)和日内波动(High-Low, Open-Close)，是效率最高的历史波动率估计量。
        o_c_lag = self.safe_log(self.open / self.close.shift(1))  # 隔夜收益
        c_o = self.safe_log(self.close / self.open)  # 日内收益
        # Rogers-Satchell项
        rs_vol = self.safe_log(self.high / self.close) * self.safe_log(self.high / self.open) + \
                 self.safe_log(self.low / self.close) * self.safe_log(self.low / self.open)
        N = 20
        k = 0.34  # 经验系数
        sigma_open = ops.ts_std(o_c_lag, N) ** 2
        sigma_close = ops.ts_std(c_o, N) ** 2
        sigma_rs = ops.ts_mean(rs_vol, N)
        self.df['ind_yang_zhang_vol'] = np.sqrt(sigma_open + k * sigma_close + (1 - k) * sigma_rs)

        # Roll Spread (Roll 价差模型)
        # 逻辑: 相邻两日价格变化的协方差如果是负的，是因为买卖价差(Bid-Ask Bounce)导致的。
        # 协方差越负，价差越大，流动性越差。
        delta_p = self.close.diff()
        cov_delta = ops.ts_cov(delta_p, delta_p.shift(1), 20)
        cov_delta[cov_delta > 0] = 0  # 理论上应为负，强行修正正值
        self.df['ind_roll_spread'] = 2 * np.sqrt(-cov_delta + 1e-9)

        # MAX Factor (博彩效应)
        # 逻辑: 过去一个月最大的单日收益率。散户喜欢买"彩票股"，导致这些股票被高估，未来收益率通常较低(异象)。
        self.df['ind_max_ret'] = ops.ts_max(self.returns, 20)

        # 成交量变异系数 (Vol CV)
        # 逻辑: 成交量的标准差 / 均值。衡量成交量是否均匀。突然的放量会导致CV升高。
        self.df['ind_vol_cv'] = ops.ts_std(self.volume, 20) / (ops.ts_mean(self.volume, 20) + 1e-9)

        # Smart Money Factor (聪明钱因子)
        # 逻辑: 这里的定义是 (K线实体长度 / 总长度) * log(成交量)。
        # 假设: 聪明的钱交易时会推动价格产生实体(实实在在的涨跌)，而噪音交易往往产生长影线(犹豫不决)。
        body_len = (self.close - self.open).abs()
        total_len = (self.high - self.low) + 1e-9
        self.df['ind_smart_money'] = (body_len / total_len) * self.safe_log(self.volume + 1)

    def _build_calendar_factors(self):
        """
        [日历效应因子] 时间嵌入
        用于神经网络学习周一效应、月末效应等。
        """
        if 'date' not in self.df.columns: return
        dates = self.df['date'].dt
        # 归一化到 (-1, 1) 区间，便于神经网络处理
        self.df['time_dow'] = (dates.dayofweek - 2) / 2.0  # Day of Week
        self.df['time_dom'] = (dates.day - 15) / 15.0  # Day of Month
        self.df['time_moy'] = (dates.month - 6.5) / 5.5  # Month of Year

    def _preprocess_factors(self):
        """
        [预处理] 数据清洗标准化
        """
        # 筛选出所有刚才计算出的因子列
        factor_cols = [c for c in self.df.columns
                       if any(c.startswith(p) for p in Config.FEATURE_PREFIXES)]

        # 1. 替换无穷大值 (Inf) 为 NaN
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # 2. 缺失值填充 0 (在Z-Score之后填0意味着填均值)
        self.df[factor_cols] = self.df[factor_cols].fillna(0)

        for col in factor_cols:
            if col.startswith('time_'): continue  # 时间因子不需要去极值
            series = self.df[col]

            # 3. Z-Score 标准化 (Rolling Window)
            # 使用过去60天的均值和方差进行标准化，防止未来数据泄露
            series = ops.zscore(series, window=60)

            # 4. 盖帽法 (Clipping)
            # 将超过4倍标准差的异常值强制拉回到4倍标准差，防止梯度爆炸
            series = series.clip(-4, 4)
            self.df[col] = series

        # 再次填充可能产生的NaN
        self.df[factor_cols] = self.df[factor_cols].fillna(0)