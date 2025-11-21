import pandas as pd
import numpy as np
from . import factor_ops as ops


class AlphaFactory:
    """
    【SOTA 多因子构建工厂 v2.0】

    架构逻辑：
    1. Raw Data (原始数据: OHLCV)
    2. -> Style Factors (风格因子: 描述资产的基本属性，如动量、波动)
    3. -> Technical Factors (技术因子: 经典的图表指标)
    4. -> SOTA Alphas (顶级私募因子: 量价结构、资金流、反直觉逻辑)
    5. -> Advanced Factors (学术前沿: 高阶矩、微观结构)
    6. -> Preprocessing (清洗: 去极值 -> 标准化)
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # 提取基础数据
        self.open = df['open']
        self.high = df['high']
        self.low = df['low']
        self.close = df['close']
        self.volume = df['volume']

        # 预计算一些通用的中间变量，避免重复计算
        self.returns = self.close.pct_change()
        # VWAP: 成交量加权平均价 (Volume Weighted Average Price)
        # 含义: 全天所有交易的平均成本。
        self.vwap = (self.volume * (self.high + self.low + self.close) / 3).cumsum() / (self.volume.cumsum() + 1e-9)
        # Log Return: 对数收益率 (数学性质比简单收益率更好)
        self.log_ret = np.log(self.close / self.close.shift(1))

    def make_factors(self) -> pd.DataFrame:
        """主入口：构建所有因子"""
        self._build_style_factors()  # 风格
        self._build_technical_factors()  # 技术
        self._build_sota_alphas()  # 私募 Alpha
        self._build_advanced_factors()  # 学术前沿
        self._preprocess_factors()  # 清洗与标准化
        return self.df

    # ==========================================================================
    # 1. 风格因子 (Style Factors) - 源自 Barra 模型
    # ==========================================================================
    def _build_style_factors(self):
        """
        构建描述股票基本属性的因子。这些通常是风险因子(Beta)，但也包含 Alpha 信息。
        """
        # --- [Momentum] 动量因子 ---
        # 逻辑: "强者恒强"。过去涨得好的，未来大概率继续涨。
        # 1个月动量 (短期), 3个月动量 (中期), 6个月动量 (长期)
        self.df['style_mom_1m'] = ops.ts_sum(self.log_ret, 20)
        self.df['style_mom_3m'] = ops.ts_sum(self.log_ret, 60)
        self.df['style_mom_6m'] = ops.ts_sum(self.log_ret, 120)

        # --- [Volatility] 波动率因子 ---
        # 逻辑: 风险厌恶。波动大的股票通常风险高，机构倾向于回避，或者要求更高的风险补偿。
        self.df['style_vol_1m'] = ops.ts_std(self.returns, 20)
        self.df['style_vol_3m'] = ops.ts_std(self.returns, 60)

        # --- [Liquidity] 流动性因子 ---
        # 逻辑: 股票越活跃(成交量越大)，越容易买卖。
        # 计算: 过去20天总成交量的对数。
        self.df['style_liquidity'] = np.log(ops.ts_sum(self.volume, 20) + 1)

        # --- [Size Proxy] 市值代理因子 ---
        # 逻辑: 小盘股通常比大盘股有更高的预期收益(小盘股溢价)。
        # 由于没有直接的 TotalShares 数据，用 Close * Volume (成交金额) 近似替代。
        self.df['style_size_proxy'] = np.log(self.close * self.volume + 1)

    # ==========================================================================
    # 2. 技术因子 (Technical Factors) - 经典图表指标
    # ==========================================================================
    def _build_technical_factors(self):
        """
        传统的图表分析指标，虽然简单，但有效捕捉了非线性形态。
        """
        # --- [MACD Hist] ---
        # 逻辑: 均线的发散与聚合。捕捉趋势的启动。
        ema_12 = self.close.ewm(span=12, adjust=False).mean()
        ema_26 = self.close.ewm(span=26, adjust=False).mean()
        diff = ema_12 - ema_26
        dea = diff.ewm(span=9, adjust=False).mean()
        self.df['tech_macd_hist'] = (diff - dea) * 2

        # --- [RSI] 相对强弱指标 ---
        # 逻辑: 衡量买卖力量的对比。>80 超买(要跌)，<20 超卖(要涨)。
        delta = self.close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        self.df['tech_rsi_14'] = 100 - (100 / (1 + rs))

        # --- [Bias] 乖离率 ---
        # 逻辑: 价格偏离均线太远，会有回归均值的动力。
        ma_20 = ops.ts_mean(self.close, 20)
        self.df['tech_bias_20'] = (self.close - ma_20) / ma_20

        # --- [Psy] 心理线 ---
        # 逻辑: 过去12天里有几天是涨的？反映市场热度。
        is_up = (self.returns > 0).astype(int)
        self.df['tech_psy_12'] = ops.ts_sum(is_up, 12) / 12

    # ==========================================================================
    # 3. SOTA Alpha (State-of-the-Art Alphas) - 顶级私募风格
    # ==========================================================================
    def _build_sota_alphas(self):
        """
        复刻 WorldQuant Alpha 101 和 常见的量化私募因子。
        特点：逻辑往往反直觉，挖掘微观结构。
        """
        # --- [Alpha 006] 量价相关性反转 ---
        # 公式: -1 * Correlation(Open, Volume, 10)
        # 深度逻辑:
        #   - 如果 开盘价涨 且 成交量大 (Corr > 0)，通常是散户冲动追高 -> 机构出货 -> 预期下跌。
        #   - 加上 -1 后，该值越小，代表上述情况越严重，看跌；反之看涨。
        self.df['alpha_006'] = -1 * ops.ts_corr(self.open, self.volume, 10)

        # --- [Alpha 012] 逆转趋势 ---
        # 公式: Sign(Delta(Vol)) * (-1 * Delta(Close))
        # 深度逻辑:
        #   - 放量(Delta Vol > 0) + 下跌(Delta Close < 0) -> 恐慌盘出清，底部特征 -> 因子为正 -> 看涨。
        #   - 放量(Delta Vol > 0) + 上涨(Delta Close > 0) -> 诱多接盘，顶部特征 -> 因子为负 -> 看跌。
        self.df['alpha_012'] = np.sign(ops.delta(self.volume, 1)) * (-1 * ops.delta(self.close, 1))

        # --- [Trend Strength] 趋势强度 ---
        # 逻辑: (最高-最低)/收盘。K线实体越长，说明当天分歧越大或趋势越强。
        self.df['alpha_trend_strength'] = ops.ts_mean((self.high - self.low) / self.close, 14)

        # --- [Price Volume Rank Corr] 量价排名相关性 ---
        # 逻辑: "价格处于高位" 和 "成交量处于高位" 是否同步？
        # 同步(Corr>0): 价涨量增，趋势健康。
        # 背离(Corr<0): 价涨量缩，或价跌量增，趋势可能反转。
        self.df['alpha_pv_corr'] = ops.ts_corr(
            ops.ts_rank(self.close, 5),
            ops.ts_rank(self.volume, 5),
            5
        )

        # --- [Money Flow] 资金流向代理 ---
        # 逻辑: 利用 K 线形态推测资金流。
        # (收盘 - 低) - (高 - 收盘) -> 收盘越接近最高价，买盘越强。
        clv = ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low + 1e-9)
        self.df['alpha_money_flow'] = ops.decay_linear(clv * self.volume, 20)

    # ==========================================================================
    # 4. 学术前沿因子 (Advanced Factors)
    # ==========================================================================
    def _build_advanced_factors(self):
        """
        基于高阶矩、信息效率和微观结构的因子。
        """
        # --- [Skewness] 偏度 (尾部风险) ---
        # 逻辑: 收益率分布是否对称？
        # 负偏度 (左偏): 经常出现小涨，但一旦跌就是暴跌 (黑天鹅风险大)。
        self.df['adv_skew_20'] = ops.ts_skew(self.returns, 20)

        # --- [Kurtosis] 峰度 (极端风险) ---
        # 逻辑: 收益率分布是不是“尖峰肥尾”？峰度高意味着极端行情频发。
        self.df['adv_kurt_20'] = ops.ts_kurt(self.returns, 20)

        # --- [Downside Volatility] 下行波动率 ---
        # 逻辑: 只有下跌的波动才是风险，上涨的波动是快乐。
        # 这是一个比普通波动率更精准的风险指标。
        downside_ret = self.returns.copy()
        downside_ret[downside_ret > 0] = 0  # 只保留负收益
        self.df['adv_downside_vol'] = ops.ts_std(downside_ret, 20)

        # --- [Kaufman Efficiency] 市场效率比 ---
        # 公式: 位移 / 路程
        # 含义: 股价是直线涨上去的(效率高, 趋势强)，还是震荡涨上去的(效率低, 噪音大)？
        change = (self.close - self.close.shift(10)).abs()  # 位移
        volatility = ops.ts_sum(self.close.diff().abs(), 10)  # 路程 (波动总和)
        self.df['adv_ker_10'] = change / (volatility + 1e-9)

        # --- [Amihud Illiquidity] 非流动性指标 ---
        # 公式: |收益率| / 成交金额
        # 含义: 花多少钱能把股价砸下去 1%？
        # 值越大，说明流动性越差(小资金就能砸盘)，股票越脆弱。
        dollar_volume = self.close * self.volume
        self.df['adv_amihud'] = self.returns.abs() / (dollar_volume + 1e-9)

        # --- [Spread Proxy] 买卖价差代理 ---
        # 逻辑: Corwin-Schultz 理论。利用 High/Low 的关系估算买一价和卖一价的差值。
        # 价差越大，交易成本越高，流动性越差。
        hl_ratio = np.log(self.high / self.low)
        self.df['adv_spread_proxy'] = ops.ts_mean(hl_ratio, 5)

        # --- [Vol of Vol] 波动率的波动率 ---
        # 逻辑: 市场的恐慌情绪本身是否稳定？
        # 如果波动率忽高忽低，说明市场处于极度不确定状态。
        vol_window = 5
        rolling_vol = ops.ts_std(self.returns, vol_window)
        self.df['adv_vol_of_vol'] = ops.ts_std(rolling_vol, 20)

    # ==========================================================================
    # 5. 预处理流水线 (Preprocessing)
    # ==========================================================================
    def _preprocess_factors(self):
        """
        将原始因子转换为模型可用的 Feature。
        步骤: 清洗 -> 去极值 -> 标准化
        """
        # 找出所有我们生成的因子列
        factor_cols = [c for c in self.df.columns
                       if any(c.startswith(p) for p in ['style_', 'tech_', 'alpha_', 'adv_'])]

        # 1. 填充 INF 和 NaN
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df[factor_cols] = self.df[factor_cols].fillna(method='ffill').fillna(0)

        for col in factor_cols:
            series = self.df[col]

            # 2. 去极值 (Winsorize)
            # 目的: 把 1% 的极端异常值拉回正常范围，防止模型梯度爆炸。
            series = ops.winsorize(series, method='mad')

            # 3. 滚动标准化 (Rolling Z-Score)
            # 目的: 将不同量纲的数据(如价格、成交量、比率)统一归一化到 N(0,1) 分布。
            # 使用 60 天滚动窗口，让模型适应市场环境的动态变化。
            series = ops.zscore(series, window=60)

            self.df[col] = series

        # 最后再次填充可能的空值 (前60天因为滚动窗口会是NaN)
        self.df[factor_cols] = self.df[factor_cols].fillna(0)