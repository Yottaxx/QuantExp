import pandas as pd
import numpy as np
from scipy.stats import rankdata


# ==============================================================================
# 第一部分：时间序列算子 (Time-Series Operators)
# 场景：针对【单只股票】的历史数据进行滑动窗口计算。
# 作用：提取该股票在时间维度上的趋势、波动和形态特征。
# ==============================================================================

def ts_mean(x: pd.Series, window: int) -> pd.Series:
    """
    【滚动均值】
    含义：计算过去 window 天的平均值。
    作用：平滑数据，消除噪音，常用于计算均线 (MA)。
    """
    return x.rolling(window=window).mean()


def ts_std(x: pd.Series, window: int) -> pd.Series:
    """
    【滚动标准差】
    含义：计算过去 window 天的数据波动剧烈程度。
    作用：衡量风险。波动越大，std 越大。
    """
    return x.rolling(window=window).std()


def ts_sum(x: pd.Series, window: int) -> pd.Series:
    """
    【滚动求和】
    含义：计算过去 window 天的总和。
    作用：常用于计算一段时间的累计成交量、累计涨幅。
    """
    return x.rolling(window=window).sum()


def ts_max(x: pd.Series, window: int) -> pd.Series:
    """
    【滚动最大值】
    含义：过去 window 天内的最高点。
    作用：寻找阶段性高点，用于计算“距离新高的距离”。
    """
    return x.rolling(window=window).max()


def ts_min(x: pd.Series, window: int) -> pd.Series:
    """
    【滚动最小值】
    含义：过去 window 天内的最低点。
    作用：寻找支撑位。
    """
    return x.rolling(window=window).min()


def delta(x: pd.Series, lag: int) -> pd.Series:
    """
    【差分算子】
    公式：x(t) - x(t-lag)
    含义：今天的值减去 lag 天前的值。
    作用：计算变化量。例如 delta(close, 1) 就是今天的涨跌额。
    """
    return x.diff(lag)


def delay(x: pd.Series, lag: int) -> pd.Series:
    """
    【滞后算子】
    公式：x(t-lag)
    含义：取 lag 天前的数据。
    作用：获取历史数据，例如“昨天的收盘价”。
    """
    return x.shift(lag)


def ts_rank(x: pd.Series, window: int) -> pd.Series:
    """
    【滚动排名】(Time-Series Rank)
    含义：今天的值，在过去 window 天里排第几名？(归一化到 0~1)
    逻辑：
        - 1.0 表示创了 window 天新高。
        - 0.0 表示创了 window 天新低。
        - 0.5 表示处于中间位置。
    作用：消除绝对数值的影响，只看“相对强弱”。
    """
    # 使用 scipy 的 rankdata 进行快速排名
    # raw=True 让 rolling 传入 numpy 数组而非 Series，速度快 10 倍
    return x.rolling(window).apply(lambda arr: (rankdata(arr)[-1] - 1) / (len(arr) - 1), raw=True)


def decay_linear(x: pd.Series, window: int) -> pd.Series:
    """
    【线性衰减加权移动平均】(Linear Decay Weighted Moving Average)
    含义：计算移动平均，但给离现在越近的数据越大的权重。
    权重分布：[1, 2, 3, ..., window] (越近权重越大)
    作用：比普通均线反应更快，更能捕捉短期趋势。
    """
    weights = np.arange(1, window + 1)
    w = weights / weights.sum()  # 归一化权重，使其和为 1
    return x.rolling(window).apply(lambda arr: np.dot(arr, w), raw=True)


def ts_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """
    【滚动相关系数】(Rolling Correlation)
    含义：计算两个序列在过去 window 天的同步程度。
    范围：-1 (完全负相关) 到 1 (完全正相关)。
    作用：著名的 Alpha 006 就是用这个算的 (价量相关性)。
    """
    return x.rolling(window).corr(y)


def ts_cov(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """
    【滚动协方差】(Rolling Covariance)
    含义：衡量两个变量总体误差的期望。
    作用：衡量两个资产或因子的联动波动性。
    """
    return x.rolling(window).cov(y)


def ts_skew(x: pd.Series, window: int) -> pd.Series:
    """
    【滚动偏度】(Skewness)
    含义：衡量数据分布的对称性。
    作用：
        - 负偏度 (左偏)：意味着大概率小涨，偶尔暴跌 (如美股大盘)。
        - 正偏度 (右偏)：意味着大概率小跌，偶尔暴涨 (如彩票、妖股)。
    """
    return x.rolling(window).skew()


def ts_kurt(x: pd.Series, window: int) -> pd.Series:
    """
    【滚动峰度】(Kurtosis)
    含义：衡量数据分布的“尖锐”程度和“肥尾”程度。
    作用：峰度越高，出现极端行情（暴涨暴跌）的概率越大。
    """
    return x.rolling(window).kurt()


# ==============================================================================
# 第二部分：截面算子 (Cross-Sectional Operators)
# 场景：针对【某一天全市场所有股票】进行计算。
# 注意：由于目前的 DataProvider 是逐只股票处理的 (TS模式)，这些算子在这里主要作为
#       库函数储备。实际使用需在 Panel 数据（日期*股票矩阵）上进行。
# ==============================================================================

def cs_rank(x: pd.Series) -> pd.Series:
    """
    【截面排名】
    含义：将该时刻所有股票的因子值排序，并归一化到 0~1。
    作用：将因子值转换为“分位数”，消除不同时间段市场波动率差异的影响。
    """
    return x.rank(pct=True)


def cs_zscore(x: pd.Series) -> pd.Series:
    """
    【截面标准化】(Cross-Sectional Z-Score)
    公式：(x - mean) / std
    作用：让因子服从标准正态分布 N(0, 1)，方便不同因子进行加权合成。
    """
    return (x - x.mean()) / x.std()


def neutralize(x: pd.Series) -> pd.Series:
    """
    【简单中性化】(Neutralization)
    含义：剔除因子的系统性偏差（如大盘影响）。
    简化版实现：减去均值 (De-mean)。
    完整版需要回归剔除行业因子和市值因子 (Res = y - (w1*Ind + w2*Cap) * Beta)。
    """
    if x.isnull().all():
        return x
    return x - x.mean()


# ==============================================================================
# 第三部分：数据清洗与缩放 (Stabilization & Scaling)
# 场景：因子计算完毕后，送入模型前的最后一步处理。
# ==============================================================================

def winsorize(x: pd.Series, method: str = "mad", limits: tuple = (0.025, 0.025)) -> pd.Series:
    """
    【去极值】(Winsorization)
    作用：把极其异常的数据（Outliers）拉回到正常范围，防止它们误导模型。
    方法：
        - "mad": 中位数去极值（最稳健，推荐）。
        - "sigma": 3倍标准差去极值。
        - "quantile": 分位数去极值（如缩尾到 1% 和 99%）。
    """
    x_clean = x.copy()

    if method == "quantile":
        lower = x.quantile(limits[0])
        upper = x.quantile(1 - limits[1])
    elif method == "mad":
        # MAD: 中位数绝对偏差 (Median Absolute Deviation)
        # 相比标准差，MAD 不受极端值影响，更适合金融数据。
        median = x.median()
        mad = (x - median).abs().median()
        # 1.4826 是常数，用于将 MAD 转换为等效的标准差
        sigma = mad * 1.4826
        lower = median - 3 * sigma
        upper = median + 3 * sigma
    elif method == "sigma":
        mean = x.mean()
        std = x.std()
        lower = mean - 3 * std
        upper = mean + 3 * std
    else:
        return x

    # 将超出范围的值强行拉回到边界值 (Clip)
    x_clean = x_clean.clip(lower=lower, upper=upper)
    return x_clean


def scale_to_vol(x: pd.Series, target_vol: float = 0.2, window: int = 20) -> pd.Series:
    """
    【波动率缩放】(Volatility Scaling)
    含义：动态调整杠杆，使得因子的波动率稳定在一个目标值 (如年化 20%)。
    作用：在市场平静时放大信号（加杠杆），在市场剧烈波动时缩小信号（降杠杆），控制风险。
    """
    # 计算历史波动率
    hist_vol = x.rolling(window).std()
    hist_vol.replace(0, np.nan, inplace=True)

    # 计算缩放系数
    leverage = target_vol / hist_vol

    # 加上杠杆上限 (例如最大只能放3倍)，防止过度放大噪音
    leverage = leverage.clip(0, 3)

    return x * leverage


def zscore(x: pd.Series, window: int) -> pd.Series:
    """
    【滚动标准化】(Rolling Z-Score)
    公式：(当前值 - 过去N天均值) / 过去N天标准差
    作用：
        将数据转换为“相对历史水平的强弱”。
        例如，成交量绝对值 100万 没意义，但 ZScore=3.0 意味着“今天是过去很难见到的放量”。
    """
    mean = x.rolling(window).mean()
    std = x.rolling(window).std()
    return (x - mean) / (std + 1e-9)