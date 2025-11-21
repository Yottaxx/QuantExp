import pandas as pd
import numpy as np
from scipy.stats import rankdata


# ==============================================================================
# 第一部分：时间序列算子 (Time-Series Operators)
# 场景：针对【单只股票】的历史数据进行滑动窗口计算。
# 作用：提取该股票在时间维度上的趋势、波动和形态特征。
# ==============================================================================

def ts_mean(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window=window).mean()

def ts_std(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window=window).std()

def ts_sum(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window=window).sum()

def ts_max(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window=window).max()

def ts_min(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window=window).min()

def delta(x: pd.Series, lag: int) -> pd.Series:
    return x.diff(lag)

def delay(x: pd.Series, lag: int) -> pd.Series:
    return x.shift(lag)

def ts_rank(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window).apply(lambda arr: (rankdata(arr)[-1] - 1) / (len(arr) - 1), raw=True)

def decay_linear(x: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1)
    w = weights / weights.sum()
    return x.rolling(window).apply(lambda arr: np.dot(arr, w), raw=True)

def ts_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    return x.rolling(window).corr(y)

def ts_cov(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    return x.rolling(window).cov(y)

def ts_skew(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window).skew()

def ts_kurt(x: pd.Series, window: int) -> pd.Series:
    return x.rolling(window).kurt()

# --- 新增/确认辅助算子 ---

def ts_returns(x: pd.Series, lag: int = 1) -> pd.Series:
    """计算收益率"""
    return x.pct_change(lag)

def log(x: pd.Series) -> pd.Series:
    """安全对数"""
    return np.log(np.abs(x) + 1e-9)

def abs_val(x: pd.Series) -> pd.Series:
    """绝对值"""
    return x.abs()

# ==============================================================================
# 截面与清洗算子 (保持不变)
# ==============================================================================

def cs_rank(x: pd.Series) -> pd.Series:
    return x.rank(pct=True)

def cs_zscore(x: pd.Series) -> pd.Series:
    return (x - x.mean()) / x.std()

def winsorize(x: pd.Series, method: str = "mad", limits: tuple = (0.025, 0.025)) -> pd.Series:
    x_clean = x.copy()
    if method == "mad":
        median = x.median()
        mad = (x - median).abs().median()
        sigma = mad * 1.4826
        lower = median - 3 * sigma
        upper = median + 3 * sigma
        x_clean = x_clean.clip(lower=lower, upper=upper)
    return x_clean

def zscore(x: pd.Series, window: int) -> pd.Series:
    mean = x.rolling(window).mean()
    std = x.rolling(window).std()
    return (x - mean) / (std + 1e-9)