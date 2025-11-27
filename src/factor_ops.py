import pandas as pd
import numpy as np
from scipy.stats import rankdata

# ==============================================================================
# 基础辅助函数 (Helper Functions)
# ==============================================================================

def _to_int(n):
    """将窗口参数强制转为整数 (兼容 WQ 公式中的浮点窗口)"""
    return int(round(n))

# ==============================================================================
# 元素级算子 (Element-wise Operators)
# ==============================================================================

def log(x: pd.Series) -> pd.Series:
    """安全对数"""
    return np.log(np.abs(x) + 1e-9)

def sign(x: pd.Series) -> pd.Series:
    """符号函数"""
    return np.sign(x)

def abs_val(x: pd.Series) -> pd.Series:
    """绝对值"""
    return x.abs()

def signed_power(x: pd.Series, e: float) -> pd.Series:
    """保持符号的幂运算: sign(x) * |x|^e"""
    return np.sign(x) * (np.abs(x) ** e)

def scale(x: pd.Series, k: float = 1) -> pd.Series:
    """缩放因子: 使得绝对值之和为 k"""
    return x.mul(k).div(np.abs(x).sum() + 1e-9)

# ==============================================================================
# 截面算子 (Cross-Sectional Operators)
# ==============================================================================

def cs_rank(x: pd.Series) -> pd.Series:
    """截面排名 (百分比 0~1)"""
    return x.rank(pct=True)

def winsorize(x: pd.Series, method: str = "mad") -> pd.Series:
    """去极值 (MAD法)"""
    x_clean = x.copy()
    if method == "mad":
        median = x.median()
        mad = (x - median).abs().median()
        sigma = mad * 1.4826
        lower = median - 3 * sigma
        upper = median + 3 * sigma
        x_clean = x_clean.clip(lower=lower, upper=upper)
    return x_clean

# ==============================================================================
# 时间序列算子 (Rolling Time-Series Operators)
# ==============================================================================

def ts_mean(x: pd.Series, window) -> pd.Series:
    return x.rolling(_to_int(window)).mean()

def ts_std(x: pd.Series, window) -> pd.Series:
    return x.rolling(_to_int(window)).std()

def ts_sum(x: pd.Series, window) -> pd.Series:
    return x.rolling(_to_int(window)).sum()

def ts_max(x: pd.Series, window) -> pd.Series:
    return x.rolling(_to_int(window)).max()

def ts_min(x: pd.Series, window) -> pd.Series:
    return x.rolling(_to_int(window)).min()

def delta(x: pd.Series, lag) -> pd.Series:
    return x.diff(_to_int(lag))

def delay(x: pd.Series, lag) -> pd.Series:
    return x.shift(_to_int(lag))

def ts_corr(x: pd.Series, y: pd.Series, window) -> pd.Series:
    return x.rolling(_to_int(window)).corr(y)

def ts_cov(x: pd.Series, y: pd.Series, window) -> pd.Series:
    return x.rolling(_to_int(window)).cov(y)

def ts_skew(x: pd.Series, window) -> pd.Series:
    return x.rolling(_to_int(window)).skew()

def ts_kurt(x: pd.Series, window) -> pd.Series:
    return x.rolling(_to_int(window)).kurt()

def ts_rank(x: pd.Series, window) -> pd.Series:
    """时间序列排名 (Rolling Rank)"""
    w = _to_int(window)
    def _rank_last(arr):
        return (rankdata(arr)[-1] - 1) / (len(arr) - 1 + 1e-9)
    return x.rolling(w).apply(_rank_last, raw=True)

def ts_argmax(x: pd.Series, window) -> pd.Series:
    """最大值所在的相对位置 (0 ~ window-1)"""
    return x.rolling(_to_int(window)).apply(np.argmax, raw=True).astype(float)

def ts_argmin(x: pd.Series, window) -> pd.Series:
    """最小值所在的相对位置 (0 ~ window-1)"""
    return x.rolling(_to_int(window)).apply(np.argmin, raw=True).astype(float)

def decay_linear(x: pd.Series, window) -> pd.Series:
    """线性衰减加权移动平均 (Weighted Moving Average)"""
    w_size = _to_int(window)
    weights = np.arange(1, w_size + 1)
    w = weights / weights.sum()
    return x.rolling(w_size).apply(lambda arr: np.dot(arr, w), raw=True)

def zscore(x: pd.Series, window) -> pd.Series:
    """Rolling Z-Score"""
    w = _to_int(window)
    mean = x.rolling(w).mean()
    std = x.rolling(w).std()
    return (x - mean) / (std + 1e-9)