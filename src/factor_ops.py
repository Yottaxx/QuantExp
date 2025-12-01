# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _to_int(n) -> int:
    return int(round(n))


def _as_by(by, like: pd.Series) -> pd.Series | None:
    if by is None:
        return None
    if isinstance(by, str):
        raise TypeError("factor_ops: `by` must be a Series (e.g. df['code']), not a column name string.")
    by = pd.Series(by)
    if len(by) != len(like):
        raise ValueError("factor_ops: `by` length mismatch.")
    return by


def _drop_gb_index(s: pd.Series) -> pd.Series:
    return s.reset_index(level=0, drop=True) if isinstance(s.index, pd.MultiIndex) else s


# ==============================================================================
# Element-wise
# ==============================================================================

def log_abs(x: pd.Series) -> pd.Series:
    """WQ兼容：log(abs(x)). 仅在确实需要时用；收益率等不要用它。"""
    x = pd.to_numeric(x, errors="coerce")
    return np.log(np.abs(x) + 1e-9)


def log1p_pos(x: pd.Series) -> pd.Series:
    """推荐：对非负量纲特征（amount/volume/dv）使用 log1p."""
    x = pd.to_numeric(x, errors="coerce")
    return np.log1p(x.clip(lower=0))


def sign(x: pd.Series) -> pd.Series:
    return np.sign(pd.to_numeric(x, errors="coerce"))


def abs_val(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").abs()


def signed_power(x: pd.Series, e: float) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    return np.sign(x) * (np.abs(x) ** e)


# ==============================================================================
# Cross-sectional (配合 groupby(date).transform 使用)
# ==============================================================================

def cs_rank(x: pd.Series) -> pd.Series:
    return x.rank(pct=True)


# ==============================================================================
# Time-series (panel-safe via `by`)
# ==============================================================================

def ts_mean(x: pd.Series, window, by=None, min_periods: int | None = 1) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    w = _to_int(window)
    by = _as_by(by, x)
    out = x.rolling(w, min_periods=min_periods).mean() if by is None else x.groupby(by, sort=False).rolling(w, min_periods=min_periods).mean()
    return _drop_gb_index(out)


def ts_std(x: pd.Series, window, by=None, min_periods: int | None = 1) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    w = _to_int(window)
    by = _as_by(by, x)
    out = x.rolling(w, min_periods=min_periods).std() if by is None else x.groupby(by, sort=False).rolling(w, min_periods=min_periods).std()
    return _drop_gb_index(out)


def ts_sum(x: pd.Series, window, by=None, min_periods: int | None = 1) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    w = _to_int(window)
    by = _as_by(by, x)
    out = x.rolling(w, min_periods=min_periods).sum() if by is None else x.groupby(by, sort=False).rolling(w, min_periods=min_periods).sum()
    return _drop_gb_index(out)


def ts_max(x: pd.Series, window, by=None, min_periods: int | None = 1) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    w = _to_int(window)
    by = _as_by(by, x)
    out = x.rolling(w, min_periods=min_periods).max() if by is None else x.groupby(by, sort=False).rolling(w, min_periods=min_periods).max()
    return _drop_gb_index(out)


def ts_min(x: pd.Series, window, by=None, min_periods: int | None = 1) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    w = _to_int(window)
    by = _as_by(by, x)
    out = x.rolling(w, min_periods=min_periods).min() if by is None else x.groupby(by, sort=False).rolling(w, min_periods=min_periods).min()
    return _drop_gb_index(out)


def delay(x: pd.Series, lag, by=None) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    l = _to_int(lag)
    by = _as_by(by, x)
    return x.shift(l) if by is None else x.groupby(by, sort=False).shift(l)


def delta(x: pd.Series, lag, by=None) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    l = _to_int(lag)
    by = _as_by(by, x)
    return x.diff(l) if by is None else x.groupby(by, sort=False).diff(l)


def ewm_mean(x: pd.Series, span: int, by=None, adjust: bool = False) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    by = _as_by(by, x)
    if by is None:
        return x.ewm(span=span, adjust=adjust).mean()
    out = x.groupby(by, sort=False).ewm(span=span, adjust=adjust).mean()
    return _drop_gb_index(out)


def ts_cov(x: pd.Series, y: pd.Series, window, by=None, min_periods: int | None = 1) -> pd.Series:
    """cov(x,y)=E[xy]-E[x]E[y]，比 rolling.cov 更稳定（panel安全）"""
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    if len(x) != len(y):
        raise ValueError("ts_cov: x/y length mismatch")
    w = _to_int(window)
    by = _as_by(by, x)

    if by is None:
        ex = x.rolling(w, min_periods=min_periods).mean()
        ey = y.rolling(w, min_periods=min_periods).mean()
        exy = (x * y).rolling(w, min_periods=min_periods).mean()
        return exy - ex * ey

    ex = _drop_gb_index(x.groupby(by, sort=False).rolling(w, min_periods=min_periods).mean())
    ey = _drop_gb_index(y.groupby(by, sort=False).rolling(w, min_periods=min_periods).mean())
    exy = _drop_gb_index((x * y).groupby(by, sort=False).rolling(w, min_periods=min_periods).mean())
    return exy - ex * ey


def ts_corr(x: pd.Series, y: pd.Series, window, by=None, min_periods: int | None = 1) -> pd.Series:
    cov = ts_cov(x, y, window, by=by, min_periods=min_periods)
    sx = ts_std(x, window, by=by, min_periods=min_periods)
    sy = ts_std(y, window, by=by, min_periods=min_periods)
    return cov / (sx * sy + 1e-9)


def ts_skew(x: pd.Series, window, by=None, min_periods: int | None = 1) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    w = _to_int(window)
    by = _as_by(by, x)
    out = x.rolling(w, min_periods=min_periods).skew() if by is None else x.groupby(by, sort=False).rolling(w, min_periods=min_periods).skew()
    return _drop_gb_index(out)


def ts_kurt(x: pd.Series, window, by=None, min_periods: int | None = 1) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    w = _to_int(window)
    by = _as_by(by, x)
    out = x.rolling(w, min_periods=min_periods).kurt() if by is None else x.groupby(by, sort=False).rolling(w, min_periods=min_periods).kurt()
    return _drop_gb_index(out)


def ts_rank(x: pd.Series, window, by=None, min_periods: int | None = None) -> pd.Series:
    """rolling rank of last element in window -> [0,1]. 注意：rolling.apply 偏慢"""
    x = pd.to_numeric(x, errors="coerce")
    w = _to_int(window)
    mp = w if min_periods is None else min_periods
    by = _as_by(by, x)

    def _rank_last(arr: np.ndarray) -> float:
        n = len(arr)
        if n <= 1:
            return 0.0
        return float((rankdata(arr)[-1] - 1) / (n - 1 + 1e-9))

    if by is None:
        return x.rolling(w, min_periods=mp).apply(_rank_last, raw=True)

    out = x.groupby(by, sort=False).rolling(w, min_periods=mp).apply(_rank_last, raw=True)
    return _drop_gb_index(out)
