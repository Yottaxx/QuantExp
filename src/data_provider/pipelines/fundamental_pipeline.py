# -*- coding: utf-8 -*-
from __future__ import annotations

import concurrent.futures
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import akshare as ak
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..clients.ak_client import AkClient
from ..core.config import DPConfig
from ..stores.paths import fundamental_dir, fundamental_path
from ..utils.code import normalize_code
from ..utils.io import atomic_save_parquet


# ----------------------------
# 0) stable schema
# ----------------------------
OUT_COLS = ("date", "roe", "rev_growth", "profit_growth", "debt_ratio", "eps", "bps", "pub_date")


@dataclass(frozen=True)
class ColPick:
    key: str
    candidates: Tuple[str, ...]


# Eastmoney 主指标字段（按报告期）：
# EPSJB 基本每股收益(元), BPS 每股净资产(元),
# ROEJQ 净资产收益率(加权)(%), TOTALOPERATEREVETZ 营业总收入同比增长(%),
# PARENTNETPROFITTZ 归属净利润同比增长(%), ZCFZL 资产负债率(%)  — 见 AkShare 数据字典 :contentReference[oaicite:3]{index=3}
EM_PICKS: Tuple[ColPick, ...] = (
    ColPick("eps", ("EPSJB", "EPSJQ", "EPS", "EPS_BASIC")),
    ColPick("bps", ("BPS",)),
    ColPick("roe", ("ROEJQ", "ROEKCJQ")),
    ColPick("rev_growth", ("TOTALOPERATEREVETZ",)),
    ColPick("profit_growth", ("PARENTNETPROFITTZ", "KCFJCXSYJLRTZ")),
    ColPick("debt_ratio", ("ZCFZL",)),
)

# 当同比缺失时的绝对值 fallback：用 abs 自算 YoY（单位：%）
EM_AUX_ABS = {
    "revenue": ("TOTALOPERATEREVE",),
    "profit": ("PARENTNETPROFIT", "KCFJCXSYJLR"),
}


# ----------------------------
# 1) small utils
# ----------------------------
def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(OUT_COLS))


def _to_em_symbol(code: str) -> Optional[str]:
    """
    6 位代码 -> 东财 SECUCODE 后缀:
    - 6/9 开头通常 SH
    - 0/3 开头通常 SZ
    - 8/4 开头通常 BJ
    若已带 .SZ/.SH/.BJ 则原样返回
    """
    c = (code or "").strip().upper()
    if not c:
        return None
    if c.endswith((".SZ", ".SH", ".BJ")):
        return c
    c6 = normalize_code(c)
    if not c6:
        return None
    if c6[0] in ("6", "9"):
        return f"{c6}.SH"
    if c6[0] in ("0", "3"):
        return f"{c6}.SZ"
    if c6[0] in ("8", "4"):
        return f"{c6}.BJ"
    # fallback：宁可错后缀也别直接丢数据（少数特殊代码）
    return f"{c6}.SZ"


def _coerce_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _first_present(df: pd.DataFrame, cols: Tuple[str, ...]) -> Optional[pd.Series]:
    for c in cols:
        if c in df.columns:
            return df[c]
    return None


def _estimate_pub_dates(report_dates: pd.Series) -> pd.Series:
    """
    估算法定披露截止日（保守、不前视）：
      0331 -> 0430
      0630 -> 0831
      0930 -> 1031
      1231 -> 次年0430
    """
    d = pd.to_datetime(report_dates, errors="coerce")
    y = d.dt.year
    m = d.dt.month
    day = d.dt.day

    out = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")

    m1 = (m == 3) & (day == 31)
    out.loc[m1] = pd.to_datetime(y.loc[m1].astype(str) + "-04-30", errors="coerce")

    m2 = (m == 6) & (day == 30)
    out.loc[m2] = pd.to_datetime(y.loc[m2].astype(str) + "-08-31", errors="coerce")

    m3 = (m == 9) & (day == 30)
    out.loc[m3] = pd.to_datetime(y.loc[m3].astype(str) + "-10-31", errors="coerce")

    m4 = (m == 12) & (day == 31)
    out.loc[m4] = pd.to_datetime((y.loc[m4] + 1).astype(str) + "-04-30", errors="coerce")

    return out


def _yoy_pct_from_abs(dates: pd.DatetimeIndex, abs_vals: np.ndarray) -> np.ndarray:
    """
    YoY% = (abs / abs_1y - 1) * 100
    """
    if abs_vals is None or len(abs_vals) == 0:
        return np.full((0,), np.nan, dtype=np.float32)
    s = pd.Series(abs_vals, index=dates, dtype="float64").sort_index()
    prev = s.reindex(s.index - pd.DateOffset(years=1)).astype("float64")
    prev = prev.replace(0.0, np.nan)
    g = (s / prev - 1.0) * 100.0
    return g.to_numpy(dtype=np.float32)


# ----------------------------
# 2) normalizer (EM indicator -> stable schema)
# ----------------------------
def normalize_fundamental_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Input: ak.stock_financial_analysis_indicator_em(symbol="xxxxxx.SZ", indicator="按报告期") 返回的 DataFrame
      - 每个 REPORT_DATE 一行，很多字段列
    Output: stable schema OUT_COLS (date=report_end, pub_date=notice/update/estimate)
    """
    if raw is None or raw.empty:
        return _empty_frame()

    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if "REPORT_DATE" not in df.columns:
        return _empty_frame()

    df["date"] = _coerce_dt(df["REPORT_DATE"])
    df = df[df["date"].notna()].copy()
    if df.empty:
        return _empty_frame()

    # pub_date 优先：NOTICE_DATE -> UPDATE_DATE -> estimate
    notice = _first_present(df, ("NOTICE_DATE",))
    update = _first_present(df, ("UPDATE_DATE",))
    pub = None
    if notice is not None:
        pub = _coerce_dt(notice)
    elif update is not None:
        pub = _coerce_dt(update)

    if pub is None:
        df["pub_date"] = _estimate_pub_dates(df["date"])
    else:
        df["pub_date"] = pub
        m = df["pub_date"].isna()
        if m.any():
            df.loc[m, "pub_date"] = _estimate_pub_dates(df.loc[m, "date"])

    # 取你要的字段
    for pick in EM_PICKS:
        s = _first_present(df, pick.candidates)
        if s is None:
            df[pick.key] = np.nan
        else:
            df[pick.key] = pd.to_numeric(s, errors="coerce")

    # YoY fallback（统一单位：%）
    dates = pd.DatetimeIndex(df["date"].astype("datetime64[ns]"))
    if df["rev_growth"].isna().all():
        rev_abs = _first_present(df, EM_AUX_ABS["revenue"])
        if rev_abs is not None:
            df["rev_growth"] = _yoy_pct_from_abs(dates, pd.to_numeric(rev_abs, errors="coerce").to_numpy())

    if df["profit_growth"].isna().all():
        prof_abs = _first_present(df, EM_AUX_ABS["profit"])
        if prof_abs is not None:
            df["profit_growth"] = _yoy_pct_from_abs(dates, pd.to_numeric(prof_abs, errors="coerce").to_numpy())

    # 同一报告期可能出现多行（口径/更新），按 UPDATE_DATE/ pub_date 选“最新一条”
    if "UPDATE_DATE" in df.columns:
        df["_upd"] = _coerce_dt(df["UPDATE_DATE"])
        df = df.sort_values(["date", "_upd", "pub_date"]).drop_duplicates("date", keep="last")
        df = df.drop(columns=["_upd"], errors="ignore")
    else:
        df = df.sort_values(["date", "pub_date"]).drop_duplicates("date", keep="last")

    out = df[["date", "roe", "rev_growth", "profit_growth", "debt_ratio", "eps", "bps", "pub_date"]].copy()
    # dtype 统一
    for c in ("roe", "rev_growth", "profit_growth", "debt_ratio", "eps", "bps"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float32)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["pub_date"] = pd.to_datetime(out["pub_date"], errors="coerce")

    return out.sort_values("date").reset_index(drop=True)


# ----------------------------
# 3) pipeline
# ----------------------------
def _fetch_em_indicator(ak_client: AkClient, em_symbol: str, indicator: str) -> pd.DataFrame:
    return ak_client.call(ak.stock_financial_analysis_indicator_em, symbol=em_symbol, indicator=indicator)


class FundamentalPipeline:
    """
    Download & cache quarterly fundamentals per-code.
    Output: {DATA_DIR}/fundamental/{code}.parquet
      columns: date(report_end), roe(%), rev_growth(%), profit_growth(%), debt_ratio(%), eps(元), bps(元), pub_date

    Data source (default): stock_financial_analysis_indicator_em (东财财务分析-主要指标) :contentReference[oaicite:4]{index=4}
    """
    SCHEMA_VER = 4  # bump: switch data source + pub_date logic

    def __init__(self, cfg: DPConfig, ak_client: AkClient, logger):
        self.cfg = cfg
        self.ak_client = ak_client
        self.logger = logger
        os.makedirs(fundamental_dir(cfg), exist_ok=True)

    def _should_skip(self, path: str) -> bool:
        days = int(self.cfg.get("FUND_TTL_DAYS", 5) or 5)
        ttl = max(1, days) * 24 * 3600
        return os.path.exists(path) and os.path.getsize(path) > 512 and (time.time() - os.path.getmtime(path)) < ttl

    def _download_one(self, code: str) -> Tuple[str, bool, str, int]:
        c6 = normalize_code(code)
        if not c6:
            return str(code), True, "BadCode", 0

        path = fundamental_path(self.cfg, c6)
        if self._should_skip(path):
            return c6, True, "Skipped", -1

        em_symbol = _to_em_symbol(code)
        if not em_symbol:
            return c6, True, "BadSymbol", 0

        indicator = str(self.cfg.get("FUND_EM_INDICATOR", "按报告期") or "按报告期")

        try:
            raw = _fetch_em_indicator(self.ak_client, em_symbol, indicator)
            out = normalize_fundamental_frame(raw)
            atomic_save_parquet(
                out,
                path,
                index=False,
                compression=str(self.cfg.get("PARQUET_COMPRESSION", "zstd") or "zstd"),
            )
            return c6, True, ("Empty" if out.empty else "Success"), int(len(out))
        except Exception as e:
            return c6, False, f"Failed({type(e).__name__})", 0

    def download(self, codes: Iterable[str]) -> None:
        if not bool(self.cfg.get("SYNC_FUNDAMENTAL", False)):
            self.logger.info("🟦 [Fundamental] SYNC_FUNDAMENTAL=False; skip.")
            return

        codes = [c for c in (normalize_code(x) for x in codes) if c]
        if not codes:
            self.logger.warning("🟦 [Fundamental] empty codes; skip.")
            return

        workers = int(self.cfg.get("FIN_WORKERS", 16) or 16)
        max_inflight = int(self.cfg.get("FIN_MAX_INFLIGHT", workers * 4) or (workers * 4))
        workers = max(1, workers)
        max_inflight = max(workers, max_inflight)

        self.logger.info(f"🟦 [Fundamental] syncing {len(codes)} codes ... workers={workers} inflight={max_inflight}")

        q = deque(codes)
        ok = bad = empty = skipped = 0

        def submit_more(ex, inflight):
            while q and len(inflight) < max_inflight:
                c = q.popleft()
                inflight[ex.submit(self._download_one, c)] = c

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            inflight: Dict[concurrent.futures.Future, str] = {}
            submit_more(ex, inflight)

            with tqdm(total=len(codes), dynamic_ncols=True, desc="Fundamental", unit="code") as pbar:
                while inflight:
                    done, _ = concurrent.futures.wait(inflight.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
                    for fut in done:
                        _ = inflight.pop(fut, None)
                        code, success, msg, _rows = fut.result()

                        if success:
                            ok += 1
                            if msg == "Empty":
                                empty += 1
                            elif msg == "Skipped":
                                skipped += 1
                        else:
                            bad += 1

                        pbar.update(1)
                        pbar.set_postfix(ok=ok, bad=bad, empty=empty, skipped=skipped, last=code)

                    submit_more(ex, inflight)

        self.logger.info(f"🟦 [Fundamental] done. ok={ok}, fail={bad}, empty={empty}, skipped={skipped}")
