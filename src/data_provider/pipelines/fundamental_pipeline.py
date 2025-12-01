from __future__ import annotations

import concurrent.futures
import datetime as _dt
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import akshare as ak
from tqdm import tqdm

from ..clients.ak_client import AkClient
from ..core.config import DPConfig
from ..utils.code import normalize_code
from ..utils.io import atomic_save_parquet
from ..stores.paths import fundamental_dir, fundamental_path


def _coerce_dt(s) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _fetch_pub_date_map(ak_client: AkClient, code: str) -> Optional[pd.DataFrame]:
    """
    Best-effort: map report_end_date -> announcement_date (both as pandas Timestamp).
    AkShare may change column names; handle common variants.
    """
    try:
        df = ak_client.call(ak.stock_financial_abstract, symbol=code)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    end_col = next((c for c in ["截止日期", "报告期", "date", "日期"] if c in df.columns), None)
    pub_col = next((c for c in ["公告日期", "公告时间", "pub_date", "披露日期"] if c in df.columns), None)
    if not end_col or not pub_col:
        return None

    out = df[[end_col, pub_col]].copy()
    out.columns = ["date", "pub_date"]
    out["date"] = _coerce_dt(out["date"])
    out["pub_date"] = _coerce_dt(out["pub_date"])
    out = out.dropna(subset=["date"])
    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return out


def _normalize_financial_indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Financial Analysis Indicator output to our stable schema:
      date, roe, rev_growth, profit_growth, debt_ratio, eps, bps
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "roe", "rev_growth", "profit_growth", "debt_ratio", "eps", "bps"])

    x = df.copy()
    x.columns = [str(c).strip() for c in x.columns]

    # AkShare columns are typically Chinese. Map widely used variants.
    mapping = {
        "日期": "date",
        "报告期": "date",
        "截止日期": "date",

        "加权净资产收益率(%)": "roe",
        "净资产收益率(%)": "roe",
        "ROE(%)": "roe",

        "主营业务收入增长率(%)": "rev_growth",
        "营业总收入同比增长率(%)": "rev_growth",
        "营业收入同比增长率(%)": "rev_growth",

        "净利润增长率(%)": "profit_growth",
        "净利润同比增长率(%)": "profit_growth",
        "归母净利润同比增长率(%)": "profit_growth",

        "资产负债率(%)": "debt_ratio",
        "资产负债率": "debt_ratio",

        "摊薄每股收益(元)": "eps",
        "基本每股收益(元)": "eps",
        "每股收益(元)": "eps",

        "每股净资产_调整后(元)": "bps",
        "每股净资产(元)": "bps",
        "每股净资产(元,调整后)": "bps",
    }

    x = x.rename(columns={k: v for k, v in mapping.items() if k in x.columns})
    if "date" not in x.columns:
        # last resort: first datetime-like col
        for c in x.columns:
            if "date" in c.lower() or "期" in c:
                x = x.rename(columns={c: "date"})
                break

    keep = ["date", "roe", "rev_growth", "profit_growth", "debt_ratio", "eps", "bps"]
    for c in keep:
        if c not in x.columns:
            x[c] = np.nan
    out = x[keep].copy()

    out["date"] = _coerce_dt(out["date"])
    out = out.dropna(subset=["date"]).sort_values("date")
    for c in keep[1:]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float32)
    out = out.drop_duplicates("date", keep="last").reset_index(drop=True)
    return out


class FundamentalPipeline:
    """
    Download & cache quarterly fundamentals per-code.
    Output: {DATA_DIR}/fundamental/{code}.parquet
      columns: date(report_end), pub_date(optional), roe, rev_growth, profit_growth, debt_ratio, eps, bps
    """
    SCHEMA_VER = 1

    def __init__(self, cfg: DPConfig, ak_client: AkClient, logger):
        self.cfg = cfg
        self.ak_client = ak_client
        self.logger = logger
        os.makedirs(fundamental_dir(cfg), exist_ok=True)

    def _should_skip(self, path: str) -> bool:
        days = int(self.cfg.get("FUND_TTL_DAYS", 5) or 5)
        ttl = max(1, days) * 24 * 3600
        return os.path.exists(path) and os.path.getsize(path) > 512 and (time.time() - os.path.getmtime(path)) < ttl

    def _download_one(self, code: str) -> Tuple[str, bool, str]:
        c = normalize_code(code)
        if not c:
            return str(code), True, "BadCode"

        path = fundamental_path(self.cfg, c)
        if self._should_skip(path):
            return c, True, "Skipped"

        start_year = str(self.cfg.get("FUNDAMENTAL_START_YEAR", "2010") or "2010")

        try:
            df = self.ak_client.call(ak.stock_financial_analysis_indicator, symbol=c, start_year=start_year)
            df = _normalize_financial_indicator_frame(df)
            if df.empty:
                atomic_save_parquet(df, path, index=False, compression=str(self.cfg.get("PARQUET_COMPRESSION","zstd") or "zstd"))
                return c, True, "Empty"

            pub = _fetch_pub_date_map(self.ak_client, c)
            if pub is not None and not pub.empty:
                df = df.merge(pub, on="date", how="left")
            else:
                df["pub_date"] = pd.NaT

            df.attrs["schema_ver"] = self.SCHEMA_VER
            atomic_save_parquet(df, path, index=False, compression=str(self.cfg.get("PARQUET_COMPRESSION","zstd") or "zstd"))
            return c, True, "Success"
        except Exception as e:
            return c, False, f"Failed({type(e).__name__})"

    def download(self, codes) -> None:
        if not bool(self.cfg.get("SYNC_FUNDAMENTAL", False)):
            self.logger.info("🟦 [Fundamental] SYNC_FUNDAMENTAL=False; skip.")
            return

        codes = [normalize_code(c) for c in codes]
        codes = [c for c in codes if c]
        if not codes:
            self.logger.warning("🟦 [Fundamental] empty codes; skip.")
            return

        self.logger.info(f"🟦 [Fundamental] syncing {len(codes)} codes ...")
        workers = int(self.cfg.get("FIN_WORKERS", 8) or 8)

        ok = 0
        bad = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(self._download_one, c) for c in codes]
            for f in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc="Fundamental"):
                code, success, msg = f.result()
                ok += int(bool(success))
                bad += int(not bool(success))
        self.logger.info(f"🟦 [Fundamental] done. ok={ok}, fail={bad}")
