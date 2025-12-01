
# -*- coding: utf-8 -*-
"""
dataprovider.py (v23-decoupled-fixed)

Fixes applied vs user v23:
- SplitPolicy is SSOT; DatasetPipeline no longer re-implements split (prevents leakage; consistent end_excl semantics)
- Remove double price normalization in PanelPipeline; introduce read-time schema coercion that preserves extra columns
- Remove half-dead TEST_RATIO logic from DatasetPipeline; handled (optionally) in SplitPolicy
- Minor de-dup: unify universe snapshot retrieval helper
- Clarify DataProvider._price_dir shadow interface signature/semantics
"""

from __future__ import annotations

import concurrent.futures
import datetime as _dt
import glob
import hashlib
import itertools
import json
import logging
import os
import pickle
import random
import re
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import akshare as ak
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm import tqdm

from .alpha_lib import AlphaFactory
from .config import Config
from .vpn_rotator import vpn_rotator

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=getattr(logging, str(getattr(Config, "LOG_LEVEL", "INFO")).upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

# =============================================================================
# Code normalization (deterministic)
# =============================================================================
_CODE6 = re.compile(r"(\d{6})")


def _normalize_code(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    m = _CODE6.search(s)
    return m.group(1) if m else None


def _norm_code_series(s: pd.Series) -> pd.Series:
    return s.map(_normalize_code)


# =============================================================================
# DP Utilities (internal; DataProvider must remain facade-only)
# =============================================================================
class DPUtils:
    VERSION = "v23"

    # -------------------------------
    # Config
    # -------------------------------
    @staticmethod
    def cfg(name: str, default: Any) -> Any:
        return getattr(Config, name, default)

    @staticmethod
    def norm_adjust(adjust: Optional[str]) -> str:
        if adjust is None:
            return "raw"
        a = str(adjust).strip().lower()
        if a in ("", "raw", "none", "null", "nan"):
            return "raw"
        if a == "qfq":
            return "qfq"
        if a == "hfq":
            return "hfq"
        raise ValueError(f"Unknown adjust={adjust!r}. Use raw/qfq/hfq")

    @staticmethod
    def ak_adjust_param(adj_norm: str):
        return "" if adj_norm == "raw" else adj_norm

    @staticmethod
    def yyyymmdd(dt: pd.Timestamp) -> str:
        return pd.to_datetime(dt).strftime("%Y%m%d")

    # -------------------------------
    # Atomic saves
    # -------------------------------
    @staticmethod
    def atomic_save_bytes(data: bytes, path: str) -> None:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        if os.name == "nt" and os.path.exists(path):
            os.remove(path)
        os.replace(tmp, path)

    @staticmethod
    def atomic_save_json(obj: Dict[str, Any], path: str) -> None:
        DPUtils.atomic_save_bytes(json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"), path)

    @staticmethod
    def atomic_save_parquet(df: pd.DataFrame, path: str, index: bool) -> None:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        kwargs = {}
        comp = DPUtils.cfg("PARQUET_COMPRESSION", "zstd")
        if comp:
            kwargs["compression"] = comp
        try:
            df.to_parquet(tmp, engine="pyarrow", index=index, **kwargs)
        except Exception:
            df.to_parquet(tmp, index=index)
        if os.name == "nt" and os.path.exists(path):
            os.remove(path)
        os.replace(tmp, path)

    # -------------------------------
    # Network/proxy
    # -------------------------------
    @staticmethod
    def setup_proxy_env() -> None:
        proxy = str(DPUtils.cfg("PROXY_URL", "") or "").strip()
        if proxy:
            os.environ["http_proxy"] = proxy
            os.environ["https_proxy"] = proxy
        if bool(DPUtils.cfg("USE_VPN_ROTATOR", False)):
            if vpn_rotator:
                logger.info("🛡️ VPN Rotator enabled.")
            else:
                logger.warning("⚠️ USE_VPN_ROTATOR=True but vpn_rotator missing.")

    # -------------------------------
    # Price paths + meta
    # -------------------------------
    @staticmethod
    def meta_path(price_path: str) -> str:
        return price_path + ".meta.json"

    @staticmethod
    def price_dir(adj_norm: str) -> str:
        root = str(DPUtils.cfg("DATA_DIR", "./data"))
        return os.path.join(root, f"price_{adj_norm}")

    @staticmethod
    def legacy_price_dirs(adj_norm: str) -> List[str]:
        root = str(DPUtils.cfg("DATA_DIR", "./data"))
        cands: List[str] = []
        if adj_norm == "qfq":
            cands.append(root)
        cands.append(os.path.join(root, "price"))
        return [d for d in cands if os.path.isdir(d)]

    @staticmethod
    def price_path(adj_norm: str, code: str) -> str:
        d = DPUtils.price_dir(adj_norm)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{code}.parquet")

    @staticmethod
    def price_glob(adj_norm: str) -> List[str]:
        d = DPUtils.price_dir(adj_norm)
        files = sorted(glob.glob(os.path.join(d, "*.parquet")))
        if files:
            return files
        for ld in DPUtils.legacy_price_dirs(adj_norm):
            files = sorted(glob.glob(os.path.join(ld, "*.parquet")))
            if files:
                logger.warning(f"[v23] Using legacy price dir for adjust={adj_norm}: {ld}")
                return files
        return []

    # -------------------------------
    # AkShare wrappers
    # -------------------------------
    @staticmethod
    def ak_hist(code: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp, adjust: str) -> pd.DataFrame:
        adj_norm = DPUtils.norm_adjust(adjust)
        start = DPUtils.yyyymmdd(start_dt)
        end = DPUtils.yyyymmdd(end_dt)
        tries = ["", None] if adj_norm == "raw" else [adj_norm]
        last_e: Optional[Exception] = None
        for t in tries:
            try:
                return _AkClient.call(
                    ak.stock_zh_a_hist,
                    symbol=code,
                    period="daily",
                    start_date=start,
                    end_date=end,
                    adjust=t,
                )
            except Exception as e:
                last_e = e
        raise last_e  # type: ignore

    # -------------------------------
    # Price normalization (write-time, strict)
    # -------------------------------
    @staticmethod
    def normalize_price_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        STRICT normalization for AkShare raw outputs.
        - Renames known columns to canonical schema
        - Keeps only canonical OHLCV(+amount+turnover) columns
        - Casts numeric columns to float32
        """
        rename_map = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover",
        }
        df = df.rename(columns=rename_map).copy()
        if "date" not in df.columns:
            df = df.reset_index().rename(columns={"index": "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")

        keep = [c for c in ["date", "open", "high", "low", "close", "volume", "amount", "turnover"] if c in df.columns]
        df = df[keep].copy()

        for c in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        for c in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
            if c in df.columns:
                df[c] = df[c].astype(np.float32, copy=False)

        if "close" in df.columns:
            df = df[df["close"] > 0].copy()
        return df

    # -------------------------------
    # Price schema coercion (read-time, non-destructive)
    # -------------------------------
    @staticmethod
    def coerce_price_schema_keep_extras(df: pd.DataFrame) -> pd.DataFrame:
        """
        NON-destructive schema coercion for cached parquet (or legacy caches).
        - Renames known columns to canonical schema
        - Ensures 'date' exists and is datetime
        - Coerces numeric columns if present
        - PRESERVES extra columns (does NOT drop)
        """
        if df is None or df.empty:
            return df

        rename_map = {
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover",
        }
        for k, v in rename_map.items():
            if k in df.columns and v not in df.columns:
                df = df.rename(columns={k: v})

        if "date" not in df.columns:
            # try common fallbacks
            if df.index.name in ("date", "日期"):
                df = df.reset_index()
                if "日期" in df.columns and "date" not in df.columns:
                    df = df.rename(columns={"日期": "date"})
            else:
                df = df.reset_index().rename(columns={"index": "date"})

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
        else:
            return df.iloc[0:0].copy()

        # Coerce canonical numeric cols if present
        for c in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32, copy=False)

        # Basic sanity: drop non-positive close if present (legacy caches)
        if "close" in df.columns:
            df = df[df["close"].fillna(0.0) > 0].copy()

        sort_cols = [c for c in ["code", "date"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols)
        return df

    # -------------------------------
    # Meta-guarded read
    # -------------------------------
    @staticmethod
    def read_price(price_file: str, expected_adj: Optional[str] = None) -> Optional[pd.DataFrame]:
        try:
            code = _normalize_code(os.path.basename(price_file).replace(".parquet", ""))
            if not code:
                return None

            # Meta guard early (no need to read full df if strict mismatch)
            if expected_adj is not None:
                exp = DPUtils.norm_adjust(expected_adj)
                meta_path = DPUtils.meta_path(price_file)
                strict = bool(DPUtils.cfg("STRICT_PRICE_META", True))
                allow_legacy = bool(DPUtils.cfg("ALLOW_LEGACY_PRICE_CACHE", False))
                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    stored = DPUtils.norm_adjust(meta.get("stored_adjust"))
                    if stored != exp:
                        logger.error(f"[CORRUPT] {price_file} stored={stored} expected={exp}. Skipping.")
                        return None
                else:
                    # If strict and not allowing legacy, reject non-raw caches without meta
                    if strict and (not allow_legacy) and exp != "raw":
                        logger.error(
                            f"[NO_META] {os.path.basename(price_file)} rejected. Expected {exp} but no meta found. "
                            f"Hint: set ALLOW_LEGACY_PRICE_CACHE=True for old data."
                        )
                        return None

            df = pd.read_parquet(price_file)
            if df is None or df.empty:
                return None

            # ensure code column exists and normalized
            if "code" not in df.columns:
                df["code"] = code
            df["code"] = _norm_code_series(df["code"])
            df = df.dropna(subset=["code"])
            df["code"] = df["code"].astype(str)

            # Coerce schema (non-destructive, preserves extras)
            df = DPUtils.coerce_price_schema_keep_extras(df)

            if "date" not in df.columns:
                return None

            df = df.dropna(subset=["date"]).sort_values(["code", "date"]).reset_index(drop=True)
            return df
        except Exception as e:
            logger.debug(f"read_price failed: {price_file} err={e}")
            return None

    # -------------------------------
    # Freshness skip
    # -------------------------------
    @staticmethod
    def should_skip_fetch(last_dt: Optional[pd.Timestamp], updated_at: Optional[pd.Timestamp], target_dt: pd.Timestamp) -> bool:
        if not bool(DPUtils.cfg("SKIP_IF_FRESH", True)):
            return False
        if last_dt is None:
            return False
        lag_days = int(DPUtils.cfg("FRESH_LAG_DAYS", 0))
        if pd.Timestamp(last_dt).normalize() < (pd.Timestamp(target_dt).normalize() - pd.Timedelta(days=lag_days)):
            return False
        if updated_at is not None and pd.Timestamp(updated_at).normalize() >= pd.Timestamp(_dt.date.today()).normalize():
            return True
        return True

    # -------------------------------
    # Panel cache keys
    # -------------------------------
    @staticmethod
    def config_fingerprint() -> str:
        keys = DPUtils.cfg(
            "PANEL_CACHE_FINGERPRINT_KEYS",
            [
                "CONTEXT_LEN",
                "PRED_LEN",
                "STRIDE",
                "PRICE_SCHEMA_VER",
                "FEATURE_PREFIXES",
                "ALPHA_BACKEND",
                "USE_FUNDAMENTAL",
                "USE_CROSS_SECTIONAL",
                "UNIVERSE_MIN_PRICE",
                "UNIVERSE_MIN_LIST_DAYS",
                "MIN_LIST_DAYS",
                "MIN_DOLLAR_VOL_FOR_TRADE",
                "MIN_PRICE",
                "LIMIT_RATE_MAIN",
                "LIMIT_RATE_ST",
                "LIMIT_RATE_GEM",
                "LIMIT_RATE_STAR",
                "LIMIT_RATE_BSE",
                "LIMIT_RATE_BSHARE",
                "LIMIT_EPS",
                "DATASET_BACKEND",
                "INCLUDE_BSE",
                "INCLUDE_BSHARE",
                "GATE_TARGET_WITH_ENTRY",
                "GATE_TARGET_WITH_EXIT",
                "ALIGN_TO_CALENDAR",
                "ENTRY_PRICE_MODE",
                "DELIST_GAP_DAYS",
                "TRAIN_RATIO",
                "VAL_RATIO",
                "TEST_RATIO",
            ],
        )
        payload: Dict[str, Any] = {"dp_ver": DPUtils.VERSION}
        for k in keys:
            payload[k] = getattr(Config, k, None)
        payload["alpha_factory"] = getattr(AlphaFactory, "VERSION", None) or getattr(AlphaFactory, "__name__", "AlphaFactory")
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:12]

    @staticmethod
    def cache_path(mode: str, adj_norm: str) -> str:
        today = _dt.date.today().strftime("%Y%m%d")
        fp = DPUtils.config_fingerprint()
        out_dir = str(DPUtils.cfg("OUTPUT_DIR", "./output"))
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, f"panel_cache_{mode}_{adj_norm}_{today}_{DPUtils.VERSION}_{fp}.pkl")

    # -------------------------------
    # Fundamentals PIT merge
    # -------------------------------
    @staticmethod
    def fund_path(code: str) -> str:
        return os.path.join(str(DPUtils.cfg("DATA_DIR", "./data")), "fundamental", f"{code}.parquet")

    @staticmethod
    def merge_fundamental_pit(price_df: pd.DataFrame) -> pd.DataFrame:
        if not bool(DPUtils.cfg("USE_FUNDAMENTAL", False)):
            return price_df
        code = _normalize_code(price_df["code"].iloc[0]) or str(price_df["code"].iloc[0])
        fp = DPUtils.fund_path(code)
        if not os.path.exists(fp) or os.path.getsize(fp) < 512:
            return price_df
        try:
            f = pd.read_parquet(fp)
        except Exception:
            return price_df
        if f is None or f.empty:
            return price_df

        if "date" not in f.columns:
            f = f.reset_index().rename(columns={"index": "date"})
        f["date"] = pd.to_datetime(f["date"], errors="coerce")
        f = f.dropna(subset=["date"]).sort_values("date")

        eff = None
        eff_col = None
        for c in ["pub_date", "公告日期", "披露日期", "发布日期", "公告日"]:
            if c in f.columns:
                eff = pd.to_datetime(f[c], errors="coerce")
                eff_col = c
                break

        if eff is not None:
            f = f.copy()
            f["_eff_date"] = eff
            f = f.dropna(subset=["_eff_date"]).sort_values("_eff_date")
            drop_cols = [c for c in ["date", eff_col] if c in f.columns]
            right = f.drop(columns=drop_cols).rename(columns={"_eff_date": "date"})
        else:
            right = f

        num_cols: List[str] = []
        for c in right.columns:
            if c == "date":
                continue
            if pd.api.types.is_numeric_dtype(right[c]):
                num_cols.append(c)
            else:
                s = pd.to_numeric(right[c], errors="coerce")
                if s.notna().mean() > 0.9:
                    right[c] = s
                    num_cols.append(c)
        if not num_cols:
            return price_df

        left = price_df.sort_values("date").copy()
        right2 = right[["date"] + num_cols].sort_values("date")
        return pd.merge_asof(left, right2, on="date", direction="backward")

    # -------------------------------
    # Meta attach + masks + target closure helpers
    # -------------------------------
    @staticmethod
    def attach_code_meta(
        df: pd.DataFrame,
        meta_map: Dict[str, Dict[str, Any]],
        sec_map: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        code = _normalize_code(df["code"].iloc[0]) or str(df["code"].iloc[0])
        m = meta_map.get(code, {})
        s = sec_map.get(code, {})

        name = str(m.get("name", s.get("name", "")))
        board = str(m.get("board", s.get("board", _CodeMeta.infer_board(code))))
        is_st = bool(m.get("is_st", _CodeMeta.is_st(name)))
        limit_rate = float(m.get("limit_rate", _CodeMeta.limit_rate(board, is_st)))

        out = df.copy()
        out["code"] = _norm_code_series(out["code"])
        out = out.dropna(subset=["code"]).copy()
        out["code"] = out["code"].astype(str)
        out["board"] = board
        out["is_st"] = bool(is_st)
        out["limit_rate"] = float(limit_rate)
        out["name"] = name

        out["first_date_hint"] = s.get("first_date", pd.NaT)
        out["last_date_hint"] = s.get("last_date", pd.NaT)
        out["is_delisted_guess"] = bool(s.get("is_delisted_guess", False))
        return out

    @staticmethod
    def align_to_calendar(df: pd.DataFrame) -> pd.DataFrame:
        if not bool(DPUtils.cfg("ALIGN_TO_CALENDAR", False)):
            return df
        cal = _CalendarStore.get_trade_dates()
        df = df.sort_values(["code", "date"]).copy()
        out_parts: List[pd.DataFrame] = []
        for code, g in df.groupby("code", sort=False):
            g = g.copy()
            dmin, dmax = pd.to_datetime(g["date"]).min(), pd.to_datetime(g["date"]).max()
            idx = cal[(cal >= dmin) & (cal <= dmax)]
            base = g.set_index("date")
            base.index = pd.to_datetime(base.index, errors="coerce")
            base = base[~base.index.isna()]
            base = base[~base.index.duplicated(keep="last")].sort_index()
            base = base.reindex(idx)
            base["code"] = code
            base = base.reset_index().rename(columns={"index": "date"})
            out_parts.append(base)
        out = pd.concat(out_parts, ignore_index=True)
        return out

    @staticmethod
    def add_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["code", "date"]).copy()
        if "amount" in df.columns:
            dv = df["amount"].fillna(0.0)
        else:
            dv = df["close"].fillna(0.0) * df.get("volume", 0.0).fillna(0.0)
        df["dollar_vol"] = dv.astype(np.float32)

        g = df.groupby("code", sort=False)
        df["adv20"] = g["dollar_vol"].transform(lambda x: x.rolling(20, min_periods=1).mean()).astype(np.float32)
        return df

    @staticmethod
    def add_trade_masks(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["code", "date"]).copy()
        g = df.groupby("code", sort=False)

        list_days = g.cumcount() + 1
        prev_close = g["close"].shift(1)
        pct_chg = df["close"] / (prev_close + 1e-9) - 1.0

        min_list_days = int(DPUtils.cfg("MIN_LIST_DAYS", 60))
        min_dv = float(DPUtils.cfg("MIN_DOLLAR_VOL_FOR_TRADE", 1e6))
        min_price = float(DPUtils.cfg("MIN_PRICE", 1.0))

        lr = pd.to_numeric(df.get("limit_rate", np.nan), errors="coerce").fillna(float(DPUtils.cfg("LIMIT_RATE_MAIN", 0.10)))
        eps = float(DPUtils.cfg("LIMIT_EPS", 0.002))

        cond_new = list_days < min_list_days
        cond_suspended = df.get("volume", 0.0).fillna(0.0) <= 0
        cond_illiquid = df.get("dollar_vol", 0.0).fillna(0.0) < min_dv
        cond_tiny_price = df["close"].fillna(0.0) < min_price

        base_noise = cond_new | cond_suspended | cond_illiquid | cond_tiny_price
        df["tradable_mask"] = np.where(base_noise, np.nan, 1.0).astype(np.float32)

        oneword = pd.Series(False, index=df.index)
        if {"high", "low"}.issubset(df.columns):
            oneword = np.isclose(df["high"].astype(float), df["low"].astype(float), rtol=0.0, atol=1e-6)

        limit_up = (pct_chg >= (lr - 1e-4)) | (pct_chg >= (lr - eps))
        limit_dn = (pct_chg <= -(lr - 1e-4)) | (pct_chg <= -(lr - eps))

        cond_buy_block = oneword & limit_up
        cond_sell_block = oneword & limit_dn

        df["buyable_mask"] = df["tradable_mask"].copy()
        df.loc[cond_buy_block, "buyable_mask"] = np.nan

        df["sellable_mask"] = df["tradable_mask"].copy()
        df.loc[cond_sell_block, "sellable_mask"] = np.nan

        df.loc[df.get("open").isna() | (df.get("open") <= 0), "buyable_mask"] = np.nan
        df.loc[df.get("close").isna() | (df.get("close") <= 0), "sellable_mask"] = np.nan

        return df

    @staticmethod
    def tag_universe(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["code", "date"]).copy()
        df["list_days_count"] = df.groupby("code")["date"].cumcount() + 1
        cond_vol = df.get("volume", 0.0).fillna(0.0) > 0
        cond_price = df["close"].fillna(0.0) >= float(DPUtils.cfg("UNIVERSE_MIN_PRICE", 2.0))
        cond_list = df["list_days_count"] > int(DPUtils.cfg("UNIVERSE_MIN_LIST_DAYS", 60))
        df["is_universe"] = (cond_vol & cond_price & cond_list).astype(bool)
        df.drop(columns=["list_days_count"], inplace=True)
        return df

    # -------------------------------
    # De-dup helper: universe snapshot retrieval
    # -------------------------------
    @staticmethod
    def get_universe_snapshot(asof: pd.Timestamp, force_refresh: bool = False) -> Tuple[pd.DataFrame, List[str]]:
        snap = _UniverseStore.get_snapshot(asof, force_refresh=force_refresh)
        codes = snap["code"].astype(str).tolist() if snap is not None and not snap.empty else []
        return snap, codes


# =============================================================================
# Manifest
# =============================================================================
@dataclass(frozen=True)
class ManifestRow:
    code: str
    adjust: str
    last_date: Optional[pd.Timestamp]
    rows: int
    updated_at: pd.Timestamp
    schema_ver: int


class ManifestStore:
    def __init__(self, path: str):
        self.path = path
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.path) or os.path.getsize(self.path) < 512:
            return pd.DataFrame(columns=["code", "adjust", "last_date", "rows", "updated_at", "schema_ver"])
        try:
            df = pd.read_parquet(self.path)
        except Exception:
            return pd.DataFrame(columns=["code", "adjust", "last_date", "rows", "updated_at", "schema_ver"])
        if "last_date" in df.columns:
            df["last_date"] = pd.to_datetime(df["last_date"], errors="coerce")
        if "updated_at" in df.columns:
            df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")
        return df

    def save(self, df: pd.DataFrame) -> None:
        df = df.sort_values(["updated_at"]).drop_duplicates(["code", "adjust"], keep="last")
        DPUtils.atomic_save_parquet(df, self.path, index=False)

    def upsert_many(self, rows: List[ManifestRow]) -> None:
        if not rows:
            return
        cur = self.load()
        add = pd.DataFrame(
            [
                {
                    "code": r.code,
                    "adjust": r.adjust,
                    "last_date": r.last_date,
                    "rows": int(r.rows),
                    "updated_at": r.updated_at,
                    "schema_ver": int(r.schema_ver),
                }
                for r in rows
            ]
        )
        self.save(pd.concat([cur, add], ignore_index=True))

    @staticmethod
    def to_map(df: pd.DataFrame) -> Dict[Tuple[str, str], Tuple[Optional[pd.Timestamp], int, Optional[pd.Timestamp]]]:
        out: Dict[Tuple[str, str], Tuple[Optional[pd.Timestamp], int, Optional[pd.Timestamp]]] = {}
        if df is None or df.empty:
            return out
        for rr in df.itertuples(index=False):
            c = _normalize_code(getattr(rr, "code"))
            if not c:
                continue
            a = DPUtils.norm_adjust(getattr(rr, "adjust"))
            ld = getattr(rr, "last_date", None)
            rows = int(getattr(rr, "rows", 0) or 0)
            ua = getattr(rr, "updated_at", None)
            out[(c, a)] = (
                pd.to_datetime(ld, errors="coerce") if ld is not None else None,
                rows,
                pd.to_datetime(ua, errors="coerce") if ua is not None else None,
            )
        return out


# =============================================================================
# Network: retry + coordinated VPN rotation
# =============================================================================
class _VPNCoordinator:
    _lock = threading.Lock()
    _last_rotate_ts = 0.0

    @classmethod
    def maybe_rotate(cls) -> bool:
        if not bool(DPUtils.cfg("USE_VPN_ROTATOR", False)):
            return False
        if not vpn_rotator:
            return False
        cooldown = int(DPUtils.cfg("VPN_ROTATE_COOLDOWN_SEC", 60))
        now = time.time()
        with cls._lock:
            if now - cls._last_rotate_ts < cooldown:
                return False
            try:
                logger.warning("🔄 VPN rotate triggered (coordinated).")
                vpn_rotator()
                cls._last_rotate_ts = time.time()
                time.sleep(float(DPUtils.cfg("VPN_POST_ROTATE_SLEEP_SEC", 3.0)))
                return True
            except Exception as e:
                logger.error(f"❌ VPN rotation failed: {e}")
                return False


class _AkErrorClassifier:
    _compiled = re.compile(
        "|".join(
            [
                r"timeout",
                r"timed out",
                r"connection",
                r"connection reset",
                r"remote end closed",
                r"proxy",
                r"tunnel",
                r"ssl",
                r"certificate",
                r"max retries exceeded",
                r"\b429\b",
                r"too many requests",
                r"\b403\b",
                r"forbidden",
                r"blocked",
                r"captcha",
                r"频繁",
                r"访问受限",
                r"拒绝",
            ]
        ),
        re.IGNORECASE,
    )

    @classmethod
    def should_rotate(cls, e: Exception) -> bool:
        return bool(cls._compiled.search(f"{type(e).__name__}: {e}"))


class _AkClient:
    @staticmethod
    def call(fn, *args, **kwargs):
        retries = int(DPUtils.cfg("AK_RETRIES", 10))
        base_sleep = float(DPUtils.cfg("AK_RETRY_BASE_SLEEP", 1.0))
        max_sleep = float(DPUtils.cfg("AK_RETRY_MAX_SLEEP", 15.0))

        last_e: Optional[Exception] = None
        for i in range(retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_e = e
                logger.warning(f"⚠️ [AkRetry {i + 1}/{retries}] {fn.__name__} failed: {e}")
                if _AkErrorClassifier.should_rotate(e):
                    _VPNCoordinator.maybe_rotate()
                sleep_time = base_sleep * (2**i) * (0.8 + random.random() * 0.4)
                time.sleep(min(sleep_time, max_sleep))
        logger.critical(f"❌ All {retries} retries failed for {fn.__name__}. Last={last_e}")
        raise last_e  # type: ignore


# =============================================================================
# Calendar
# =============================================================================
class _CalendarStore:
    @staticmethod
    def _path() -> str:
        root = str(DPUtils.cfg("DATA_DIR", "./data"))
        os.makedirs(os.path.join(root, "calendar"), exist_ok=True)
        sym = str(DPUtils.cfg("MARKET_INDEX_SYMBOL", "sh000001"))
        return os.path.join(root, "calendar", f"index_{sym}.parquet")

    @staticmethod
    def _is_fresh(path: str, ttl: int) -> bool:
        return os.path.exists(path) and os.path.getsize(path) >= 512 and (time.time() - os.path.getmtime(path)) < ttl

    @classmethod
    def get_trade_dates(cls) -> pd.DatetimeIndex:
        path = cls._path()
        ttl = int(DPUtils.cfg("CALENDAR_TTL_SEC", 7 * 24 * 3600))
        if cls._is_fresh(path, ttl):
            try:
                df = pd.read_parquet(path)
                d = pd.to_datetime(df["date"], errors="coerce").dropna()
                return pd.DatetimeIndex(sorted(d.unique()))
            except Exception:
                pass

        sym = str(DPUtils.cfg("MARKET_INDEX_SYMBOL", "sh000001"))
        df = _AkClient.call(ak.stock_zh_index_daily, symbol=sym)
        if df is None or df.empty:
            raise RuntimeError("Index daily is empty, cannot build calendar.")
        d = pd.to_datetime(df["date"], errors="coerce").dropna().sort_values()
        DPUtils.atomic_save_parquet(pd.DataFrame({"date": d}), path, index=False)
        return pd.DatetimeIndex(sorted(d.unique()))

    @classmethod
    def latest_trade_date(cls, today: Optional[pd.Timestamp] = None) -> pd.Timestamp:
        if today is None:
            today = pd.Timestamp(_dt.date.today())
        try:
            dates = cls.get_trade_dates()
            eligible = dates[dates <= pd.Timestamp(today).normalize()]
            if len(eligible):
                return pd.Timestamp(eligible[-1])
        except Exception as e:
            logger.warning(f"[Calendar] fallback due to error: {e}")

        d = pd.Timestamp(today).normalize()
        while d.weekday() >= 5:
            d -= pd.Timedelta(days=1)
        return d


# =============================================================================
# Universe + Security Master (reduce survivorship bias)
# =============================================================================
class _CodeMeta:
    @staticmethod
    def infer_board(code: str) -> str:
        c = str(code).strip()
        if c.startswith(("300", "301")):
            return "GEM"
        if c.startswith(("688", "689")):
            return "STAR"
        if c.startswith(("8", "4")):
            return "BSE"
        if c.startswith(("200", "900")):
            return "BSHARE"
        return "MAIN"

    @staticmethod
    def is_st(name: Optional[str]) -> bool:
        if not name:
            return False
        n = str(name).upper()
        return "ST" in n

    @staticmethod
    def limit_rate(board: str, is_st: bool) -> float:
        main = float(DPUtils.cfg("LIMIT_RATE_MAIN", 0.10))
        st = float(DPUtils.cfg("LIMIT_RATE_ST", 0.05))
        gem = float(DPUtils.cfg("LIMIT_RATE_GEM", 0.20))
        star = float(DPUtils.cfg("LIMIT_RATE_STAR", 0.20))
        bse = float(DPUtils.cfg("LIMIT_RATE_BSE", 0.30))
        bshare = float(DPUtils.cfg("LIMIT_RATE_BSHARE", 0.10))
        if board == "BSE":
            return bse
        if board == "GEM":
            return gem
        if board == "STAR":
            return star
        if board == "BSHARE":
            return bshare
        return st if is_st else main


class _UniverseStore:
    @staticmethod
    def _dir() -> str:
        root = str(DPUtils.cfg("DATA_DIR", "./data"))
        d = os.path.join(root, "universe")
        os.makedirs(d, exist_ok=True)
        return d

    @classmethod
    def _snapshot_path(cls, date: pd.Timestamp) -> str:
        return os.path.join(cls._dir(), f"universe_{pd.Timestamp(date).strftime('%Y%m%d')}.parquet")

    @classmethod
    def _master_path(cls) -> str:
        return os.path.join(cls._dir(), "code_master.parquet")

    @staticmethod
    def _is_fresh(path: str, ttl: int) -> bool:
        return os.path.exists(path) and os.path.getsize(path) >= 512 and (time.time() - os.path.getmtime(path)) < ttl

    @staticmethod
    def _extract_code_name(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["code", "name"])
        code_col = None
        name_col = None
        for c in ["代码", "证券代码", "code", "symbol"]:
            if c in df.columns:
                code_col = c
                break
        for c in ["名称", "证券简称", "name", "short_name"]:
            if c in df.columns:
                name_col = c
                break
        if code_col is None:
            return pd.DataFrame(columns=["code", "name"])
        out = pd.DataFrame({"code": df[code_col]})
        out["name"] = df[name_col].astype(str) if name_col is not None else ""
        out["code"] = _norm_code_series(out["code"])
        out = out.dropna(subset=["code"]).drop_duplicates("code").reset_index(drop=True)
        out["code"] = out["code"].astype(str)
        out["name"] = out["name"].fillna("").astype(str)
        return out

    @staticmethod
    def _try_all_code_lists() -> pd.DataFrame:
        cands = [
            "stock_info_a_code_name",
            "stock_info_sz_name_code",
            "stock_info_sh_name_code",
        ]
        for fn_name in cands:
            fn = getattr(ak, fn_name, None)
            if callable(fn):
                try:
                    df = _AkClient.call(fn)
                    out = _UniverseStore._extract_code_name(df)
                    if not out.empty:
                        out["source"] = fn_name
                        return out
                except Exception:
                    continue
        return pd.DataFrame(columns=["code", "name", "source"])

    @staticmethod
    def _cached_price_codes() -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for adj in ["raw", "qfq", "hfq"]:
            for p in DPUtils.price_glob(adj):
                code = _normalize_code(os.path.basename(p).replace(".parquet", ""))
                if not code:
                    continue
                rows.append({"code": code, "name": "", "source": f"price_cache_{adj}"})
        if not rows:
            return pd.DataFrame(columns=["code", "name", "source"])
        return pd.DataFrame(rows).drop_duplicates("code").reset_index(drop=True)

    @staticmethod
    def _extra_codes_file() -> pd.DataFrame:
        path = str(DPUtils.cfg("UNIVERSE_EXTRA_CODES_FILE", "") or "").strip()
        if not path:
            return pd.DataFrame(columns=["code", "name", "source"])
        if not os.path.exists(path):
            logger.warning(f"[Universe] UNIVERSE_EXTRA_CODES_FILE not found: {path}")
            return pd.DataFrame(columns=["code", "name", "source"])
        try:
            if path.lower().endswith(".csv"):
                df = pd.read_csv(path)
                out = _UniverseStore._extract_code_name(df)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    codes = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
                out = pd.DataFrame({"code": _norm_code_series(pd.Series(codes)), "name": ""})
                out = out.dropna(subset=["code"])
                out["code"] = out["code"].astype(str)
            if out.empty:
                return pd.DataFrame(columns=["code", "name", "source"])
            out["source"] = "extra_codes_file"
            return out
        except Exception as e:
            logger.warning(f"[Universe] read extra codes failed: {e}")
            return pd.DataFrame(columns=["code", "name", "source"])

    @staticmethod
    def _apply_market_filters(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        include_bse = bool(DPUtils.cfg("INCLUDE_BSE", False))
        include_bshare = bool(DPUtils.cfg("INCLUDE_BSHARE", False))
        board = df["code"].map(_CodeMeta.infer_board)

        keep = pd.Series(True, index=df.index)
        if not include_bse:
            keep &= board != "BSE"
        if not include_bshare:
            keep &= board != "BSHARE"

        allow_prefixes = DPUtils.cfg("UNIVERSE_ALLOW_PREFIXES", None)
        if allow_prefixes:
            allow_prefixes = [str(x) for x in allow_prefixes]
            ok = pd.Series(False, index=df.index)
            for p in allow_prefixes:
                ok |= df["code"].str.startswith(p)
            keep &= ok

        return df.loc[keep].reset_index(drop=True)

    @classmethod
    def get_snapshot(cls, date: pd.Timestamp, force_refresh: bool = False) -> pd.DataFrame:
        date = pd.Timestamp(date).normalize()
        path = cls._snapshot_path(date)
        ttl = int(DPUtils.cfg("UNIVERSE_TTL_SEC", 24 * 3600))
        if (not force_refresh) and cls._is_fresh(path, ttl):
            try:
                return pd.read_parquet(path)
            except Exception:
                pass

        spot = _AkClient.call(ak.stock_zh_a_spot_em)
        spot2 = cls._extract_code_name(spot)
        if not spot2.empty:
            spot2["source"] = "spot_em"
        else:
            spot2 = pd.DataFrame(columns=["code", "name", "source"])

        all_codes = cls._try_all_code_lists()
        cached = cls._cached_price_codes()
        extra = cls._extra_codes_file()

        uni = pd.concat([spot2, all_codes, cached, extra], ignore_index=True)
        if uni.empty:
            return uni

        uni["code"] = _norm_code_series(uni["code"])
        uni = uni.dropna(subset=["code"])
        uni["code"] = uni["code"].astype(str)
        uni["name"] = uni.get("name", "").fillna("").astype(str)

        uni["_name_ok"] = (uni["name"].str.len() > 0).astype(int)
        uni = (
            uni.sort_values(["code", "_name_ok", "source"])
            .drop_duplicates("code", keep="last")
            .drop(columns=["_name_ok"])
        )

        uni = cls._apply_market_filters(uni)

        uni["board"] = uni["code"].map(_CodeMeta.infer_board)
        uni["is_st"] = uni["name"].map(_CodeMeta.is_st)
        uni["limit_rate"] = [
            float(_CodeMeta.limit_rate(b, bool(s))) for b, s in zip(uni["board"].tolist(), uni["is_st"].tolist())
        ]
        uni["asof"] = date

        try:
            DPUtils.atomic_save_parquet(uni.reset_index(drop=True), path, index=False)
        except Exception as e:
            logger.warning(f"[Universe] persist snapshot failed: {e}")

        cls._upsert_master(uni)
        return uni.reset_index(drop=True)

    @classmethod
    def _upsert_master(cls, snapshot: pd.DataFrame) -> None:
        mp = cls._master_path()
        cur = None
        if os.path.exists(mp) and os.path.getsize(mp) >= 512:
            try:
                cur = pd.read_parquet(mp)
            except Exception:
                cur = None
        add = snapshot.copy()
        out = add if cur is None or cur.empty else pd.concat([cur, add], ignore_index=True)
        out["asof"] = pd.to_datetime(out["asof"], errors="coerce")
        out = out.sort_values(["asof"]).drop_duplicates(["code"], keep="last")
        try:
            DPUtils.atomic_save_parquet(out.reset_index(drop=True), mp, index=False)
        except Exception as e:
            logger.warning(f"[Universe] persist master failed: {e}")

    @classmethod
    def meta_map_for_asof(cls, date: pd.Timestamp) -> Dict[str, Dict[str, Any]]:
        date = pd.Timestamp(date).normalize()
        sp = cls._snapshot_path(date)
        df = None
        if os.path.exists(sp) and os.path.getsize(sp) >= 512:
            try:
                df = pd.read_parquet(sp)
            except Exception:
                df = None
        if df is None or df.empty:
            mp = cls._master_path()
            if os.path.exists(mp) and os.path.getsize(mp) >= 512:
                try:
                    df = pd.read_parquet(mp)
                    logger.warning(f"[Meta] snapshot missing for {date.date()}, fallback to master (ST/limits may bias).")
                except Exception:
                    df = None

        if df is None or df.empty:
            return {}

        df = df.drop_duplicates("code", keep="last")
        out: Dict[str, Dict[str, Any]] = {}
        for r in df.itertuples(index=False):
            code = _normalize_code(getattr(r, "code"))
            if not code:
                continue
            out[code] = {
                "name": getattr(r, "name", ""),
                "board": getattr(r, "board", _CodeMeta.infer_board(code)),
                "is_st": bool(getattr(r, "is_st", False)),
                "limit_rate": float(getattr(r, "limit_rate", _CodeMeta.limit_rate(_CodeMeta.infer_board(code), False))),
            }
        return out

    @classmethod
    def get_codes(cls, date: pd.Timestamp) -> List[str]:
        snap = cls.get_snapshot(date)
        if snap is None or snap.empty:
            return []
        return snap["code"].astype(str).tolist()


class _SecurityMasterStore:
    """Security master inferred from local history; optional PIT events can be merged later."""
    @staticmethod
    def _dir() -> str:
        root = str(DPUtils.cfg("DATA_DIR", "./data"))
        d = os.path.join(root, "security_master")
        os.makedirs(d, exist_ok=True)
        return d

    @classmethod
    def path(cls) -> str:
        return os.path.join(cls._dir(), "security_master.parquet")

    @staticmethod
    def _is_fresh(path: str, ttl: int) -> bool:
        return os.path.exists(path) and os.path.getsize(path) >= 512 and (time.time() - os.path.getmtime(path)) < ttl

    @classmethod
    def load(cls) -> pd.DataFrame:
        p = cls.path()
        if not os.path.exists(p) or os.path.getsize(p) < 512:
            return pd.DataFrame(columns=["code", "board", "name", "first_date", "last_date", "is_delisted_guess"])
        try:
            df = pd.read_parquet(p)
            df["first_date"] = pd.to_datetime(df.get("first_date"), errors="coerce")
            df["last_date"] = pd.to_datetime(df.get("last_date"), errors="coerce")
            return df
        except Exception:
            return pd.DataFrame(columns=["code", "board", "name", "first_date", "last_date", "is_delisted_guess"])

    @classmethod
    def build_or_update(
        cls,
        codes: List[str],
        adj_norm_for_scan: str,
        target_dt: pd.Timestamp,
        meta_map: Dict[str, Dict[str, Any]],
        force: bool = False,
    ) -> pd.DataFrame:
        ttl = int(DPUtils.cfg("SECMASTER_TTL_SEC", 7 * 24 * 3600))
        p = cls.path()
        if (not force) and cls._is_fresh(p, ttl):
            return cls.load()

        scan_price = bool(DPUtils.cfg("SECMASTER_SCAN_PRICE", True))
        max_scan = int(DPUtils.cfg("SECMASTER_MAX_SCAN_FILES", 0))  # 0 = no cap

        rows: List[Dict[str, Any]] = []
        if scan_price:
            files = DPUtils.price_glob(adj_norm_for_scan)
            if max_scan and len(files) > max_scan:
                files = files[:max_scan]
            for pf in tqdm(files, desc="SecMasterScan"):
                code = _normalize_code(os.path.basename(pf).replace(".parquet", ""))
                if not code:
                    continue
                try:
                    d = pd.read_parquet(pf, columns=["date"])
                    dd = pd.to_datetime(d["date"], errors="coerce").dropna()
                    if dd.empty:
                        continue
                    first_date = dd.min()
                    last_date = dd.max()
                except Exception:
                    continue

                mm = meta_map.get(code, {})
                name = str(mm.get("name", ""))
                board = str(mm.get("board", _CodeMeta.infer_board(code)))
                delist_gap = int(DPUtils.cfg("DELIST_GAP_DAYS", 90))
                is_delisted_guess = bool(last_date < (pd.Timestamp(target_dt) - pd.Timedelta(days=delist_gap)))

                rows.append(
                    {
                        "code": code,
                        "board": board,
                        "name": name,
                        "first_date": first_date,
                        "last_date": last_date,
                        "is_delisted_guess": is_delisted_guess,
                    }
                )

        seen = {r["code"] for r in rows}
        for c in codes:
            cc = _normalize_code(c)
            if not cc:
                continue
            if cc in seen:
                continue
            mm = meta_map.get(cc, {})
            rows.append(
                {
                    "code": cc,
                    "board": str(mm.get("board", _CodeMeta.infer_board(cc))),
                    "name": str(mm.get("name", "")),
                    "first_date": pd.NaT,
                    "last_date": pd.NaT,
                    "is_delisted_guess": False,
                }
            )

        df = pd.DataFrame(rows).drop_duplicates("code", keep="last")
        try:
            DPUtils.atomic_save_parquet(df.reset_index(drop=True), p, index=False)
        except Exception as e:
            logger.warning(f"[SecMaster] persist failed: {e}")
        return df.reset_index(drop=True)

    @classmethod
    def to_map(cls, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        if df is None or df.empty:
            return out
        for r in df.itertuples(index=False):
            code = _normalize_code(getattr(r, "code"))
            if not code:
                continue
            out[code] = {
                "board": getattr(r, "board", _CodeMeta.infer_board(code)),
                "name": getattr(r, "name", ""),
                "first_date": getattr(r, "first_date", pd.NaT),
                "last_date": getattr(r, "last_date", pd.NaT),
                "is_delisted_guess": bool(getattr(r, "is_delisted_guess", False)),
            }
        return out


# =============================================================================
# Dataset memmap store (keyed safely)
# =============================================================================
class _MemmapStore:
    @staticmethod
    def _dir() -> str:
        d = os.path.join(str(DPUtils.cfg("OUTPUT_DIR", "./output")), "dataset_store")
        os.makedirs(d, exist_ok=True)
        return d

    @staticmethod
    def _feature_hash(feature_cols: List[str]) -> str:
        raw = ",".join(feature_cols).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:10]

    @classmethod
    def key(cls, mode: str, adj_norm: str, fp: str, label_col: str, feature_cols: List[str]) -> str:
        today = _dt.date.today().strftime("%Y%m%d")
        fh = cls._feature_hash(feature_cols)
        return f"{mode}_{adj_norm}_{today}_{DPUtils.VERSION}_{fp}_{label_col}_{fh}"

    @classmethod
    def paths(cls, key: str) -> Dict[str, str]:
        base = os.path.join(cls._dir(), key)
        os.makedirs(base, exist_ok=True)
        return {
            "base": base,
            "meta": os.path.join(base, "meta.json"),
            "features": os.path.join(base, "features.f32.bin"),
            "labels": os.path.join(base, "labels.f32.bin"),
            "dates": os.path.join(base, "dates.i8.bin"),
            "code_ids": os.path.join(base, "code_ids.i4.bin"),
        }

    @classmethod
    def exists(cls, key: str) -> bool:
        p = cls.paths(key)
        need = ["meta", "features", "labels", "dates", "code_ids"]
        return all(os.path.exists(p[k]) and os.path.getsize(p[k]) >= 512 for k in need)

    @classmethod
    def build(cls, panel_df: pd.DataFrame, feature_cols: List[str], label_col: str, key: str) -> Dict[str, Any]:
        p = cls.paths(key)
        panel_df = panel_df.sort_values(["code", "date"]).reset_index(drop=True)
        panel_df["date"] = pd.to_datetime(panel_df["date"], errors="coerce")
        panel_df = panel_df.dropna(subset=["date"]).reset_index(drop=True)

        n = int(len(panel_df))
        fdim = int(len(feature_cols))
        if n <= 0 or fdim <= 0:
            raise ValueError("Empty panel_df or feature_cols.")

        cat = pd.Categorical(panel_df["code"].astype(str))
        code_ids = cat.codes.astype(np.int32, copy=False)
        code_vocab = [str(x) for x in cat.categories.tolist()]

        dates_i8 = panel_df["date"].values.astype("datetime64[ns]").astype(np.int64)

        feat_mm = np.memmap(p["features"], mode="w+", dtype=np.float32, shape=(n, fdim))
        lab_mm = np.memmap(p["labels"], mode="w+", dtype=np.float32, shape=(n,))
        date_mm = np.memmap(p["dates"], mode="w+", dtype=np.int64, shape=(n,))
        cid_mm = np.memmap(p["code_ids"], mode="w+", dtype=np.int32, shape=(n,))

        cid_mm[:] = code_ids
        date_mm[:] = dates_i8

        lab = pd.to_numeric(panel_df[label_col], errors="coerce").fillna(0.0).astype(np.float32).values
        lab_mm[:] = lab

        chunk = int(DPUtils.cfg("MEMMAP_WRITE_CHUNK", 250_000))
        for st in range(0, n, chunk):
            ed = min(n, st + chunk)
            feat_mm[st:ed, :] = panel_df.loc[st:ed - 1, feature_cols].astype(np.float32).values

        feat_mm.flush()
        lab_mm.flush()
        date_mm.flush()
        cid_mm.flush()

        meta = {
            "key": key,
            "n": n,
            "fdim": fdim,
            "feature_cols": feature_cols,
            "label_col": label_col,
            "code_vocab": code_vocab,
            "created_at_utc": _dt.datetime.utcnow().isoformat(),
        }
        DPUtils.atomic_save_json(meta, p["meta"])
        return meta

    @classmethod
    def open(cls, key: str) -> Tuple[Dict[str, Any], Dict[str, np.memmap]]:
        p = cls.paths(key)
        with open(p["meta"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        n = int(meta["n"])
        fdim = int(meta["fdim"])
        mms = {
            "features": np.memmap(p["features"], mode="r", dtype=np.float32, shape=(n, fdim)),
            "labels": np.memmap(p["labels"], mode="r", dtype=np.float32, shape=(n,)),
            "dates": np.memmap(p["dates"], mode="r", dtype=np.int64, shape=(n,)),
            "code_ids": np.memmap(p["code_ids"], mode="r", dtype=np.int32, shape=(n,)),
        }
        return meta, mms


# =============================================================================
# Worker init for ProcessPool (avoid pickling big maps each task)
# =============================================================================
_G_META_MAP: Optional[Dict[str, Dict[str, Any]]] = None
_G_SECMASTER_MAP: Optional[Dict[str, Dict[str, Any]]] = None


def _worker_init(meta_map: Dict[str, Dict[str, Any]], sec_map: Dict[str, Dict[str, Any]]) -> None:
    global _G_META_MAP, _G_SECMASTER_MAP
    _G_META_MAP = meta_map
    _G_SECMASTER_MAP = sec_map


# =============================================================================
# QC (hard contracts, explicit rejection)
# =============================================================================
class _QC:
    @staticmethod
    def validate_price_frame(df: pd.DataFrame) -> Tuple[bool, str]:
        need_cols = ["code", "date", "open", "high", "low", "close"]
        for c in need_cols:
            if c not in df.columns:
                return False, f"MissingCol({c})"

        if df["code"].isna().any():
            return False, "BadCode(NaN)"
        if df["date"].isna().any():
            return False, "BadDate(NaT)"

        g = df.groupby("code", sort=False)
        dup = g["date"].apply(lambda x: x.duplicated().any())
        if bool(dup.any()):
            return False, "DupDate"

        for c in ["open", "high", "low", "close"]:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.isna().mean() > 0.1:
                return False, f"TooManyNaN({c})"
            if (s <= 0).mean() > 0.01:
                return False, f"NonPositive({c})"

        oh = np.maximum(df["open"].astype(float), df["close"].astype(float))
        ol = np.minimum(df["open"].astype(float), df["close"].astype(float))
        if ((df["high"].astype(float) + 1e-9) < oh).mean() > 0.001:
            return False, "OHLCHighInconsistent"
        if ((df["low"].astype(float) - 1e-9) > ol).mean() > 0.001:
            return False, "OHLCLowInconsistent"
        return True, "OK"


# =============================================================================
# Pipelines
# =============================================================================
class PricePipeline:
    """
    Download/update caches for price and fundamentals.
    """

    @staticmethod
    def _download_price_incremental(
        code: str,
        adjust: str,
        target_dt: pd.Timestamp,
        manifest_map: Dict[Tuple[str, str], Tuple[Optional[pd.Timestamp], int, Optional[pd.Timestamp]]],
    ) -> Tuple[str, str, bool, str, Optional[pd.Timestamp], int]:
        code = _normalize_code(code) or str(code)
        adj_norm = DPUtils.norm_adjust(adjust)
        schema_ver = int(DPUtils.cfg("PRICE_SCHEMA_VER", 2))
        path = DPUtils.price_path(adj_norm, code)

        last_dt, rows_hint, updated_at = (None, 0, None)
        key = (code, adj_norm)
        if key in manifest_map:
            last_dt, rows_hint, updated_at = manifest_map.get(key, (None, 0, None))
            last_dt = pd.to_datetime(last_dt, errors="coerce") if last_dt is not None else None

        if DPUtils.should_skip_fetch(last_dt, updated_at, target_dt):
            return code, adj_norm, True, "SkipFresh", last_dt, int(rows_hint)

        if last_dt is None and os.path.exists(path) and os.path.getsize(path) > 512:
            old_df = DPUtils.read_price(path, expected_adj=adj_norm)
            if old_df is not None and not old_df.empty:
                last_dt = pd.to_datetime(old_df["date"], errors="coerce").max()

        if last_dt is None:
            years = int(DPUtils.cfg("PRICE_BACKFILL_YEARS", 10))
            start_dt = pd.Timestamp(target_dt) - pd.Timedelta(days=365 * years)
        else:
            overlap = int(DPUtils.cfg("PRICE_OVERLAP_DAYS", 3))
            start_dt = pd.Timestamp(last_dt) - pd.Timedelta(days=overlap)

        try:
            raw = DPUtils.ak_hist(code, start_dt, pd.Timestamp(target_dt), adjust=adj_norm)
        except Exception as e:
            if os.path.exists(path) and os.path.getsize(path) > 512:
                return code, adj_norm, True, f"UsingCache({type(e).__name__})", last_dt, int(rows_hint)
            return code, adj_norm, False, f"FetchFail({type(e).__name__}:{e})", last_dt, 0

        if raw is None or raw.empty:
            return code, adj_norm, True, "NoNewRows", last_dt, int(rows_hint)

        df_new = DPUtils.normalize_price_df(raw)
        df_new["code"] = code

        df_old = None
        if os.path.exists(path) and os.path.getsize(path) > 512:
            df_old = DPUtils.read_price(path, expected_adj=adj_norm)

        if df_old is not None and not df_old.empty:
            merged = pd.concat([df_old, df_new], ignore_index=True)
            merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
            merged = merged.dropna(subset=["date"]).sort_values(["code", "date"])
            merged = merged.drop_duplicates(["code", "date"], keep="last").reset_index(drop=True)
        else:
            merged = df_new.drop_duplicates(["code", "date"], keep="last").reset_index(drop=True)

        DPUtils.atomic_save_parquet(merged, path, index=False)
        DPUtils.atomic_save_json(
            {
                "schema_ver": schema_ver,
                "created_by": DPUtils.VERSION,
                "code": code,
                "stored_adjust": adj_norm,
                "ak_adjust": DPUtils.ak_adjust_param(adj_norm),
                "updated_at_utc": _dt.datetime.utcnow().isoformat(),
                "rows": int(len(merged)),
                "last_date": pd.to_datetime(merged["date"]).max().isoformat() if len(merged) else None,
            },
            DPUtils.meta_path(path),
        )

        last = pd.to_datetime(merged["date"]).max() if len(merged) else last_dt
        return code, adj_norm, True, "OK", last, int(len(merged))

    @staticmethod
    def _download_finance_worker(code: str) -> Tuple[str, bool, str]:
        code = _normalize_code(code) or str(code)
        fund_dir = os.path.join(str(DPUtils.cfg("DATA_DIR", "./data")), "fundamental")
        os.makedirs(fund_dir, exist_ok=True)
        path = os.path.join(fund_dir, f"{code}.parquet")

        curr_month = _dt.date.today().month
        ttl_seconds = 12 * 3600 if curr_month in [4, 8, 10] else 72 * 3600

        if os.path.exists(path) and os.path.getsize(path) > 512 and (time.time() - os.path.getmtime(path)) < ttl_seconds:
            return code, True, "Skipped(Cache)"

        try:
            df = _AkClient.call(ak.stock_financial_analysis_indicator, symbol=code)
        except Exception as e:
            return code, False, f"FetchFail({type(e).__name__}:{e})"

        try:
            if df is None or df.empty:
                return code, True, "Empty"

            if "date" not in df.columns:
                for c in ["公告日期", "披露日期", "报告期", "报告日期", "日期"]:
                    if c in df.columns:
                        df = df.rename(columns={c: "date"})
                        break
            if "date" not in df.columns:
                df = df.reset_index().rename(columns={"index": "date"})

            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")

            keep = ["date"]
            for c in df.columns:
                if c == "date":
                    continue
                if pd.api.types.is_numeric_dtype(df[c]):
                    keep.append(c)
                else:
                    s = pd.to_numeric(df[c], errors="coerce")
                    if s.notna().mean() > 0.9:
                        df[c] = s
                        keep.append(c)

            DPUtils.atomic_save_parquet(df[keep].copy(), path, index=False)
            return code, True, "OK"
        except Exception as e:
            return code, False, f"ParseFail({type(e).__name__}:{e})"

    @staticmethod
    def download_data(adjusts: Optional[List[str]] = None) -> None:
        logger.info(f"\n{'=' * 60}\n>>> [ETL] Data Pipeline Initiated ({DPUtils.VERSION})\n{'=' * 60}")
        DPUtils.setup_proxy_env()
        os.makedirs(str(DPUtils.cfg("DATA_DIR", "./data")), exist_ok=True)

        target_dt = _CalendarStore.latest_trade_date()
        logger.info(f"📅 Target trading date = {target_dt.date()}")

        if adjusts is None:
            adjusts = DPUtils.cfg("PRICE_ADJUSTS", ["qfq"])
        adjusts = [DPUtils.norm_adjust(a) for a in adjusts]

        logger.info("📋 Building universe snapshot (normalized + historical-complete union)...")
        snap, codes = DPUtils.get_universe_snapshot(target_dt, force_refresh=bool(DPUtils.cfg("FORCE_UNIVERSE_REFRESH", False)))
        if not codes:
            logger.critical("❌ Universe is empty. Abort.")
            return
        logger.info(f"✅ Universe size = {len(codes)}")

        mstore = ManifestStore(os.path.join(str(DPUtils.cfg("DATA_DIR", "./data")), "manifest", "price_manifest.parquet"))
        manifest_map = ManifestStore.to_map(mstore.load())

        todo = [(c, a) for c in codes for a in adjusts]

        def _task(args):
            c, a = args
            return PricePipeline._download_price_incremental(c, a, target_dt=target_dt, manifest_map=manifest_map)

        workers = int(DPUtils.cfg("PRICE_WORKERS", min(24, (os.cpu_count() or 8) + 8)))
        updates: List[ManifestRow] = []
        ok = 0
        bad = 0

        logger.info(f"🚀 Starting price download with {workers} workers... (todo={len(todo)})")
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for code, adj, succ, msg, last_dt, rows in tqdm(ex.map(_task, todo), total=len(todo), desc="Downloading"):
                if succ:
                    ok += 1
                    updates.append(
                        ManifestRow(
                            code=code,
                            adjust=adj,
                            last_date=last_dt,
                            rows=int(rows),
                            updated_at=pd.Timestamp.utcnow(),
                            schema_ver=int(DPUtils.cfg("PRICE_SCHEMA_VER", 2)),
                        )
                    )
                else:
                    bad += 1
                    logger.warning(f"[Price] {code}/{adj} failed: {msg}")

        mstore.upsert_many(updates)
        logger.info(f"✅ Price update done. OK={ok} Bad={bad}")

        if bool(DPUtils.cfg("SYNC_FUNDAMENTAL", False)):
            logger.info("📋 Syncing Fundamental Data...")
            fin_workers = int(DPUtils.cfg("FIN_WORKERS", 8))
            with concurrent.futures.ThreadPoolExecutor(max_workers=fin_workers) as ex:
                futures = {ex.submit(PricePipeline._download_finance_worker, c): c for c in codes}
                for fut in tqdm(concurrent.futures.as_completed(futures), total=len(codes), desc="Finance"):
                    code, succ, msg = fut.result()
                    if not succ:
                        logger.warning(f"[Fund] {code} failed: {msg}")

        logger.info("✅ ETL Pipeline Completed.")


class PanelPipeline:
    """
    Build panel_df by processing each code file:
    - read & schema-coerce (non-destructive)
    - attach meta
    - QC
    - optional PIT merge
    - liquidity features + trade masks
    - Alpha factors
    - closed-loop label gating
    """

    @staticmethod
    def _process_one_code_file(
        price_file: str,
        mode: str,
        adjust: str,
        meta_map: Dict[str, Dict[str, Any]],
        sec_map: Dict[str, Dict[str, Any]],
    ) -> Optional[pd.DataFrame]:
        adj_norm = DPUtils.norm_adjust(adjust)

        df = DPUtils.read_price(price_file, expected_adj=adj_norm)
        if df is None or df.empty:
            return None

        code = _normalize_code(df["code"].iloc[0]) or _normalize_code(os.path.basename(price_file)) or ""
        if not code:
            return None
        df["code"] = code  # enforce single-code
        df = df.sort_values("date").reset_index(drop=True)

        df = DPUtils.attach_code_meta(df, meta_map, sec_map)

        df = DPUtils.align_to_calendar(df)

        ok, _reason = _QC.validate_price_frame(df)
        if not ok:
            return None

        context_len = int(DPUtils.cfg("CONTEXT_LEN", 60))
        if len(df) <= context_len:
            return None

        df = DPUtils.merge_fundamental_pit(df)

        df = DPUtils.add_liquidity_features(df)
        df = DPUtils.add_trade_masks(df)

        df = df.sort_values(["code", "date"]).reset_index(drop=True)
        df = AlphaFactory(df).make_factors()

        pred_len = int(DPUtils.cfg("PRED_LEN", 5))
        g = df.groupby("code", sort=False)

        entry_price_mode = str(DPUtils.cfg("ENTRY_PRICE_MODE", "open")).lower().strip()
        if entry_price_mode == "open":
            df["entry_price"] = g["open"].shift(-1)
        elif entry_price_mode == "vwap":
            vwap = (df.get("amount", 0.0).fillna(0.0) / (df.get("volume", 0.0).fillna(0.0) + 1e-9)).astype(np.float32)
            df["_vwap"] = vwap
            df["entry_price"] = g["_vwap"].shift(-1)
            df["entry_price"] = df["entry_price"].fillna(g["open"].shift(-1))
            df.drop(columns=["_vwap"], inplace=True, errors="ignore")
        elif entry_price_mode == "close":
            df["entry_price"] = g["close"].shift(-1)
        else:
            df["entry_price"] = g["open"].shift(-1)

        df["future_close"] = g["close"].shift(-pred_len)
        df["target"] = df["future_close"] / (df["entry_price"] + 1e-9) - 1.0

        gate_entry = bool(DPUtils.cfg("GATE_TARGET_WITH_ENTRY", True))
        gate_exit = bool(DPUtils.cfg("GATE_TARGET_WITH_EXIT", False))

        if gate_entry:
            entry_ok = g["buyable_mask"].shift(-1).notna()
            df.loc[~entry_ok, "target"] = np.nan

        if gate_exit and pred_len > 0:
            exit_ok = g["sellable_mask"].shift(-pred_len).notna()
            df.loc[~exit_ok, "target"] = np.nan

        df.loc[~np.isfinite(df["target"].astype(float)), "target"] = np.nan
        if mode == "train":
            df = df.dropna(subset=["target"]).reset_index(drop=True)

        return df

    @staticmethod
    def build_panel(
        mode: str = "train",
        force_refresh: bool = False,
        adjust: str = "qfq",
        backend: Optional[str] = None,
        debug: bool = False,
    ):
        adj_norm = DPUtils.norm_adjust(adjust)
        cache_path = DPUtils.cache_path(mode, adj_norm)

        if not force_refresh and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    panel_df, feat_cols = pickle.load(f)
                if isinstance(panel_df, pd.DataFrame):
                    panel_df.attrs.setdefault("adjust", adj_norm)
                    panel_df.attrs.setdefault("mode", mode)
                    panel_df.attrs.setdefault("fingerprint", DPUtils.config_fingerprint())
                return panel_df, feat_cols
            except Exception:
                pass

        if backend is None:
            backend = str(DPUtils.cfg("ALPHA_BACKEND", "process"))
        backend = backend.lower().strip()
        if debug or bool(DPUtils.cfg("DEBUG", False)):
            backend = "serial"

        universe_asof = _CalendarStore.latest_trade_date()
        logger.info(f">>> [Processing] Building Panel (Mode={mode}, adjust={adj_norm}, backend={backend}, universe_asof={universe_asof.date()})")

        # Unified snapshot retrieval
        snap, codes = DPUtils.get_universe_snapshot(universe_asof, force_refresh=bool(DPUtils.cfg("FORCE_UNIVERSE_REFRESH", False)))
        meta_map = _UniverseStore.meta_map_for_asof(universe_asof)

        sm_df = _SecurityMasterStore.build_or_update(
            codes=codes,
            adj_norm_for_scan=adj_norm,
            target_dt=universe_asof,
            meta_map=meta_map,
            force=bool(DPUtils.cfg("FORCE_SECMASTER_REFRESH", False)),
        )
        sec_map = _SecurityMasterStore.to_map(sm_df)

        price_files = DPUtils.price_glob(adj_norm)
        if not price_files:
            raise RuntimeError(f"❌ No price data found for adjust={adj_norm}. Run download_data() first.")

        if backend == "serial":
            max_files = int(DPUtils.cfg("DEBUG_MAX_FILES", 10))
            if debug:
                price_files = price_files[:max_files]

        part_dir = os.path.join(
            str(DPUtils.cfg("OUTPUT_DIR", "./output")),
            "panel_parts",
            f"{DPUtils.VERSION}_{mode}_{adj_norm}_{DPUtils.config_fingerprint()}",
        )
        os.makedirs(part_dir, exist_ok=True)

        flush_n = int(DPUtils.cfg("PANEL_FLUSH_N", 200))
        buffer: List[pd.DataFrame] = []
        part_paths: List[str] = []

        def _flush() -> None:
            if not buffer:
                return
            part_path = os.path.join(part_dir, f"part_{len(part_paths):05d}.parquet")
            tmp = pd.concat(buffer, ignore_index=True)
            DPUtils.atomic_save_parquet(tmp, part_path, index=False)
            part_paths.append(part_path)
            buffer.clear()

        fail_fast = bool(DPUtils.cfg("FAIL_FAST", True))
        rej = {"empty": 0, "ok": 0, "exc": 0}

        def _iter_outputs() -> Iterable[Optional[pd.DataFrame]]:
            if backend == "serial":
                for pf in tqdm(price_files, desc="Alpha+Label(serial)"):
                    try:
                        out = PanelPipeline._process_one_code_file(pf, mode, adj_norm, meta_map, sec_map)
                        yield out
                    except Exception as e:
                        rej["exc"] += 1
                        logger.exception(f"[serial] Failed on file={pf}: {e}")
                        if fail_fast:
                            raise
                        yield None
            elif backend == "process":
                alpha_workers = int(DPUtils.cfg("ALPHA_WORKERS", max(1, (os.cpu_count() or 8) - 1)))
                chunksize = int(DPUtils.cfg("ALPHA_MAP_CHUNKSIZE", 8))
                logger.info(f"⚙️  Alpha per-code with {alpha_workers} processes ...")
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=alpha_workers,
                    initializer=_worker_init,
                    initargs=(meta_map, sec_map),
                ) as ex:
                    it = ex.map(
                        _process_one_code_file_worker,
                        price_files,
                        itertools.repeat(mode),
                        itertools.repeat(adj_norm),
                        chunksize=chunksize,
                    )
                    for out in tqdm(it, total=len(price_files), desc="Alpha+Label(process)"):
                        yield out
            else:
                raise ValueError(f"Unknown backend={backend}. Use 'serial' or 'process'.")

        for out in _iter_outputs():
            if out is None or out.empty:
                rej["empty"] += 1
                continue
            buffer.append(out)
            rej["ok"] += 1
            if len(buffer) >= flush_n:
                _flush()
        _flush()

        logger.info(f"[Panel] per-code results: ok={rej['ok']} empty={rej['empty']} exc={rej['exc']}")

        if not part_paths:
            logger.error("❌ No valid data generated. Hints:")
            logger.error("   1) Check price meta files (or set ALLOW_LEGACY_PRICE_CACHE=True).")
            logger.error("   2) Try ALIGN_TO_CALENDAR=False if calendar alignment is too strict.")
            logger.error("   3) Run DEBUG=True to see rejection details.")
            raise ValueError("Not enough valid price/factor data to build panel.")

        merge_batch = int(DPUtils.cfg("PANEL_MERGE_BATCH", 24))
        merged_chunks: List[pd.DataFrame] = []
        for i in range(0, len(part_paths), merge_batch):
            batch = [pd.read_parquet(p) for p in part_paths[i : i + merge_batch]]
            merged_chunks.append(pd.concat(batch, ignore_index=True))
        panel_df = pd.concat(merged_chunks, ignore_index=True) if len(merged_chunks) > 1 else merged_chunks[0]

        panel_df["date"] = pd.to_datetime(panel_df["date"], errors="coerce")
        panel_df = panel_df.dropna(subset=["date"]).reset_index(drop=True)

        panel_df = DPUtils.tag_universe(panel_df)

        if bool(DPUtils.cfg("USE_CROSS_SECTIONAL", True)):
            try:
                panel_df = AlphaFactory.add_cross_sectional_factors(panel_df)
            except Exception as e:
                logger.warning(f"Cross-sectional factors skipped (error: {e})")

        prefixes = list(DPUtils.cfg("FEATURE_PREFIXES", ["alpha_", "fac_", "cs_"]))
        feat_cols = [c for c in panel_df.columns if any(str(c).startswith(p) for p in prefixes)]
        if feat_cols:
            panel_df[feat_cols] = (
                panel_df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32, copy=False)
            )

        panel_df.attrs["adjust"] = adj_norm
        panel_df.attrs["mode"] = mode
        panel_df.attrs["fingerprint"] = DPUtils.config_fingerprint()
        panel_df.attrs["universe_asof"] = str(universe_asof.date())
        panel_df.attrs["created_by"] = DPUtils.VERSION

        with open(cache_path, "wb") as f:
            pickle.dump((panel_df, feat_cols), f)

        logger.info(f"✅ Panel Ready. Shape={panel_df.shape} Features={len(feat_cols)} Cache={cache_path}")
        return panel_df, feat_cols


def _process_one_code_file_worker(price_file: str, mode: str, adjust: str) -> Optional[pd.DataFrame]:
    mm = _G_META_MAP or {}
    sm = _G_SECMASTER_MAP or {}
    return PanelPipeline._process_one_code_file(price_file, mode, adjust, mm, sm)


class SplitPolicy:
    """Single source of truth for Train/Val/Test date splits (end is EXCLUSIVE)."""

    @staticmethod
    def _normalize_ratios() -> Tuple[float, float, Optional[float]]:
        train_ratio = float(getattr(Config, "TRAIN_RATIO", 0.7))
        val_ratio = float(getattr(Config, "VAL_RATIO", 0.15))
        test_ratio = getattr(Config, "TEST_RATIO", None)
        test_ratio_f = float(test_ratio) if test_ratio is not None else None

        if test_ratio_f is not None:
            s = train_ratio + val_ratio + test_ratio_f
            if s > 0:
                train_ratio, val_ratio, test_ratio_f = train_ratio / s, val_ratio / s, test_ratio_f / s
        else:
            s = train_ratio + val_ratio
            if s > 0:
                train_ratio, val_ratio = train_ratio / s, val_ratio / s

        return train_ratio, val_ratio, test_ratio_f

    @staticmethod
    def get_date_splits(panel_df: pd.DataFrame, seq_len: Optional[int] = None) -> Dict[str, pd.Timestamp]:
        if "date" not in panel_df.columns:
            raise ValueError("panel_df must contain 'date' column")

        dates = pd.to_datetime(panel_df["date"], errors="coerce").dropna().unique()
        unique_dates = np.sort(dates)
        n_dates = int(len(unique_dates))
        if n_dates == 0:
            raise ValueError("Empty dates in panel_df")

        train_ratio, val_ratio, _test_ratio = SplitPolicy._normalize_ratios()

        gap = int(seq_len or getattr(Config, "CONTEXT_LEN", 60))
        gap = max(0, gap)

        # boundary indices for exclusive ends
        cut1 = max(1, min(n_dates - 2, int(n_dates * train_ratio)))
        cut2 = max(cut1 + 1, min(n_dates - 1, int(n_dates * (train_ratio + val_ratio))))

        train_start = pd.to_datetime(unique_dates[0])
        train_end_excl = pd.to_datetime(unique_dates[cut1])

        val_start_idx = min(cut1 + gap, n_dates - 1)
        val_end_excl = pd.to_datetime(unique_dates[cut2])
        val_start = pd.to_datetime(unique_dates[val_start_idx])

        test_start_idx = min(cut2 + gap, n_dates - 1)
        test_start = pd.to_datetime(unique_dates[test_start_idx])

        last_date = pd.to_datetime(unique_dates[-1])
        test_end_excl = last_date + pd.Timedelta(days=1)

        return {
            "train_start": train_start,
            "train_end_excl": train_end_excl,
            "val_start": val_start,
            "val_end_excl": val_end_excl,
            "test_start": test_start,
            "test_end_excl": test_end_excl,
            "last_date": last_date,
            "gap_days": gap,
            "cut1": pd.to_datetime(unique_dates[cut1 - 1]),
            "cut2": pd.to_datetime(unique_dates[cut2 - 1]),
        }


class DatasetPipeline:
    """
    Construct HF DatasetDict (memmap preferred).
    """

    @staticmethod
    def build_dataset(panel_df: pd.DataFrame, feature_cols: List[str]):
        logger.info(">>> [Dataset] Constructing dataset ...")

        panel_df = panel_df.sort_values(["code", "date"]).reset_index(drop=True)
        panel_df["date"] = pd.to_datetime(panel_df["date"], errors="coerce")
        panel_df = panel_df.dropna(subset=["date"]).reset_index(drop=True)

        seq_len = int(DPUtils.cfg("CONTEXT_LEN", 60))
        stride = int(DPUtils.cfg("STRIDE", 1))
        if seq_len <= 1:
            raise ValueError("CONTEXT_LEN must be > 1")

        label_col = "rank_label" if "rank_label" in panel_df.columns else "target"

        backend = str(DPUtils.cfg("DATASET_BACKEND", "hf_memmap")).lower().strip()
        fp = str(panel_df.attrs.get("fingerprint") or DPUtils.config_fingerprint())
        adj_norm = str(panel_df.attrs.get("adjust") or "unknown")
        mode = str(panel_df.attrs.get("mode") or "train")

        # SSOT: splits from SplitPolicy (prevents leakage via gap)
        splits = SplitPolicy.get_date_splits(panel_df, seq_len=seq_len)
        train_end_excl_ns = pd.Timestamp(splits["train_end_excl"]).value
        val_start_ns = pd.Timestamp(splits["val_start"]).value
        val_end_excl_ns = pd.Timestamp(splits["val_end_excl"]).value
        test_start_ns = pd.Timestamp(splits["test_start"]).value
        test_end_excl_ns = pd.Timestamp(splits["test_end_excl"]).value

        if backend in ("hf_memmap", "memmap"):
            key = _MemmapStore.key(mode=mode, adj_norm=adj_norm, fp=fp, label_col=label_col, feature_cols=feature_cols)
            if not _MemmapStore.exists(key) or bool(DPUtils.cfg("FORCE_DATASET_REBUILD", False)):
                logger.info(f"🧠 Building memmap dataset store: key={key}")
                _MemmapStore.build(panel_df, feature_cols, label_col, key)

            meta, mms = _MemmapStore.open(key)
            feats = mms["features"]
            labels = mms["labels"]
            code_ids = mms["code_ids"]
            dates = mms["dates"]
            n = int(meta["n"])

            changes = np.flatnonzero(code_ids[:-1] != code_ids[1:]) + 1
            start_idx = np.concatenate(([0], changes)).astype(np.int64)
            end_idx = np.concatenate((changes, [n])).astype(np.int64)

            valid_starts: List[int] = []
            for st, ed in zip(start_idx.tolist(), end_idx.tolist()):
                ln = ed - st
                if ln < seq_len:
                    continue
                last = ed - seq_len
                valid_starts.extend(range(st, last + 1, stride))
            if not valid_starts:
                raise ValueError("No valid sequences. Check CONTEXT_LEN / STRIDE / panel size.")
            valid_starts = np.array(valid_starts, dtype=np.int64)

            pred_ns = dates[valid_starts + (seq_len - 1)]

            idx_train = valid_starts[pred_ns < train_end_excl_ns]
            idx_valid = valid_starts[(pred_ns >= val_start_ns) & (pred_ns < val_end_excl_ns)]
            idx_test = valid_starts[(pred_ns >= test_start_ns) & (pred_ns < test_end_excl_ns)]

            def lazy_transform(batch: Dict[str, List[int]]) -> Dict[str, Any]:
                s_list = batch["start_idx"]
                past_values = []
                y = []
                for s in s_list:
                    s = int(s)
                    e = s + seq_len
                    past_values.append(np.asarray(feats[s:e, :]))
                    y.append(float(labels[e - 1]))
                return {"past_values": past_values, "labels": y}

            ds = DatasetDict(
                {
                    "train": Dataset.from_dict({"start_idx": idx_train.tolist()}),
                    "validation": Dataset.from_dict({"start_idx": idx_valid.tolist()}),
                    "test": Dataset.from_dict({"start_idx": idx_test.tolist()}),
                }
            )
            ds.set_transform(lazy_transform)
            return ds, int(meta["fdim"])

        logger.warning("DATASET_BACKEND != memmap; falling back to in-RAM feature_matrix (may OOM).")
        feature_matrix = np.ascontiguousarray(panel_df[feature_cols].values.astype(np.float32))
        target_array = pd.to_numeric(panel_df[label_col], errors="coerce").fillna(0.0).values.astype(np.float32)
        codes = panel_df["code"].astype(str).values
        dates_ns = panel_df["date"].values.astype("datetime64[ns]").astype(np.int64)

        code_changes = np.where(codes[:-1] != codes[1:])[0] + 1
        start_indices = np.concatenate(([0], code_changes))
        end_indices = np.concatenate((code_changes, [len(codes)]))

        valid_start_indices: List[int] = []
        for st, ed in zip(start_indices, end_indices):
            ln = ed - st
            if ln < seq_len:
                continue
            last_start = ed - seq_len
            valid_start_indices.extend(range(st, last_start + 1, stride))
        valid_start_indices = np.array(valid_start_indices, dtype=np.int64)

        pred_ns = dates_ns[valid_start_indices + (seq_len - 1)]

        idx_train = valid_start_indices[pred_ns < train_end_excl_ns]
        idx_valid = valid_start_indices[(pred_ns >= val_start_ns) & (pred_ns < val_end_excl_ns)]
        idx_test = valid_start_indices[(pred_ns >= test_start_ns) & (pred_ns < test_end_excl_ns)]

        def lazy_transform(batch: Dict[str, List[int]]) -> Dict[str, Any]:
            start_idx = batch["start_idx"]
            past_values = []
            labels = []
            for s in start_idx:
                s = int(s)
                e = s + seq_len
                past_values.append(feature_matrix[s:e])
                labels.append(float(target_array[e - 1]))
            return {"past_values": past_values, "labels": labels}

        ds = DatasetDict(
            {
                "train": Dataset.from_dict({"start_idx": idx_train.tolist()}),
                "validation": Dataset.from_dict({"start_idx": idx_valid.tolist()}),
                "test": Dataset.from_dict({"start_idx": idx_test.tolist()}),
            }
        )
        ds.set_transform(lazy_transform)
        return ds, len(feature_cols)


# =============================================================================
# Facade (public API) — DataProvider does NOT implement internals
# =============================================================================
class DataProvider:
    VERSION = DPUtils.VERSION
    # ---------------------------------------------------------------------
    # Shadow interfaces (backward-compat): do NOT implement logic here.
    # ---------------------------------------------------------------------
    @staticmethod
    def _norm_adjust(adjust: Optional[str]) -> str:
        return DPUtils.norm_adjust(adjust)

    @staticmethod
    def _price_dir(adjust: str) -> str:
        return DPUtils.price_dir(DPUtils.norm_adjust(adjust))

    @staticmethod
    def _add_trade_masks(df: pd.DataFrame) -> pd.DataFrame:
        return DPUtils.add_trade_masks(df)

    @staticmethod
    def _get_date_splits(panel_df: pd.DataFrame, seq_len: Optional[int] = None) -> Dict[str, pd.Timestamp]:
        return SplitPolicy.get_date_splits(panel_df, seq_len=seq_len)

    @staticmethod
    def download_data(adjusts: Optional[List[str]] = None) -> None:
        return PricePipeline.download_data(adjusts=adjusts)

    @staticmethod
    def load_and_process_panel(
        mode: str = "train",
        force_refresh: bool = False,
        adjust: str = "qfq",
        backend: Optional[str] = None,
        debug: bool = False,
    ):
        return PanelPipeline.build_panel(
            mode=mode,
            force_refresh=force_refresh,
            adjust=adjust,
            backend=backend,
            debug=debug,
        )

    @staticmethod
    def make_dataset(panel_df: pd.DataFrame, feature_cols: List[str]):
        return DatasetPipeline.build_dataset(panel_df, feature_cols)


def get_dataset(force_refresh: bool = False, adjust: str = "qfq"):
    panel_df, feature_cols = DataProvider.load_and_process_panel(mode="train", force_refresh=force_refresh, adjust=adjust)
    return DataProvider.make_dataset(panel_df, feature_cols)
