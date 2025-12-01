from __future__ import annotations
import concurrent.futures
import datetime as _dt
import os
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
import akshare as ak

from ..core.config import DPConfig
from ..core.version import VERSION
from ..utils.code import normalize_code
from ..utils.io import atomic_save_json, atomic_save_parquet
from ..rules.normalize import normalize_ak_hist_df
from ..stores.paths import norm_adjust, price_path, meta_path
from ..stores.manifest_store import ManifestRow, ManifestStore

def yyyymmdd(dt: pd.Timestamp) -> str:
    return pd.to_datetime(dt).strftime("%Y%m%d")

def should_skip_fetch(cfg: DPConfig, last_dt: Optional[pd.Timestamp], updated_at: Optional[pd.Timestamp], target_dt: pd.Timestamp) -> bool:
    # Clear, explicit logic (avoid the previous "always True" feel).
    if not bool(cfg.get("SKIP_IF_FRESH", True)):
        return False
    if last_dt is None:
        return False
    lag_days = int(cfg.get("FRESH_LAG_DAYS", 0) or 0)
    if pd.Timestamp(last_dt).normalize() < (pd.Timestamp(target_dt).normalize() - pd.Timedelta(days=lag_days)):
        return False
    if updated_at is None:
        # last_dt is fresh enough; allow skipping.
        return True
    # updated today UTC/local isn't reliable; treat recent mtime equivalently.
    return True

class PricePipeline:
    def __init__(self, cfg: DPConfig, ak_client, calendar_store, universe_store, logger):
        self.cfg = cfg
        self.ak_client = ak_client
        self.calendar = calendar_store
        self.universe = universe_store
        self.logger = logger

    def _ak_hist(self, code: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp, adj_norm: str):
        start = yyyymmdd(start_dt)
        end = yyyymmdd(end_dt)
        ak_adjust = "" if adj_norm == "raw" else adj_norm
        return self.ak_client.call(
            ak.stock_zh_a_hist,
            symbol=code,
            period="daily",
            start_date=start,
            end_date=end,
            adjust=ak_adjust,
        )

    def _download_one(self, code: str, adjust: str, target_dt: pd.Timestamp,
                      manifest_map: Dict[Tuple[str,str], Tuple[Optional[pd.Timestamp], int, Optional[pd.Timestamp]]]):
        code = normalize_code(code) or str(code)
        adj_norm = norm_adjust(adjust)
        schema_ver = int(self.cfg.get("PRICE_SCHEMA_VER", 2) or 2)
        path = price_path(self.cfg, adj_norm, code)

        last_dt, rows_hint, updated_at = (None, 0, None)
        key = (code, adj_norm)
        if key in manifest_map:
            last_dt, rows_hint, updated_at = manifest_map[key]
            last_dt = pd.to_datetime(last_dt, errors="coerce") if last_dt is not None else None

        if should_skip_fetch(self.cfg, last_dt, updated_at, target_dt):
            return code, adj_norm, True, "SkipFresh", last_dt, int(rows_hint)

        if last_dt is None and os.path.exists(path) and os.path.getsize(path) > 512:
            try:
                old = pd.read_parquet(path, columns=["date"])
                dd = pd.to_datetime(old["date"], errors="coerce").dropna()
                if not dd.empty:
                    last_dt = dd.max()
            except Exception:
                pass

        if last_dt is None:
            years = int(self.cfg.get("PRICE_BACKFILL_YEARS", 10) or 10)
            start_dt = pd.Timestamp(target_dt) - pd.Timedelta(days=365*years)
        else:
            overlap = int(self.cfg.get("PRICE_OVERLAP_DAYS", 3) or 3)
            start_dt = pd.Timestamp(last_dt) - pd.Timedelta(days=overlap)

        try:
            raw = self._ak_hist(code, start_dt, pd.Timestamp(target_dt), adj_norm=adj_norm)
        except Exception as e:
            if os.path.exists(path) and os.path.getsize(path) > 512:
                return code, adj_norm, True, f"UsingCache({type(e).__name__})", last_dt, int(rows_hint)
            return code, adj_norm, False, f"FetchFail({type(e).__name__}:{e})", last_dt, 0

        if raw is None or raw.empty:
            return code, adj_norm, True, "NoNewRows", last_dt, int(rows_hint)

        df_new = normalize_ak_hist_df(raw)
        df_new["code"] = code

        df_old = None
        if os.path.exists(path) and os.path.getsize(path) > 512:
            try:
                df_old = pd.read_parquet(path)
            except Exception:
                df_old = None

        if df_old is not None and not df_old.empty:
            merged = pd.concat([df_old, df_new], ignore_index=True)
            merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
            merged = merged.dropna(subset=["date"]).sort_values(["code","date"])
            merged = merged.drop_duplicates(["code","date"], keep="last").reset_index(drop=True)
        else:
            merged = df_new.drop_duplicates(["code","date"], keep="last").reset_index(drop=True)

        atomic_save_parquet(merged, path, index=False, compression=str(self.cfg.get("PARQUET_COMPRESSION","zstd") or "zstd"))
        atomic_save_json({
            "schema_ver": schema_ver,
            "created_by": VERSION,
            "code": code,
            "stored_adjust": adj_norm,
            "updated_at_utc": _dt.datetime.utcnow().isoformat(),
            "rows": int(len(merged)),
            "last_date": pd.to_datetime(merged["date"]).max().isoformat() if len(merged) else None,
        }, meta_path(path))

        last = pd.to_datetime(merged["date"]).max() if len(merged) else last_dt
        return code, adj_norm, True, "OK", last, int(len(merged))

    def download_data(self, adjusts: Optional[List[str]] = None) -> None:
        self.logger.info(f"\n{'='*60}\n>>> [ETL] Data Pipeline Initiated ({VERSION})\n{'='*60}")

        target_dt = self.calendar.latest_trade_date()
        self.logger.info(f"📅 Target trading date = {target_dt.date()}")

        if adjusts is None:
            adjusts = self.cfg.get("PRICE_ADJUSTS", ["qfq"]) or ["qfq"]
        adjusts = [norm_adjust(a) for a in adjusts]

        snap = self.universe.get_snapshot(target_dt, force_refresh=bool(self.cfg.get("FORCE_UNIVERSE_REFRESH", False)))
        codes = snap["code"].astype(str).tolist() if snap is not None and not snap.empty else []
        if not codes:
            self.logger.critical("❌ Universe is empty. Abort.")
            return
        self.logger.info(f"✅ Universe size = {len(codes)}")

        mstore = ManifestStore(
            os.path.join(str(self.cfg.get("DATA_DIR","./data") or "./data"), "manifest", "price_manifest.parquet"),
            parquet_compression=str(self.cfg.get("PARQUET_COMPRESSION","zstd") or "zstd"),
        )
        manifest_map = ManifestStore.to_map(mstore.load())

        todo = [(c,a) for c in codes for a in adjusts]
        workers = int(self.cfg.get("PRICE_WORKERS", min(24, (os.cpu_count() or 8)+8)) or 16)
        updates: List[ManifestRow] = []
        ok = bad = 0

        def _task(args):
            c,a = args
            return self._download_one(c, a, target_dt=target_dt, manifest_map=manifest_map)

        self.logger.info(f"🚀 Starting price download with {workers} workers... (todo={len(todo)})")
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for code, adj, succ, msg, last_dt, rows in ex.map(_task, todo):
                if succ:
                    ok += 1
                    updates.append(ManifestRow(
                        code=code, adjust=adj, last_date=last_dt, rows=int(rows),
                        updated_at=pd.Timestamp.utcnow(),
                        schema_ver=int(self.cfg.get("PRICE_SCHEMA_VER", 2) or 2),
                    ))
                else:
                    bad += 1
                    self.logger.warning(f"[Price] {code}/{adj} failed: {msg}")

        mstore.upsert_many(updates)
        self.logger.info(f"✅ Price update done. OK={ok} Bad={bad}")
