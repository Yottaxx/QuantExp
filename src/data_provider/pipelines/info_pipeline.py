from __future__ import annotations

import concurrent.futures
import os
import time
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import akshare as ak
from tqdm import tqdm

from ..clients.ak_client import AkClient
from ..core.config import DPConfig
from ..utils.code import normalize_code
from ..utils.io import atomic_save_parquet
from ..stores.info_store import InfoStore
from ..stores.paths import info_path, info_dir


class InfoPipeline:
    """
    Download & cache per-code static info.
    Output: {DATA_DIR}/info/{code}.parquet (one-row)
      columns: code, name, industry, industry_cat, list_date, total_mkt_cap
    """
    SCHEMA_VER = 1

    def __init__(self, cfg: DPConfig, ak_client: AkClient, logger):
        self.cfg = cfg
        self.ak_client = ak_client
        self.logger = logger
        self.store = InfoStore(cfg, logger)
        self.encoder = self.store.industry_encoder()
        os.makedirs(info_dir(cfg), exist_ok=True)

    def _should_skip(self, path: str) -> bool:
        days = int(self.cfg.get("INFO_TTL_DAYS", 30) or 30)
        ttl = max(1, days) * 24 * 3600
        return os.path.exists(path) and os.path.getsize(path) > 256 and (time.time() - os.path.getmtime(path)) < ttl

    def _download_one(self, code: str) -> Tuple[str, bool, str]:
        c = normalize_code(code)
        if not c:
            return str(code), True, "BadCode"
        path = info_path(self.cfg, c)
        if self._should_skip(path):
            return c, True, "Skipped"

        try:
            df = self.ak_client.call(ak.stock_individual_info_em, symbol=c)
            if df is None or df.empty:
                out = pd.DataFrame([{
                    "code": c, "name": "", "industry": "Unknown", "industry_cat": 0,
                    "list_date": "19900101", "total_mkt_cap": 0.0,
                }])
                atomic_save_parquet(out, path, index=False, compression=str(self.cfg.get("PARQUET_COMPRESSION","zstd") or "zstd"))
                return c, True, "Empty"

            # df schema: columns [item, value]
            item_col = next((k for k in ["item","项目","指标名称"] if k in df.columns), None)
            val_col = next((k for k in ["value","值","指标值"] if k in df.columns), None)
            if not item_col or not val_col:
                item_col, val_col = df.columns[:2]

            info = dict(zip(df[item_col].astype(str), df[val_col]))

            name = str(info.get("股票简称", "") or info.get("证券简称", "") or "")
            industry = str(info.get("行业", "") or "Unknown")
            list_date = InfoStore.sanitize_list_date(info.get("上市时间", "") or info.get("上市日期", ""))
            mkt_cap = InfoStore.parse_mkt_cap(info.get("总市值", 0) or info.get("总市值(元)", 0) or 0)

            ind_cat = self.encoder.encode(industry)

            out = pd.DataFrame([{
                "code": c,
                "name": name,
                "industry": industry,
                "industry_cat": int(ind_cat),
                "list_date": list_date,
                "total_mkt_cap": float(mkt_cap),
            }])
            out.attrs["schema_ver"] = self.SCHEMA_VER
            atomic_save_parquet(out, path, index=False, compression=str(self.cfg.get("PARQUET_COMPRESSION","zstd") or "zstd"))
            return c, True, "Success"
        except Exception as e:
            return c, False, f"Failed({type(e).__name__})"

    def download(self, codes) -> None:
        flag = self.cfg.get("SYNC_INFO", None)
        if flag is False:
            self.logger.info("🟩 [Info] SYNC_INFO=False; skip.")
            return

        codes = [normalize_code(c) for c in codes]
        codes = [c for c in codes if c]
        if not codes:
            self.logger.warning("🟩 [Info] empty codes; skip.")
            return

        self.logger.info(f"🟩 [Info] syncing {len(codes)} codes ...")
        workers = int(self.cfg.get("PRICE_WORKERS", 8) or 8)

        ok = 0
        bad = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(self._download_one, c) for c in codes]
            for f in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc="Info"):
                code, success, msg = f.result()
                ok += int(bool(success))
                bad += int(not bool(success))
        self.logger.info(f"🟩 [Info] done. ok={ok}, fail={bad}")
