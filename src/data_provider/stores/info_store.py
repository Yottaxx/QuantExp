from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.config import DPConfig
from ..utils.code import normalize_code
from ..utils.io import atomic_save_json, atomic_save_parquet
from .paths import industry_map_path, info_dir, info_master_path


@dataclass
class IndustryEncoder:
    """
    Persistent industry -> int mapping.

    Why: usually you want industry IDs to be stable across incremental updates.
    """
    path: str

    def __post_init__(self):
        import threading
        self._lock = threading.Lock()
        self.mapping: Dict[str, int] = self._load()

    def _load(self) -> Dict[str, int]:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                if isinstance(m, dict) and "Unknown" in m:
                    return {str(k): int(v) for k, v in m.items()}
            except Exception:
                pass
        return {"Unknown": 0}

    def encode(self, industry: str) -> int:
        ind = str(industry or "").strip() or "Unknown"
        if ind not in self.mapping:
            with self._lock:
                if ind not in self.mapping:
                    self.mapping[ind] = (max(self.mapping.values()) + 1) if self.mapping else 1
                    atomic_save_json(self.mapping, self.path)
        return int(self.mapping.get(ind, 0))


class InfoStore:
    """
    Static info cache (per-code + optional master snapshot).
    Files:
      - {DATA_DIR}/info/{code}.parquet : one-row info for each code
      - {DATA_DIR}/info/_master.parquet: concatenated snapshot for fast lookup
      - {DATA_DIR}/info/industry_map.json: stable industry encoding
    """
    def __init__(self, cfg: DPConfig, logger=None):
        self.cfg = cfg
        self.logger = logger

    def _ttl_sec(self) -> int:
        days = int(self.cfg.get("INFO_TTL_DAYS", 30) or 30)
        return max(1, days) * 24 * 3600

    def industry_encoder(self) -> IndustryEncoder:
        return IndustryEncoder(industry_map_path(self.cfg))

    def load_master(self, force: bool = False) -> pd.DataFrame:
        p = info_master_path(self.cfg)
        ttl = self._ttl_sec()
        if (not force) and os.path.exists(p) and os.path.getsize(p) > 512 and (time.time() - os.path.getmtime(p)) < ttl:
            try:
                return pd.read_parquet(p)
            except Exception:
                pass

        # Build master from shards
        root = info_dir(self.cfg)
        files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".parquet") and not f.startswith("_")]
        rows = []
        for fp in files:
            try:
                d = pd.read_parquet(fp)
                if d is None or d.empty:
                    continue
                rows.append(d)
            except Exception:
                continue
        out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=["code", "name", "industry", "industry_cat", "list_date", "total_mkt_cap"]
        )
        if not out.empty:
            out["code"] = out["code"].map(normalize_code).astype(str)
            out = out.dropna(subset=["code"]).drop_duplicates("code").reset_index(drop=True)

        atomic_save_parquet(out, p, index=False, compression=str(self.cfg.get("PARQUET_COMPRESSION", "zstd") or "zstd"))
        return out

    def to_map(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        if df is None or df.empty:
            return out
        for r in df.itertuples(index=False):
            code = normalize_code(getattr(r, "code", None))
            if not code:
                continue
            out[code] = {
                "industry": getattr(r, "industry_cat", 0) if hasattr(r, "industry_cat") else getattr(r, "industry", "Unknown"),
                "industry_cat": int(getattr(r, "industry_cat", 0) or 0),
                "industry_name": str(getattr(r, "industry", "Unknown") or "Unknown"),
                "list_date": str(getattr(r, "list_date", "") or ""),
                "total_mkt_cap": float(getattr(r, "total_mkt_cap", 0.0) or 0.0),
                "name": str(getattr(r, "name", "") or ""),
            }
        return out

    @staticmethod
    def parse_mkt_cap(x: Any) -> float:
        # best-effort: "1234.56亿" / "1.2万亿" / plain number
        if x is None:
            return 0.0
        s = str(x).strip().replace(",", "")
        if not s:
            return 0.0
        mul = 1.0
        if "万亿" in s:
            mul = 1e12
            s = s.replace("万亿", "")
        elif "亿" in s:
            mul = 1e8
            s = s.replace("亿", "")
        elif "万" in s:
            mul = 1e4
            s = s.replace("万", "")
        try:
            return float(s) * mul
        except Exception:
            return 0.0

    @staticmethod
    def sanitize_list_date(x: Any) -> str:
        s = str(x or "").strip()
        # keep as YYYYMMDD if possible
        s = s.replace("-", "").replace("/", "")
        if len(s) >= 8 and s[:8].isdigit():
            return s[:8]
        return "19900101"
