from __future__ import annotations
import os, time
from typing import Any, Dict, List
import pandas as pd
from ..core.config import DPConfig
from ..utils.io import atomic_save_parquet
from ..utils.code import normalize_code
from .universe_store import CodeMeta

class SecurityMasterStore:
    def __init__(self, cfg: DPConfig, logger, price_glob_fn):
        self.cfg = cfg
        self.logger = logger
        self.price_glob = price_glob_fn

    def _dir(self) -> str:
        root = str(self.cfg.get("DATA_DIR","./data") or "./data")
        d = os.path.join(root, "security_master")
        os.makedirs(d, exist_ok=True)
        return d

    def path(self) -> str:
        return os.path.join(self._dir(), "security_master.parquet")

    def _is_fresh(self, path: str, ttl: int) -> bool:
        return os.path.exists(path) and os.path.getsize(path) >= 512 and (time.time()-os.path.getmtime(path)) < ttl

    def load(self) -> pd.DataFrame:
        p = self.path()
        if not os.path.exists(p) or os.path.getsize(p) < 512:
            return pd.DataFrame(columns=["code","board","name","first_date","last_date","is_delisted_guess"])
        try:
            df = pd.read_parquet(p)
            df["first_date"] = pd.to_datetime(df.get("first_date"), errors="coerce")
            df["last_date"] = pd.to_datetime(df.get("last_date"), errors="coerce")
            return df
        except Exception:
            return pd.DataFrame(columns=["code","board","name","first_date","last_date","is_delisted_guess"])

    def build_or_update(self, codes: List[str], adj_norm_for_scan: str, target_dt: pd.Timestamp, meta_map: Dict[str, Dict[str, Any]], force: bool=False) -> pd.DataFrame:
        ttl = int(self.cfg.get("SECMASTER_TTL_SEC", 7*24*3600) or 7*24*3600)
        p = self.path()
        if (not force) and self._is_fresh(p, ttl):
            return self.load()

        scan = bool(self.cfg.get("SECMASTER_SCAN_PRICE", True))
        max_scan = int(self.cfg.get("SECMASTER_MAX_SCAN_FILES", 0) or 0)
        delist_gap = int(self.cfg.get("DELIST_GAP_DAYS", 90) or 90)

        rows: List[Dict[str, Any]] = []
        if scan:
            files = self.price_glob(adj_norm_for_scan)
            if max_scan and len(files) > max_scan:
                files = files[:max_scan]
            for pf in files:
                code = normalize_code(os.path.basename(pf).replace(".parquet",""))
                if not code: continue
                try:
                    d = pd.read_parquet(pf, columns=["date"])
                    dd = pd.to_datetime(d["date"], errors="coerce").dropna()
                    if dd.empty: continue
                    first_date = dd.min()
                    last_date = dd.max()
                except Exception:
                    continue
                mm = meta_map.get(code, {})
                name = str(mm.get("name",""))
                board = str(mm.get("board", CodeMeta.infer_board(code)))
                is_delisted_guess = bool(last_date < (pd.Timestamp(target_dt) - pd.Timedelta(days=delist_gap)))
                rows.append({
                    "code": code, "board": board, "name": name,
                    "first_date": first_date, "last_date": last_date,
                    "is_delisted_guess": is_delisted_guess,
                })

        seen = {r["code"] for r in rows}
        for c in codes:
            cc = normalize_code(c)
            if not cc or cc in seen:
                continue
            mm = meta_map.get(cc, {})
            rows.append({
                "code": cc,
                "board": str(mm.get("board", CodeMeta.infer_board(cc))),
                "name": str(mm.get("name","")),
                "first_date": pd.NaT,
                "last_date": pd.NaT,
                "is_delisted_guess": False,
            })

        df = pd.DataFrame(rows).drop_duplicates("code", keep="last")
        try:
            atomic_save_parquet(df.reset_index(drop=True), p, index=False, compression=str(self.cfg.get("PARQUET_COMPRESSION","zstd") or "zstd"))
        except Exception as e:
            self.logger.warning(f"[SecMaster] persist failed: {e}")
        return df.reset_index(drop=True)

    @staticmethod
    def to_map(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        if df is None or df.empty:
            return out
        for r in df.itertuples(index=False):
            code = normalize_code(getattr(r,"code",None))
            if not code: continue
            out[code] = {
                "board": getattr(r,"board", CodeMeta.infer_board(code)),
                "name": getattr(r,"name",""),
                "first_date": getattr(r,"first_date", pd.NaT),
                "last_date": getattr(r,"last_date", pd.NaT),
                "is_delisted_guess": bool(getattr(r,"is_delisted_guess", False)),
            }
        return out
