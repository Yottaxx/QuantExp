from __future__ import annotations
import glob, json, os
from typing import Optional, List
import pandas as pd
from ..core.config import DPConfig
from ..utils.code import normalize_code, norm_code_series
from .paths import meta_path, legacy_price_dirs, price_dir
from ..core.version import VERSION

def price_glob(cfg: DPConfig, adj_norm: str, logger) -> List[str]:
    d = price_dir(cfg, adj_norm)
    files = sorted(glob.glob(os.path.join(d, "*.parquet")))
    if files:
        return files
    for ld in legacy_price_dirs(cfg, adj_norm):
        files = sorted(glob.glob(os.path.join(ld, "*.parquet")))
        if files:
            logger.warning(f"[{VERSION}] Using legacy price dir for adjust={adj_norm}: {ld}")
            return files
    return []

def read_price(cfg: DPConfig, price_file: str, expected_adj: Optional[str], logger) -> Optional[pd.DataFrame]:
    """Read cached price parquet with strict adjust meta guard."""
    try:
        df = pd.read_parquet(price_file)
        if df is None or df.empty:
            return None

        code = normalize_code(os.path.basename(price_file).replace(".parquet",""))
        if not code:
            return None
        if "code" not in df.columns:
            df["code"] = code

        df["code"] = norm_code_series(df["code"])
        df = df.dropna(subset=["code"]).copy()
        df["code"] = df["code"].astype(str)

        if "date" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values(["code","date"]).reset_index(drop=True)

        if expected_adj is not None:
            exp = str(expected_adj)
            mp = meta_path(price_file)
            strict = bool(cfg.get("STRICT_PRICE_META", True))
            allow_legacy = bool(cfg.get("ALLOW_LEGACY_PRICE_CACHE", False))
            if os.path.exists(mp):
                with open(mp, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                stored = str(meta.get("stored_adjust","raw"))
                if stored != exp:
                    logger.error(f"[CORRUPT] {price_file} stored={stored} expected={exp}. Skipping.")
                    return None
            else:
                if strict and (not allow_legacy) and exp != "raw":
                    logger.error(f"[NO_META] {os.path.basename(price_file)} rejected. Expected {exp} but no meta found.")
                    return None
        return df
    except Exception as e:
        logger.debug(f"read_price failed: {price_file} err={e}")
        return None
