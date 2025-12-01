from __future__ import annotations
import os
from typing import List
from ..core.config import DPConfig

def norm_adjust(adjust: str | None) -> str:
    if adjust is None:
        return "raw"
    a = str(adjust).strip().lower()
    if a in ("", "raw", "none", "null", "nan"):
        return "raw"
    if a in ("qfq","hfq"):
        return a
    raise ValueError(f"Unknown adjust={adjust!r}. Use raw/qfq/hfq")

def price_dir(cfg: DPConfig, adj_norm: str) -> str:
    root = str(cfg.get("DATA_DIR", "./data") or "./data")
    return os.path.join(root, f"price_{adj_norm}")

def legacy_price_dirs(cfg: DPConfig, adj_norm: str) -> List[str]:
    root = str(cfg.get("DATA_DIR", "./data") or "./data")
    cands: List[str] = []
    if adj_norm == "qfq":
        cands.append(root)
    cands.append(os.path.join(root, "price"))
    return [d for d in cands if os.path.isdir(d)]

def price_path(cfg: DPConfig, adj_norm: str, code: str) -> str:
    d = price_dir(cfg, adj_norm)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{code}.parquet")

def meta_path(price_path: str) -> str:
    return price_path + ".meta.json"



def info_dir(cfg: DPConfig) -> str:
    root = str(cfg.get("DATA_DIR", "./data") or "./data")
    d = os.path.join(root, "info")
    os.makedirs(d, exist_ok=True)
    return d

def info_path(cfg: DPConfig, code: str) -> str:
    return os.path.join(info_dir(cfg), f"{code}.parquet")

def info_master_path(cfg: DPConfig) -> str:
    return os.path.join(info_dir(cfg), "_master.parquet")

def industry_map_path(cfg: DPConfig) -> str:
    return os.path.join(info_dir(cfg), "industry_map.json")

def fundamental_dir(cfg: DPConfig) -> str:
    root = str(cfg.get("DATA_DIR", "./data") or "./data")
    d = os.path.join(root, "fundamental")
    os.makedirs(d, exist_ok=True)
    return d

def fundamental_path(cfg: DPConfig, code: str) -> str:
    return os.path.join(fundamental_dir(cfg), f"{code}.parquet")
