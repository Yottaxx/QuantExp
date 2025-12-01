from __future__ import annotations
import json, os
from typing import Any, Dict, Optional
import pandas as pd

def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def atomic_save_bytes(data: bytes, path: str) -> None:
    ensure_dir(path)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    if os.name == "nt" and os.path.exists(path):
        os.remove(path)
    os.replace(tmp, path)

def atomic_save_json(obj: Dict[str, Any], path: str) -> None:
    atomic_save_bytes(json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"), path)

def atomic_save_parquet(df: pd.DataFrame, path: str, index: bool, compression: Optional[str] = "zstd") -> None:
    ensure_dir(path)
    tmp = path + ".tmp"
    kwargs = {}
    if compression:
        kwargs["compression"] = compression
    try:
        df.to_parquet(tmp, engine="pyarrow", index=index, **kwargs)
    except Exception:
        df.to_parquet(tmp, index=index)
    if os.name == "nt" and os.path.exists(path):
        os.remove(path)
    os.replace(tmp, path)
