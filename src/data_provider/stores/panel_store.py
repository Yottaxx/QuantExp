from __future__ import annotations
import json, os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
import pandas as pd
from ..core.config import DPConfig
from ..utils.io import atomic_save_json

@dataclass(frozen=True)
class PanelMeta:
    mode: str
    adjust: str
    fingerprint: str
    universe_asof: str
    feature_cols: List[str]
    part_paths: List[str]
    created_by: str
    split_manifest: Optional[Dict] = None

class PanelStore:
    def __init__(self, cfg: DPConfig):
        self.cfg = cfg

    def parts_dir(self, mode: str, adj_norm: str, fp: str) -> str:
        out = str(self.cfg.get("OUTPUT_DIR","./output") or "./output")
        d = os.path.join(out, "panel_parts", f"{mode}_{adj_norm}_{fp}")
        os.makedirs(d, exist_ok=True)
        return d

    def meta_path(self, parts_dir: str) -> str:
        return os.path.join(parts_dir, "panel_meta.json")

    def write_meta(self, parts_dir: str, meta: PanelMeta) -> None:
        atomic_save_json(meta.__dict__, self.meta_path(parts_dir))

    def read_meta(self, parts_dir: str) -> PanelMeta:
        with open(self.meta_path(parts_dir), "r", encoding="utf-8") as f:
            obj = json.load(f)
        return PanelMeta(**obj)

    def iter_parts(self, part_paths: List[str], columns: Optional[List[str]] = None) -> Iterable[pd.DataFrame]:
        for p in part_paths:
            if columns is None:
                yield pd.read_parquet(p)
            else:
                yield pd.read_parquet(p, columns=columns)
