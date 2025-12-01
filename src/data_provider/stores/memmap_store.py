from __future__ import annotations
import hashlib, json, os
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from ..core.config import DPConfig
from ..utils.io import atomic_save_json

class MemmapStore:
    def __init__(self, cfg: DPConfig):
        self.cfg = cfg

    def _dir(self) -> str:
        d = os.path.join(str(self.cfg.get("OUTPUT_DIR","./output") or "./output"), "dataset_store")
        os.makedirs(d, exist_ok=True)
        return d

    @staticmethod
    def _feature_hash(feature_cols: List[str]) -> str:
        return hashlib.sha1(",".join(feature_cols).encode("utf-8")).hexdigest()[:10]

    def key(self, *, mode: str, adj_norm: str, fp: str, universe_asof: str, label_col: str, feature_cols: List[str]) -> str:
        fh = self._feature_hash(feature_cols)
        return f"{mode}_{adj_norm}_{universe_asof}_{fp}_{label_col}_{fh}"

    def paths(self, key: str) -> Dict[str, str]:
        base = os.path.join(self._dir(), key)
        os.makedirs(base, exist_ok=True)
        return {
            "base": base,
            "meta": os.path.join(base, "meta.json"),
            "features": os.path.join(base, "features.f32.bin"),
            "labels": os.path.join(base, "labels.f32.bin"),
            "dates": os.path.join(base, "dates.i8.bin"),
            "code_ids": os.path.join(base, "code_ids.i4.bin"),
        }

    def exists(self, key: str) -> bool:
        p = self.paths(key)
        need = ["meta","features","labels","dates","code_ids"]
        return all(os.path.exists(p[k]) and os.path.getsize(p[k]) >= 512 for k in need)

    def build_from_parts(self, *, part_paths: List[str], feature_cols: List[str], label_col: str, key: str, logger) -> Dict[str, Any]:
        p = self.paths(key)

        # First pass: count rows and collect codes / dates ordering after sorting per-part.
        total = 0
        code_list: List[str] = []
        date_list: List[np.ndarray] = []
        code_id_list: List[np.ndarray] = []
        feat_chunks: List[np.ndarray] = []
        lab_chunks: List[np.ndarray] = []

        # To ensure stable code->id mapping we build categories in a pre-pass.
        codes_set = set()
        for df in self._iter_sorted_parts(part_paths, columns=["code","date"]):
            codes_set.update(df["code"].astype(str).unique().tolist())
        code_vocab = sorted(codes_set)
        code_to_id = {c:i for i,c in enumerate(code_vocab)}

        # Now stream write to memmap in chunks.
        # Compute total length to allocate
        for df in self._iter_sorted_parts(part_paths, columns=["code","date"]):
            total += len(df)
        n = int(total)
        fdim = int(len(feature_cols))
        if n <= 0 or fdim <= 0:
            raise ValueError("Empty panel parts or feature_cols.")

        feat_mm = np.memmap(p["features"], mode="w+", dtype=np.float32, shape=(n, fdim))
        lab_mm = np.memmap(p["labels"], mode="w+", dtype=np.float32, shape=(n,))
        date_mm = np.memmap(p["dates"], mode="w+", dtype=np.int64, shape=(n,))
        cid_mm = np.memmap(p["code_ids"], mode="w+", dtype=np.int32, shape=(n,))

        chunk = int(self.cfg.get("MEMMAP_WRITE_CHUNK", 250_000) or 250_000)
        pos = 0
        for df in self._iter_sorted_parts(part_paths, columns=["code","date"] + feature_cols + [label_col]):
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values(["code","date"]).reset_index(drop=True)
            m = len(df)
            if m == 0:
                continue

            codes = df["code"].astype(str).values
            cids = np.fromiter((code_to_id[c] for c in codes), dtype=np.int32, count=m)
            dates_i8 = df["date"].values.astype("datetime64[ns]").astype(np.int64)
            feats = df[feature_cols].astype(np.float32).values
            labs = pd.to_numeric(df[label_col], errors="coerce").fillna(0.0).astype(np.float32).values

            feat_mm[pos:pos+m,:] = feats
            lab_mm[pos:pos+m] = labs
            date_mm[pos:pos+m] = dates_i8
            cid_mm[pos:pos+m] = cids
            pos += m

        feat_mm.flush(); lab_mm.flush(); date_mm.flush(); cid_mm.flush()
        meta = {
            "key": key,
            "n": int(pos),
            "fdim": fdim,
            "feature_cols": feature_cols,
            "label_col": label_col,
            "code_vocab": code_vocab,
        }
        atomic_save_json(meta, p["meta"])
        logger.info(f"[Memmap] built key={key} n={pos} fdim={fdim}")
        return meta

    def open(self, key: str) -> Tuple[Dict[str, Any], Dict[str, np.memmap]]:
        p = self.paths(key)
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

    def _iter_sorted_parts(self, part_paths: List[str], columns: List[str]):
        for p in part_paths:
            df = pd.read_parquet(p, columns=list(dict.fromkeys(columns)))
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date","code"]).sort_values(["code","date"]).reset_index(drop=True)
            yield df
