from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

from ..core.config import DPConfig
from ..rules.split_policy import SplitPolicy
from ..stores.memmap_store import MemmapStore
from ..stores.panel_store import PanelStore, PanelMeta

class DatasetPipeline:
    def __init__(self, cfg: DPConfig, logger):
        self.cfg = cfg
        self.logger = logger
        self.memmap = MemmapStore(cfg)

    def build_dataset(self, panel_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[DatasetDict, int]:
        panel_df = panel_df.sort_values(["code","date"]).reset_index(drop=True)
        panel_df["date"] = pd.to_datetime(panel_df["date"], errors="coerce")
        panel_df = panel_df.dropna(subset=["date","code"]).reset_index(drop=True)

        label_col = "rank_label" if "rank_label" in panel_df.columns else "target"
        seq_len = int(self.cfg.get("CONTEXT_LEN", 60) or 60)
        stride = int(self.cfg.get("STRIDE", 1) or 1)
        gap = int(self.cfg.get("SPLIT_GAP", seq_len) or seq_len)

        dates_ns = panel_df["date"].values.astype("datetime64[ns]").astype(np.int64)
        unique_dates = np.unique(dates_ns).astype("datetime64[ns]")
        split = SplitPolicy.compute(unique_dates, seq_len=seq_len, stride=stride, gap=gap,
                                    train_ratio=float(self.cfg.get("TRAIN_RATIO",0.7) or 0.7),
                                    val_ratio=float(self.cfg.get("VAL_RATIO",0.15) or 0.15),
                                    test_ratio=float(self.cfg.get("TEST_RATIO", None)) if self.cfg.get("TEST_RATIO", None) is not None else None).to_dict()

        train_end = np.datetime64(split["train_end_excl"])
        val_end = np.datetime64(split["val_end_excl"])

        backend = str(self.cfg.get("DATASET_BACKEND","hf_memmap") or "hf_memmap").lower().strip()
        if backend in ("hf_memmap","memmap"):
            # Build from in-memory df - ok for train subsets; for full market use build_dataset_from_parts
            fp = str(panel_df.attrs.get("fingerprint") or "nofp")
            universe_asof = str(panel_df.attrs.get("universe_asof") or "unknown")
            adj_norm = str(panel_df.attrs.get("adjust") or "unknown")
            mode = str(panel_df.attrs.get("mode") or "train")
            key = self.memmap.key(mode=mode, adj_norm=adj_norm, fp=fp, universe_asof=universe_asof, label_col=label_col, feature_cols=feature_cols)
            if not self.memmap.exists(key) or bool(self.cfg.get("FORCE_DATASET_REBUILD", False)):
                # write temp single part to reuse part-stream builder
                raise RuntimeError("Use build_dataset_from_parts for memmap in v24 (avoid big in-memory).")
            meta, mms = self.memmap.open(key)
            return self._build_hf_from_memmap(meta, mms, seq_len=seq_len, stride=stride, train_end=train_end, val_end=val_end), int(meta["fdim"])

        # fallback in-RAM
        return self._build_hf_from_ram(panel_df, feature_cols, label_col, seq_len, stride, train_end, val_end), len(feature_cols)

    def build_dataset_from_parts(self, parts_dir: str) -> Tuple[DatasetDict, int]:
        store = PanelStore(self.cfg)
        meta: PanelMeta = store.read_meta(parts_dir)
        feature_cols = meta.feature_cols
        label_col = "rank_label"  # prefer
        # decide label col by probing first part
        probe = pd.read_parquet(meta.part_paths[0], nrows=5)
        if "rank_label" in probe.columns:
            label_col = "rank_label"
        else:
            label_col = "target"

        seq_len = int(self.cfg.get("CONTEXT_LEN", 60) or 60)
        stride = int(self.cfg.get("STRIDE", 1) or 1)
        gap = int(self.cfg.get("SPLIT_GAP", seq_len) or seq_len)

        # Split manifest already computed and persisted; consume it (single source of truth).
        if meta.split_manifest:
            split = meta.split_manifest
        else:
            # fallback: compute from dates in parts
            dates = []
            for df in store.iter_parts(meta.part_paths, columns=["date"]):
                d = pd.to_datetime(df["date"], errors="coerce").dropna().unique()
                if len(d): dates.append(d)
            unique_dates = np.unique(np.concatenate(dates)) if dates else np.array([], dtype="datetime64[ns]")
            split = SplitPolicy.compute(unique_dates, seq_len=seq_len, stride=stride, gap=gap,
                                        train_ratio=float(self.cfg.get("TRAIN_RATIO",0.7) or 0.7),
                                        val_ratio=float(self.cfg.get("VAL_RATIO",0.15) or 0.15),
                                        test_ratio=float(self.cfg.get("TEST_RATIO", None)) if self.cfg.get("TEST_RATIO", None) is not None else None).to_dict()

        train_end = np.datetime64(split["train_end_excl"])
        val_end = np.datetime64(split["val_end_excl"])

        key = self.memmap.key(mode=meta.mode, adj_norm=meta.adjust, fp=meta.fingerprint, universe_asof=meta.universe_asof, label_col=label_col, feature_cols=feature_cols)
        if not self.memmap.exists(key) or bool(self.cfg.get("FORCE_DATASET_REBUILD", False)):
            self.logger.info(f"🧠 Building memmap dataset store from parts: key={key}")
            self.memmap.build_from_parts(part_paths=meta.part_paths, feature_cols=feature_cols, label_col=label_col, key=key, logger=self.logger)

        mm_meta, mms = self.memmap.open(key)
        ds = self._build_hf_from_memmap(mm_meta, mms, seq_len=seq_len, stride=stride, train_end=train_end, val_end=val_end)
        return ds, int(mm_meta["fdim"])

    def _build_hf_from_memmap(self, meta: Dict[str, Any], mms: Dict[str, np.memmap], *, seq_len: int, stride: int, train_end: np.datetime64, val_end: np.datetime64) -> DatasetDict:
        feats = mms["features"]; labels = mms["labels"]; code_ids = mms["code_ids"]; dates = mms["dates"]
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
            valid_starts.extend(range(st, last+1, stride))
        valid_starts = np.array(valid_starts, dtype=np.int64)
        pred_dates = dates[valid_starts + (seq_len - 1)].astype("datetime64[ns]")
        idx_train = valid_starts[pred_dates <= train_end]
        idx_valid = valid_starts[(pred_dates > train_end) & (pred_dates <= val_end)]
        idx_test = valid_starts[pred_dates > val_end]

        def transform(batch: Dict[str, List[int]]) -> Dict[str, Any]:
            s_list = batch["start_idx"]
            past_values = []
            y = []
            for s in s_list:
                s = int(s); e = s + seq_len
                past_values.append(np.asarray(feats[s:e, :]))
                y.append(float(labels[e-1]))
            return {"past_values": past_values, "labels": y}

        ds = DatasetDict({
            "train": Dataset.from_dict({"start_idx": idx_train.tolist()}),
            "validation": Dataset.from_dict({"start_idx": idx_valid.tolist()}),
            "test": Dataset.from_dict({"start_idx": idx_test.tolist()}),
        })
        ds.set_transform(transform)
        return ds

    def _build_hf_from_ram(self, panel_df: pd.DataFrame, feature_cols: List[str], label_col: str, seq_len: int, stride: int, train_end: np.datetime64, val_end: np.datetime64) -> Tuple[DatasetDict, int]:
        feature_matrix = np.ascontiguousarray(panel_df[feature_cols].values.astype(np.float32))
        target_array = pd.to_numeric(panel_df[label_col], errors="coerce").fillna(0.0).values.astype(np.float32)
        codes = panel_df["code"].astype(str).values
        dates = panel_df["date"].values.astype("datetime64[ns]")

        changes = np.where(codes[:-1] != codes[1:])[0] + 1
        start_idx = np.concatenate(([0], changes))
        end_idx = np.concatenate((changes, [len(codes)]))

        valid_starts = []
        for st, ed in zip(start_idx, end_idx):
            ln = ed - st
            if ln < seq_len: continue
            last = ed - seq_len
            valid_starts.extend(range(st, last+1, stride))
        valid_starts = np.array(valid_starts, dtype=np.int64)
        pred_dates = dates[valid_starts + (seq_len - 1)]
        idx_train = valid_starts[pred_dates <= train_end]
        idx_valid = valid_starts[(pred_dates > train_end) & (pred_dates <= val_end)]
        idx_test = valid_starts[pred_dates > val_end]

        def transform(batch: Dict[str, List[int]]) -> Dict[str, Any]:
            s_list = batch["start_idx"]
            past_values = []
            y = []
            for s in s_list:
                s = int(s); e = s + seq_len
                past_values.append(feature_matrix[s:e])
                y.append(float(target_array[e-1]))
            return {"past_values": past_values, "labels": y}

        ds = DatasetDict({
            "train": Dataset.from_dict({"start_idx": idx_train.tolist()}),
            "validation": Dataset.from_dict({"start_idx": idx_valid.tolist()}),
            "test": Dataset.from_dict({"start_idx": idx_test.tolist()}),
        })
        ds.set_transform(transform)
        return ds, len(feature_cols)
