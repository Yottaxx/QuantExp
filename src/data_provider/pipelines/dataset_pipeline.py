from __future__ import annotations

import json
import shutil
import time
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
import os


from ..core.config import DPConfig
from ..rules.split_policy import SplitPolicy
from ..stores.memmap_store import MemmapStore
from ..stores.panel_store import PanelStore, PanelMeta


class DatasetPipeline:
    def __init__(self, cfg: DPConfig, logger):
        self.cfg = cfg
        self.logger = logger
        self.memmap = MemmapStore(cfg)

    def save_to_disk_atomic(
        self,
        ds: DatasetDict,
        out_dir: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save DatasetDict to disk in an atomic-ish way:
        write to sibling tmp dir then rename into place.
        """
        parent = os.path.dirname(out_dir.rstrip("/\\"))
        os.makedirs(parent, exist_ok=True)

        tmp = f"{out_dir}.tmp_{os.getpid()}_{int(time.time())}"
        if os.path.exists(tmp):
            shutil.rmtree(tmp, ignore_errors=True)

        ds.save_to_disk(tmp)

        if meta is not None:
            with open(os.path.join(tmp, "dataset_meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        if os.path.exists(out_dir):
            shutil.rmtree(out_dir, ignore_errors=True)

        os.rename(tmp, out_dir)  # same parent => same filesystem => rename is atomic on POSIX
        self.logger.info(f"[Dataset] saved to {out_dir}")
        return out_dir

    # -------------------------
    # helpers: date_id convert
    # -------------------------
    @staticmethod
    def _ns_to_yyyymmdd_int(ns_i8: np.ndarray) -> np.ndarray:
        """
        ns_i8: int64 array of unix-ns timestamps
        returns: int64 array of YYYYMMDD; NaT -> 0
        """
        ns_i8 = np.asarray(ns_i8, dtype=np.int64)
        dt = pd.to_datetime(ns_i8, unit="ns", errors="coerce")
        # vectorized format -> numeric
        out = pd.to_numeric(dt.strftime("%Y%m%d"), errors="coerce").fillna(0).astype(np.int64).to_numpy()
        return out

    @staticmethod
    def _dt64_to_yyyymmdd_int(dt64: np.datetime64) -> int:
        """
        np.datetime64 -> YYYYMMDD int (end_excl boundary)
        """
        return int(pd.Timestamp(dt64).strftime("%Y%m%d"))

    # -------------------------
    # public APIs
    # -------------------------
    def build_dataset(self, panel_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[DatasetDict, int]:
        panel_df = panel_df.sort_values(["code", "date"]).reset_index(drop=True)
        panel_df["date"] = pd.to_datetime(panel_df["date"], errors="coerce")
        panel_df = panel_df.dropna(subset=["date", "code"]).reset_index(drop=True)

        label_col = "rank_label" if "rank_label" in panel_df.columns else "target"
        seq_len = int(self.cfg.get("CONTEXT_LEN", 60) or 60)
        stride = int(self.cfg.get("STRIDE", 1) or 1)
        gap = int(self.cfg.get("SPLIT_GAP", seq_len) or seq_len)

        dates_ns = panel_df["date"].values.astype("datetime64[ns]").astype(np.int64)
        unique_dates = np.unique(dates_ns).astype("datetime64[ns]")

        split = SplitPolicy.compute(
            unique_dates,
            seq_len=seq_len,
            stride=stride,
            gap=gap,
            train_ratio=float(self.cfg.get("TRAIN_RATIO", 0.7) or 0.7),
            val_ratio=float(self.cfg.get("VAL_RATIO", 0.15) or 0.15),
            test_ratio=float(self.cfg.get("TEST_RATIO", None)) if self.cfg.get("TEST_RATIO", None) is not None else None,
        ).to_dict()

        train_end = np.datetime64(split["train_end_excl"])
        val_end = np.datetime64(split["val_end_excl"])

        backend = str(self.cfg.get("DATASET_BACKEND", "hf_memmap") or "hf_memmap").lower().strip()
        if backend in ("hf_memmap", "memmap"):
            # 与你原实现一致：in-memory 不负责 build memmap
            fp = str(panel_df.attrs.get("fingerprint") or "nofp")
            universe_asof = str(panel_df.attrs.get("universe_asof") or "unknown")
            adj_norm = str(panel_df.attrs.get("adjust") or "unknown")
            mode = str(panel_df.attrs.get("mode") or "train")
            key = self.memmap.key(
                mode=mode,
                adj_norm=adj_norm,
                fp=fp,
                universe_asof=universe_asof,
                label_col=label_col,
                feature_cols=feature_cols,
            )
            if not self.memmap.exists(key) or bool(self.cfg.get("FORCE_DATASET_REBUILD", False)):
                raise RuntimeError("Use build_dataset_from_parts for memmap in v24 (avoid big in-memory).")
            meta, mms = self.memmap.open(key)
            ds = self._build_hf_from_memmap(meta, mms, seq_len=seq_len, stride=stride, train_end=train_end, val_end=val_end)
            return ds, int(meta["fdim"])

        # fallback: RAM
        ds, fdim = self._build_hf_from_ram(panel_df, feature_cols, label_col, seq_len, stride, train_end, val_end)
        return ds, fdim

    def build_dataset_from_parts(self, parts_dir: str) -> Tuple[DatasetDict, int]:
        store = PanelStore(self.cfg)
        meta: PanelMeta = store.read_meta(parts_dir)
        feature_cols = meta.feature_cols

        # decide label col by probing
        probe = pd.read_parquet(meta.part_paths[0], nrows=5)
        label_col = "rank_label" if "rank_label" in probe.columns else "target"

        seq_len = int(self.cfg.get("CONTEXT_LEN", 60) or 60)
        stride = int(self.cfg.get("STRIDE", 1) or 1)
        gap = int(self.cfg.get("SPLIT_GAP", seq_len) or seq_len)

        # single source of truth: split_manifest
        if meta.split_manifest:
            split = meta.split_manifest
        else:
            # fallback compute
            dates = []
            for df in store.iter_parts(meta.part_paths, columns=["date"]):
                d = pd.to_datetime(df["date"], errors="coerce").dropna().unique()
                if len(d):
                    dates.append(d)
            unique_dates = np.unique(np.concatenate(dates)) if dates else np.array([], dtype="datetime64[ns]")
            split = SplitPolicy.compute(
                unique_dates,
                seq_len=seq_len,
                stride=stride,
                gap=gap,
                train_ratio=float(self.cfg.get("TRAIN_RATIO", 0.7) or 0.7),
                val_ratio=float(self.cfg.get("VAL_RATIO", 0.15) or 0.15),
                test_ratio=float(self.cfg.get("TEST_RATIO", None)) if self.cfg.get("TEST_RATIO", None) is not None else None,
            ).to_dict()

        train_end = np.datetime64(split["train_end_excl"])
        val_end = np.datetime64(split["val_end_excl"])

        key = self.memmap.key(
            mode=meta.mode,
            adj_norm=meta.adjust,
            fp=meta.fingerprint,
            universe_asof=meta.universe_asof,
            label_col=label_col,
            feature_cols=feature_cols,
        )

        if not self.memmap.exists(key) or bool(self.cfg.get("FORCE_DATASET_REBUILD", False)):
            self.logger.info(f"🧠 Building memmap dataset store from parts: key={key}")
            self.memmap.build_from_parts(
                part_paths=meta.part_paths,
                feature_cols=feature_cols,
                label_col=label_col,
                key=key,
                logger=self.logger,
            )

        mm_meta, mms = self.memmap.open(key)
        ds = self._build_hf_from_memmap(mm_meta, mms, seq_len=seq_len, stride=stride, train_end=train_end, val_end=val_end)
        return ds, int(mm_meta["fdim"])

    # -------------------------
    # builders
    # -------------------------
    def _build_hf_from_memmap(
        self,
        meta: Dict[str, Any],
        mms: Dict[str, np.memmap],
        *,
        seq_len: int,
        stride: int,
        train_end: np.datetime64,
        val_end: np.datetime64,
    ) -> DatasetDict:
        feats = mms["features"]
        labels = mms["labels"]
        code_ids = mms["code_ids"]
        dates_i8 = mms["dates"]
        date_ids = mms.get("date_ids", None)  # optional row-level YYYYMMDD int32

        n = int(meta["n"])
        changes = np.flatnonzero(code_ids[:-1] != code_ids[1:]) + 1
        start_idx = np.concatenate(([0], changes)).astype(np.int64)
        end_idx = np.concatenate((changes, [n])).astype(np.int64)

        # valid window starts per code
        valid_starts: List[int] = []
        for st, ed in zip(start_idx.tolist(), end_idx.tolist()):
            ln = ed - st
            if ln < seq_len:
                continue
            last = ed - seq_len
            valid_starts.extend(range(st, last + 1, stride))
        valid_starts = np.asarray(valid_starts, dtype=np.int64)

        asof_pos = valid_starts + (seq_len - 1)

        # window end (as-of) date_id
        if date_ids is not None:
            pred_date_id = np.asarray(date_ids[asof_pos], dtype=np.int64)
        else:
            pred_ns = np.asarray(dates_i8[asof_pos], dtype=np.int64)
            pred_date_id = self._ns_to_yyyymmdd_int(pred_ns)

        # end_excl boundaries => YYYYMMDD
        train_end_id = self._dt64_to_yyyymmdd_int(train_end)
        val_end_id = self._dt64_to_yyyymmdd_int(val_end)

        m_train = pred_date_id < train_end_id
        m_val = (pred_date_id >= train_end_id) & (pred_date_id < val_end_id)
        m_test = pred_date_id >= val_end_id

        idx_train = valid_starts[m_train]
        did_train = pred_date_id[m_train]
        idx_valid = valid_starts[m_val]
        did_valid = pred_date_id[m_val]
        idx_test = valid_starts[m_test]
        did_test = pred_date_id[m_test]

        def transform(batch: Dict[str, List[int]]) -> Dict[str, Any]:
            s_list = batch["start_idx"]
            pv, y = [], []
            for s in s_list:
                s = int(s)
                e = s + seq_len
                pv.append(np.asarray(feats[s:e, :], dtype=np.float32))
                y.append(float(labels[e - 1]))
            # 直接透传 date_id 给 sampler/collator/trainer
            return {"past_values": pv, "labels": y, "date_id": batch["date_id"]}

        ds = DatasetDict(
            {
                "train": Dataset.from_dict({"start_idx": idx_train, "date_id": did_train}),
                "validation": Dataset.from_dict({"start_idx": idx_valid, "date_id": did_valid}),
                "test": Dataset.from_dict({"start_idx": idx_test, "date_id": did_test}),
            }
        )
        ds.set_transform(transform)
        return ds

    def _build_hf_from_ram(
        self,
        panel_df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str,
        seq_len: int,
        stride: int,
        train_end: np.datetime64,
        val_end: np.datetime64,
    ) -> Tuple[DatasetDict, int]:
        panel_df = panel_df.sort_values(["code", "date"]).reset_index(drop=True)
        panel_df["date"] = pd.to_datetime(panel_df["date"], errors="coerce")
        panel_df = panel_df.dropna(subset=["date", "code"]).reset_index(drop=True)

        feats = np.ascontiguousarray(panel_df[feature_cols].astype(np.float32).values)
        labels = pd.to_numeric(panel_df[label_col], errors="coerce").fillna(0.0).astype(np.float32).values
        codes = panel_df["code"].astype(str).values
        dates_ns = panel_df["date"].values.astype("datetime64[ns]").astype(np.int64)

        # per-code segments
        changes = np.where(codes[:-1] != codes[1:])[0] + 1
        start_idx = np.concatenate(([0], changes)).astype(np.int64)
        end_idx = np.concatenate((changes, [len(codes)])).astype(np.int64)

        valid_starts: List[int] = []
        for st, ed in zip(start_idx.tolist(), end_idx.tolist()):
            ln = ed - st
            if ln < seq_len:
                continue
            last = ed - seq_len
            valid_starts.extend(range(st, last + 1, stride))
        valid_starts = np.asarray(valid_starts, dtype=np.int64)

        asof_pos = valid_starts + (seq_len - 1)
        pred_date_id = self._ns_to_yyyymmdd_int(dates_ns[asof_pos])

        train_end_id = self._dt64_to_yyyymmdd_int(train_end)
        val_end_id = self._dt64_to_yyyymmdd_int(val_end)

        m_train = pred_date_id < train_end_id
        m_val = (pred_date_id >= train_end_id) & (pred_date_id < val_end_id)
        m_test = pred_date_id >= val_end_id

        idx_train = valid_starts[m_train]
        did_train = pred_date_id[m_train]
        idx_valid = valid_starts[m_val]
        did_valid = pred_date_id[m_val]
        idx_test = valid_starts[m_test]
        did_test = pred_date_id[m_test]

        def transform(batch: Dict[str, List[int]]) -> Dict[str, Any]:
            s_list = batch["start_idx"]
            pv, y = [], []
            for s in s_list:
                s = int(s)
                e = s + seq_len
                pv.append(np.asarray(feats[s:e, :], dtype=np.float32))
                y.append(float(labels[e - 1]))
            return {"past_values": pv, "labels": y, "date_id": batch["date_id"]}

        ds = DatasetDict(
            {
                "train": Dataset.from_dict({"start_idx": idx_train, "date_id": did_train}),
                "validation": Dataset.from_dict({"start_idx": idx_valid, "date_id": did_valid}),
                "test": Dataset.from_dict({"start_idx": idx_test, "date_id": did_test}),
            }
        )
        ds.set_transform(transform)
        return ds, len(feature_cols)
