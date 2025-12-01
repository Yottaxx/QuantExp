# src/train/date_batching.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler

from transformers import Trainer


def _dist_info() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _as_np_int64(x) -> np.ndarray:
    # HuggingFace Dataset column: list[int] / np.ndarray / pyarrow scalar list
    arr = np.asarray(x)
    if arr.dtype == object:
        arr = arr.astype(np.int64)
    return arr.astype(np.int64, copy=False)


def _group_indices_by_date(date_ids: Sequence[int]) -> tuple[np.ndarray, List[np.ndarray]]:
    """
    Return:
      unique_dates (sorted),
      groups: list of np.ndarray indices for each unique date
    """
    date_ids = _as_np_int64(date_ids)
    if date_ids.size == 0:
        return np.array([], dtype=np.int64), []

    order = np.argsort(date_ids, kind="mergesort")
    d_sorted = date_ids[order]
    uniq, starts = np.unique(d_sorted, return_index=True)
    ends = np.concatenate([starts[1:], np.array([len(d_sorted)], dtype=np.int64)])

    groups = [order[st:ed] for st, ed in zip(starts, ends)]
    return uniq.astype(np.int64, copy=False), groups


@dataclass
class DateBatchingConfig:
    date_key: str = "date_id"
    # None => 每日一个 batch（全横截面）；设置为 int => 每日按 batch_size 切块（可能更省显存但排序信号变弱）
    batch_size_per_date: Optional[int] = None
    shuffle_dates: bool = True
    shuffle_within_date: bool = False
    drop_last_in_date: bool = False  # 仅对“切块”生效
    seed: int = 42


class DateGroupedBatchSampler(Sampler[List[int]]):
    """
    Yield batches where each batch contains samples from the same date_id.

    DDP behavior:
      - build all batches globally (after shuffling dates),
      - shard by batches (rank::world_size),
      - pad/truncate so every rank has identical number of batches -> no DDP hang.
    """

    def __init__(self, dataset, cfg: DateBatchingConfig):
        self.dataset = dataset
        self.cfg = cfg
        self.epoch = 0

        # NOTE: 依赖 dataset 存在列 date_id
        if not hasattr(dataset, "__getitem__"):
            raise TypeError("dataset must be indexable")
        if not hasattr(dataset, "column_names") and not isinstance(dataset, dict):
            # torch dataset fallback not supported here
            raise TypeError("This sampler expects a HuggingFace Dataset split with a 'date_id' column.")

        date_ids = dataset[cfg.date_key]  # HF Dataset fast column access
        self._uniq_dates, self._groups = _group_indices_by_date(date_ids)

        self._rank, self._world = _dist_info()
        self._len = self._compute_len()  # epoch-independent upper bound (DDP padded)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self) -> np.random.Generator:
        # all ranks use same rng for global ordering, then shard deterministically
        return np.random.default_rng(self.cfg.seed + self.epoch)

    def _build_all_batches(self) -> List[List[int]]:
        rng = self._rng()
        date_order = np.arange(len(self._uniq_dates), dtype=np.int64)

        if self.cfg.shuffle_dates and len(date_order) > 1:
            rng.shuffle(date_order)

        all_batches: List[List[int]] = []
        bs = self.cfg.batch_size_per_date

        for di in date_order:
            idxs = self._groups[int(di)]
            if idxs.size == 0:
                continue

            if self.cfg.shuffle_within_date and idxs.size > 1:
                idxs = idxs.copy()
                rng.shuffle(idxs)

            if bs is None or bs <= 0 or bs >= idxs.size:
                all_batches.append(idxs.astype(np.int64, copy=False).tolist())
            else:
                # chunk within date
                n = idxs.size
                full = (n // bs) * bs
                if full > 0:
                    for s in range(0, full, bs):
                        all_batches.append(idxs[s : s + bs].tolist())
                if not self.cfg.drop_last_in_date and full < n:
                    all_batches.append(idxs[full:n].tolist())

        return all_batches

    def _shard_and_pad(self, all_batches: List[List[int]]) -> List[List[int]]:
        if self._world <= 1:
            self._len = len(all_batches)
            return all_batches

        if len(all_batches) == 0:
            self._len = 0
            return []

        # shard by batches
        shard = all_batches[self._rank :: self._world]
        # pad to equal length per rank
        per_rank = int(math.ceil(len(all_batches) / self._world))

        if len(shard) == 0:
            # rare: more ranks than batches
            shard = [all_batches[self._rank % len(all_batches)]]

        # deterministic padding
        i = 0
        while len(shard) < per_rank:
            shard.append(shard[i % len(shard)])
            i += 1
        if len(shard) > per_rank:
            shard = shard[:per_rank]

        self._len = per_rank
        return shard

    def _compute_len(self) -> int:
        # epoch 0 estimate; real len fixed each epoch after sharding/padding
        all_batches = self._build_all_batches()
        if self._world <= 1:
            return len(all_batches)
        return int(math.ceil(len(all_batches) / self._world))

    def __iter__(self) -> Iterator[List[int]]:
        all_batches = self._build_all_batches()
        shard = self._shard_and_pad(all_batches)
        yield from shard

    def __len__(self) -> int:
        return int(self._len)
