# src/train/date_batching.py (continue)
from typing import Optional

from torch.utils.data import DataLoader
from transformers import Trainer

from src.data_provider.loader.sampler_date_batch import DateBatchingConfig, DateGroupedBatchSampler


class _DateBatchingEpochCallback:
    """Call batch_sampler.set_epoch(epoch) on epoch begin (works for batch_sampler, not just sampler)."""
    def on_epoch_begin(self, args, state, control, **kwargs):
        dl = kwargs.get("train_dataloader", None)
        if dl is None:
            return
        bs = getattr(dl, "batch_sampler", None)
        if bs is not None and hasattr(bs, "set_epoch") and state.epoch is not None:
            bs.set_epoch(int(state.epoch))


class DateAwareTrainer(Trainer):
    """
    A Trainer that:
      - keeps date_id column (avoid remove_unused_columns killing it)
      - uses date-grouped batch_sampler for train/eval/test
      - supports DDP sharding by date (actually at batch-level after date grouping)
    """

    def __init__(
        self,
        *args,
        date_batching: Optional[DateBatchingConfig] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.date_batching = date_batching or DateBatchingConfig()
        # 强制保留 date_id，否则 sampler 会炸
        if getattr(self.args, "remove_unused_columns", False):
            self.args.remove_unused_columns = False
        self.add_callback(_DateBatchingEpochCallback())

    def _make_loader(self, dataset, *, train: bool) -> DataLoader:
        cfg = self.date_batching
        if train:
            # train shuffle by default
            cfg = DateBatchingConfig(**{**cfg.__dict__, "shuffle_dates": cfg.shuffle_dates})
        else:
            # eval/test 默认不 shuffle，保证可复现
            cfg = DateBatchingConfig(**{**cfg.__dict__, "shuffle_dates": False, "shuffle_within_date": False})

        batch_sampler = DateGroupedBatchSampler(dataset, cfg)

        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=int(getattr(self.args, "dataloader_num_workers", 0) or 0),
            pin_memory=bool(getattr(self.args, "dataloader_pin_memory", True)),
            persistent_workers=bool(getattr(self.args, "dataloader_num_workers", 0)) > 0,
            prefetch_factor=2 if int(getattr(self.args, "dataloader_num_workers", 0) or 0) > 0 else None,
        )

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: train_dataset is None.")
        dl = self._make_loader(self.train_dataset, train=True)
        # accelerate / ddp prepare
        if hasattr(self, "accelerator") and self.accelerator is not None:
            dl = self.accelerator.prepare(dl)
        return dl

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: eval_dataset is None.")
        dl = self._make_loader(eval_dataset, train=False)
        if hasattr(self, "accelerator") and self.accelerator is not None:
            dl = self.accelerator.prepare(dl)
        return dl

    def get_test_dataloader(self, test_dataset) -> DataLoader:
        if test_dataset is None:
            raise ValueError("Trainer: test_dataset is None.")
        dl = self._make_loader(test_dataset, train=False)
        if hasattr(self, "accelerator") and self.accelerator is not None:
            dl = self.accelerator.prepare(dl)
        return dl
