
# -*- coding: utf-8 -*-
"""
HuggingFace-style training entrypoint for date-grouped cross-sectional batching.

- Model init is *project-native* (PatchTSTForStock + SotaConfig from your codebase).
- Dataset can be either:
  1) loaded via `datasets.load_from_disk(--dataset_path)` (recommended for reproducibility), or
  2) built on-the-fly via your `get_dataset()` helper when --dataset_path is omitted.

Notes
- We keep `remove_unused_columns=False` so `date_id` will not be dropped.
- `per_device_*_batch_size=1` is a placeholder; real batch is controlled by DateGroupedBatchSampler.
"""
from __future__ import annotations

import importlib
import inspect
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from transformers import EarlyStoppingCallback, HfArgumentParser, TrainingArguments, set_seed

from src.config import Config
from src.core.trainer_quant import DateAwareTrainer
from src.data_provider.loader.collator_quant import StockDateCollator
from src.data_provider.loader.sampler_date_batch import DateBatchingConfig


# --------------------------
# Args
# --------------------------

@dataclass
class DataArgs:
    dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a DatasetDict saved by `datasets.DatasetDict.save_to_disk`."},
    )
    train_split: str = "train"
    eval_split: str = "validation"
    test_split: str = "test"
    require_keys: Tuple[str, ...] = ("past_values", "labels", "date_id")


@dataclass
class DateArgs:
    date_key: str = "date_id"
    batch_size_per_date: Optional[int] = None  # None => full cross-section per date as one batch
    shuffle_dates: bool = True
    shuffle_within_date: bool = False
    drop_last_in_date: bool = False


@dataclass
class RunArgs:
    resume_from_checkpoint: Optional[str] = None
    # if >0, enable early stopping and load_best_model_at_end
    early_stopping_patience: int = 0


# --------------------------
# Metrics (Spearman IC)
# --------------------------

def _spearman_ic(preds: np.ndarray, labs: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr  # type: ignore
        ic, _ = spearmanr(preds, labs)
        return float(ic) if np.isfinite(ic) else 0.0
    except Exception:
        # Fallback: rank-corr via pandas-style ranks
        p = preds.astype(np.float64, copy=False)
        y = labs.astype(np.float64, copy=False)
        # rank
        pr = p.argsort().argsort().astype(np.float64)
        yr = y.argsort().argsort().astype(np.float64)
        pr -= pr.mean()
        yr -= yr.mean()
        denom = (np.sqrt((pr * pr).sum()) * np.sqrt((yr * yr).sum())) + 1e-12
        return float((pr * yr).sum() / denom)


def compute_metrics(eval_pred) -> Dict[str, float]:
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    preds = np.asarray(predictions).reshape(-1)
    labs = np.asarray(labels).reshape(-1)
    preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
    labs = np.nan_to_num(labs, nan=0.0, posinf=0.0, neginf=0.0)
    return {"ic": _spearman_ic(preds, labs)}


# --------------------------
# Utilities
# --------------------------

def _infer_num_features(ds: DatasetDict, split: str) -> int:
    row = ds[split][0]
    pv = np.asarray(row["past_values"])
    if pv.ndim == 0:
        return 1
    if pv.ndim == 1:
        return 1
    return int(pv.shape[-1])


def _import_model_classes():
    """
    Import PatchTSTForStock + SotaConfig from your codebase without model_fn plumbing.
    Adjust the module list if you move files.
    """
    for mod_name in ("src.model", "src.models.model", "src.models.patchtst", "src.train.model"):
        try:
            m = importlib.import_module(mod_name)
        except Exception:
            continue
        if hasattr(m, "PatchTSTForStock") and hasattr(m, "SotaConfig"):
            return m.PatchTSTForStock, m.SotaConfig
    raise ImportError("Cannot import PatchTSTForStock/SotaConfig. "
                      "Expected in one of: src.model / src.models.model / src.models.patchtst / src.train.model")


def _load_or_build_dataset(data_args: DataArgs) -> DatasetDict:
    if data_args.dataset_path:
        ds = load_from_disk(data_args.dataset_path)
        if not isinstance(ds, DatasetDict):
            raise TypeError(f"load_from_disk returned {type(ds)}; expected DatasetDict.")
        return ds

    # fallback: build on-the-fly from your pipeline
    for mod_name, fn_name in (("src.data_provider", "get_dataset"), ("src.data_provider.data_provider", "get_dataset")):
        try:
            m = importlib.import_module(mod_name)
            fn = getattr(m, fn_name, None)
            if callable(fn):
                out = fn()
                if isinstance(out, tuple):
                    out = out[0]
                if not isinstance(out, DatasetDict):
                    raise TypeError(f"{mod_name}.{fn_name} returned {type(out)}; expected DatasetDict.")
                return out
        except Exception:
            continue
    raise ValueError("No --dataset_path provided and cannot import get_dataset(). "
                     "Provide --dataset_path or expose get_dataset in src.data_provider.")


def _assert_dataset_contract(ds: DatasetDict, data_args: DataArgs, date_key: str) -> None:
    for split_name in [data_args.train_split, data_args.eval_split]:
        if split_name not in ds:
            raise KeyError(f"DatasetDict missing split '{split_name}'. Available: {list(ds.keys())}")
        split = ds[split_name]
        cols = getattr(split, "column_names", [])
        missing = [k for k in data_args.require_keys if k not in cols]
        if missing:
            raise KeyError(f"Split '{split_name}' missing columns {missing}. Found: {cols}")
        if date_key not in cols:
            raise KeyError(f"Split '{split_name}' must contain '{date_key}' for date batching.")
        # __getitem__ contract quick check
        row0 = split[0]
        for k in data_args.require_keys:
            if k not in row0:
                raise KeyError(f"Split '{split_name}' row missing key '{k}'. Got keys: {list(row0.keys())}")


class _SmartCollator:
    """
    StockDateCollator returns {"past_values","labels","date_id"}.
    Most models won't accept date_id -> drop it unless forward supports it.
    """
    def __init__(self, base: StockDateCollator, model: torch.nn.Module, date_key: str = "date_id"):
        self.base = base
        self.date_key = date_key

        keep = True
        try:
            sig = inspect.signature(model.forward)  # type: ignore[attr-defined]
            ps = sig.parameters.values()
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in ps):
                keep = True
            else:
                keep = (date_key in sig.parameters)
        except Exception:
            keep = True
        self._keep_date = keep

    def __call__(self, features):
        batch = self.base(features)
        if not self._keep_date:
            batch.pop(self.date_key, None)
        return batch


def _maybe_set_defaults_from_config(args: TrainingArguments) -> None:
    # Keep this tiny: make CLI overrides win, but give good project defaults.
    args.output_dir = args.output_dir or Config.OUTPUT_DIR
    args.learning_rate = float(getattr(Config, "LR", args.learning_rate))
    args.num_train_epochs = float(getattr(Config, "EPOCHS", args.num_train_epochs))
    args.max_grad_norm = float(getattr(Config, "MAX_GRAD_NORM", args.max_grad_norm))

    # required for date batching
    args.remove_unused_columns = False

    # batch_size is controlled by batch_sampler, so keep placeholder=1
    args.per_device_train_batch_size = 1
    args.per_device_eval_batch_size = 1

    # dataloader perf knobs (DateAwareTrainer reads these)
    if getattr(args, "dataloader_num_workers", None) is None:
        args.dataloader_num_workers = 0
    if getattr(args, "dataloader_pin_memory", None) is None:
        args.dataloader_pin_memory = bool(torch.cuda.is_available())

    # keep logs readable
    if getattr(args, "logging_steps", 500) == 500:
        args.logging_steps = 50
    if getattr(args, "report_to", None) in (None, "all"):
        args.report_to = "none"


# --------------------------
# Main
# --------------------------

def main() -> None:
    parser = HfArgumentParser((DataArgs, DateArgs, RunArgs, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        data_args, date_args, run_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        data_args, date_args, run_args, training_args = parser.parse_args_into_dataclasses()

    _maybe_set_defaults_from_config(training_args)
    set_seed(int(getattr(training_args, "seed", getattr(Config, "SEED", 42)) or 42))

    ds = _load_or_build_dataset(data_args)
    _assert_dataset_contract(ds, data_args, date_args.date_key)
    num_features = _infer_num_features(ds, data_args.train_split)

    PatchTSTForStock, SotaConfig = _import_model_classes()
    model_config = SotaConfig(
        num_input_channels=int(num_features),
        context_length=int(getattr(Config, "CONTEXT_LEN", 64)),
        patch_length=int(getattr(Config, "PATCH_LEN", 8)),
        stride=int(getattr(Config, "STRIDE", 4)),
        d_model=int(getattr(Config, "D_MODEL", 128)),
        num_hidden_layers=int(getattr(Config, "NUM_HIDDEN_LAYERS", 3)),
        n_heads=int(getattr(Config, "N_HEADS", 4)),
        dropout=float(getattr(Config, "DROPOUT", 0.2)),
        mse_weight=float(getattr(Config, "MSE_WEIGHT", 0.1)),
        rank_weight=float(getattr(Config, "RANK_WEIGHT", 1.0)),
    )
    model = PatchTSTForStock(model_config)

    base_collator = StockDateCollator(assert_single_date=True)
    collator = _SmartCollator(base_collator, model, date_key=date_args.date_key)

    date_cfg = DateBatchingConfig(
        date_key=date_args.date_key,
        batch_size_per_date=date_args.batch_size_per_date,
        shuffle_dates=bool(date_args.shuffle_dates),
        shuffle_within_date=bool(date_args.shuffle_within_date),
        drop_last_in_date=bool(date_args.drop_last_in_date),
        seed=int(getattr(training_args, "seed", getattr(Config, "SEED", 42)) or 42),
    )

    callbacks = []
        # ensure eval happens during training
    if getattr(training_args, "evaluation_strategy", "no") == "no":
        training_args.evaluation_strategy = "steps"
        training_args.eval_steps = int(getattr(Config, "EVAL_STEPS", 500))
    if getattr(training_args, "save_strategy", "steps") == "no":
        training_args.save_strategy = str(training_args.evaluation_strategy)
    training_args.load_best_model_at_end = True
    training_args.metric_for_best_model = "ic"
    training_args.greater_is_better = True

    trainer = DateAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=ds[data_args.train_split],
        eval_dataset=ds[data_args.eval_split],
        data_collator=collator,
        date_batching=date_cfg,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=run_args.resume_from_checkpoint)
        trainer.save_model()

    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict and data_args.test_split in ds:
        preds = trainer.predict(ds[data_args.test_split])
        trainer.log_metrics("test", preds.metrics)
        trainer.save_metrics("test", preds.metrics)


if __name__ == "__main__":
    main()
