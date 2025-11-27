# main.py
from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import Any, Dict, Optional

from utils.utils_func import (
    apply_config_overrides,
    debug_print_config,
    ensure_dirs,
    parse_codes_arg,
    parse_kv_pairs,
    patch_dataprovider_defaults,
    setup_debug_mode,
)


def _import_src(name: str):
    """
    Import src module lazily. Requires src/ be a package (best: src/__init__.py).
    """
    return importlib.import_module(f"src.{name}")


def _set_seed(seed: int) -> None:
    """
    Prefer utils/seed_utils.py if exists, else fallback minimal.
    """
    try:
        seed_mod = importlib.import_module("utils.seed_utils")
        if hasattr(seed_mod, "set_global_seed"):
            seed_mod.set_global_seed(seed)  # type: ignore
            return
    except Exception:
        pass

    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Quant Platform (root main.py, uses src/ + utils/)")
    p.add_argument("--debug", action="store_true", help="Debug mode (single worker, serial backend hints)")
    p.add_argument("--print-config", action="store_true", help="Print effective config before running")
    p.add_argument("--strict-config", action="store_true", help="Unknown --set KEY raises error")
    p.add_argument("--set", action="append", default=[], help="Override Config: KEY=VAL (repeatable)")
    p.add_argument("--seed", type=int, default=None, help="Override Config.SEED (or use existing)")

    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("config", help="Print config and exit")
    sub.add_parser("download", help="Run DataProvider.download_data()")

    panel = sub.add_parser("panel", help="Build panel cache (load_and_process_panel)")
    panel.add_argument("--mode", default="train", choices=["train", "predict"])
    panel.add_argument("--adjust", default="qfq")
    panel.add_argument("--force-refresh", action="store_true")

    sub.add_parser("train", help="Train model (run_training)")

    pred = sub.add_parser("predict", help="Run inference (run_inference)")
    pred.add_argument("--top-k", type=int, default=None)
    pred.add_argument("--min-score", type=float, default=None)

    bt = sub.add_parser("backtest", help="Backtest (run_single_backtest or inference->run_backtest)")
    bt.add_argument("--codes", default="", help="Comma codes. Empty => use inference picks")
    bt.add_argument("--with-fees", action="store_true", help="Enable A-share fees (commission+stamp duty)")
    bt.add_argument("--initial-cash", type=float, default=1_000_000.0)
    bt.add_argument("--top-k", type=int, default=None)
    bt.add_argument("--min-score", type=float, default=None)

    wf = sub.add_parser("walkforward", help="Walk-forward backtest (run_walk_forward_backtest)")
    wf.add_argument("--start", required=True, help="YYYY-MM-DD")
    wf.add_argument("--end", required=True, help="YYYY-MM-DD")
    wf.add_argument("--initial-cash", type=float, default=1_000_000.0)
    wf.add_argument("--top-k", type=int, default=None)

    ana = sub.add_parser("analysis", help="Analysis report (BacktestAnalyzer)")
    ana.add_argument("--target-set", default="test", choices=["test", "validation", "val", "eval", "train", "custom"])
    ana.add_argument("--start", default=None)
    ana.add_argument("--end", default=None)

    fe = sub.add_parser("factor-eval", help="Factor evaluation (AlphaEvaluator)")
    fe.add_argument("--mode", default="train", choices=["train", "predict"])
    fe.add_argument("--adjust", default="qfq")
    fe.add_argument("--force-refresh", action="store_true")

    return p


def apply_runtime_config(args: argparse.Namespace) -> Any:
    # Import Config first, override before importing modules that bind Config at import-time (e.g. backtest Strategy.params)
    cfg_mod = _import_src("config")
    Config = cfg_mod.Config

    # CLI overrides
    overrides: Dict[str, Any] = parse_kv_pairs(args.set)
    apply_config_overrides(Config, overrides, strict=args.strict_config)

    # debug -> enforce debug profile
    if args.debug:
        setup_debug_mode(Config)

    # seed
    seed = args.seed if args.seed is not None else getattr(Config, "SEED", 42)
    _set_seed(int(seed))

    # ensure dirs after overrides
    ensure_dirs(Config)

    if args.print_config or args.cmd == "config" or args.debug:
        debug_print_config(Config)
        dp_mod = _import_src("data_provider")
        dp_mod.DataProvider.debug_print_config()

    return Config


def cmd_download() -> None:
    dp_mod = _import_src("data_provider")
    dp_mod.DataProvider.download_data()


def cmd_panel(mode: str, adjust: str, force_refresh: bool) -> None:
    dp_mod = _import_src("data_provider")
    dp_mod.DataProvider.load_and_process_panel(mode=mode, adjust=adjust, force_refresh=force_refresh)


def cmd_train() -> None:
    tr_mod = _import_src("train")
    tr_mod.run_training()


def cmd_predict(top_k: Optional[int], min_score: Optional[float]) -> list:
    cfg_mod = _import_src("config")
    Config = cfg_mod.Config

    inf_mod = _import_src("inference")
    k = int(top_k if top_k is not None else Config.TOP_K)
    thr = float(min_score if min_score is not None else Config.MIN_SCORE_THRESHOLD)
    return inf_mod.run_inference(top_k=k, min_score_threshold=thr)


def cmd_backtest(
    codes: str,
    with_fees: bool,
    initial_cash: float,
    top_k: Optional[int],
    min_score: Optional[float],
) -> None:
    cfg_mod = _import_src("config")
    Config = cfg_mod.Config
    bt_mod = _import_src("backtest")

    code_list = parse_codes_arg(codes)
    if not code_list:
        picks = cmd_predict(top_k=top_k, min_score=min_score)
        # picks: [(code, score, pe), ...]
        code_list = [c for (c, *_rest) in picks]

    k = int(top_k if top_k is not None else Config.TOP_K)
    bt_mod.run_single_backtest(code_list, with_fees=with_fees, initial_cash=initial_cash, top_k=k)


def cmd_walkforward(start: str, end: str, initial_cash: float, top_k: Optional[int]) -> None:
    cfg_mod = _import_src("config")
    Config = cfg_mod.Config
    bt_mod = _import_src("backtest")

    k = int(top_k if top_k is not None else Config.TOP_K)
    bt_mod.run_walk_forward_backtest(start, end, initial_cash, top_k=k)


def cmd_analysis(target_set: str, start: Optional[str], end: Optional[str]) -> None:
    an_mod = _import_src("analysis")
    analyzer = an_mod.BacktestAnalyzer(target_set=target_set, start_date=start, end_date=end)
    analyzer.generate_historical_predictions()
    analyzer.analyze_performance()


def cmd_factor_eval(mode: str, adjust: str, force_refresh: bool) -> None:
    dp_mod = _import_src("data_provider")
    ev_mod = _import_src("evaluator")

    panel_df, feat_cols = dp_mod.DataProvider.load_and_process_panel(mode=mode, adjust=adjust, force_refresh=force_refresh)
    valid = ev_mod.AlphaEvaluator.evaluate(panel_df, feat_cols, target_col="target")
    print(f"✅ Valid factors: {len(valid)}/{len(feat_cols)}")


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    Config = apply_runtime_config(args)

    # Optional: patch DataProvider defaults so inference/backtest/analysis internal calls
    # DataProvider.load_and_process_panel(mode='predict') also respects CLI adjust/force.
    dp_mod = _import_src("data_provider")
    patch_dataprovider_defaults(
        dp_mod.DataProvider,
        adjust=getattr(args, "adjust", None),  # only panel/factor-eval provides adjust; others not
        force_refresh=getattr(args, "force_refresh", None),
    )

    if args.cmd == "config":
        return
    if args.cmd == "download":
        cmd_download()
        return
    if args.cmd == "panel":
        cmd_panel(args.mode, args.adjust, args.force_refresh)
        return
    if args.cmd == "train":
        cmd_train()
        return
    if args.cmd == "predict":
        cmd_predict(args.top_k, args.min_score)
        return
    if args.cmd == "backtest":
        cmd_backtest(args.codes, args.with_fees, args.initial_cash, args.top_k, args.min_score)
        return
    if args.cmd == "walkforward":
        cmd_walkforward(args.start, args.end, args.initial_cash, args.top_k)
        return
    if args.cmd == "analysis":
        cmd_analysis(args.target_set, args.start, args.end)
        return
    if args.cmd == "factor-eval":
        cmd_factor_eval(args.mode, args.adjust, args.force_refresh)
        return

    raise RuntimeError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    # Ensure project root is on sys.path (usually already when `python main.py`)
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)
    main()
