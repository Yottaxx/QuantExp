# main.py (v2) - unified CLI entry for the quant platform
from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import Any, Dict, Optional

from utils.logging_utils import get_logger, init_logger

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
    Import module from the src/ package.
    """
    return importlib.import_module(f"src.{name}")


def _set_seed(seed: int) -> None:
    """
    Prefer utils/seed_utils.set_global_seed; fall back to minimal seeding.
    """
    try:
        seed_mod = importlib.import_module("utils.seed_utils")
        if hasattr(seed_mod, "set_global_seed"):
            seed_mod.set_global_seed(seed)  # type: ignore[attr-defined]
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
    """
    Build the top-level CLI.

    命令分层：
      - config     : 查看配置
      - download   : 下载/更新原始行情&财务数据
      - panel      : 构建特征 panel 缓存
      - train      : 训练深度学习模型
      - predict    : 最新交易日选股
      - backtest   : 单次回测（固定标的或基于当日选股）
      - walkforward: 滚动窗口 Walk-Forward 回测
      - analysis   : 历史预测生成 + 回测分析（可视化、统计）
      - factor-eval: 因子 IC/IR 评价（需要 evaluator.AlphaEvaluator）
      - debug      : 端到端 smoke test（data -> model -> inference -> backtest）
    """
    p = argparse.ArgumentParser(
        "Quant Platform",
        description="Institutional-grade A-share DL + multi-factor quant platform",
    )
    # global options
    p.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: 单进程/少量样本/串行 alpha，便于本地断点调试",
    )
    p.add_argument(
        "--print-config",
        action="store_true",
        help="在执行前打印有效 Config",
    )
    p.add_argument(
        "--strict-config",
        action="store_true",
        help="遇到未知的 --set KEY 报错而不是静默忽略",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        help="覆盖 Config：KEY=VAL（可重复）",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子覆盖（默认使用 Config.SEED 或 42）",
    )
    p.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="实验名称，会拼接到本次运行时间生成的 logger 名称中",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    # 1) Config / data
    sub.add_parser("config", help="打印有效 Config 并退出")
    sub.add_parser("download", help="运行 DataProvider.download_data()")

    panel = sub.add_parser("panel", help="构建 panel 缓存 (load_and_process_panel)")
    panel.add_argument("--mode", default="train", choices=["train", "predict"])
    panel.add_argument("--adjust", default="qfq")
    panel.add_argument("--force-refresh", action="store_true")

    # 2) 模型训练 / 推理 / 回测
    sub.add_parser("train", help="训练模型 (run_training)")

    pred = sub.add_parser("predict", help="运行推理 (run_inference)")
    pred.add_argument("--top-k", type=int, default=None)
    pred.add_argument("--min-score", type=float, default=None)
    pred.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="如指定路径，则将选股结果保存为 CSV",
    )

    bt = sub.add_parser(
        "backtest",
        help="回测 (run_single_backtest 或 基于当日选股的回测)",
    )
    bt.add_argument(
        "--codes",
        default="",
        help="逗号分隔的股票代码列表；为空则使用 predict 得到的当日选股",
    )
    bt.add_argument(
        "--with-fees",
        action="store_true",
        help="是否启用 A 股交易费用（佣金+印花税）",
    )
    bt.add_argument(
        "--initial-cash",
        type=float,
        default=1_000_000.0,
        help="初始资金",
    )
    bt.add_argument("--top-k", type=int, default=None)
    bt.add_argument("--min-score", type=float, default=None)

    wf = sub.add_parser("walkforward", help="Walk-forward 回测 (run_walk_forward_backtest)")
    wf.add_argument("--start", required=True, help="YYYY-MM-DD")
    wf.add_argument("--end", required=True, help="YYYY-MM-DD")
    wf.add_argument(
        "--initial-cash",
        type=float,
        default=1_000_000.0,
        help="初始资金",
    )
    wf.add_argument("--top-k", type=int, default=None)

    ana = sub.add_parser(
        "analysis",
        help="历史预测 + 回测分析 (BacktestAnalyzer)",
        aliases=["test", "eval"],
    )
    ana.add_argument(
        "--target-set",
        default="test",
        choices=["test", "validation", "val", "eval", "train", "custom"],
    )
    ana.add_argument("--start", default=None)
    ana.add_argument("--end", default=None)

    fe = sub.add_parser("factor-eval", help="因子评估 (AlphaEvaluator)")
    fe.add_argument("--mode", default="train", choices=["train", "predict"])
    fe.add_argument("--adjust", default="qfq")
    fe.add_argument("--force-refresh", action="store_true")

    dbg = sub.add_parser(
        "debug",
        help="端到端 smoke test：panel -> inference -> backtest（配合 --debug 更适合本地）",
    )
    dbg.add_argument("--top-k", type=int, default=None)
    dbg.add_argument("--min-score", type=float, default=None)
    dbg.add_argument(
        "--initial-cash",
        type=float,
        default=100_000.0,
        help="debug 回测初始资金",
    )

    return p


def apply_runtime_config(args: argparse.Namespace) -> Any:
    """
    统一处理 Config 覆盖 / debug 模式 / 随机种子 / 目录创建。
    """
    # 1) 导入 Config，并在其它模块 import 之前完成覆盖
    cfg_mod = _import_src("config")
    Config = cfg_mod.Config

    # 2) CLI 覆盖
    overrides: Dict[str, Any] = parse_kv_pairs(args.set)
    apply_config_overrides(Config, overrides, strict=args.strict_config)

    # 2.1) 日志
    level = str(getattr(Config, "LOG_LEVEL", "INFO") or "INFO")
    exp_name = args.exp_name or getattr(Config, "EXPERIMENT_NAME", "default")
    global _RUN_LOGGER
    _RUN_LOGGER = init_logger(exp_name, level=level)
    log = _RUN_LOGGER

    # 3) debug profile
    if args.debug:
        setup_debug_mode(Config)

    # 4) 随机种子
    seed = args.seed if args.seed is not None else getattr(Config, "SEED", 42)
    _set_seed(int(seed))

    # 5) 确保目录存在
    ensure_dirs(Config)

    # 6) 打印 Config 概览
    if args.print_config or args.cmd == "config" or args.debug:
        debug_print_config(Config, logger=log)
        # DataProvider 可能没有 debug_print_config，新版仅打印 VERSION 即可
        try:
            dp_mod = _import_src("data_provider")
            DP = getattr(dp_mod, "DataProvider", None)
            if DP is not None and hasattr(DP, "VERSION"):
                log.info(f"[DataProvider] VERSION = {getattr(DP, 'VERSION')}")
        except Exception as e:  # noqa: BLE001
            log.warning(f"[warn] DataProvider introspection failed: {e}")

    return Config


def cmd_download() -> None:
    log = _get_run_logger()
    dp_mod = _import_src("data_provider")
    dp_mod.DataProvider.download_data()


def cmd_panel(mode: str, adjust: str, force_refresh: bool) -> None:
    log = _get_run_logger()
    dp_mod = _import_src("data_provider")
    panel_df, feature_cols = dp_mod.DataProvider.load_and_process_panel(
        mode=mode,
        adjust=adjust,
        force_refresh=force_refresh,
    )
    log.info(
        f"✅ Panel ready: shape={panel_df.shape}, features={len(feature_cols)} (mode={mode}, adjust={adjust})"
    )


def cmd_train() -> None:
    log = _get_run_logger()
    tr_mod = _import_src("train")
    tr_mod.run_training()


def cmd_predict(
    top_k: Optional[int],
    min_score: Optional[float],
    save_csv: Optional[str] = None,
):
    log = _get_run_logger()
    cfg_mod = _import_src("config")
    Config = cfg_mod.Config

    inf_mod = _import_src("inference")
    k = int(top_k if top_k is not None else Config.TOP_K)
    thr = float(min_score if min_score is not None else Config.MIN_SCORE_THRESHOLD)

    picks = inf_mod.run_inference(top_k=k, min_score_threshold=thr)

    if save_csv:
        import pandas as pd

        df = pd.DataFrame(picks, columns=["code", "score", "pe"])
        os.makedirs(os.path.dirname(save_csv) or ".", exist_ok=True)
        df.to_csv(save_csv, index=False, encoding="utf-8-sig")
        log.info(f"💾 Picks saved to {save_csv}")

    return picks


def cmd_backtest(
    codes: str,
    with_fees: bool,
    initial_cash: float,
    top_k: Optional[int],
    min_score: Optional[float],
) -> None:
    log = _get_run_logger()
    cfg_mod = _import_src("config")
    Config = cfg_mod.Config
    bt_mod = _import_src("backtest")

    code_list = parse_codes_arg(codes)
    if not code_list:
        picks = cmd_predict(top_k=top_k, min_score=min_score, save_csv=None)
        code_list = [c for (c, *_rest) in picks]

    if not code_list:
        log.warning("⚠️ No codes to backtest, exiting.")
        return

    k = int(top_k if top_k is not None else Config.TOP_K)
    bt_mod.run_single_backtest(
        code_list,
        with_fees=with_fees,
        initial_cash=initial_cash,
        top_k=k,
    )


def cmd_walkforward(start: str, end: str, initial_cash: float, top_k: Optional[int]) -> None:
    cfg_mod = _import_src("config")
    Config = cfg_mod.Config
    bt_mod = _import_src("backtest")

    k = int(top_k if top_k is not None else Config.TOP_K)
    bt_mod.run_walk_forward_backtest(start, end, initial_cash, top_k=k)


def cmd_analysis(target_set: str, start: Optional[str], end: Optional[str]) -> None:
    log = _get_run_logger()
    an_mod = _import_src("analysis")
    analyzer = an_mod.BacktestAnalyzer(
        target_set=target_set,
        start_date=start,
        end_date=end,
    )
    analyzer.generate_historical_predictions()
    analyzer.analyze_performance()


def cmd_factor_eval(mode: str, adjust: str, force_refresh: bool) -> None:
    dp_mod = _import_src("data_provider")
    ev_mod = _import_src("evaluator")

    panel_df, feat_cols = dp_mod.DataProvider.load_and_process_panel(
        mode=mode,
        adjust=adjust,
        force_refresh=force_refresh,
    )
    valid = ev_mod.AlphaEvaluator.evaluate(panel_df, feat_cols, target_col="target")
    log.info(f"✅ Valid factors: {len(valid)}/{len(feat_cols)}")


def cmd_debug(
    top_k: Optional[int],
    min_score: Optional[float],
    initial_cash: float,
) -> None:
    """
    端到端 smoke test，方便检查整个流水线是否闭环可跑：
      DataProvider -> SignalEngine / inference -> Backtrader
    """
    cfg_mod = _import_src("config")
    Config = cfg_mod.Config

    log = _get_run_logger()
    log.info("=" * 80)
    log.info("🧪 DEBUG PIPELINE: panel -> inference -> backtest")
    log.info("=" * 80)

    # Step 1: 尝试构建一份 train panel（如果 DataProvider 支持 debug 参数，会由 main 统一打补丁）
    try:
        dp_mod = _import_src("data_provider")
        panel_df, feature_cols = dp_mod.DataProvider.load_and_process_panel(mode="train")
        log.info(
            f"[1/3] panel_df.shape={panel_df.shape}, features={len(feature_cols)}  ✅",
        )
    except Exception as e:  # noqa: BLE001
        log.error(f"[1/3] ❌ DataProvider.load_and_process_panel failed: {e}")
        return

    # Step 2: 运行一次 inference（通常是最近交易日）
    try:
        k = int(top_k if top_k is not None else Config.TOP_K)
        thr = float(min_score if min_score is not None else Config.MIN_SCORE_THRESHOLD)
        inf_mod = _import_src("inference")
        picks = inf_mod.run_inference(top_k=k, min_score_threshold=thr)
        log.info(f"[2/3] inference picks={len(picks)}  ✅")
    except Exception as e:  # noqa: BLE001
        log.error(f"[2/3] ❌ run_inference failed: {e}")
        return

    if not picks:
        log.warning("[2/3] ⚠️ No picks returned; skip backtest.")
        return

    # Step 3: 对这批标的做一轮快速回测（不含费用）
    try:
        codes = [c for (c, *_rest) in picks]
        bt_mod = _import_src("backtest")
        k_bt = min(len(codes), int(getattr(Config, "TOP_K", len(codes))))
        log.info(
            f"[3/3] Running debug backtest on {k_bt} codes, initial_cash={initial_cash:.0f} ...",
        )
        bt_mod.run_single_backtest(
            codes,
            with_fees=False,
            initial_cash=initial_cash,
            top_k=k_bt,
        )
        log.info("[3/3] debug backtest finished  ✅")
    except Exception as e:  # noqa: BLE001
        log.error(f"[3/3] ❌ debug backtest failed: {e}")


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # 先处理 Config / 随机种子 / 目录等
    Config = apply_runtime_config(args)

    # 对 DataProvider.load_and_process_panel 打补丁，让内部调用也能继承 CLI 选择
    try:
        dp_mod = _import_src("data_provider")
        patch_dataprovider_defaults(
            dp_mod.DataProvider,
            adjust=getattr(args, "adjust", None),  # panel/factor-eval 提供
            force_refresh=getattr(args, "force_refresh", None),
            debug_flag=getattr(args, "debug", False),
        )
    except Exception as e:  # noqa: BLE001
        _get_run_logger().warning(f"[warn] patch_dataprovider_defaults failed: {e}")

    # 根据子命令分发逻辑
    if args.cmd == "config":
        # 已在 apply_runtime_config 中打印 Config，这里直接退出
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
        cmd_predict(args.top_k, args.min_score, args.save_csv)
        return
    if args.cmd == "backtest":
        cmd_backtest(args.codes, args.with_fees, args.initial_cash, args.top_k, args.min_score)
        return
    if args.cmd == "walkforward":
        cmd_walkforward(args.start, args.end, args.initial_cash, args.top_k)
        return
    if args.cmd in {"analysis", "test", "eval"}:
        cmd_analysis(args.target_set, args.start, args.end)
        return
    if args.cmd == "factor-eval":
        cmd_factor_eval(args.mode, args.adjust, args.force_refresh)
        return
    if args.cmd == "debug":
        cmd_debug(args.top_k, args.min_score, args.initial_cash)
        return

    raise RuntimeError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    # Ensure project root is on sys.path (usually already when `python main.py`)
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)
    main()

_RUN_LOGGER = None


def _get_run_logger():
    global _RUN_LOGGER
    if _RUN_LOGGER is None:
        _RUN_LOGGER = get_logger()
    return _RUN_LOGGER
