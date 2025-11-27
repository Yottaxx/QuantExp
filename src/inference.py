# -*- coding: utf-8 -*-
# inference.py (v4, reuse SignalEngine; no duplicated window/batch logic)

from __future__ import annotations

import pandas as pd
import numpy as np

from .config import Config
from .core.signal_engine import SignalEngine


def _run_latest_via_engine(
    *,
    adjust: str,
    mode: str,
    top_k: int,
    min_score_threshold: float,
    batch_size: int,
) -> tuple[list[tuple[str, float, float | None]], dict]:
    """
    Prefer calling SignalEngine.score_latest_day() if it exists.
    Otherwise fallback to composing existing SignalEngine primitives:
      load_model + load_panel + score_date_range (+ merge pe/regime)
    """
    # --- Preferred path: your engine already provides score_latest_day ---
    if hasattr(SignalEngine, "score_latest_day"):
        picks, meta = SignalEngine.score_latest_day(  # type: ignore[attr-defined]
            adjust=adjust,
            mode=mode,
            top_k=top_k,
            min_score_threshold=min_score_threshold,
            batch_size=batch_size,
        )

        # normalize picks to tuples (code, score, pe)
        final_picks: list[tuple[str, float, float | None]] = []
        for p in picks:
            if isinstance(p, tuple) and len(p) >= 2:
                code = str(p[0])
                score = float(p[1])
                pe = float(p[2]) if len(p) >= 3 and p[2] is not None else None
                final_picks.append((code, score, pe))
            else:
                # dataclass ScoredPick
                code = str(getattr(p, "code"))
                score = float(getattr(p, "score"))
                pe = getattr(p, "pe_ttm", None)
                pe = float(pe) if pe is not None else None
                final_picks.append((code, score, pe))

        # also keep raw_topk for pretty printing if provided
        raw_topk = meta.get("raw_topk", [])
        if raw_topk and not isinstance(raw_topk[0], tuple):
            meta["raw_topk"] = [(str(x.code), float(x.score), getattr(x, "pe_ttm", None)) for x in raw_topk]

        return final_picks, meta

    # --- Fallback: compose SSOT primitives (works with the engine you pasted earlier) ---
    model = SignalEngine.load_model()
    panel_df, feature_cols = SignalEngine.load_panel(adjust=adjust, mode=mode)

    if panel_df.empty:
        raise RuntimeError("panel_df empty")

    last_date = pd.to_datetime(panel_df["date"]).max().normalize()

    # score only last_date (expects panel_df already includes enough lookback for seq_len)
    scored = SignalEngine.score_date_range(
        model=model,
        panel_df=panel_df,
        feature_cols=feature_cols,
        start_date=last_date,
        end_date=last_date,
        seq_len=int(getattr(Config, "CONTEXT_LEN", 64)),
        batch_size=int(batch_size),
        desc="ScoringLatestDay",
    )
    if scored.empty:
        raise RuntimeError("No scores produced for latest day (maybe insufficient lookback).")

    scored["date"] = pd.to_datetime(scored["date"]).dt.normalize()
    scored["code"] = scored["code"].astype(str)
    scored["score"] = pd.to_numeric(scored["score"], errors="coerce").fillna(-1.0)

    # attach PE (optional)
    pe_map = None
    if "pe_ttm" in panel_df.columns:
        daily = panel_df[pd.to_datetime(panel_df["date"]).dt.normalize() == last_date][["code", "pe_ttm"]].copy()
        daily["code"] = daily["code"].astype(str)
        daily["pe_ttm"] = pd.to_numeric(daily["pe_ttm"], errors="coerce")
        pe_map = daily.drop_duplicates("code", keep="last").set_index("code")["pe_ttm"].to_dict()

    # market regime
    regime, mom = SignalEngine.check_market_regime(panel_df, last_date)

    # bear-day + threshold filtering (single day version)
    daily_mean = float(scored["score"].mean()) if len(scored) else -1.0
    bear_th = float(getattr(Config, "BEAR_MEAN_THRESHOLD", 0.45))
    if daily_mean < bear_th:
        scored["score"] = -1.0
    scored.loc[scored["score"] < float(min_score_threshold), "score"] = -1.0

    # raw topk (before “advice”)
    raw = scored.sort_values("score", ascending=False).head(top_k)
    raw_topk = []
    for _, r in raw.iterrows():
        code = str(r["code"])
        s = float(r["score"])
        pe = None if pe_map is None else pe_map.get(code, None)
        pe = float(pe) if pe is not None and np.isfinite(pe) else None
        raw_topk.append((code, s, pe))

    top_score = float(raw_topk[0][1]) if raw_topk else -1.0

    # final picks: only those eligible to "buy" given regime + threshold (same semantics as your printer)
    final_picks = []
    for code, s, pe in raw_topk:
        if regime == "Bear":
            continue
        if s < float(min_score_threshold):
            continue
        final_picks.append((code, s, pe))

    meta = {
        "last_date": last_date,
        "regime": regime,
        "mom": float(mom),
        "top_score": top_score,
        "raw_topk": raw_topk,
        "daily_mean": daily_mean,
        "bear_mean_th": bear_th,
    }
    return final_picks, meta


def run_inference(top_k=Config.TOP_K, min_score_threshold=Config.MIN_SCORE_THRESHOLD, adjust="qfq"):
    print("\n" + "=" * 50)
    print(f">>> 启动全市场每日选股 [Adjust={adjust}]")
    print("=" * 50)

    try:
        final_picks, meta = _run_latest_via_engine(
            adjust=adjust,
            mode="predict",
            top_k=int(top_k),
            min_score_threshold=float(min_score_threshold),
            batch_size=int(getattr(Config, "INFERENCE_BATCH_SIZE", 2048)),
        )
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        return []

    last_date = meta.get("last_date")
    regime = meta.get("regime", "Unknown")
    mom = float(meta.get("mom", 0.0))
    top_score = float(meta.get("top_score", -1.0))
    raw_topk = meta.get("raw_topk", [])

    print(f"📅 最新交易日: {pd.to_datetime(last_date).date()}")
    print(f"📊 市场状态: {regime} | mom={mom:.4f} | top_score={top_score:.4f}")

    if regime == "Bear":
        print("⚠️ 熊市特征明显：建议空仓或极低仓位。")

    print("-" * 60)
    print(f"{'排名':<5} | {'代码':<10} | {'AI预测分':<10} | {'PE(TTM)':<10} | {'建议'}")
    print("-" * 60)

    # 打印 raw_topk，再按 advice 过滤后返回 final_picks（与你原来的交互一致）
    picks_out: list[tuple[str, float, float | None]] = []
    final_set = {c for c, _, _ in final_picks}

    for rank, item in enumerate(raw_topk[: int(top_k)], 1):
        if isinstance(item, tuple) and len(item) >= 2:
            code = str(item[0])
            score = float(item[1])
            pe = item[2] if len(item) >= 3 else None
            pe = float(pe) if pe is not None and pe != 0 else None
        else:
            code = str(getattr(item, "code"))
            score = float(getattr(item, "score"))
            pe = getattr(item, "pe_ttm", None)
            pe = float(pe) if pe is not None and pe != 0 else None

        pe_str = f"{pe:.2f}" if (pe is not None) else "-"
        advice = "买入"
        if regime == "Bear":
            advice = "慎买"
        if score < float(min_score_threshold):
            advice = "观望"
        if code not in final_set:
            # 被风控/阈值过滤（但仍展示）
            if regime == "Bear":
                advice = "空仓"
            elif score < float(min_score_threshold):
                advice = "观望"
            else:
                advice = "过滤"

        print(f"{rank:<5} | {code:<10} | {score:.6f}     | {pe_str:<10} | {advice}")

        if code in final_set:
            picks_out.append((code, score, pe))

    print("=" * 60)
    if len(picks_out) < min(int(top_k), len(raw_topk)):
        print(f"💡 风控生效：{min(int(top_k), len(raw_topk))} -> {len(picks_out)}")
    if not picks_out:
        print("🛡️ 最终决策：空仓")

    return picks_out


if __name__ == "__main__":
    run_inference()
