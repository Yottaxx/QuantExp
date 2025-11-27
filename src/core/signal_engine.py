# -*- coding: utf-8 -*-
# signal_engine.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..config import Config
from ..model import PatchTSTForStock
from ..data_provider import DataProvider


@dataclass(frozen=True)
class ScoredPick:
    code: str
    score: float
    pe_ttm: float | None = None


class SignalEngine:
    """Single Source of Truth for:
    - model loading
    - panel loading
    - batch scoring (latest day / date range)
    - signal matrix construction
    """

    @staticmethod
    def load_model(model_path: Optional[str] = None) -> torch.nn.Module:
        device = Config.DEVICE
        model_path = model_path or f"{Config.OUTPUT_DIR}/final_model"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = PatchTSTForStock.from_pretrained(model_path).to(device)
        model.eval()
        return model

    @staticmethod
    def load_panel(adjust: str, mode: str) -> Tuple[pd.DataFrame, List[str]]:
        panel_df, feature_cols = DataProvider.load_and_process_panel(mode=mode, adjust=adjust)
        panel_df = panel_df.copy()
        panel_df["date"] = pd.to_datetime(panel_df["date"], errors="coerce")
        panel_df["code"] = panel_df["code"].astype(str)
        panel_df = panel_df.dropna(subset=["date", "code"]).sort_values(["code", "date"]).reset_index(drop=True)
        return panel_df, feature_cols

    @staticmethod
    def check_market_regime(panel_df: pd.DataFrame, last_date: pd.Timestamp) -> Tuple[str, float]:
        if "style_mom_1m" not in panel_df.columns:
            return "Unknown", 0.0
        daily = panel_df[panel_df["date"] == last_date]
        if daily.empty:
            return "Unknown", 0.0
        up_ratio = float((daily["style_mom_1m"] > 0).mean())
        median_mom = float(daily["style_mom_1m"].median())
        if up_ratio < 0.4 or median_mom < -0.02:
            return "Bear", median_mom
        if up_ratio > 0.6:
            return "Bull", median_mom
        return "Shock", median_mom

    @staticmethod
    def _batch_score(model: torch.nn.Module, windows: np.ndarray, batch_size: int) -> np.ndarray:
        device = Config.DEVICE
        outs: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                x = torch.from_numpy(windows[i:i + batch_size]).float().to(device)
                logits = model(past_values=x).logits
                logits = logits.squeeze()
                s = logits.detach().cpu().numpy()
                if np.ndim(s) == 0:
                    s = np.array([float(s)], dtype=np.float32)
                outs.append(s.astype(np.float32))
        return np.concatenate(outs, axis=0) if outs else np.array([], dtype=np.float32)

    @staticmethod
    def score_date_range(
        *,
        model: torch.nn.Module,
        panel_df: pd.DataFrame,
        feature_cols: List[str],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        seq_len: int,
        batch_size: int,
        desc: str = "Scoring",
    ) -> pd.DataFrame:
        """
        对区间内 (start_date~end_date) 做历史推理。
        输出 DataFrame: [date, code, score]
        注意：
          - panel_df 需要已经包含足够的 lookback，否则 start_date 前 seq_len-1 天无法预测
        """
        start_date = pd.to_datetime(start_date).normalize()
        end_date = pd.to_datetime(end_date).normalize()

        results: list[tuple[pd.Timestamp, str, float]] = []
        buf_x: list[np.ndarray] = []
        buf_meta: list[tuple[pd.Timestamp, str]] = []

        # 这里按 code 分组，避免全局滑窗爆内存
        for code, g in tqdm(panel_df.groupby("code", sort=False), desc=desc):
            g = g.sort_values("date")
            if len(g) < seq_len:
                continue

            dates = pd.to_datetime(g["date"]).values.astype("datetime64[ns]")
            feats = g[feature_cols].values.astype(np.float32, copy=False)

            # 可预测的窗口末端位置
            # end_idx 是窗口末端（含），窗口区间 [end_idx-seq_len+1, end_idx]
            for end_idx in range(seq_len - 1, len(g)):
                pred_dt = pd.to_datetime(dates[end_idx]).normalize()
                if pred_dt < start_date or pred_dt > end_date:
                    continue

                w = feats[end_idx - seq_len + 1: end_idx + 1]
                if w.shape[0] != seq_len:
                    continue

                buf_x.append(w)
                buf_meta.append((pred_dt, str(code)))

                if len(buf_x) >= batch_size:
                    x_np = np.ascontiguousarray(np.stack(buf_x))
                    scores = SignalEngine._batch_score(model, x_np, batch_size=batch_size)
                    for (d, c), sc in zip(buf_meta, scores):
                        results.append((d, c, float(sc)))
                    buf_x.clear()
                    buf_meta.clear()

        if buf_x:
            x_np = np.ascontiguousarray(np.stack(buf_x))
            scores = SignalEngine._batch_score(model, x_np, batch_size=batch_size)
            for (d, c), sc in zip(buf_meta, scores):
                results.append((d, c, float(sc)))

        out = pd.DataFrame(results, columns=["date", "code", "score"])
        if not out.empty:
            out["date"] = pd.to_datetime(out["date"]).dt.normalize()
            out["code"] = out["code"].astype(str)
        return out

    @staticmethod
    def scores_to_signal_matrix(
        scored_rows: pd.DataFrame,
        *,
        min_score_threshold: float,
        bear_mean_th: float = 0.45,
    ) -> pd.DataFrame:
        """
        scored_rows: [date, code, score]
        统一生成信号矩阵：低分/熊市日 -> -1
        """
        df = scored_rows.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["code"] = df["code"].astype(str)
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(-1.0)

        daily_mean = df.groupby("date")["score"].mean()
        bear_days = daily_mean[daily_mean < bear_mean_th].index
        df.loc[df["date"].isin(bear_days), "score"] = -1
        df.loc[df["score"] < float(min_score_threshold), "score"] = -1

        return df.pivot(index="date", columns="code", values="score").sort_index()
