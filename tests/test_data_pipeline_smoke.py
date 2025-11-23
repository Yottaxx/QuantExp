import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")

import numpy as np
import pandas as pd

from src.alpha_lib import AlphaFactory
from src.config import Config
from src.data_provider import DataProvider


def build_synthetic_panel():
    # Reduce context to keep the synthetic dataset small and fast
    Config.CONTEXT_LEN = 5
    Config.PRED_LEN = 2
    Config.STRIDE = 2

    dates = pd.date_range("2024-01-02", periods=12, freq="B")
    frames = []
    for offset, code in enumerate(["000001", "000002"]):
        base = np.arange(len(dates)) + offset
        df = pd.DataFrame(
            {
                "date": dates,
                "open": 10 + base * 0.1,
                "high": 10.2 + base * 0.1,
                "low": 9.8 + base * 0.1,
                "close": 10 + base * 0.1 + 0.05,
                "volume": 1e6 + base * 1000,
                "code": code,
            }
        )
        frames.append(AlphaFactory(df).make_factors())

    panel_df = pd.concat(frames, ignore_index=True)

    panel_df["next_open"] = panel_df.groupby("code")["open"].shift(-1)
    panel_df["future_close"] = panel_df.groupby("code")["close"].shift(-Config.PRED_LEN)
    panel_df["target"] = panel_df["future_close"] / panel_df["next_open"] - 1
    panel_df.drop(columns=["next_open", "future_close"], inplace=True)

    panel_df["is_universe"] = True
    panel_df = panel_df.sort_values(["code", "date"])
    panel_df = panel_df.set_index("date")
    panel_df = AlphaFactory.add_cross_sectional_factors(panel_df)
    feature_cols = [
        c for c in panel_df.columns if any(c.startswith(p) for p in Config.FEATURE_PREFIXES)
    ]

    panel_df[feature_cols] = (
        panel_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32)
    )
    panel_df = panel_df.reset_index()
    return panel_df, feature_cols


def test_dataset_creation_runs_end_to_end():
    panel_df, feature_cols = build_synthetic_panel()

    # Ensure we have enough samples for both splits
    np.random.seed(42)
    ds, num_features = DataProvider.make_dataset(panel_df, feature_cols)

    assert num_features == len(feature_cols)
    assert len(ds["train"]) > 0
    assert len(ds["test"]) > 0

    sample = ds["train"][0]
    assert sample["past_values"].shape[1] == num_features
    assert np.isfinite(sample["labels"]).all()
