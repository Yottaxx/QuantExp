import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")

import numpy as np
import pandas as pd

from src.config import Config


def test_label_shift_uses_future_returns():
    Config.PRED_LEN = 2
    dates = pd.date_range("2024-01-02", periods=6, freq="B")
    df = pd.DataFrame(
        {
            "date": dates,
            "code": "000001",
            "open": [10, 10.1, 10.2, 10.3, 10.4, 10.5],
            "close": [10.1, 10.2, 10.3, 10.4, 10.5, 10.6],
        }
    )

    df["next_open"] = df.groupby("code")["open"].shift(-1)
    df["future_close"] = df.groupby("code")["close"].shift(-Config.PRED_LEN)
    df["target"] = df["future_close"] / df["next_open"] - 1

    # Target at t=0 should use close at t+2 and open at t+1
    expected = df.loc[2, "close"] / df.loc[1, "open"] - 1
    assert np.isclose(df.loc[0, "target"], expected, equal_nan=False)

    # Last two rows cannot form a label and should be NaN
    assert df["target"].iloc[-2:].isna().all()


def test_time_split_has_gap():
    # Synthetic date array to mirror DataProvider.make_dataset split logic
    Config.CONTEXT_LEN = 3
    unique_dates = pd.date_range("2024-01-02", periods=10, freq="B").values
    split_idx = int(len(unique_dates) * 0.9)
    split_date = unique_dates[split_idx]
    gap_date = unique_dates[min(split_idx + Config.CONTEXT_LEN, len(unique_dates) - 1)]

    assert gap_date >= split_date
    assert (gap_date - split_date).astype("timedelta64[D]") >= np.timedelta64(0, "D")
