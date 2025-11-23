import importlib

import pytest

# Optional heavy dependencies are required for the main package; skip the smoke import
# test entirely when they are unavailable in the CI/runtime environment.
pytest.importorskip("pandas")
pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("backtrader")

MODULES = [
    "src.data_provider",
    "src.alpha_lib",
    "src.factor_ops",
    "src.model",
    "src.train",
    "src.inference",
    "src.backtest",
    "src.analysis",
]


def test_core_imports():
    for module in MODULES:
        imported = importlib.import_module(module)
        assert imported is not None, f"{module} should import successfully"
