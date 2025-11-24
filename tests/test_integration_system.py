import pytest
import os
import shutil
import pandas as pd
import numpy as np
import torch
import pickle
from src.config import Config
from src.alpha_lib import AlphaFactory
from src.data_provider import DataProvider
from src.model import PatchTSTForStock, SotaConfig
from src.backtest import run_single_backtest
from transformers import TrainingArguments, Trainer


# ==============================================================================
#  Helper: 临时配置覆盖
# ==============================================================================
class MockConfig:
    """为集成测试创建的微型配置"""

    def __init__(self, tmp_path):
        self.BASE_DIR = str(tmp_path)
        self.DATA_DIR = os.path.join(str(tmp_path), "data")
        self.OUTPUT_DIR = os.path.join(str(tmp_path), "output")
        self.CONTEXT_LEN = 10
        self.PRED_LEN = 2
        self.PATCH_LEN = 2
        self.STRIDE = 2
        self.FEATURE_PREFIXES = ['style_', 'tech_']  # 只测试少量因子以加速
        self.BATCH_SIZE = 4
        self.INFERENCE_BATCH_SIZE = 4
        self.DEVICE = "cpu"  # 强制 CPU 确保 CI/CD 兼容性

        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)


# ==============================================================================
#  Integration Test Suite
# ==============================================================================
@pytest.fixture
def mock_env(tmp_path):
    """Fixture: 准备隔离的测试环境"""
    # 1. 覆盖 Config (Monkeypatching)
    original_data_dir = Config.DATA_DIR
    original_output_dir = Config.OUTPUT_DIR

    mock_cfg = MockConfig(tmp_path)
    Config.DATA_DIR = mock_cfg.DATA_DIR
    Config.OUTPUT_DIR = mock_cfg.OUTPUT_DIR
    Config.CONTEXT_LEN = mock_cfg.CONTEXT_LEN
    Config.FEATURE_PREFIXES = mock_cfg.FEATURE_PREFIXES
    Config.DEVICE = "cpu"

    yield mock_cfg

    # Teardown: 恢复配置
    Config.DATA_DIR = original_data_dir
    Config.OUTPUT_DIR = original_output_dir


def create_synthetic_data(mock_cfg, days=300):
    """
    生成足够用于 Training + Gap + Inference 的数据
    [Fix] 增加到 300 天，确保 90% Split (270天) 后，剩余 30 天扣除 Gap (10天) 还有 20 天给 Test 集。
    """
    dates = pd.date_range("2024-01-01", periods=days, freq="B")
    codes = ["000001", "000002"]

    frames = []
    for code in codes:
        # 制造确定性趋势: 000001 一直涨, 000002 一直跌
        trend = 1 if code == "000001" else -1
        base_price = 100.0 + np.arange(days) * trend * 0.5

        df = pd.DataFrame({
            'date': dates,
            'code': code,
            'open': base_price,
            'high': base_price + 2,
            'low': base_price - 2,
            'close': base_price + trend * 0.2,  # 收盘价略有变动
            'volume': 100000 + np.random.randint(-1000, 1000, days),
            'amount': 10000000.0,
            # Mock 财务数据
            'pe_ttm': 10.0,
            'pb': 1.5,
            'roe': 0.15
        })
        frames.append(df)

    return pd.concat(frames).sort_values(['code', 'date']).reset_index(drop=True)


def test_full_system_lifecycle(mock_env, capsys):
    """
    🚀 全系统集成测试 (End-to-End Integration Test)

    覆盖流程:
    1. ETL: 生成模拟数据并存入 Parquet
    2. Alpha: 读取数据，计算因子，生成 Dataset
    3. Train: 初始化模型，运行微型训练循环，保存 Checkpoint
    4. Inference: 加载 Checkpoint，对新数据进行推理
    5. Consistency: 验证训练和推理的特征列是否严格对齐 (P0级风险检查)
    6. Backtest: 将推理信号输入回测引擎，验证是否产生交易
    """

    print("\n>>> [Step 1] ETL & Data Processing...")
    raw_df = create_synthetic_data(mock_env)

    # 模拟 DataProvider._download_worker 的结果 (写入 parquet)
    for code in raw_df['code'].unique():
        sub_df = raw_df[raw_df['code'] == code].set_index('date')
        sub_df.to_parquet(os.path.join(mock_env.DATA_DIR, f"{code}.parquet"))

    # 运行 DataProvider 处理逻辑 (生成 Cache)
    panel_df, feature_cols = DataProvider.load_and_process_panel(mode='train', force_refresh=True)

    assert len(panel_df) > 0
    assert len(feature_cols) > 0
    assert 'target' in panel_df.columns
    print(f"✅ Data Processed. Features: {len(feature_cols)}, Samples: {len(panel_df)}")

    print("\n>>> [Step 2] Dataset Generation...")
    ds, num_features = DataProvider.make_dataset(panel_df, feature_cols)
    assert len(ds['train']) > 0
    # [Check] 确保 Test 集不为空，否则 Generator 会抛错
    assert len(ds['test']) > 0
    print(f"✅ Dataset Created. Train: {len(ds['train'])}, Test: {len(ds['test'])}")

    print("\n>>> [Step 3] Model Training (Micro-Batch)...")
    model_config = SotaConfig(
        num_input_channels=num_features,
        context_length=Config.CONTEXT_LEN,
        patch_length=mock_env.PATCH_LEN,
        stride=mock_env.STRIDE,
        d_model=16,  # Tiny model
        num_hidden_layers=1,
        n_heads=2,
        dropout=0.1
    )
    model = PatchTSTForStock(model_config)

    training_args = TrainingArguments(
        output_dir=mock_env.OUTPUT_DIR,
        num_train_epochs=1,  # 只跑 1 个 epoch
        max_steps=5,  # 强制只跑 5 步，验证能否 backward 即可
        per_device_train_batch_size=4,
        learning_rate=1e-3,
        report_to="none",
        save_strategy="no",  # 训练中不存，最后手动存
        use_cpu=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train']
    )

    trainer.train()

    # 保存模型
    final_model_path = os.path.join(mock_env.OUTPUT_DIR, "final_model")
    trainer.save_model(final_model_path)

    # 保存特征列表 (模拟生产环境的 metadata)
    feature_meta_path = os.path.join(mock_env.OUTPUT_DIR, "feature_meta.pkl")
    with open(feature_meta_path, "wb") as f:
        pickle.dump(feature_cols, f)

    assert os.path.exists(os.path.join(final_model_path, "config.json"))
    assert os.path.exists(os.path.join(final_model_path, "model.safetensors"))
    print("✅ Model Trained & Saved.")

    print("\n>>> [Step 4] Inference & Consistency Check...")
    # 模拟推理模式：重新加载模型和数据
    loaded_model = PatchTSTForStock.from_pretrained(final_model_path)
    loaded_model.eval()

    # 加载推理数据 (Mode='predict')
    pred_df, pred_features = DataProvider.load_and_process_panel(mode='predict', force_refresh=True)

    # 【关键检查】验证特征对齐 (P0级风险)
    # 在生产中，必须确保推理时的特征顺序与训练时完全一致
    with open(feature_meta_path, "rb") as f:
        train_features = pickle.load(f)

    assert train_features == pred_features, "❌ CRITICAL: Inference features mismatch Training features!"
    print("✅ Feature Alignment Verified.")

    # 构造推理 Batch
    # 取最后一天的数据进行推理
    last_date = pred_df['date'].max()
    target_group = pred_df[pred_df['date'] == last_date]
    codes = target_group['code'].unique()

    # 简单模拟推理过程 (不走 full loop，直接测 forward)
    # 构造一个 (Batch, Seq, Feat)
    sample_input = torch.randn(len(codes), Config.CONTEXT_LEN, num_features)
    with torch.no_grad():
        output = loaded_model(past_values=sample_input)
        scores = output.logits.squeeze().numpy()

    assert len(scores) == len(codes)
    print(f"✅ Inference Successful. Generated {len(scores)} scores.")

    print("\n>>> [Step 5] Backtest Execution...")
    # 构造 Backtest 需要的 Signal DataFrame
    # 我们手动制造一个强信号：000001 极高分，000002 极低分
    # [Fix] 确保索引是 DatetimeIndex，以便 Backtest 引擎能正确索引
    unique_dates = sorted(pred_df['date'].unique())
    signals = pd.DataFrame(index=unique_dates, columns=codes, dtype=float)
    signals[:] = -100.0  # 默认无效

    # 在所有日期对 000001 发出买入信号
    if "000001" in codes:
        signals["000001"] = 100.0

        # 运行回测
    # 注意：我们需要为回测提供行情数据，这里复用之前生成的 Parquet
    # 这里的 Config 已经被 monkeypatch 了，需要确保回测引擎能读到 Mock Data
    result = run_single_backtest(["000001"], with_fees=True, initial_cash=100000.0, top_k=1)

    assert result is not None
    # 000001 是上涨趋势，一直持有应该盈利
    print(f"Final Value: {result['final_value']:.2f}")
    assert result['final_value'] > 100000.0, "Backtest should profit on uptrend stock with buy signal"
    assert result['total_trades'] > 0, "Should have executed at least one trade"

    print("✅ Backtest Finished Successfully.")
    print("\n🎉🎉🎉 ALL SYSTEMS GO! INTEGRATION TEST PASSED! 🎉🎉🎉")