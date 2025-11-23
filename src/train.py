import torch
import numpy as np
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from scipy.stats import spearmanr
from .config import Config
from .model import PatchTSTForStock, SotaConfig
from .data_provider import get_dataset


def compute_metrics(eval_pred):
    """
    计算验证集指标
    注意：在 HF Trainer 中直接计算 Daily Rank IC 比较困难（缺失 Date 信息）。
    这里使用 Flatten 后的 Spearman IC 作为近似代理，
    更严谨的 Daily IC 会在 Analysis 阶段通过 analysis.py 计算。
    """
    predictions, labels = eval_pred
    # 确保是 Numpy 数组
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    preds = predictions.flatten()
    labs = labels.flatten()

    # 异常值处理 (防止 NaN 导致 crash)
    preds = np.nan_to_num(preds)
    labs = np.nan_to_num(labs)

    # 计算 IC
    ic, p_value = spearmanr(preds, labs)
    return {"ic": ic}


def run_training():
    print("\n" + "=" * 60)
    print(">>> 启动模型训练 (Training Pipeline)")
    print(f">>> Device: {Config.DEVICE}")
    print("=" * 60)

    ds, num_features = get_dataset()

    print(f"Feature Dim: {num_features}")
    print(f"Train Samples: {len(ds['train'])} | Test Samples: {len(ds['test'])}")

    # 2. 配置模型
    model_config = SotaConfig(
        num_input_channels=num_features,
        context_length=Config.CONTEXT_LEN,
        patch_length=Config.PATCH_LEN,
        stride=Config.STRIDE,
        d_model=128,
        num_hidden_layers=3,
        n_heads=4,
        dropout=Config.DROPOUT,
        mse_weight=Config.MSE_WEIGHT
    )

    model = PatchTSTForStock(model_config)

    # 3. 训练参数 (Production Grade)
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=Config.EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.INFERENCE_BATCH_SIZE,

        # 优化器配置
        learning_rate=Config.LR,
        weight_decay=1e-4,
        max_grad_norm=Config.MAX_GRAD_NORM,  # 梯度裁剪

        # 评估策略
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,  # 只保留最近2个Checkpoint，节省空间

        logging_steps=50,
        fp16=torch.cuda.is_available(),  # 自动开启混合精度
        dataloader_num_workers=0,  # 避免多进程死锁 (特别是 DataLoader 在 AkShare 环境下)

        load_best_model_at_end=True,
        metric_for_best_model="ic",
        greater_is_better=True,

        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    print(f"🚀 开始训练 (Epochs={Config.EPOCHS}, Batch={Config.BATCH_SIZE})...")
    trainer.train()

    final_path = f"{Config.OUTPUT_DIR}/final_model"
    trainer.save_model(final_path)
    print(f"✅ 模型训练完成，已保存至: {final_path}")