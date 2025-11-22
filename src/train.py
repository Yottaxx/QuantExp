import torch
import numpy as np
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from scipy.stats import spearmanr
from .config import Config
from .model import PatchTSTForStock, SotaConfig
from .data_provider import get_dataset


def compute_metrics(eval_pred):
    """在验证集上计算 IC"""
    predictions, labels = eval_pred
    preds = predictions.flatten()
    labs = labels.flatten()
    ic, _ = spearmanr(preds, labs)
    return {"ic": ic}


def run_training():
    print("\n" + "=" * 50)
    print(">>> 启动模型训练 (Training Pipeline)")
    print("=" * 50)

    ds, num_features = get_dataset()

    print(f"Input Features: {num_features}")
    print(f"Train Size: {len(ds['train'])} | Test Size: {len(ds['test'])}")

    # 2. 配置模型
    model_config = SotaConfig(
        num_input_channels=num_features,
        context_length=Config.CONTEXT_LEN,
        patch_length=8,
        stride=4,
        d_model=128,
        num_hidden_layers=3,
        n_heads=4,

        # 【核心修改】使用 Config 中的 Dropout
        dropout=Config.DROPOUT,

        # 传入 MSE 权重
        mse_weight=Config.MSE_WEIGHT
    )

    model = PatchTSTForStock(model_config)

    # 3. 训练参数
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=Config.EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=256,

        learning_rate=Config.LR,
        weight_decay=1e-4,

        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=3,

        logging_steps=50,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,

        load_best_model_at_end=True,
        metric_for_best_model="ic",
        greater_is_better=True,

        remove_unused_columns=False,
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

    print(f"🚀 开始训练 (MSE Weight={Config.MSE_WEIGHT}, Dropout={Config.DROPOUT})...")
    trainer.train()

    final_path = f"{Config.OUTPUT_DIR}/final_model"
    trainer.save_model(final_path)
    print(f"✅ 模型训练完成，已保存至: {final_path}")