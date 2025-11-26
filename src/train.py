import torch
import numpy as np
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from scipy.stats import spearmanr
from .config import Config
from .model import PatchTSTForStock, SotaConfig
from .data_provider import get_dataset


def compute_metrics(eval_pred):
    """
    计算 Validation 集指标 (用于 Early Stopping)
    """
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    preds = predictions.flatten()
    labs = labels.flatten()
    preds = np.nan_to_num(preds)
    labs = np.nan_to_num(labs)

    ic, p_value = spearmanr(preds, labs)
    return {"ic": ic}


def run_training():
    print("\n" + "=" * 60)
    print(">>> 启动模型训练 (Train / Validation Split)")
    print("=" * 60)

    # 获取包含 train, validation, test 的数据集
    ds, num_features = get_dataset()

    print(f"Feature Dim: {num_features}")
    # [Check] 确保只使用 Train 和 Validation
    print(f"Training on: {len(ds['train'])} samples")
    print(f"Evaluating on: {len(ds['validation'])} samples (Early Stopping)")
    print(f"Held-out Test: {len(ds['test'])} samples (Ignored during training)")

    model_config = SotaConfig(
        num_input_channels=num_features,
        context_length=Config.CONTEXT_LEN,
        patch_length=Config.PATCH_LEN,
        stride=Config.STRIDE,
        d_model=Config.D_MODEL,
        num_hidden_layers=3,
        n_heads=4,
        dropout=Config.DROPOUT,
        mse_weight=Config.MSE_WEIGHT
    )

    model = PatchTSTForStock(model_config)

    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=Config.EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.INFERENCE_BATCH_SIZE,

        learning_rate=Config.LR,
        weight_decay=1e-4,
        max_grad_norm=Config.MAX_GRAD_NORM,

        eval_strategy="steps",
        eval_steps=1000,  # 每500步验证一次
        save_steps=1000,
        save_total_limit=2,

        logging_steps=100,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,

        load_best_model_at_end=True,
        metric_for_best_model="ic",  # 监控 Validation Set 的 IC
        greater_is_better=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],  # 训练集
        eval_dataset=ds['validation'],  # 验证集 (Eval Set)
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    print(f"🚀 开始训练...")
    trainer.train()

    final_path = f"{Config.OUTPUT_DIR}/final_model"
    trainer.save_model(final_path)
    print(f"✅ 模型已保存: {final_path}")