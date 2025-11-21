import torch
import numpy as np
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from scipy.stats import spearmanr
from .config import Config
from .model import PatchTSTForStock, SotaConfig
from .data_provider import get_dataset


def compute_metrics(eval_pred):
    """
    【新增】自定义评估函数
    在验证集上计算 IC (Information Coefficient)
    """
    predictions, labels = eval_pred
    # predictions shape: [Batch, 1] or [Batch]
    # labels shape: [Batch]

    preds = predictions.flatten()
    labs = labels.flatten()

    # 计算 Spearman Rank IC
    ic, _ = spearmanr(preds, labs)

    # 计算 Pearson IC
    # pearson_ic = np.corrcoef(preds, labs)[0, 1]

    return {
        "ic": ic,
        # "pearson_ic": pearson_ic
    }


def run_training():
    print("\n" + "=" * 50)
    print(">>> 启动模型训练 (Training Pipeline)")
    print("=" * 50)

    # 1. 获取数据 (自动调用全内存加载 + 缓存)
    # ds 包含 {'train': ..., 'test': ...}
    ds, num_features = get_dataset()

    print(f"Input Features: {num_features}")
    print(f"Train Size: {len(ds['train'])} | Test Size: {len(ds['test'])}")

    # 2. 配置模型
    model_config = SotaConfig(
        num_input_channels=num_features,
        context_length=Config.CONTEXT_LEN,
        patch_length=8,  # PatchTST 核心参数
        stride=4,  # Patch 步长
        d_model=128,  # 隐层维度
        num_hidden_layers=3,  # 层数
        n_heads=4,
        dropout=0.2
    )

    model = PatchTSTForStock(model_config)

    # 3. 训练参数 (工业级配置)
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=10,  # 适当增加 Epoch，因为有 EarlyStopping
        per_device_train_batch_size=64,
        per_device_eval_batch_size=256,  # 验证集 Batch 可以大一点

        learning_rate=1e-4,
        weight_decay=1e-4,  # L2 正则化

        evaluation_strategy="steps",  # 按步数评估
        eval_steps=200,  # 每 200 步验证一次
        save_steps=200,  # 每 200 步保存一次 Checkpoint
        save_total_limit=3,  # 最多保留 3 个 Checkpoint

        logging_steps=50,
        fp16=True,  # 开启混合精度加速
        dataloader_num_workers=0,  # Windows/Mac 有时多进程会报错，设为0最稳，Linux可设为4

        load_best_model_at_end=True,  # 训练结束后加载最好的模型
        metric_for_best_model="ic",  # 【关键】以 IC 作为最优模型的评判标准
        greater_is_better=True,  # IC 越大越好

        remove_unused_columns=False,  # 防止 feature 列被自动过滤
        report_to="none"  # 不上传 WandB
    )

    # 4. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        compute_metrics=compute_metrics,  # 挂载自定义评估指标
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]  # 连续5次 IC 不提升则停止
    )

    # 5. 开始训练
    print("🚀 开始训练...")
    trainer.train()

    # 6. 保存最终模型
    final_path = f"{Config.OUTPUT_DIR}/final_model"
    trainer.save_model(final_path)
    print(f"✅ 模型训练完成，已保存至: {final_path}")