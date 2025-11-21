from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from .config import Config
from .model import PatchTSTForStock, SotaConfig
from .data_provider import get_dataset


def run_training():
    # 1. 数据
    dataset, num_features = get_dataset()
    print(f"Features: {num_features}")

    # 2. 配置
    model_config = SotaConfig(
        num_input_channels=num_features,
        context_length=Config.CONTEXT_LEN,
        patch_length=Config.PATCH_LEN,
        stride=Config.STRIDE,
        d_model=Config.D_MODEL,
        dropout=Config.DROPOUT
    )
    model = PatchTSTForStock(model_config)

    # 3. 训练参数
    args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE * 2,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        fp16=True,  # 混合精度
        logging_steps=50,
        learning_rate=Config.LR,
        save_total_limit=2,
        load_best_model_at_end=True,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()
    trainer.save_model(f"{Config.OUTPUT_DIR}/final_model")
    print("训练完成。")