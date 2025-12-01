from transformers import TrainingArguments
from src.train.date_batching import DateAwareTrainer, DateBatchingConfig, StockDateCollator

# ds: DatasetDict = DatasetPipeline(...).build_dataset_from_parts(...)[0]
# ds["train"] / ds["validation"] / ds["test"] 每条样本必须有 date_id，且 __getitem__ 返回 past_values/labels/date_id

args = TrainingArguments(
    output_dir="./output/run1",
    per_device_train_batch_size=1,  # 这里不用它来分 batch；batch_sampler 决定 batch
    per_device_eval_batch_size=1,
    remove_unused_columns=False,     # 强烈建议
    dataloader_num_workers=0,        # 先稳，再提
    fp16=False,
)

collator = StockDateCollator(assert_single_date=True)

date_cfg = DateBatchingConfig(
    date_key="date_id",
    batch_size_per_date=None,     # None => 每日全横截面一个 batch（排序最干净）；怕 OOM 再设 512/1024...
    shuffle_dates=True,
    shuffle_within_date=False,
    drop_last_in_date=False,
    seed=args.seed,
)

trainer = DateAwareTrainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    data_collator=collator,
    date_batching=date_cfg,
)

trainer.train()
