import torch
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from dataclasses import dataclass

# 假设引用您的项目模块
from src.config import Config
from src.model import PatchTSTForStock
from src.data_provider import DataProvider


@dataclass
class PredictionResult:
    code: str
    score: float
    target_date: pd.Timestamp


class RealTimePredictor:
    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        self.seq_len = Config.CONTEXT_LEN
        self.model_path = os.path.join(Config.OUTPUT_DIR, "final_model")
        self.model = self._load_model()

    def _load_model(self) -> PatchTSTForStock:
        """加载模型并设置为评估模式，确保无梯度计算"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"❌ Critical: Model artifact not found at {self.model_path}")

        # 显式使用 CPU/GPU 映射，防止跨设备加载错误
        model = PatchTSTForStock.from_pretrained(self.model_path).to(self.device)
        model.eval()
        return model

    def predict_next_day(self, date_str: str) -> pd.DataFrame:
        """
        核心入口：给定日期 T (date_str)，预测 T+1 日的表现。
        逻辑：
        1. 获取 [T - Lookback, T] 的数据。
        2. 严格筛选最后一天必须是 T 的股票。
        3. 批量推理。
        """
        target_date = pd.to_datetime(date_str)
        print(f"\n⚡ [Inference] Target Date (T): {target_date.date()} | Predicting for: T+1")

        # 1. 数据准备 (Data Preparation)
        # 策略：向前多取一些数据 (2.5倍 Context)，以应对非交易日或停牌带来的Gap
        # 这样能保证绝大多数股票都能凑齐 seq_len 长度的交易日数据
        lookback_days = int(self.seq_len * 2.5)
        start_date = target_date - pd.Timedelta(days=lookback_days)

        # 调用 DataProvider (需用户实现该接口支持日期过滤)
        # 关键：这里获取的数据必须已经做过 归一化(Normalization) 处理，且与训练时一致！
        panel_df, feature_cols = DataProvider.load_and_process_panel(
            start_date=start_date,
            end_date=target_date
        )

        if panel_df.empty:
            print(f"⚠️ Warning: No data found ending on {target_date.date()}. Is it a trading day?")
            return pd.DataFrame()

        # 2. 张量构建 (Tensor Construction)
        inputs, meta_data = self._build_inference_batches(
            panel_df, feature_cols, target_date
        )

        if not inputs:
            print("⚠️ Warning: No valid sequences constructed. Check data integrity or date matching.")
            return pd.DataFrame()

        # 3. 模型推理 (Model Inference)
        scores = self._run_inference(inputs)

        # 4. 结果整合 (Result Aggregation)
        results = []
        for meta, score in zip(meta_data, scores):
            results.append({
                'date': target_date,
                'code': meta['code'],
                'score': score
            })

        df_res = pd.DataFrame(results).sort_values(by='score', ascending=False).reset_index(drop=True)
        return df_res

    def _build_inference_batches(
            self,
            df: pd.DataFrame,
            cols: List[str],
            target_date: pd.Timestamp
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        构建推理批次。
        核心逻辑：
        使用 Numpy 向量化操作进行切片，避免 Pandas GroupBy 的低效循环。
        严格校验：Sequence 的最后一天必须等于 Target Date。
        """
        # 确保数据按 code, date 排序
        df = df.sort_values(['code', 'date'])

        feat_vals = df[cols].values.astype(np.float32)
        codes = df['code'].values
        dates = df['date'].values

        # 找到每个 code 的切分点
        unique_codes, code_indices = np.unique(codes, return_index=True)
        code_indices = np.append(code_indices, len(codes))  # 添加末尾索引

        valid_inputs = []
        valid_meta = []

        # 使用 tqdm 显示进度，因为股票数量可能很大
        for k in range(len(unique_codes)):
            start_idx = code_indices[k]
            end_idx = code_indices[k + 1]

            curr_len = end_idx - start_idx

            # Check 1: 长度不足
            if curr_len < self.seq_len:
                continue

            # Check 2: (Critical) 锚定检查
            # 最后一个数据点的时间戳必须严格等于 target_date
            # 如果不等于，说明该股票在 target_date 停牌或数据缺失，不可预测
            last_date = dates[end_idx - 1]
            if last_date != np.datetime64(target_date):
                # Optional: 记录日志 "Stock {unique_codes[k]} skipped: Last date {last_date} != {target_date}"
                continue

            # 构造切片
            # 取最后 seq_len 行
            slice_start = end_idx - self.seq_len
            slice_end = end_idx

            seq = feat_vals[slice_start:slice_end]

            # Check 3: (Optional but Safe) 检查 NaN
            # 如果输入包含 NaN，模型输出也会是 NaN
            if np.isnan(seq).any():
                continue

            valid_inputs.append(seq)
            valid_meta.append({'code': unique_codes[k]})

        return valid_inputs, valid_meta

    def _run_inference(self, inputs_list: List[np.ndarray]) -> np.ndarray:
        """批量执行推理，优化显存使用"""
        batch_size = Config.ANALYSIS_BATCH_SIZE
        all_scores = []

        # 将 list 转为 tensor 并不是最高效的，如果内存允许，可以预分配大 Tensor
        # 这里为了通用性使用 batch 迭代
        total_samples = len(inputs_list)

        with torch.no_grad():
            for i in range(0, total_samples, batch_size):
                batch_data = inputs_list[i: i + batch_size]

                # 转换为 Tensor [Batch, Seq, Feat]
                tensor_batch = torch.tensor(np.array(batch_data), dtype=torch.float32).to(self.device)

                # Forward
                outputs = self.model(past_values=tensor_batch)

                # 处理 logits
                logits = outputs.logits.squeeze()
                if logits.ndim == 0:
                    logits = logits.unsqueeze(0)

                all_scores.append(logits.cpu().numpy())

        if not all_scores:
            return np.array([])

        return np.concatenate(all_scores)


# --- 接口模拟与调用示例 ---

if __name__ == "__main__":
    # 模拟配置 (实际使用时不需要)
    # 用户需要设置的预测日期 (即拥有完整收盘数据的最后一天)
    TARGET_DATE = '2025-11-15'

    try:
        predictor = RealTimePredictor()

        # 执行预测
        df_rank = predictor.predict_next_day(TARGET_DATE)

        if not df_rank.empty:
            print("\n" + "=" * 50)
            print(f"🚀 Top 10 Predictions for Next Trading Day (Based on {TARGET_DATE})")
            print("=" * 50)
            print(df_rank.head(10).to_markdown(index=False))

            # 这里的 Top 1 就是模型最看好的股票
        else:
            print("No predictions generated.")

    except Exception as e:
        import traceback

        traceback.print_exc()