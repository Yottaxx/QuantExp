import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from .config import Config
from .model import PatchTSTForStock
from .data_provider import DataProvider

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class BacktestAnalyzer:
    def __init__(self, use_test_set_only=True):
        """
        :param use_test_set_only: 如果为 True，自动覆盖 start_date 为测试集起始日
        """
        self.device = Config.DEVICE
        # self.model_path = f"{Config.OUTPUT_DIR}/final_model"
        self.model_path= "/Users/yotta/PycharmProjects/QuantExp/output/checkpoints/checkpoint-3000"

        self.results_df = None
        self.use_test_set_only = use_test_set_only

        # 默认先占位，稍后在加载数据时动态修正
        self.start_date = pd.to_datetime(Config.START_DATE)
        self.end_date = pd.to_datetime("2099-12-31")

    def generate_historical_predictions(self):
        print("\n" + "=" * 60)
        print(">>> [Analysis] 启动全量截面分析")
        print("=" * 60)

        if not os.path.exists(self.model_path):
            print(f"❌ 模型未找到: {self.model_path}")
            return

        model = PatchTSTForStock.from_pretrained(self.model_path).to(self.device)
        model.eval()

        # 1. 加载全量带 Label 的数据
        print("Loading Full Panel Data (with labels)...")
        panel_df, feature_cols = DataProvider.load_and_process_panel(mode='train')

        # 2. [关键修改] 自动定位测试集范围
        unique_dates = np.sort(panel_df['date'].unique())

        if self.use_test_set_only:
            # 复用 DataProvider 中的切分逻辑 (90% 训练, 10% 测试)
            split_idx = int(len(unique_dates) * 0.9)

            # 加上 Gap 防止数据泄露 (Context Len)
            test_start_idx = min(split_idx + Config.CONTEXT_LEN, len(unique_dates) - 1)

            self.start_date = pd.to_datetime(unique_dates[test_start_idx])
            self.end_date = pd.to_datetime(unique_dates[-1])

            print(f"\n🔒 [Auto-Split] 已锁定样本外测试集 (Out-of-Sample):")
            print(f"   训练集范围: {unique_dates[0]} ~ {unique_dates[split_idx]}")
            print(f"   测试集范围: {self.start_date.date()} ~ {self.end_date.date()}")
        else:
            # 如果想看全量，则使用 config 的时间
            print(f"\n⚠️ [Warning] 正在分析全量数据 (含训练集)，结果可能虚高！")

        # 3. 筛选时间窗口
        # 需要预留 Context Length 的数据用于 Lookback，所以物理读取的 start 要前推
        read_start_date = self.start_date - pd.Timedelta(days=Config.CONTEXT_LEN * 2 + 60)

        mask_date = (panel_df['date'] >= read_start_date) & (panel_df['date'] <= self.end_date)
        df_sub = panel_df[mask_date].copy()

        if df_sub.empty:
            print("❌ 选定区间无有效数据")
            return

        print("Start Batch Inference...")
        all_results = []
        batch_inputs, batch_meta = [], []

        feat_vals = df_sub[feature_cols].values.astype(np.float32)
        dates = df_sub['date'].values
        codes = df_sub['code'].values

        # 优先使用 rank_label
        if 'rank_label' in df_sub.columns:
            labels = df_sub['rank_label'].values
        else:
            labels = df_sub['target'].values

        has_excess = 'excess_label' in df_sub.columns
        excess_vals = df_sub['excess_label'].values if has_excess else df_sub['target'].values

        unique_codes, code_indices = np.unique(codes, return_index=True)
        code_indices = np.append(code_indices, len(codes))

        seq_len = Config.CONTEXT_LEN
        batch_size = Config.ANALYSIS_BATCH_SIZE

        for k in tqdm(range(len(unique_codes)), desc="Processing Stocks"):
            start_pos = code_indices[k]
            end_pos = code_indices[k + 1]
            if end_pos - start_pos < seq_len: continue

            # 筛选只在 Analysis 区间内的日期进行预测
            curr_dates = dates[start_pos + seq_len - 1: end_pos]
            valid_mask = (curr_dates >= np.datetime64(self.start_date)) & \
                         (curr_dates <= np.datetime64(self.end_date))

            if not np.any(valid_mask): continue

            valid_offsets = np.where(valid_mask)[0]

            for offset in valid_offsets:
                pred_idx = start_pos + seq_len - 1 + offset
                window_start = start_pos + offset
                window_end = window_start + seq_len

                batch_inputs.append(feat_vals[window_start:window_end])
                batch_meta.append({
                    'date': dates[pred_idx],
                    'code': codes[pred_idx],
                    'rank_label': labels[pred_idx],
                    'excess_label': excess_vals[pred_idx]
                })

                if len(batch_inputs) >= batch_size:
                    self._flush_batch(model, batch_inputs, batch_meta, all_results)
                    batch_inputs, batch_meta = [], []

        if batch_inputs:
            self._flush_batch(model, batch_inputs, batch_meta, all_results)

        self.results_df = pd.DataFrame(all_results)
        self.results_df['date'] = pd.to_datetime(self.results_df['date'])
        print(f"✅ 推理完成，生成 {len(self.results_df)} 条预测记录。")

    def _flush_batch(self, model, inputs, meta, results_list):
        tensor = torch.tensor(np.array(inputs), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = model(past_values=tensor)
            scores = outputs.logits.squeeze().cpu().numpy()

        if scores.ndim == 0: scores = [scores]
        limit = min(len(meta), len(scores))
        for i in range(limit):
            meta[i]['score'] = float(scores[i])
            results_list.append(meta[i])

    def analyze_performance(self):
        if self.results_df is None or self.results_df.empty:
            print("⚠️ 结果集为空")
            return

        df = self.results_df.copy()

        # 1. 计算 Rank IC
        df['score_rank'] = df.groupby('date')['score'].rank(pct=True)
        df['label_rank'] = df.groupby('date')['rank_label'].rank(pct=True)

        daily_ic = df.groupby('date').apply(lambda x: x['score_rank'].corr(x['label_rank']))

        # 2. 统计
        ic_mean = daily_ic.mean()
        ic_std = daily_ic.std()
        icir = ic_mean / (ic_std + 1e-9) * np.sqrt(252)
        ic_win_rate = (daily_ic > 0).mean()

        print("-" * 50)
        # 打印当前分析的时间段，再次确认
        print(f"📊 【因子深度绩效报告】 (区间: {self.start_date.date()} ~ {self.end_date.date()})")
        print("-" * 50)
        print(f"Rank IC (Mean) : {ic_mean:.4f}")
        print(f"ICIR (Annual)  : {icir:.4f}")
        print(f"IC Win Rate    : {ic_win_rate:.2%}")
        print("-" * 50)

        self._plot_results(df, daily_ic, ic_mean, icir, ic_win_rate)

    def _plot_results(self, df, daily_ic, ic_mean, icir, ic_win_rate):
        # ... (绘图代码保持不变，请直接复用之前的 _plot_results) ...
        # 为节省篇幅，此处省略绘图部分，逻辑完全一致
        pass


if __name__ == "__main__":
    # 默认开启 True，只分析测试集
    analyzer = BacktestAnalyzer(use_test_set_only=True)
    analyzer.generate_historical_predictions()
    analyzer.analyze_performance()