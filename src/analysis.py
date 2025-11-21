import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from tqdm import tqdm
from .config import Config
from .model import PatchTSTForStock
from .data_provider import DataProvider


class BacktestAnalyzer:
    def __init__(self, start_date='2024-01-01', end_date='2025-12-31'):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.device = Config.DEVICE
        self.model_path = f"{Config.OUTPUT_DIR}/final_model"
        self.results_df = None

    def generate_historical_predictions(self):
        """全量历史回溯推理"""
        print("\n" + "=" * 60)
        print(">>> 启动全量截面分析 (Full Cross-Sectional Analysis)")
        print("=" * 60)

        if not os.path.exists(self.model_path):
            print(f"❌ 模型未找到: {self.model_path}")
            return

        print(f"加载模型: {self.model_path}")
        model = PatchTSTForStock.from_pretrained(self.model_path).to(self.device)
        model.eval()

        # 1. 加载数据 (Train 模式以获取 Target，利用缓存)
        print("加载全市场 Panel 数据...")
        panel_df, feature_cols = DataProvider.load_and_process_panel(mode='train')

        # 2. 筛选时间段 (为了回测效率，只取目标区间)
        # 注意：要多取 Config.CONTEXT_LEN 天，以便为 start_date 生成窗口
        mask_date = (panel_df['date'] >= (self.start_date - pd.Timedelta(days=60))) & \
                    (panel_df['date'] <= self.end_date)
        df_sub = panel_df[mask_date].copy()

        if df_sub.empty:
            print("❌ 选定区间无数据")
            return

        # 3. 批量推理
        # 策略：按 Code 分组，利用 Numpy 快速切片构建 Batch
        print("正在构建时序窗口并推理...")

        all_results = []
        batch_size = 2048
        batch_inputs = []
        batch_meta = []  # (date, code, label, excess)

        grouped = df_sub.groupby('code')

        for code, group in tqdm(grouped, desc="Processing Stocks"):
            if len(group) < Config.CONTEXT_LEN: continue

            # 提取 Numpy 数组
            feats = group[feature_cols].values.astype(np.float32)
            dates = group['date'].values

            # 目标值 (优先用 rank_label 验证模型能力，用 excess_label 验证赚钱能力)
            # 注意：Panel 中可能包含 rank_label, excess_label, target
            # 我们这里主要记录 excess_label 用于分层回测
            if 'excess_label' in group.columns:
                labels = group['excess_label'].values
            else:
                labels = group['target'].values

            # 滑动窗口切片
            # 我们需要预测的时间点是从 start_date 开始的
            # 窗口 i 对应的数据是 [i : i+seq_len]，预测的是 i+seq_len-1 那个时间点的 Label

            seq_len = Config.CONTEXT_LEN

            # 找到符合时间范围的起始索引
            # dates[i + seq_len - 1] >= self.start_date

            valid_indices = []
            for i in range(len(group) - seq_len + 1):
                pred_date = pd.to_datetime(dates[i + seq_len - 1])
                if pred_date < self.start_date or pred_date > self.end_date:
                    continue

                # 加入 Batch
                batch_inputs.append(feats[i: i + seq_len])
                batch_meta.append({
                    'date': pred_date,
                    'code': code,
                    'label': labels[i + seq_len - 1]
                })

                if len(batch_inputs) >= batch_size:
                    self._flush_batch(model, batch_inputs, batch_meta, all_results)
                    batch_inputs = []
                    batch_meta = []

        # 处理剩余 Batch
        if batch_inputs:
            self._flush_batch(model, batch_inputs, batch_meta, all_results)

        self.results_df = pd.DataFrame(all_results)
        print(f"推理完成，生成 {len(self.results_df)} 条预测记录。")

    def _flush_batch(self, model, inputs, meta, results_list):
        tensor = torch.tensor(np.array(inputs), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = model(past_values=tensor)
            scores = outputs.logits.squeeze().cpu().numpy()

        if scores.ndim == 0: scores = [scores]

        for i, score in enumerate(scores):
            item = meta[i]
            item['score'] = float(score)
            results_list.append(item)

    def analyze_performance(self):
        if self.results_df is None or self.results_df.empty: return

        df = self.results_df.sort_values(['date', 'score'], ascending=[True, False])

        print("\n计算截面 IC 指标...")
        # Rank IC: 预测分 vs 实际超额收益
        daily_ic = df.groupby('date').apply(
            lambda x: spearmanr(x['score'], x['label'])[0]
        )

        ic_mean = daily_ic.mean()
        icir = ic_mean / (daily_ic.std() + 1e-9) * np.sqrt(252)

        print("-" * 40)
        print(f"📊 【因子绩效报告】")
        print(f"Rank IC (Mean): {ic_mean:.4f}")
        print(f"ICIR (Annual) : {icir:.4f}")
        print(f"IC Win Rate   : {(daily_ic > 0).mean():.2%}")
        print("-" * 40)

        # 分层回测
        def get_layer_ret(g):
            try:
                # 分5组，label=4是最高分(Long)，label=0是最低分(Short)
                g['group'] = pd.qcut(g['score'], 5, labels=False, duplicates='drop')
                return g.groupby('group')['label'].mean()
            except:
                return None

        layer_ret = df.groupby('date').apply(get_layer_ret)

        if layer_ret is not None:
            cum_ret = (1 + layer_ret).cumprod()
            long_short = (1 + (layer_ret[4] - layer_ret[0])).cumprod()

            plt.figure(figsize=(14, 8))
            plt.subplot(2, 1, 1)
            colors = ['green', 'lime', 'grey', 'orange', 'red']
            for i in range(5):
                if i in cum_ret.columns:
                    label = "Top 20% (Long)" if i == 4 else f"Group {i}"
                    label = "Bottom 20% (Short)" if i == 0 else label
                    plt.plot(cum_ret.index, cum_ret[i], label=label, color=colors[i])

            plt.plot(long_short.index, long_short, label='Long-Short (Alpha)', color='blue', linestyle='--',
                     linewidth=2)
            plt.title('Layered Backtest (Cumulative Excess Return)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 1, 2)
            plt.bar(daily_ic.index, daily_ic.values, color='orange', alpha=0.5, label='Daily IC')
            plt.axhline(ic_mean, color='red', linestyle='--')
            plt.title('Daily Rank IC')
            plt.legend()

            save_path = os.path.join(Config.OUTPUT_DIR, "cross_section_analysis.png")
            plt.tight_layout()
            plt.savefig(save_path)
            print(f"📈 报表已保存: {save_path}")