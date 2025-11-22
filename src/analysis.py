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
        print("\n" + "=" * 60)
        print(">>> 启动全量截面分析 (Full Cross-Sectional Analysis)")
        print("=" * 60)

        if not os.path.exists(self.model_path):
            print(f"❌ 模型未找到: {self.model_path}")
            return

        print(f"加载模型: {self.model_path}")
        model = PatchTSTForStock.from_pretrained(self.model_path).to(self.device)
        model.eval()

        print("加载全市场 Panel 数据 (验证模式)...")
        panel_df, feature_cols = DataProvider.load_and_process_panel(mode='train')

        mask_date = (panel_df['date'] >= (self.start_date - pd.Timedelta(days=Config.CONTEXT_LEN * 2))) & \
                    (panel_df['date'] <= self.end_date)
        df_sub = panel_df[mask_date].copy()

        if df_sub.empty:
            print("❌ 选定区间无数据")
            return

        print("正在构建时序窗口并推理...")

        all_results = []
        batch_size = Config.ANALYSIS_BATCH_SIZE
        batch_inputs = []
        batch_meta = []

        grouped = df_sub.groupby('code')

        for code, group in tqdm(grouped, desc="Processing Stocks"):
            if len(group) < Config.CONTEXT_LEN: continue

            feats = group[feature_cols].values.astype(np.float32)
            dates = group['date'].values

            # 优先获取预计算好的标签
            ranks = group['rank_label'].values if 'rank_label' in group.columns else np.zeros(len(group))
            excess = group['excess_label'].values if 'excess_label' in group.columns else group['target'].values

            seq_len = Config.CONTEXT_LEN

            for i in range(len(group) - seq_len + 1):
                pred_date_ts = dates[i + seq_len - 1]
                pred_date = pd.to_datetime(pred_date_ts)

                if pred_date < self.start_date or pred_date > self.end_date:
                    continue

                batch_inputs.append(feats[i: i + seq_len])
                batch_meta.append({
                    'date': pred_date,
                    'code': code,
                    'rank_label': ranks[i + seq_len - 1],
                    'excess_label': excess[i + seq_len - 1]
                })

                if len(batch_inputs) >= batch_size:
                    self._flush_batch(model, batch_inputs, batch_meta, all_results)
                    batch_inputs = []
                    batch_meta = []

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
        daily_ic = df.groupby('date').apply(
            lambda x: spearmanr(x['score'], x['rank_label'])[0]
        )

        ic_mean = daily_ic.mean()
        ic_std = daily_ic.std()
        icir = ic_mean / (ic_std + 1e-9) * np.sqrt(252)

        print("-" * 40)
        print(f"📊 【因子绩效报告】")
        print(f"Rank IC (Mean): {ic_mean:.4f}")
        print(f"ICIR (Annual) : {icir:.4f}")
        print(f"IC Win Rate   : {(daily_ic > 0).mean():.2%}")
        print("-" * 40)

        def get_layer_ret(g):
            try:
                g['group'] = pd.qcut(g['score'], 5, labels=False, duplicates='drop')
                return g.groupby('group')['excess_label'].mean()
            except:
                return None

        layer_ret = df.groupby('date').apply(get_layer_ret)

        if layer_ret is not None:
            cum_ret = (1 + layer_ret).cumprod()
            long_short = (1 + (layer_ret[4] - layer_ret[0])).cumprod()

            plt.figure(figsize=(14, 8))
            plt.subplot(2, 1, 1)
            colors = ['green', 'lime', 'grey', 'orange', 'red']
            labels = ['Bottom 20%', '40%-60%', 'Middle', '60%-80%', 'Top 20%']

            for i in range(5):
                if i in cum_ret.columns:
                    plt.plot(cum_ret.index, cum_ret[i], label=labels[i], color=colors[i], alpha=0.8)

            plt.plot(long_short.index, long_short, label='Long-Short (Alpha)', color='blue', linestyle='--',
                     linewidth=2)
            plt.title('Layered Backtest (Cumulative Excess Return)')
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3)

            plt.subplot(2, 1, 2)
            plt.bar(daily_ic.index, daily_ic.values, color='orange', alpha=0.5, label='Daily IC')
            plt.axhline(ic_mean, color='red', linestyle='--', label=f'Mean IC: {ic_mean:.3f}')
            plt.title('Daily Rank IC Series')
            plt.legend()

            save_path = os.path.join(Config.OUTPUT_DIR, "cross_section_analysis.png")
            plt.tight_layout()
            plt.savefig(save_path)
            print(f"📈 报表已保存: {save_path}")