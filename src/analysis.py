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
        """
        全量历史回溯推理
        逻辑：使用训练好的模型，对过去每一天的全市场股票进行打分
        """
        print("\n" + "=" * 60)
        print(">>> 启动全量截面分析 (Full Cross-Sectional Analysis)")
        print("=" * 60)

        if not os.path.exists(self.model_path):
            print(f"❌ 模型未找到: {self.model_path}")
            return

        print(f"加载模型: {self.model_path}")
        model = PatchTSTForStock.from_pretrained(self.model_path).to(self.device)
        model.eval()

        # 1. 加载数据 (Train 模式，因为我们需要 Target/Label 来验证效果)
        # 利用缓存加速
        print("加载全市场 Panel 数据 (验证模式)...")
        panel_df, feature_cols = DataProvider.load_and_process_panel(mode='train')

        # 2. 筛选时间段 (为了回测效率，只取目标区间 + 窗口期)
        # start_date 往前推 Context_Len 天，确保第一天就能构建窗口
        mask_date = (panel_df['date'] >= (self.start_date - pd.Timedelta(days=Config.CONTEXT_LEN * 2))) & \
                    (panel_df['date'] <= self.end_date)
        df_sub = panel_df[mask_date].copy()

        if df_sub.empty:
            print("❌ 选定区间无数据")
            return

        print("正在构建时序窗口并推理...")

        all_results = []
        batch_size = 2048  # 根据显存调整
        batch_inputs = []
        batch_meta = []  # (date, code, rank_label, excess_label)

        grouped = df_sub.groupby('code')

        for code, group in tqdm(grouped, desc="Processing Stocks"):
            if len(group) < Config.CONTEXT_LEN: continue

            # 提取 Numpy 数组 (Float32)
            feats = group[feature_cols].values.astype(np.float32)
            dates = group['date'].values

            # 获取验证用的真实标签
            # rank_label: 0~1, 用于计算 IC
            # excess_label: 真实超额收益, 用于画资金曲线
            # target: 实盘绝对收益 (Close_N / Open_1 - 1)

            # 优先获取预计算好的标签，如果没有则 fallback
            ranks = group['rank_label'].values if 'rank_label' in group.columns else np.zeros(len(group))
            excess = group['excess_label'].values if 'excess_label' in group.columns else group['target'].values

            # 滑动窗口切片
            # i 是窗口起点，预测的是 i + seq_len - 1 这个时间点的表现
            seq_len = Config.CONTEXT_LEN

            for i in range(len(group) - seq_len + 1):
                # 预测日期是窗口的最后一天
                pred_date_ts = dates[i + seq_len - 1]
                pred_date = pd.to_datetime(pred_date_ts)

                if pred_date < self.start_date or pred_date > self.end_date:
                    continue

                # 加入 Batch
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

        # 按日期和分数排序
        df = self.results_df.sort_values(['date', 'score'], ascending=[True, False])

        print("\n计算截面 IC 指标...")

        # 1. 计算 Rank IC (Spearman Correlation)
        # 预测分(score) vs 真实排名(rank_label)
        # 如果模型好，预测分高的地方，真实排名也应该靠前(接近1.0)
        daily_ic = df.groupby('date').apply(
            lambda x: spearmanr(x['score'], x['rank_label'])[0]
        )

        ic_mean = daily_ic.mean()
        ic_std = daily_ic.std()
        # 年化 ICIR = IC均值 / IC波动率 * sqrt(252)
        icir = ic_mean / (ic_std + 1e-9) * np.sqrt(252)

        print("-" * 40)
        print(f"📊 【因子绩效报告】")
        print(f"Rank IC (Mean): {ic_mean:.4f}  (>0.05 优秀)")
        print(f"ICIR (Annual) : {icir:.4f}    (>2.0 稳定)")
        print(f"IC Win Rate   : {(daily_ic > 0).mean():.2%}")
        print("-" * 40)

        # 2. 分层回测 (Layered Backtest)
        # 将每日股票按分数分为 5 组，看每组的平均超额收益
        def get_layer_ret(g):
            try:
                # 分5组，label=4是最高分(Long)，label=0是最低分(Short)
                g['group'] = pd.qcut(g['score'], 5, labels=False, duplicates='drop')
                # 计算每组的平均超额收益
                return g.groupby('group')['excess_label'].mean()
            except:
                return None

        layer_ret = df.groupby('date').apply(get_layer_ret)

        if layer_ret is not None:
            # 累积收益
            cum_ret = (1 + layer_ret).cumprod()
            # 多空收益 = Top - Bottom
            long_short = (1 + (layer_ret[4] - layer_ret[0])).cumprod()

            plt.figure(figsize=(14, 8))

            # 子图1: 分层收益曲线
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

            # 子图2: 每日 IC 柱状图
            plt.subplot(2, 1, 2)
            plt.bar(daily_ic.index, daily_ic.values, color='orange', alpha=0.5, label='Daily IC')
            plt.axhline(ic_mean, color='red', linestyle='--', label=f'Mean IC: {ic_mean:.3f}')
            plt.title('Daily Rank IC Series')
            plt.legend()
            plt.grid(True, alpha=0.3)

            save_path = os.path.join(Config.OUTPUT_DIR, "cross_section_analysis.png")
            plt.tight_layout()
            plt.savefig(save_path)
            print(f"📈 报表已保存: {save_path}")