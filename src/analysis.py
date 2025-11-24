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
        # 预留足够的 Lookback 窗口
        start_buffer = self.start_date - pd.Timedelta(days=Config.CONTEXT_LEN * 2 + 100)
        mask_date = (panel_df['date'] >= start_buffer) & (panel_df['date'] <= self.end_date)
        df_sub = panel_df[mask_date].copy()

        if df_sub.empty:
            print("❌ 选定区间无数据")
            return

        print("正在构建时序窗口并推理...")

        all_results = []
        batch_size = Config.ANALYSIS_BATCH_SIZE
        batch_inputs = []
        batch_meta = []  # (date, code, rank_label, excess_label)

        grouped = df_sub.groupby('code')

        # [Optim] 使用 tqdm 显示总体进度，而不是每只股票刷屏
        for code, group in tqdm(grouped, desc="Processing Stocks"):
            if len(group) < Config.CONTEXT_LEN: continue

            # 提取 Numpy 数组 (Float32)
            feats = group[feature_cols].values.astype(np.float32)
            dates = group['date'].values

            # 优先获取预计算好的标签
            ranks = group['rank_label'].values if 'rank_label' in group.columns else np.zeros(len(group))
            excess = group['excess_label'].values if 'excess_label' in group.columns else group['target'].values

            seq_len = Config.CONTEXT_LEN

            # [Fix] 优化循环边界，确保索引不越界
            # 我们需要 feats[i: i+seq_len] 作为输入
            # 对应的预测日期是 dates[i + seq_len - 1]
            valid_indices = range(len(group) - seq_len + 1)

            for i in valid_indices:
                pred_date_ts = dates[i + seq_len - 1]
                pred_date = pd.to_datetime(pred_date_ts)

                # 只保留用户指定时间范围内的结果
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

        # Handle batch size 1 or scalar output
        if scores.ndim == 0: scores = [scores]
        if len(meta) > 1 and len(scores) != len(meta):
            # 极少数情况下可能维度不匹配，做个防御
            scores = np.resize(scores, len(meta))

        for i, score in enumerate(scores):
            # [Safe] 确保 meta 索引不过界
            if i >= len(meta): break
            item = meta[i]
            item['score'] = float(score)
            results_list.append(item)

    def analyze_performance(self):
        if self.results_df is None or self.results_df.empty:
            print("⚠️ 无分析结果数据")
            return

        # 按日期和分数排序
        df = self.results_df.sort_values(['date', 'score'], ascending=[True, False])

        print("\n计算截面 IC 指标...")

        # 1. 计算 Rank IC
        # [Fix] 兼容 Pandas 新版 groupby apply
        daily_ic = df.groupby('date')[['score', 'rank_label']].apply(
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

        # 2. 分层回测 (Layered Backtest)
        def get_layer_ret(g):
            try:
                # 尝试分5组
                g['group'] = pd.qcut(g['score'], 5, labels=False, duplicates='drop')
                return g.groupby('group')['excess_label'].mean()
            except:
                return None

        # [Fix] 兼容 Pandas 新版
        layer_ret_raw = df.groupby('date').apply(get_layer_ret)

        if layer_ret_raw is not None and not layer_ret_raw.empty:
            if isinstance(layer_ret_raw, pd.Series):
                layer_ret = layer_ret_raw.unstack(level=-1)
            else:
                layer_ret = layer_ret_raw

            layer_ret = layer_ret.fillna(0.0)

            # ==================================================================
            # [CRITICAL FIX] 修正重叠收益计算逻辑
            # ==================================================================
            # 因为 Config.PRED_LEN = 5，也就是说 excess_label 是未来5天的累计收益。
            # 如果每天都累乘这个收益，会导致收益被重复计算 5 次。
            #
            # 修正方法：
            # 将 N 日累计收益平摊到单日，近似为：Daily_Ret = Total_Ret / N
            # 这是一个线性近似，用于分层对比足够了。严谨的回测请参考 backtest.py。
            # ==================================================================
            if Config.PRED_LEN > 1:
                print(f"⚠️ 检测到多日预测 (PRED_LEN={Config.PRED_LEN})，正在对收益率进行平摊修正...")
                layer_ret = layer_ret / Config.PRED_LEN

            # 计算各组累积收益
            cum_ret = (1 + layer_ret).cumprod()

            plt.figure(figsize=(14, 8))

            # 子图1: 分层收益曲线
            plt.subplot(2, 1, 1)

            # 获取实际存在的组号
            available_groups = sorted(layer_ret.columns)

            # 颜色映射：根据组数动态生成或固定
            cmap = plt.get_cmap('RdYlGn_r')  # 红(好) -> 绿(差)

            for idx, i in enumerate(available_groups):
                # 动态生成标签
                if i == available_groups[-1]:
                    label = "Top Group (Long)"
                    c = 'red'
                    alpha = 1.0
                    lw = 2
                elif i == available_groups[0]:
                    label = "Bottom Group (Short)"
                    c = 'green'
                    alpha = 1.0
                    lw = 1.5
                else:
                    label = f"Group {i}"
                    c = 'grey'
                    alpha = 0.3
                    lw = 1

                plt.plot(cum_ret.index, cum_ret[i], label=label, color=c, alpha=alpha, linewidth=lw)

            # 动态计算多空曲线
            if len(available_groups) >= 2:
                top_grp = available_groups[-1]
                bot_grp = available_groups[0]
                # 多空收益也需要平摊吗？是的，因为 layer_ret 已经平摊过了，直接相减即可
                long_short_ret = layer_ret[top_grp] - layer_ret[bot_grp]
                long_short_cum = (1 + long_short_ret).cumprod()
                plt.plot(long_short_cum.index, long_short_cum, label='Long-Short (Alpha)', color='blue', linestyle='--',
                         linewidth=2)

            plt.title(f'Layered Backtest (Avg Daily Return derived from {Config.PRED_LEN}-Day Horizon)')
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3)

            # 子图2: 每日 IC 柱状图
            plt.subplot(2, 1, 2)
            plt.bar(daily_ic.index, daily_ic.values, color='orange', alpha=0.5, label='Daily IC')
            plt.axhline(ic_mean, color='red', linestyle='--', label=f'Mean IC: {ic_mean:.3f}')
            plt.title('Daily Rank IC Series')
            plt.legend()

            save_path = os.path.join(Config.OUTPUT_DIR, "cross_section_analysis.png")
            plt.tight_layout()
            plt.savefig(save_path)
            print(f"📈 报表已保存: {save_path}")