import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from .config import Config
from .model import PatchTSTForStock
from .data_provider import DataProvider

# 设置 matplotlib 风格，防止中文乱码
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class BacktestAnalyzer:
    def __init__(self, start_date='2024-01-01', end_date='2025-12-31'):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.device = Config.DEVICE
        self.model_path = f"{Config.OUTPUT_DIR}/final_model"
        self.results_df = None

    def generate_historical_predictions(self):
        """
        [Step 1] 全量历史回溯推理
        使用训练好的模型，对目标区间内的全市场股票进行滚动预测
        """
        print("\n" + "=" * 60)
        print(">>> [Analysis] 启动全量截面分析 (Full Cross-Sectional Inference)")
        print("=" * 60)

        if not os.path.exists(self.model_path):
            print(f"❌ 模型未找到: {self.model_path}，请先运行 train.py")
            return

        print(f"Loading Model: {self.model_path}")
        model = PatchTSTForStock.from_pretrained(self.model_path).to(self.device)
        model.eval()

        # 加载数据 (Train 模式包含 Label，用于后续验证)
        # 强制刷新缓存以确保数据是最新的，或者根据需求 remove force_refresh
        print("Loading Panel Data (Train Mode)...")
        panel_df, feature_cols = DataProvider.load_and_process_panel(mode='train')

        # 筛选时间窗口：需要预留 Context Length 的数据用于 Lookback
        start_buffer = self.start_date - pd.Timedelta(days=Config.CONTEXT_LEN * 2 + 60)
        mask_date = (panel_df['date'] >= start_buffer) & (panel_df['date'] <= self.end_date)
        df_sub = panel_df[mask_date].copy()

        if df_sub.empty:
            print("❌ 选定区间无有效数据")
            return

        print(f"Inference Range: {self.start_date.date()} ~ {self.end_date.date()}")
        print("Start Batch Inference...")

        all_results = []
        batch_inputs = []
        batch_meta = []  # 存储元数据 (date, code, label)

        # 预计算字段索引，避免在循环中频繁字符串查找
        feat_vals = df_sub[feature_cols].values.astype(np.float32)
        dates = df_sub['date'].values
        codes = df_sub['code'].values

        # 优先使用 rank_label (如有)，否则用 raw target
        if 'rank_label' in df_sub.columns:
            labels = df_sub['rank_label'].values
        else:
            labels = df_sub['target'].values

        # 识别 excess_label
        has_excess = 'excess_label' in df_sub.columns
        excess_vals = df_sub['excess_label'].values if has_excess else df_sub['target'].values

        # 获取每个 code 的切片位置，替代 groupby 以提升性能
        # 前提：df_sub 已经按 code, date 排序 (DataProvider 保证了这点)
        # 利用 pandas 的 index 特性或者 numpy diff 找边界
        unique_codes, code_indices = np.unique(codes, return_index=True)
        # 追加最后一个索引作为结束边界
        code_indices = np.append(code_indices, len(codes))

        seq_len = Config.CONTEXT_LEN
        batch_size = Config.ANALYSIS_BATCH_SIZE

        # 遍历每只股票
        for k in tqdm(range(len(unique_codes)), desc="Processing Stocks"):
            start_pos = code_indices[k]
            end_pos = code_indices[k + 1]

            # 该股票的数据长度
            series_len = end_pos - start_pos
            if series_len < seq_len:
                continue

            # 向量化构建切片索引
            # 我们需要预测的时间点索引：从 (start + seq_len - 1) 到 (end - 1)
            # 对应的输入窗口起始点：从 start 到 (end - seq_len)

            # 筛选符合 date 范围的索引
            curr_dates = dates[start_pos + seq_len - 1: end_pos]
            valid_mask = (curr_dates >= np.datetime64(self.start_date)) & \
                         (curr_dates <= np.datetime64(self.end_date))

            if not np.any(valid_mask):
                continue

            # 相对偏移量
            valid_offsets = np.where(valid_mask)[0]

            # 构建 Batch
            for offset in valid_offsets:
                # 绝对索引
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
                    batch_inputs = []
                    batch_meta = []

        # 处理剩余尾部数据
        if batch_inputs:
            self._flush_batch(model, batch_inputs, batch_meta, all_results)

        self.results_df = pd.DataFrame(all_results)
        # 转换日期格式确保对齐
        self.results_df['date'] = pd.to_datetime(self.results_df['date'])
        print(f"✅ 推理完成，生成 {len(self.results_df)} 条预测记录。")

    def _flush_batch(self, model, inputs, meta, results_list):
        """批量推理并回填结果"""
        tensor = torch.tensor(np.array(inputs), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = model(past_values=tensor)
            # 兼容不同维度的输出
            scores = outputs.logits
            if scores.dim() > 1:
                scores = scores.squeeze()
            scores = scores.cpu().numpy()

        # 处理标量或单样本情况
        if scores.ndim == 0:
            scores = [scores]

        # 安全对齐
        limit = min(len(meta), len(scores))
        for i in range(limit):
            meta[i]['score'] = float(scores[i])
            results_list.append(meta[i])

    def analyze_performance(self):
        """
        [Step 2] 计算核心指标 (IC, ICIR, WinRate) 并绘图
        """
        if self.results_df is None or self.results_df.empty:
            print("⚠️ 结果集为空，无法分析")
            return

        df = self.results_df.copy()

        print("\n>>> [Metrics] 计算每日截面指标...")

        # ----------------------------------------------------------------------
        # 1. 高效 IC 计算 (使用 Groupby + Rank + Corr 替代 循环 Spearmanr)
        # Spearman IC 本质上就是 Rank 后的 Pearson IC
        # ----------------------------------------------------------------------
        # 先在组内计算 Rank
        df['score_rank'] = df.groupby('date')['score'].rank(pct=True)
        df['label_rank'] = df.groupby('date')['rank_label'].rank(pct=True)

        # 计算每日相关性 (Rank IC)
        daily_ic = df.groupby('date').apply(
            lambda x: x['score_rank'].corr(x['label_rank'])
        )

        # ----------------------------------------------------------------------
        # 2. 核心指标统计
        # ----------------------------------------------------------------------
        ic_mean = daily_ic.mean()
        ic_std = daily_ic.std()

        # 年化 ICIR = Mean / Std * sqrt(252)
        icir = ic_mean / (ic_std + 1e-9) * np.sqrt(252)

        # IC 胜率
        ic_win_rate = (daily_ic > 0).mean()

        # 打印体检报告
        print("-" * 50)
        print(f"📊 【因子深度绩效报告】 ({self.start_date.date()} ~ {self.end_date.date()})")
        print("-" * 50)
        print(f"Rank IC (Mean) : {ic_mean:.4f}   (参考: >0.03 优秀)")
        print(f"ICIR (Annual)  : {icir:.4f}     (参考: >1.00 稳定)")
        print(f"IC Win Rate    : {ic_win_rate:.2%}   (参考: >55%  胜率)")
        print(f"IC Std Dev     : {ic_std:.4f}")
        print("-" * 50)

        # ----------------------------------------------------------------------
        # 3. 生成可视化报表
        # ----------------------------------------------------------------------
        self._plot_results(df, daily_ic, ic_mean, icir, ic_win_rate)

    def _plot_results(self, df, daily_ic, ic_mean, icir, ic_win_rate):
        """
        [Step 3] 绘制深度分析图表
        """
        plt.figure(figsize=(16, 12))

        # --- 子图 1: 累积 IC 曲线 (Cumulative IC) ---
        # 它是判断因子稳定性的金标准，斜率越稳定越好
        ax1 = plt.subplot(3, 1, 1)
        daily_ic_cumsum = daily_ic.cumsum()
        ax1.plot(daily_ic_cumsum.index, daily_ic_cumsum.values, label='Cumulative Rank IC', color='#4B0082',
                 linewidth=1.5)
        ax1.set_title(f'Cumulative Rank IC (ICIR={icir:.2f})', fontsize=12, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.4)
        ax1.legend(loc='upper left')

        # --- 子图 2: 每日 IC 分布柱状图 ---
        ax2 = plt.subplot(3, 1, 2)
        colors = ['#d32f2f' if v < 0 else '#388e3c' for v in daily_ic.values]  # 红绿柱
        ax2.bar(daily_ic.index, daily_ic.values, color=colors, alpha=0.6, width=1.0, label='Daily IC')
        ax2.axhline(ic_mean, color='blue', linestyle='--', linewidth=1.5, label=f'Mean IC: {ic_mean:.3f}')
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_title(f'Daily IC Distribution (Win Rate={ic_win_rate:.1%})', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, axis='y', linestyle='--', alpha=0.4)

        # --- 子图 3: 分层累计收益曲线 (Layered Backtest) ---
        ax3 = plt.subplot(3, 1, 3)

        # 计算分层收益
        # 将 score 分成 5 组 (Group 0: Worst, Group 4: Best)
        # 注意：duplicates='drop' 防止分数过于集中导致切分失败
        df['group'] = df.groupby('date')['score'].transform(
            lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
        )

        # 计算每组每日的平均 excess_label
        layer_ret = df.groupby(['date', 'group'])['excess_label'].mean().unstack()

        # [Critical Fix] 修正多日预测带来的收益重叠
        # 如果预测的是未来 5 日收益，每日累乘会导致收益被放大 5 倍
        # 这里进行简单的线性平摊，模拟日频收益
        if Config.PRED_LEN > 1:
            layer_ret = layer_ret / Config.PRED_LEN

        layer_ret = layer_ret.fillna(0)
        cum_ret = (1 + layer_ret).cumprod()

        # 绘图逻辑
        groups = sorted(layer_ret.columns)
        cmap = plt.get_cmap('RdYlGn_r')  # 逆序：红(Top) -> 绿(Bottom)

        for idx, g in enumerate(groups):
            if g == groups[-1]:
                label, c, lw, alpha = "Top 20% (Long)", "red", 2.0, 1.0
            elif g == groups[0]:
                label, c, lw, alpha = "Bottom 20% (Short)", "green", 1.5, 0.8
            else:
                label, c, lw, alpha = f"Group {g}", "gray", 0.8, 0.3

            ax3.plot(cum_ret.index, cum_ret[g], label=label, color=c, linewidth=lw, alpha=alpha)

        # 绘制多空曲线 (Long - Short)
        if len(groups) >= 2:
            ls_ret = layer_ret[groups[-1]] - layer_ret[groups[0]]
            ls_cum = (1 + ls_ret).cumprod()
            ax3.plot(ls_cum.index, ls_cum, label='Long-Short Alpha', color='blue', linestyle='--', linewidth=1.5)

        ax3.set_title(f'Layered Backtest (Avg Daily Return derived from {Config.PRED_LEN}-Day Horizon)', fontsize=12,
                      fontweight='bold')
        ax3.legend(loc='upper left', ncol=2)
        ax3.grid(True, linestyle='--', alpha=0.4)

        plt.tight_layout()
        save_path = os.path.join(Config.OUTPUT_DIR, "factor_comprehensive_report.png")
        plt.savefig(save_path, dpi=150)
        print(f"📈 图表已保存至: {save_path}")


if __name__ == "__main__":
    # 单元测试
    analyzer = BacktestAnalyzer(start_date='2024-01-01', end_date='2024-12-31')
    analyzer.generate_historical_predictions()
    analyzer.analyze_performance()