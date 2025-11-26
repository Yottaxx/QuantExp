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
    def __init__(self, start_date=None, end_date=None, use_test_set_only=True):
        """
        初始化分析器
        :param start_date: 自定义开始日期 (仅在 use_test_set_only=False 时生效)
        :param end_date: 自定义结束日期 (仅在 use_test_set_only=False 时生效)
        :param use_test_set_only: 是否强制使用测试集 (默认 True，优先级最高)
        """
        self.device = Config.DEVICE
        self.model_path = f"{Config.OUTPUT_DIR}/final_model"
        self.results_df = None

        self.use_test_set_only = use_test_set_only
        self.user_start_date = start_date
        self.user_end_date = end_date

        # 实际分析的起止日期 (将在加载数据后计算)
        self.analysis_start_date = None
        self.analysis_end_date = None

    def _resolve_analysis_range(self, panel_df):
        """
        根据模式解析实际的分析时间范围
        """
        unique_dates = np.sort(panel_df['date'].unique())
        n_dates = len(unique_dates)

        if self.use_test_set_only:
            # --- 模式 A: 自动锁定测试集 (严格防泄漏) ---
            train_end_idx = int(n_dates * Config.TRAIN_RATIO)
            val_end_idx = int(n_dates * (Config.TRAIN_RATIO + Config.VAL_RATIO))

            # Test Set Start = Valid End + Gap (Context Len)
            # 必须跳过 Gap，防止 Valid 集末尾的数据作为 History 泄漏给 Test 集开头
            test_start_idx = min(val_end_idx + Config.CONTEXT_LEN, n_dates - 1)

            self.analysis_start_date = pd.to_datetime(unique_dates[test_start_idx])
            self.analysis_end_date = pd.to_datetime(unique_dates[-1])

            print(f"\n🔒 [Mode: Test Set Only] 已自动锁定样本外区间:")
            print(f"   范围: {self.analysis_start_date.date()} ~ {self.analysis_end_date.date()}")

        else:
            # --- 模式 B: 用户自定义 (灵活分析) ---
            # 如果用户未指定，默认使用 Config 或全量范围
            s_date = self.user_start_date or Config.START_DATE
            e_date = self.user_end_date or "2099-12-31"

            self.analysis_start_date = pd.to_datetime(s_date)
            self.analysis_end_date = pd.to_datetime(e_date)

            print(f"\n🔓 [Mode: Custom Range] 使用自定义区间:")
            print(f"   范围: {self.analysis_start_date.date()} ~ {self.analysis_end_date.date()}")

            # 警告：如果自定义区间覆盖了训练集，提示风险
            train_limit_date = pd.to_datetime(unique_dates[int(n_dates * Config.TRAIN_RATIO)])
            if self.analysis_start_date < train_limit_date:
                print(f"   ⚠️ 警告: 该区间包含训练集数据 ({train_limit_date.date()} 之前)，结果可能存在过拟合!")

    def generate_historical_predictions(self):
        """
        执行推理
        """
        print("\n" + "=" * 60)
        print(">>> [Analysis] 启动截面分析与推理")
        print("=" * 60)

        if not os.path.exists(self.model_path):
            print(f"❌ 模型未找到: {self.model_path}")
            return

        # 1. 加载模型
        print(f"Loading Model: {self.model_path}")
        model = PatchTSTForStock.from_pretrained(self.model_path).to(self.device)
        model.eval()

        # 2. 加载全量数据 (用于定位日期和提取特征)
        # mode='train' 仅表示加载包含 Label 的数据结构，并非只加载训练集
        print("Loading Full Panel Data...")
        panel_df, feature_cols = DataProvider.load_and_process_panel(mode='train')

        # 3. 解析时间范围
        self._resolve_analysis_range(panel_df)

        # 4. 数据切片 (物理读取范围)
        # 为了预测 T 日，我们需要 T - Context_Len 的历史数据
        # 所以物理读取的 Start Date 必须比 Analysis Start Date 早
        lookback_buffer = Config.CONTEXT_LEN * 2 + 60  # 预留充足 buffer
        read_start_date = self.analysis_start_date - pd.Timedelta(days=lookback_buffer)

        mask_date = (panel_df['date'] >= read_start_date) & (panel_df['date'] <= self.analysis_end_date)
        df_sub = panel_df[mask_date].copy()

        if df_sub.empty:
            print("❌ 选定区间无有效数据")
            return

        print("Start Batch Inference...")
        all_results = []
        batch_inputs, batch_meta = [], []

        # 预处理数据加速读取
        feat_vals = df_sub[feature_cols].values.astype(np.float32)
        dates = df_sub['date'].values
        codes = df_sub['code'].values

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

        # 遍历所有股票
        for k in tqdm(range(len(unique_codes)), desc="Processing Stocks"):
            start_pos = code_indices[k]
            end_pos = code_indices[k + 1]

            # 数据长度不足以构建一个窗口
            if end_pos - start_pos < seq_len: continue

            # 筛选出 [Analysis Start, Analysis End] 区间内的日期索引
            curr_dates = dates[start_pos + seq_len - 1: end_pos]
            valid_mask = (curr_dates >= np.datetime64(self.analysis_start_date)) & \
                         (curr_dates <= np.datetime64(self.analysis_end_date))

            if not np.any(valid_mask): continue

            # 获取相对偏移量
            valid_offsets = np.where(valid_mask)[0]

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
                    batch_inputs, batch_meta = [], []

        # 处理剩余 Batch
        if batch_inputs:
            self._flush_batch(model, batch_inputs, batch_meta, all_results)

        self.results_df = pd.DataFrame(all_results)
        if not self.results_df.empty:
            self.results_df['date'] = pd.to_datetime(self.results_df['date'])
            print(f"✅ 推理完成，生成 {len(self.results_df)} 条预测记录。")
        else:
            print("❌ 未生成任何预测记录，请检查日期范围或数据完整性。")

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
        """
        计算核心指标
        """
        if self.results_df is None or self.results_df.empty:
            print("⚠️ 结果集为空，无法分析")
            return

        df = self.results_df.copy()

        # 1. 计算 Rank IC
        # 在每个日期截面上计算 score 和 label 的相关性
        df['score_rank'] = df.groupby('date')['score'].rank(pct=True)
        df['label_rank'] = df.groupby('date')['rank_label'].rank(pct=True)

        daily_ic = df.groupby('date').apply(
            lambda x: x['score_rank'].corr(x['label_rank'])
        )

        # 2. 统计指标
        ic_mean = daily_ic.mean()
        ic_std = daily_ic.std()
        # 年化 ICIR
        icir = ic_mean / (ic_std + 1e-9) * np.sqrt(252)
        # 胜率
        ic_win_rate = (daily_ic > 0).mean()

        print("-" * 50)
        print(f"📊 【因子深度绩效报告】")
        print(f"   分析区间: {self.analysis_start_date.date()} ~ {self.analysis_end_date.date()}")
        print("-" * 50)
        print(f"Rank IC (Mean) : {ic_mean:.4f}   (参考: >0.03 优秀)")
        print(f"ICIR (Annual)  : {icir:.4f}     (参考: >1.00 稳定)")
        print(f"IC Win Rate    : {ic_win_rate:.2%}   (参考: >55%  胜率)")
        print("-" * 50)

        self._plot_results(df, daily_ic, ic_mean, icir, ic_win_rate)

    def _plot_results(self, df, daily_ic, ic_mean, icir, ic_win_rate):
        """
        绘制分析图表
        """
        plt.figure(figsize=(16, 12))

        # Subplot 1: 累积 IC
        ax1 = plt.subplot(3, 1, 1)
        daily_ic_cumsum = daily_ic.cumsum()
        ax1.plot(daily_ic_cumsum.index, daily_ic_cumsum.values, label='Cumulative Rank IC', color='#4B0082',
                 linewidth=1.5)
        ax1.set_title(f'Cumulative Rank IC (ICIR={icir:.2f})', fontsize=12, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.4)
        ax1.legend(loc='upper left')

        # Subplot 2: 每日 IC
        ax2 = plt.subplot(3, 1, 2)
        colors = ['#d32f2f' if v < 0 else '#388e3c' for v in daily_ic.values]
        ax2.bar(daily_ic.index, daily_ic.values, color=colors, alpha=0.6, width=1.0, label='Daily IC')
        ax2.axhline(ic_mean, color='blue', linestyle='--', linewidth=1.5, label=f'Mean IC: {ic_mean:.3f}')
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_title(f'Daily IC Distribution (Win Rate={ic_win_rate:.1%})', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, axis='y', linestyle='--', alpha=0.4)

        # Subplot 3: 分层回测
        ax3 = plt.subplot(3, 1, 3)

        # 分组
        df['group'] = df.groupby('date')['score'].transform(
            lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
        )

        # 计算每日各组平均超额收益
        layer_ret = df.groupby(['date', 'group'])['excess_label'].mean().unstack()

        # 简单平摊多日收益
        if Config.PRED_LEN > 1:
            layer_ret = layer_ret / Config.PRED_LEN

        layer_ret = layer_ret.fillna(0)
        cum_ret = (1 + layer_ret).cumprod()

        # 绘图
        groups = sorted(layer_ret.columns)
        for idx, g in enumerate(groups):
            if g == groups[-1]:
                label, c, lw, alpha = "Top 20% (Long)", "#d32f2f", 2.0, 1.0  # Red
            elif g == groups[0]:
                label, c, lw, alpha = "Bottom 20% (Short)", "#388e3c", 1.5, 0.8  # Green
            else:
                label, c, lw, alpha = f"Group {g}", "gray", 0.8, 0.3

            ax3.plot(cum_ret.index, cum_ret[g], label=label, color=c, linewidth=lw, alpha=alpha)

        # 多空曲线
        if len(groups) >= 2:
            ls_ret = layer_ret[groups[-1]] - layer_ret[groups[0]]
            ls_cum = (1 + ls_ret).cumprod()
            ax3.plot(ls_cum.index, ls_cum, label='Long-Short Alpha', color='blue', linestyle='--', linewidth=1.5)

        ax3.set_title('Layered Backtest & Long-Short Alpha', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper left', ncol=2)
        ax3.grid(True, linestyle='--', alpha=0.4)

        plt.tight_layout()
        save_path = os.path.join(Config.OUTPUT_DIR, "factor_comprehensive_report.png")
        plt.savefig(save_path, dpi=150)
        print(f"📈 图表已保存至: {save_path}")


if __name__ == "__main__":
    # === 使用示例 ===

    # 场景 1: 严格验证 (推荐)
    # 自动计算 Test Set 范围，防止数据泄漏
    print(">>> Mode 1: Auto Test Set")
    analyzer = BacktestAnalyzer(use_test_set_only=True)
    analyzer.generate_historical_predictions()
    analyzer.analyze_performance()

    # 场景 2: 灵活分析 (用于复盘特定历史时期)
    # print("\n>>> Mode 2: Custom Range")
    # analyzer = BacktestAnalyzer(start_date='2023-01-01', end_date='2023-06-30', use_test_set_only=False)
    # analyzer.generate_historical_predictions()
    # analyzer.analyze_performance()