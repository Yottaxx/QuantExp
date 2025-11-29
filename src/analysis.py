import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from .config import Config
from .model import PatchTSTForStock
from .data_provider import DataProvider

# 设置 matplotlib 风格
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class BacktestAnalyzer:
    def __init__(self, target_set='test', start_date=None, end_date=None):
        """
        初始化分析器
        :param target_set: 目标数据集模式
               - 'test': (默认) 分析测试集 (基于 Config 比例计算)
               - 'validation' / 'eval': 分析验证集 (基于 Config 比例计算)
               - 'train': 分析训练集
               - 'custom': 自定义范围
        """
        self.device = Config.DEVICE
        self.model_path = f"{Config.OUTPUT_DIR}/final_model"
        self.results_df = None

        self.target_set = target_set.lower()
        self.user_start_date = start_date
        self.user_end_date = end_date

        self.analysis_start_date = None
        self.analysis_end_date = None

    def _resolve_analysis_range(self, panel_df):
        """
        【核心修正】严格根据 Config 比例解析分析范围
        杜绝硬编码 (如 0.9) 导致的 Train/Val/Test 错位
        """
        unique_dates = np.sort(panel_df['date'].unique())
        n_dates = len(unique_dates)

        # 1. 从 Config 读取比例 (Single Source of Truth)
        train_ratio = Config.TRAIN_RATIO
        val_ratio = Config.VAL_RATIO
        # test_ratio = 1.0 - train_ratio - val_ratio (隐式)

        # 2. 计算严格的索引边界 (与 data_provider.py make_dataset 逻辑保持原子级一致)
        train_end_idx = int(n_dates * train_ratio)
        val_end_idx = int(n_dates * (train_ratio + val_ratio))

        # GAP = Context Length (防止时序预测中的数据泄漏)
        gap = Config.CONTEXT_LEN

        if self.target_set == 'test':
            # Test Set: 从 (Train + Val) 结束的位置开始，跳过 Gap
            # 这里的 val_end_idx 就是 Train+Val 的总长度
            start_idx = min(val_end_idx + gap, n_dates - 1)

            self.analysis_start_date = pd.to_datetime(unique_dates[start_idx])
            self.analysis_end_date = pd.to_datetime(unique_dates[-1])

            print(f"\n🔒 [Target: TEST SET] 样本外测试集 (Strict Split):")
            print(f"   配置比例: Train({train_ratio:.0%}) + Val({val_ratio:.0%}) -> Test")

        elif self.target_set in ['validation', 'eval', 'val']:
            # Validation Set: 从 Train 结束的位置开始，跳过 Gap
            start_idx = min(train_end_idx + gap, n_dates - 1)
            end_idx = min(val_end_idx, n_dates - 1)

            self.analysis_start_date = pd.to_datetime(unique_dates[start_idx])
            self.analysis_end_date = pd.to_datetime(unique_dates[end_idx])

            print(f"\n🔓 [Target: VALIDATION SET] 验证集 (Strict Split):")
            print(f"   配置比例: Train({train_ratio:.0%}) -> Val")

        elif self.target_set == 'train':
            # Train Set: 从头开始，到 train_end_idx 结束
            # 注意：如果要严格防止 Leakage，分析 Train 时不需要 Gap，因为它是起点
            self.analysis_start_date = pd.to_datetime(unique_dates[0])
            self.analysis_end_date = pd.to_datetime(unique_dates[train_end_idx])
            print(f"\n📈 [Target: TRAIN SET] 训练集 (In-Sample):")

        else:
            # Custom Range
            s_date = self.user_start_date or Config.START_DATE
            e_date = self.user_end_date or "2099-12-31"

            self.analysis_start_date = pd.to_datetime(s_date)
            self.analysis_end_date = pd.to_datetime(e_date)
            print(f"\n🛠️ [Target: CUSTOM] 自定义时间范围:")

        print(f"   分析区间: {self.analysis_start_date.date()} ~ {self.analysis_end_date.date()}")

    def generate_historical_predictions(self):
        print("\n" + "=" * 60)
        print(f">>> [Analysis] 启动截面分析 (Target: {self.target_set})")
        print("=" * 60)

        if not os.path.exists(self.model_path):
            print(f"❌ 模型未找到: {self.model_path}")
            return

        print(f"Loading Model: {self.model_path}")
        model = PatchTSTForStock.from_pretrained(self.model_path).to(self.device)
        model.eval()

        print("Loading Full Panel Data...")
        # 加载全量数据用于定位日期索引
        panel_df, feature_cols = DataProvider.load_and_process_panel(mode='train')

        # 解析分析区间
        self._resolve_analysis_range(panel_df)

        # 物理数据切片 (向前回溯 Context Length 以确保第一天能预测)
        lookback_buffer = Config.CONTEXT_LEN * 2 + 60
        read_start_date = self.analysis_start_date - pd.Timedelta(days=lookback_buffer)

        mask_date = (panel_df['date'] >= read_start_date) & (panel_df['date'] <= self.analysis_end_date)
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

        for k in tqdm(range(len(unique_codes)), desc="Processing"):
            start_pos = code_indices[k]
            end_pos = code_indices[k + 1]

            if end_pos - start_pos < seq_len: continue

            # 严格筛选日期：只对 Analysis 区间内的 T 时刻进行预测
            curr_dates = dates[start_pos + seq_len - 1: end_pos]
            valid_mask = (curr_dates >= np.datetime64(self.analysis_start_date)) & \
                         (curr_dates <= np.datetime64(self.analysis_end_date))

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
        if not self.results_df.empty:
            self.results_df['date'] = pd.to_datetime(self.results_df['date'])
            print(f"✅ 推理完成，生成 {len(self.results_df)} 条预测记录。")
        else:
            print("❌ 未生成预测记录 (可能是数据Gap导致区间内无样本)")

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

        df['score_rank'] = df.groupby('date')['score'].rank(pct=True)
        df['label_rank'] = df.groupby('date')['rank_label'].rank(pct=True)

        daily_ic = df.groupby('date').apply(
            lambda x: x['score_rank'].corr(x['label_rank'])
        )

        ic_mean = daily_ic.mean()
        ic_std = daily_ic.std()
        icir = ic_mean / (ic_std + 1e-9) * np.sqrt(252)
        ic_win_rate = (daily_ic > 0).mean()

        print("-" * 50)
        print(f"📊 【因子深度绩效报告】 (Set: {self.target_set.upper()})")
        print("-" * 50)
        print(f"Rank IC (Mean) : {ic_mean:.4f}")
        print(f"ICIR (Annual)  : {icir:.4f}")
        print(f"IC Win Rate    : {ic_win_rate:.2%}")
        print("-" * 50)

        self._plot_results(df, daily_ic, ic_mean, icir, ic_win_rate)

    def _plot_results(self, df, daily_ic, ic_mean, icir, ic_win_rate):
        plt.figure(figsize=(16, 12))

        # 1. Cumulative IC
        ax1 = plt.subplot(3, 1, 1)
        daily_ic_cumsum = daily_ic.cumsum()
        ax1.plot(daily_ic_cumsum.index, daily_ic_cumsum.values, label='Cumulative Rank IC', color='#4B0082',
                 linewidth=1.5)
        ax1.set_title(f'Cumulative Rank IC (ICIR={icir:.2f}) - {self.target_set.upper()}', fontsize=12,
                      fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.4)
        ax1.legend(loc='upper left')

        # 2. Daily IC
        ax2 = plt.subplot(3, 1, 2)
        colors = ['#d32f2f' if v < 0 else '#388e3c' for v in daily_ic.values]
        ax2.bar(daily_ic.index, daily_ic.values, color=colors, alpha=0.6, width=1.0, label='Daily IC')
        ax2.axhline(ic_mean, color='blue', linestyle='--', linewidth=1.5, label=f'Mean IC: {ic_mean:.3f}')
        ax2.axhline(0, color='black', linewidth=0.8)
        ax2.set_title(f'Daily IC Distribution (Win Rate={ic_win_rate:.1%})', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, axis='y', linestyle='--', alpha=0.4)

        # 3. Layered Backtest
        ax3 = plt.subplot(3, 1, 3)
        df['group'] = df.groupby('date')['score'].transform(
            lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')
        )
        layer_ret = df.groupby(['date', 'group'])['excess_label'].mean().unstack()

        if Config.PRED_LEN > 1:
            layer_ret = layer_ret / Config.PRED_LEN

        layer_ret = layer_ret.fillna(0)
        cum_ret = (1 + layer_ret).cumprod()

        groups = sorted(layer_ret.columns)
        for idx, g in enumerate(groups):
            if g == groups[-1]:
                label, c, lw, alpha = "Top 20% (Long)", "#d32f2f", 2.0, 1.0
            elif g == groups[0]:
                label, c, lw, alpha = "Bottom 20% (Short)", "#388e3c", 1.5, 0.8
            else:
                label, c, lw, alpha = f"Group {g}", "gray", 0.8, 0.3

            ax3.plot(cum_ret.index, cum_ret[g], label=label, color=c, linewidth=lw, alpha=alpha)

        if len(groups) >= 2:
            ls_ret = layer_ret[groups[-1]] - layer_ret[groups[0]]
            ls_cum = (1 + ls_ret).cumprod()
            ax3.plot(ls_cum.index, ls_cum, label='Long-Short Alpha', color='blue', linestyle='--', linewidth=1.5)

        ax3.set_title(f'Layered Backtest ({self.target_set.upper()})', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper left', ncol=2)
        ax3.grid(True, linestyle='--', alpha=0.4)

        plt.tight_layout()
        save_path = os.path.join(Config.OUTPUT_DIR, f"report_{self.target_set}.png")
        plt.savefig(save_path, dpi=150)
        print(f"📈 图表已保存至: {save_path}")


if __name__ == "__main__":
    # === 使用示例 ===

    # 1. 分析测试集 (Out-of-Sample) [默认]
    # print(">>> Mode: Test Set")
    # analyzer = BacktestAnalyzer(target_set='test')
    # analyzer.generate_historical_predictions()
    # analyzer.analyze_performance()

    # 2. 分析验证集 (Eval Set - 检查是否过拟合)
    print(">>> Mode: Test Set")
    analyzer = BacktestAnalyzer(target_set='eval')
    analyzer.generate_historical_predictions()
    analyzer.analyze_performance()