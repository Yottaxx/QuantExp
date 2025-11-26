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
    def __init__(self, target_set='test'):
        """
        :param target_set: 'test' (默认，最后10%), 'validation' (中间10%), 'train' (前80%)
        """
        self.device = Config.DEVICE
        self.model_path = f"{Config.OUTPUT_DIR}/final_model"
        self.results_df = None
        self.target_set = target_set

        self.start_date = None
        self.end_date = None

    def _auto_set_date_range(self, panel_df):
        """根据 Config 比例自动定位 Test Set 时间段"""
        unique_dates = np.sort(panel_df['date'].unique())
        n_dates = len(unique_dates)

        train_end_idx = int(n_dates * Config.TRAIN_RATIO)
        val_end_idx = int(n_dates * (Config.TRAIN_RATIO + Config.VAL_RATIO))

        # 计算带 Gap 的索引
        val_start_idx = min(train_end_idx + Config.CONTEXT_LEN, n_dates - 1)
        test_start_idx = min(val_end_idx + Config.CONTEXT_LEN, n_dates - 1)

        if self.target_set == 'test':
            self.start_date = pd.to_datetime(unique_dates[test_start_idx])
            self.end_date = pd.to_datetime(unique_dates[-1])
        elif self.target_set == 'validation':
            self.start_date = pd.to_datetime(unique_dates[val_start_idx])
            self.end_date = pd.to_datetime(unique_dates[val_end_idx])
        else:
            # Train
            self.start_date = pd.to_datetime(unique_dates[0])
            self.end_date = pd.to_datetime(unique_dates[train_end_idx])

        print(f"\n🔒 [Target Set: {self.target_set.upper()}] 分析区间锁定:")
        print(f"   Range: {self.start_date.date()} ~ {self.end_date.date()}")

    def generate_historical_predictions(self):
        print("\n" + "=" * 60)
        print(">>> [Analysis] 启动截面分析")
        print("=" * 60)

        if not os.path.exists(self.model_path):
            print(f"❌ 模型未找到: {self.model_path}")
            return

        model = PatchTSTForStock.from_pretrained(self.model_path).to(self.device)
        model.eval()

        # 加载全量数据用于定位日期
        print("Loading Full Panel Data...")
        panel_df, feature_cols = DataProvider.load_and_process_panel(mode='train')

        # 自动定位 Test Set
        self._auto_set_date_range(panel_df)

        # 物理读取需要往前预留 Context Length，否则 Test Set 第一天无法预测
        read_start_date = self.start_date - pd.Timedelta(days=Config.CONTEXT_LEN * 2 + 60)

        mask_date = (panel_df['date'] >= read_start_date) & (panel_df['date'] <= self.end_date)
        df_sub = panel_df[mask_date].copy()

        if df_sub.empty:
            print("❌ 无有效数据")
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

            # 只预测在 Target Set 区间内的日期
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
        print(f"✅ 推理完成: {len(self.results_df)} 条记录")

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

        # Rank IC
        df['score_rank'] = df.groupby('date')['score'].rank(pct=True)
        df['label_rank'] = df.groupby('date')['rank_label'].rank(pct=True)
        daily_ic = df.groupby('date').apply(lambda x: x['score_rank'].corr(x['label_rank']))

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
        # ... (绘图代码与之前一致，请复用) ...
        pass


if __name__ == "__main__":
    # 强制只分析 Test Set
    analyzer = BacktestAnalyzer(target_set='test')
    analyzer.generate_historical_predictions()
    analyzer.analyze_performance()