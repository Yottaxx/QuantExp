import backtrader as bt
import pandas as pd
import numpy as np
import os
import akshare as ak
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from .config import Config
from .model import PatchTSTForStock
from .data_provider import DataProvider

# 设置 Matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
#  策略类：接收外部信号驱动 (Signal Driven Strategy)
# ==============================================================================
class ModelDrivenStrategy(bt.Strategy):
    params = (
        ('signals', None),  # 外部传入的信号 DataFrame: index=date, columns=codes, value=rank/score
        ('top_k', 5),
        ('hold_days', 5),
    )

    def __init__(self):
        self.hold_time = {}  # 记录持仓天数
        self.rebalance_days = 0  # 记录调仓计数

    def next(self):
        current_date = self.data.datetime.date(0)

        # 1. 检查卖出 (持有期满)
        for data in self.datas:
            pos = self.getposition(data).size
            if pos > 0:
                name = data._name
                self.hold_time[name] = self.hold_time.get(name, 0) + 1
                if self.hold_time[name] >= self.p.hold_days:
                    self.close(data=data)
                    self.hold_time[name] = 0

        # 2. 检查买入 (根据模型信号)
        # 从 signals 中获取当天的目标股票
        if self.p.signals is None: return

        # 转换 current_date 为 pandas timestamp 以便索引
        try:
            ts = pd.Timestamp(current_date)
            if ts not in self.p.signals.index:
                return  # 当天无信号

            # 获取当天的 Top K 代码
            daily_ranks = self.p.signals.loc[ts]
            # 假设 signals 存的是 score，我们取最大的 Top K
            # daily_ranks 是一个 Series: index=code, value=score
            top_targets = daily_ranks.nlargest(self.p.top_k).index.tolist()

        except Exception as e:
            # print(f"Signal lookup error: {e}")
            return

        # 执行买入
        cash = self.broker.get_cash()
        if cash < 5000: return

        current_positions = len([d for d in self.datas if self.getposition(d).size > 0])
        slots = self.p.top_k - current_positions
        if slots <= 0: return

        target_val = cash / slots * 0.98

        for target_code in top_targets:
            # 找到对应的 data feed
            data = self.getdatabyname(target_code)
            if data is None: continue  # 数据可能缺失

            pos = self.getposition(data).size
            if pos == 0:
                price = data.close[0]
                if price <= 0: continue
                size = int(target_val / price / 100) * 100
                if size >= 100:
                    self.buy(data=data, size=size)
                    self.hold_time[target_code] = 0


# ==============================================================================
#  滚动预测引擎 (Walk-Forward Predictor)
# ==============================================================================
class WalkForwardEngine:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.device = Config.DEVICE
        self.model_path = f"{Config.OUTPUT_DIR}/final_model"

    def generate_signals(self):
        """
        生成全历史的模型预测信号
        """
        print(">>> [Walk-Forward] 正在生成历史预测信号...")

        # 1. 加载模型
        if not os.path.exists(self.model_path):
            print("❌ 模型未找到")
            return None
        model = PatchTSTForStock.from_pretrained(self.model_path).to(self.device)
        model.eval()

        # 2. 加载数据 (使用 predict 模式保留最新数据，且需要全量数据来构建窗口)
        # 为了回测历史，我们需要覆盖 start_date 之前 Config.CONTEXT_LEN 的数据
        panel_df, feature_cols = DataProvider.load_and_process_panel(mode='predict')

        # 3. 滚动预测
        # 这里的逻辑和 analysis.py 类似，但我们需要把结果整理成 Backtrader 可用的格式
        # 即：DataFrame, Index=Date, Columns=Codes, Values=Score

        # 为了速度，我们还是使用 Batch 推理
        # ... (复用 analysis.py 的推理逻辑) ...
        # 这里为了代码简洁，直接调用 analysis 模块的逻辑，或者重写一遍
        # 我们重写一遍简化的，只返回信号矩阵

        # 筛选时间：start_date 往前推 60 天用于窗口构建
        s_date = pd.to_datetime(self.start_date) - pd.Timedelta(days=60)
        e_date = pd.to_datetime(self.end_date)
        mask = (panel_df['date'] >= s_date) & (panel_df['date'] <= e_date)
        df_sub = panel_df[mask].copy()

        results = []
        batch_inputs = []
        batch_meta = []

        grouped = df_sub.groupby('code')
        print("正在批量推理...")

        for code, group in tqdm(grouped):
            if len(group) < Config.CONTEXT_LEN: continue
            feats = group[feature_cols].values.astype(np.float32)
            dates = group['date'].values

            for i in range(len(group) - Config.CONTEXT_LEN + 1):
                # 预测日期是窗口最后一天
                pred_date = pd.to_datetime(dates[i + Config.CONTEXT_LEN - 1])
                if pred_date < pd.to_datetime(self.start_date): continue

                batch_inputs.append(feats[i: i + Config.CONTEXT_LEN])
                batch_meta.append((pred_date, code))

                if len(batch_inputs) >= 2048:
                    self._flush(model, batch_inputs, batch_meta, results)
                    batch_inputs = []
                    batch_meta = []

        if batch_inputs:
            self._flush(model, batch_inputs, batch_meta, results)

        # 转换为信号矩阵 (Pivot Table)
        print("正在构建信号矩阵...")
        res_df = pd.DataFrame(results, columns=['date', 'code', 'score'])
        # pivot: index=date, columns=code, values=score
        signal_matrix = res_df.pivot(index='date', columns='code', values='score')
        return signal_matrix

    def _flush(self, model, inputs, meta, results):
        tensor = torch.tensor(np.array(inputs), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            scores = model(past_values=tensor).logits.squeeze().cpu().numpy()
        if scores.ndim == 0: scores = [scores]
        for i, s in enumerate(scores):
            results.append((meta[i][0], meta[i][1], float(s)))


# ==============================================================================
#  回测主程序
# ==============================================================================
def run_backtest(start_date='2024-01-01', end_date='2024-12-31', initial_cash=1000000.0):
    print(f"\n>>> 启动模型驱动的 Walk-Forward 回测 ({start_date} ~ {end_date})")

    # 1. 生成信号
    engine = WalkForwardEngine(start_date, end_date)
    signal_matrix = engine.generate_signals()

    if signal_matrix is None or signal_matrix.empty:
        print("❌ 未生成有效信号")
        return

    # 2. 初始化 Cerebro
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)

    # 费率
    class AShareCommission(bt.CommInfoBase):
        params = (('stocklike', True), ('commtype', bt.CommInfoBase.COMM_PERC),
                  ('perc', 0.0003), ('stamp_duty', 0.0005), ('min_comm', 5.0))

        def _getcommission(self, size, price, pseudoexec):
            if size > 0:
                return max(abs(size) * price * self.p.perc, self.p.min_comm)
            elif size < 0:
                return max(abs(size) * price * self.p.perc, self.p.min_comm) + abs(size) * price * self.p.stamp_duty
            return 0.0

    cerebro.broker.addcommissioninfo(AShareCommission())

    # 3. 加载数据 (只加载信号矩阵中涉及到的股票，且在时间范围内)
    # 这里的 DataProvider 需要能快速加载指定股票的行情
    # 为了简化，我们重新加载一遍 panel (或者您可以优化让 DataProvider 提供 get_price_data 接口)
    print("正在加载回测行情数据...")

    # 找出所有涉及到的股票代码
    involved_codes = signal_matrix.columns.tolist()
    # 为了演示，只取 Top 50 活跃的股票 (否则几千只加载进 Backtrader 会非常慢)
    # 实际生产中可以使用数据库按需加载

    # 简易方案：只加载 signal_matrix 中曾经进入过 Top 5 的股票
    # 这是一种优化技巧：没被选中的股票不需要行情数据

    top_k_mask = signal_matrix.rank(axis=1, ascending=False) <= 5
    active_codes = signal_matrix.columns[top_k_mask.any()].tolist()
    print(f"回测涉及活跃股票数: {len(active_codes)}")

    loaded_count = 0
    for code in tqdm(active_codes):
        fpath = os.path.join(Config.DATA_DIR, f"{code}.parquet")
        if not os.path.exists(fpath): continue
        try:
            df = pd.read_parquet(fpath)
            # 过滤时间
            df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
            if df.empty: continue

            data = bt.feeds.PandasData(dataname=df, name=code, plot=False)
            cerebro.adddata(data)
            loaded_count += 1
        except:
            continue

    if loaded_count == 0:
        print("❌ 无有效回测数据")
        return

    # 4. 注入策略
    cerebro.addstrategy(ModelDrivenStrategy, signals=signal_matrix, top_k=5, hold_days=5)

    # 5. 添加分析器
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    # 6. 运行
    print("⏳ 开始回测 (这可能需要几分钟)...")
    results = cerebro.run()
    strat = results[0]

    # 7. 报告
    final_val = cerebro.broker.getvalue()
    ret = (final_val - initial_cash) / initial_cash
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
    max_dd = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)

    print("\n" + "=" * 40)
    print("📊 [Walk-Forward 回测报告]")
    print(f"回测区间: {start_date} ~ {end_date}")
    print(f"初始资金: {initial_cash:,.0f}")
    print(f"最终资金: {final_val:,.2f}")
    print(f"累计收益: {ret:.2%}")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"最大回撤: {max_dd:.2%}")
    print("=" * 40)

    # 绘图
    returns = pd.Series(strat.analyzers.returns.get_analysis())
    (1 + returns).cumprod().plot(title="Strategy Equity Curve", figsize=(10, 6))
    plt.savefig(os.path.join(Config.OUTPUT_DIR, "walk_forward_result.png"))
    print("📈 曲线图已保存。")