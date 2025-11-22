import backtrader as bt
import pandas as pd
import numpy as np
import os
import torch
import akshare as ak
import matplotlib.pyplot as plt
from tqdm import tqdm
from .config import Config
from .model import PatchTSTForStock
from .data_provider import DataProvider

# 设置 Matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==============================================================================
#  1. 费率模型 (保持不变)
# ==============================================================================
class AShareCommission(bt.CommInfoBase):
    params = (('stocklike', True), ('commtype', bt.CommInfoBase.COMM_PERC),
              ('perc', 0.0003), ('stamp_duty', 0.0005), ('min_comm', 5.0))

    def _getcommission(self, size, price, pseudoexec):
        if size > 0:
            return max(abs(size) * price * self.p.perc, self.p.min_comm)
        elif size < 0:
            return max(abs(size) * price * self.p.perc, self.p.min_comm) + abs(size) * price * self.p.stamp_duty
        return 0.0


# ==============================================================================
#  2. 核心策略：信号驱动型 (Signal Driven)
# ==============================================================================
class ModelDrivenStrategy(bt.Strategy):
    """
    【真·回测策略】
    不再持有固定股票，而是根据传入的 `signals` DataFrame 每日动态换仓。
    signals 格式: Index=Date, Columns=Codes, Values=Score/Rank
    """
    params = (
        ('signals', None),  # 信号矩阵
        ('top_k', 5),
        ('hold_days', 5),
        ('min_volume_percent', 0.02),
    )

    def __init__(self):
        self.hold_time = {}  # 记录持仓天数 {code: days}
        # 将信号转换为字典以便快速查找: {date: [top_codes]}
        self.signal_dict = {}
        if self.p.signals is not None:
            print("正在解析交易信号...")
            for date, row in self.p.signals.iterrows():
                # 选出当天得分最高的 Top K
                top_codes = row.nlargest(self.p.top_k).index.tolist()
                self.signal_dict[date.date()] = top_codes

    def next(self):
        current_date = self.data.datetime.date(0)

        # --- 1. 卖出逻辑 (持有期满) ---
        for data in self.datas:
            if self.getposition(data).size > 0:
                name = data._name
                self.hold_time[name] = self.hold_time.get(name, 0) + 1
                if self.hold_time[name] >= self.p.hold_days:
                    self.close(data=data)
                    self.hold_time[name] = 0

        # --- 2. 买入逻辑 (根据历史信号) ---
        # 获取当天的目标持仓
        target_codes = self.signal_dict.get(current_date, [])
        if not target_codes: return

        cash = self.broker.get_cash()
        if cash < 5000: return

        # 计算可用槽位
        current_pos_count = len([d for d in self.datas if self.getposition(d).size > 0])
        slots_available = self.p.top_k - current_pos_count
        if slots_available <= 0: return

        target_val = cash / slots_available * 0.98  # 预留现金防滑点

        buy_count = 0
        for code in target_codes:
            if buy_count >= slots_available: break

            # 从 backtrader 数据流中找到对应的 feed
            data = self.getdatabyname(code)
            # 如果当天该股票停牌或数据缺失，可能拿不到 data
            if data is None: continue

            # 检查是否已有持仓
            if self.getposition(data).size == 0:
                price = data.close[0]
                vol = data.volume[0]
                if price <= 0 or vol <= 0: continue

                size = int(target_val / price / 100) * 100

                # 风控
                if size < 100: continue
                if size > vol * 100 * self.p.min_volume_percent:
                    size = int(vol * 100 * self.p.min_volume_percent / 100) * 100

                if size >= 100:
                    self.buy(data=data, size=size)
                    self.hold_time[code] = 0
                    buy_count += 1


# ==============================================================================
#  3. 滚动回测引擎 (Walk-Forward Engine)
# ==============================================================================
class WalkForwardBacktester:
    def __init__(self, start_date, end_date, initial_cash=1000000.0):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.device = Config.DEVICE
        self.model_path = f"{Config.OUTPUT_DIR}/final_model"

    def generate_signal_matrix(self):
        """生成全市场的历史预测分矩阵"""
        print(f"⏳ [Signal Gen] 正在生成历史信号 ({self.start_date} ~ {self.end_date})...")

        if not os.path.exists(self.model_path):
            print("❌ 模型文件不存在，请先训练")
            return None

        model = PatchTSTForStock.from_pretrained(self.model_path).to(self.device)
        model.eval()

        # 加载全量数据 (predict模式保留最新数据，虽然回测用不到最新的一天，但为了逻辑统一)
        panel_df, feature_cols = DataProvider.load_and_process_panel(mode='predict')

        # 筛选时间窗口 (Start往前推 Context_Len)
        s_dt = pd.to_datetime(self.start_date) - pd.Timedelta(days=Config.CONTEXT_LEN * 2)
        e_dt = pd.to_datetime(self.end_date)
        mask = (panel_df['date'] >= s_dt) & (panel_df['date'] <= e_dt)
        df_sub = panel_df[mask].copy()

        results = []
        batch_inputs, batch_meta = [], []

        print("正在批量推理历史数据...")
        grouped = df_sub.groupby('code')

        for code, group in tqdm(grouped, desc="Inference"):
            if len(group) < Config.CONTEXT_LEN: continue

            feats = group[feature_cols].values.astype(np.float32)
            dates = group['date'].values

            # 滚动生成每一天的预测
            for i in range(len(group) - Config.CONTEXT_LEN + 1):
                # 预测日期 = 窗口最后一天
                pred_date = pd.to_datetime(dates[i + Config.CONTEXT_LEN - 1])
                if pred_date < pd.to_datetime(self.start_date): continue

                batch_inputs.append(feats[i: i + Config.CONTEXT_LEN])
                # 记录 (date, code)
                batch_meta.append((pred_date, code))

                if len(batch_inputs) >= 2048:
                    self._flush_batch(model, batch_inputs, batch_meta, results)
                    batch_inputs, batch_meta = [], []

        if batch_inputs:
            self._flush_batch(model, batch_inputs, batch_meta, results)

        if not results:
            print("❌ 未生成任何信号")
            return None

        # 转换为矩阵: Index=Date, Columns=Code, Values=Score
        print("正在重构信号矩阵...")
        res_df = pd.DataFrame(results, columns=['date', 'code', 'score'])
        # pivot 可能会消耗大量内存，注意优化
        signal_matrix = res_df.pivot(index='date', columns='code', values='score')
        signal_matrix = signal_matrix.sort_index()

        return signal_matrix

    def _flush_batch(self, model, inputs, meta, res):
        t = torch.tensor(np.array(inputs), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            s = model(past_values=t).logits.squeeze().cpu().numpy()
        if s.ndim == 0: s = [s]
        for i, score in enumerate(s):
            res.append((meta[i][0], meta[i][1], float(score)))

    def run(self, top_k=5):
        # 1. 生成信号
        signals = self.generate_signal_matrix()
        if signals is None: return

        # 2. 确定回测池 (只加载曾经入选 Top K 的股票，优化内存)
        print("正在筛选活跃股票池...")
        # 对每天的 score 排序，只要进过前 Top K * 2 (放宽一点) 的股票都加载
        # 使用 rank 方法
        daily_ranks = signals.rank(axis=1, ascending=False)
        active_mask = (daily_ranks <= top_k * 2).any(axis=0)
        active_codes = signals.columns[active_mask].tolist()

        print(f"回测涉及股票数量: {len(active_codes)}")
        if not active_codes: return

        # 3. 初始化 Backtrader
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.addcommissioninfo(AShareCommission())

        print("正在加载回测行情数据 (这可能需要一点时间)...")
        loaded_cnt = 0
        for code in tqdm(active_codes):
            fpath = os.path.join(Config.DATA_DIR, f"{code}.parquet")
            if not os.path.exists(fpath): continue
            try:
                df = pd.read_parquet(fpath)
                # 截取时间
                df = df[(df.index >= pd.to_datetime(self.start_date)) & (df.index <= pd.to_datetime(self.end_date))]
                if df.empty: continue

                data = bt.feeds.PandasData(dataname=df, name=code, plot=False)
                cerebro.adddata(data)
                loaded_cnt += 1
            except:
                continue

        if loaded_cnt == 0:
            print("❌ 无有效行情数据")
            return

        # 4. 运行回测
        print(f"🚀 开始 Walk-Forward 回测 (Top {top_k})...")
        cerebro.addstrategy(ModelDrivenStrategy, signals=signals, top_k=top_k, hold_days=5)

        # 添加分析器
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='returns')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

        results = cerebro.run()
        strat = results[0]

        # 5. 报告
        self._generate_report(strat, cerebro)

    def _generate_report(self, strat, cerebro):
        final_val = cerebro.broker.getvalue()
        ret = (final_val - self.initial_cash) / self.initial_cash

        sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0)
        max_dd = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)

        print("\n" + "=" * 40)
        print(f"📊 [Walk-Forward 真实回测报告]")
        print(f"区间: {self.start_date} ~ {self.end_date}")
        print(f"资金: {self.initial_cash:,.0f} -> {final_val:,.2f}")
        print(f"收益: {ret:.2%}")
        print(f"夏普: {sharpe:.2f}")
        print(f"回撤: {max_dd:.2%}")
        print("=" * 40)

        # 绘图
        ret_series = pd.Series(strat.analyzers.returns.get_analysis())
        cumulative = (1 + ret_series).cumprod()

        # 尝试获取基准
        try:
            bench = ak.stock_zh_index_daily(symbol="sh000300")
            bench['date'] = pd.to_datetime(bench['date'])
            bench.set_index('date', inplace=True)
            bench_ret = bench['close'].pct_change().reindex(ret_series.index).fillna(0)
            bench_cum = (1 + bench_ret).cumprod()

            plt.figure(figsize=(12, 6))
            plt.plot(cumulative.index, cumulative, label='Strategy', color='red')
            plt.plot(bench_cum.index, bench_cum, label='CSI 300', color='gray', linestyle='--')
        except:
            cumulative.plot(figsize=(12, 6), label='Strategy')

        plt.title('Walk-Forward Equity Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(Config.OUTPUT_DIR, "walk_forward_result.png"))
        print("📈 曲线已保存。")


# 外部调用入口
def run_walk_forward_backtest(start_date, end_date, initial_cash, top_k):
    engine = WalkForwardBacktester(start_date, end_date, initial_cash)
    engine.run(top_k=top_k)