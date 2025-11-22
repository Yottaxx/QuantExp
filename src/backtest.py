import backtrader as bt
import pandas as pd
import os
import math
import numpy as np
import akshare as ak
import matplotlib.pyplot as plt
from .config import Config

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class AShareCommission(bt.CommInfoBase):
    params = (('stocklike', True), ('commtype', bt.CommInfoBase.COMM_PERC),
              ('perc', 0.0003), ('stamp_duty', 0.0005), ('min_comm', 5.0))
    def _getcommission(self, size, price, pseudoexec):
        if size > 0: return max(abs(size) * price * self.p.perc, self.p.min_comm)
        elif size < 0: return max(abs(size) * price * self.p.perc, self.p.min_comm) + abs(size) * price * self.p.stamp_duty
        return 0.0

class TopKStrategy(bt.Strategy):
    params = (('top_k', 5), ('hold_days', 5), ('min_volume_percent', 0.02))
    def __init__(self):
        self.hold_time = {}
    def next(self):
        for data in self.datas:
            if self.getposition(data).size > 0:
                self.hold_time[data._name] = self.hold_time.get(data._name, 0) + 1
                if self.hold_time[data._name] >= self.p.hold_days:
                    self.close(data=data); self.hold_time[data._name] = 0
        cash = self.broker.get_cash()
        if cash < 5000: return
        current_pos = len([d for d in self.datas if self.getposition(d).size > 0])
        slots = self.p.top_k - current_pos
        if slots <= 0: return
        target = cash / slots * 0.98
        buy_cnt = 0
        for data in self.datas:
            if buy_cnt >= slots: break
            if self.getposition(data).size == 0:
                price = data.close[0]; vol = data.volume[0]
                if price <= 0 or vol <= 0: continue
                size = int(target / price / 100) * 100
                if size < 100: continue
                if size > vol * 100 * self.p.min_volume_percent:
                    size = int(vol * 100 * self.p.min_volume_percent / 100) * 100
                if size >= 100:
                    self.buy(data=data, size=size); self.hold_time[data._name] = 0; buy_cnt += 1

class PerformanceAnalyzer:
    @staticmethod
    def get_benchmark(start_date, end_date):
        try:
            df = ak.stock_zh_index_daily(symbol="sh000300")
            df['date'] = pd.to_datetime(df['date']); df.set_index('date', inplace=True)
            mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
            return df.loc[mask, 'close'].pct_change().fillna(0)
        except: return None

    @staticmethod
    def calculate_metrics(strategy_returns, benchmark_returns):
        df = pd.concat([strategy_returns, benchmark_returns], axis=1, join='inner')
        df.columns = ['Strategy', 'Benchmark']
        if len(df) < 10: return None
        Rp, Rm = df['Strategy'], df['Benchmark']
        days = len(df)
        ann_p = (1 + Rp).prod() ** (252 / days) - 1
        ann_m = (1 + Rm).prod() ** (252 / days) - 1
        vol = Rp.std() * np.sqrt(252)
        sharpe = (ann_p - 0.03) / (vol + 1e-9)
        cum = (1 + Rp).cumprod()
        dd = ((cum.cummax() - cum) / cum.cummax()).max()
        cov = np.cov(Rp, Rm); beta = cov[0, 1] / (cov[1, 1] + 1e-9)
        alpha = ann_p - (0.03 + beta * (ann_m - 0.03))
        return {"Ann. Return": ann_p, "Benchmark Ret": ann_m, "Alpha": alpha, "Beta": beta, "Sharpe": sharpe, "Max Drawdown": dd, "Win Rate": (Rp > 0).mean()}

    @staticmethod
    def plot_curve(strategy_returns, benchmark_returns):
        df = pd.concat([strategy_returns, benchmark_returns], axis=1, join='inner')
        df.columns = ['Strategy', 'CSI 300']
        (1 + df).cumprod().plot(figsize=(12, 6), grid=True)
        plt.savefig(os.path.join(Config.OUTPUT_DIR, "backtest_result.png"))

def run_single_backtest(codes, with_fees=True, initial_cash=1000000.0):
    cerebro = bt.Cerebro(); cerebro.broker.setcash(initial_cash)
    if with_fees: cerebro.broker.addcommissioninfo(AShareCommission())
    else: cerebro.broker.setcommission(commission=0.0)
    loaded = False
    for code in codes:
        fpath = os.path.join(Config.DATA_DIR, f"{code}.parquet")
        if not os.path.exists(fpath): continue
        try:
            df = pd.read_parquet(fpath); start = pd.to_datetime(Config.START_DATE)
            if len(df) > 250: df = df.iloc[-250:]
            data = bt.feeds.PandasData(dataname=df, fromdate=df.index[0], plot=False)
            cerebro.adddata(data, name=code); loaded = True
        except: continue
    if not loaded: return None
    cerebro.addstrategy(TopKStrategy, top_k=5, hold_days=5)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='ret')
    strat = cerebro.run()[0]
    final = cerebro.broker.getvalue(); ret_s = pd.Series(strat.analyzers.ret.get_analysis())
    ret_s.index = pd.to_datetime(ret_s.index)
    trade = strat.analyzers.trade.get_analysis()
    total = trade.total.closed if 'total' in trade else 0
    won = trade.won.total if 'won' in trade else 0
    return {"final_value": final, "profit_rate": (final - initial_cash)/initial_cash, "win_rate": won/total if total>0 else 0, "total_trades": total, "returns": ret_s, "start_date": strat.data.datetime.date(0), "end_date": strat.data.datetime.date(-1)}

def run_backtest(top_stocks_list, initial_cash=1000000.0):
    print(f"\n>>> 启动回测 (初始资金: {initial_cash:,.0f})")
    codes = [x[0] for x in top_stocks_list[:5]]
    if not codes: return
    res_fees = run_single_backtest(codes, True, initial_cash)
    res_no = run_single_backtest(codes, False, initial_cash)
    if not res_fees: return
    print(f"{'指标':<15} | {'含费':<15} | {'无费':<15}")
    print("-" * 50)
    print(f"{'最终市值':<15} | {res_fees['final_value']:<15,.2f} | {res_no['final_value']:<15,.2f}")
    print(f"{'收益率':<15} | {res_fees['profit_rate']:<15.2%} | {res_no['profit_rate']:<15.2%}")
    bench = PerformanceAnalyzer.get_benchmark(res_fees['start_date'], res_fees['end_date'])
    if bench is not None:
        m = PerformanceAnalyzer.calculate_metrics(res_fees['returns'], bench)
        if m:
            print(f"\n📊 绩效: Alpha {m['Alpha']:.4f} | Sharpe {m['Sharpe']:.4f} | Excess {m['Ann. Return']-m['Benchmark Ret']:.2%}")
            PerformanceAnalyzer.plot_curve(res_fees['returns'], bench)