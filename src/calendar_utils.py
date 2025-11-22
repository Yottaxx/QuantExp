import os
import pandas as pd
import akshare as ak
from .config import Config


class TradingCalendar:
    """
    统一的 A 股交易日历管理器，避免自然日填充导致的数据错位。
    - 优先加载本地缓存，若不存在则通过 AkShare 获取交易日列表。
    - 提供交易日索引切片与对齐方法，便于行情/因子统一按交易日对齐。
    """

    def __init__(self, start_date: str):
        self.start_date = pd.to_datetime(start_date)
        self._calendar = self._load_calendar()

    def _load_calendar(self) -> pd.DatetimeIndex:
        cache_path = Config.CALENDAR_CACHE
        if os.path.exists(cache_path):
            try:
                cache = pd.read_parquet(cache_path)
                return pd.to_datetime(cache['trade_date']).sort_values().unique()
            except Exception:
                pass

        df = ak.tool_trade_date_hist_sina()
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        df = df[df['trade_date'] >= self.start_date]
        df = df.sort_values('trade_date')

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df[['trade_date']].to_parquet(cache_path, index=False)
        return df['trade_date'].unique()

    @property
    def calendar(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self._calendar)

    def slice(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        mask = (self.calendar >= start) & (self.calendar <= end)
        return self.calendar[mask]

    def align_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        将行情 DataFrame 对齐到统一交易日历。
        - 对缺失日做前向填充，但额外记录 is_trading/suspend 标记，避免后续回测误用。
        """
        if df.empty:
            return df

        df = df.copy()
        df.index = pd.to_datetime(df.index)
        aligned_index = self.slice(df.index.min(), df.index.max())
        df = df.reindex(aligned_index)

        df['is_trading'] = ~df['close'].isna()
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].ffill()
        df['volume'] = df['volume'].fillna(0)
        df['suspend'] = (~df['is_trading']) | (df['volume'] <= 0)

        df['prev_close'] = df['close'].shift(1)
        df['limit_up'] = df['prev_close'] * (1 + Config.DAILY_LIMIT)
        df['limit_down'] = df['prev_close'] * (1 - Config.DAILY_LIMIT)
        return df

