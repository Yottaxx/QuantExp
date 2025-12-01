from __future__ import annotations
import os, time
from typing import Optional
import pandas as pd

from ..core.config import DPConfig
from ..utils.io import atomic_save_parquet

class CalendarStore:
    def __init__(self, cfg: DPConfig, ak_client, logger, ak_module):
        self.cfg = cfg
        self.ak = ak_module
        self.ak_client = ak_client
        self.logger = logger

    def _path(self) -> str:
        root = str(self.cfg.get("DATA_DIR", "./data") or "./data")
        os.makedirs(os.path.join(root,"calendar"), exist_ok=True)
        sym = str(self.cfg.get("MARKET_INDEX_SYMBOL","sh000001") or "sh000001")
        return os.path.join(root,"calendar", f"index_{sym}.parquet")

    def _is_fresh(self, path: str, ttl: int) -> bool:
        return os.path.exists(path) and os.path.getsize(path) >= 512 and (time.time()-os.path.getmtime(path)) < ttl

    def get_trade_dates(self) -> pd.DatetimeIndex:
        path = self._path()
        ttl = int(self.cfg.get("CALENDAR_TTL_SEC", 7*24*3600) or 7*24*3600)
        if self._is_fresh(path, ttl):
            try:
                df = pd.read_parquet(path)
                d = pd.to_datetime(df["date"], errors="coerce").dropna()
                return pd.DatetimeIndex(sorted(d.unique()))
            except Exception:
                pass

        sym = str(self.cfg.get("MARKET_INDEX_SYMBOL","sh000001") or "sh000001")
        df = self.ak_client.call(self.ak.stock_zh_index_daily, symbol=sym)
        if df is None or df.empty:
            raise RuntimeError("Index daily is empty, cannot build calendar.")
        d = pd.to_datetime(df["date"], errors="coerce").dropna().sort_values()
        atomic_save_parquet(pd.DataFrame({"date": d}), path, index=False, compression=str(self.cfg.get("PARQUET_COMPRESSION","zstd") or "zstd"))
        return pd.DatetimeIndex(sorted(d.unique()))

    def latest_trade_date(self, today: Optional[pd.Timestamp] = None) -> pd.Timestamp:
        if today is None:
            today = pd.Timestamp(pd.Timestamp.now().date())
        try:
            dates = self.get_trade_dates()
            eligible = dates[dates <= pd.Timestamp(today).normalize()]
            if len(eligible):
                return pd.Timestamp(eligible[-1])
        except Exception as e:
            self.logger.warning(f"[Calendar] fallback due to error: {e}")

        d = pd.Timestamp(today).normalize()
        while d.weekday() >= 5:
            d -= pd.Timedelta(days=1)
        return d
