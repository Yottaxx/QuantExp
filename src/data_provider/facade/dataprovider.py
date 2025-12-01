from __future__ import annotations
import os
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
import akshare as ak

from ..core.config import DPConfig
from ..core.logging import setup_logger
from ..core.version import VERSION
from ..clients.ak_client import AkClient, VPNCoordinator
from ..stores.paths import norm_adjust, price_dir
from ..stores.price_store import price_glob
from ..stores.calendar_store import CalendarStore
from ..stores.universe_store import UniverseStore
from ..stores.secmaster_store import SecurityMasterStore
from ..pipelines.price_pipeline import PricePipeline
from ..pipelines.fundamental_pipeline import FundamentalPipeline
from ..pipelines.info_pipeline import InfoPipeline
from ..pipelines.panel_pipeline import PanelPipeline
from ..pipelines.dataset_pipeline import DatasetPipeline
from ..rules.split_policy import SplitPolicy
from ..rules.trading_rules import TradingParams, add_trade_masks

# sibling modules from src/
from .. import __init__ as _noop  # noqa: F401

class DataProvider:
    VERSION = VERSION

    def __init__(self, Config: Any = None, AlphaFactory: Any = None, vpn_rotator: Optional[Any] = None):
        if Config is None:
            from ...config import Config  # type: ignore
        if AlphaFactory is None:
            from ...alpha_lib import AlphaFactory  # type: ignore
        self.cfg = DPConfig.from_module(Config)
        self.logger = setup_logger("data_provider", level=str(self.cfg.get("LOG_LEVEL","INFO") or "INFO"))

        proxy = str(self.cfg.get("PROXY_URL","") or "").strip()
        if proxy:
            os.environ["http_proxy"] = proxy
            os.environ["https_proxy"] = proxy

        vpn = VPNCoordinator(self.cfg, vpn_rotator)
        self.ak_client = AkClient(self.cfg, vpn, self.logger)
        self.calendar = CalendarStore(self.cfg, self.ak_client, self.logger, ak)
        self._price_glob = lambda adj: price_glob(self.cfg, adj, self.logger)

        self.universe = UniverseStore(self.cfg, self.ak_client, self.logger, ak, price_glob_fn=self._price_glob)
        self.secmaster = SecurityMasterStore(self.cfg, self.logger, price_glob_fn=self._price_glob)

        self.price_pipeline = PricePipeline(self.cfg, self.ak_client, self.calendar, self.universe, self.logger)
        self.info_pipeline = InfoPipeline(self.cfg, self.ak_client, self.logger)
        self.fundamental_pipeline = FundamentalPipeline(self.cfg, self.ak_client, self.logger)
        self.panel_pipeline = PanelPipeline(self.cfg, self.logger, self.calendar, self.universe, self.secmaster, AlphaFactory)
        self.dataset_pipeline = DatasetPipeline(self.cfg, self.logger)

        self.AlphaFactory = AlphaFactory

    # -------- shadow interfaces (backward compat) --------
    @staticmethod
    def _norm_adjust(adjust: Optional[str]) -> str:
        return norm_adjust(adjust)

    def _price_dir(self, adj_norm: str) -> str:
        return price_dir(self.cfg, norm_adjust(adj_norm))

    @staticmethod
    def _add_trade_masks(df: pd.DataFrame) -> pd.DataFrame:
        # parameters come from Config in the real pipeline; here keep a stable default.
        params = TradingParams()
        return add_trade_masks(df, params)

    @staticmethod
    def _get_date_splits(panel_df: pd.DataFrame, seq_len: Optional[int] = None) -> Dict[str, pd.Timestamp]:
        if "date" not in panel_df.columns:
            raise ValueError("panel_df must contain 'date'")
        seq = int(seq_len or 60)
        dates = pd.to_datetime(panel_df["date"], errors="coerce").dropna().unique()
        sm = SplitPolicy.compute(np.array(dates), seq_len=seq, stride=1, gap=seq, train_ratio=0.7, val_ratio=0.15).to_dict()
        # return timestamps
        return {k: pd.Timestamp(v) for k, v in sm.items() if k.endswith(("start","excl","date"))}

    # -------- public API --------
    def download_data(self, adjusts: Optional[List[str]] = None) -> None:
        # Price is the main dependency (universe/security master inferred from it).
        self.price_pipeline.download_data(adjusts=adjusts)
        # Extra streams are optional and cached by TTL.
        try:
            target_dt = self.calendar.latest_trade_date()
            snap = self.universe.get_snapshot(target_dt, force_refresh=bool(self.cfg.get("FORCE_UNIVERSE_REFRESH", False)))
            codes = snap["code"].astype(str).tolist() if snap is not None and not snap.empty else []
        except Exception:
            codes = []
        if codes:
            self.info_pipeline.download(codes)
            self.fundamental_pipeline.download(codes)
        return None

    def load_and_process_panel(
        self,
        mode: str = "train",
        force_refresh: bool = False,
        adjust: str = "qfq",
        backend: Optional[str] = None,
        debug: bool = False,
    ):
        adj_norm = norm_adjust(adjust)
        universe_asof = self.calendar.latest_trade_date()
        meta_map = self.universe.meta_map_for_asof(universe_asof)
        snap = self.universe.get_snapshot(universe_asof, force_refresh=False)
        codes = snap["code"].astype(str).tolist() if snap is not None and not snap.empty else []
        sm_df = self.secmaster.build_or_update(
            codes=codes,
            adj_norm_for_scan=adj_norm,
            target_dt=universe_asof,
            meta_map=meta_map,
            force=bool(self.cfg.get("FORCE_SECMASTER_REFRESH", False)),
        )
        sec_map = self.secmaster.to_map(sm_df)

        part_paths, feat_cols, reject, parts_dir, fp = self.panel_pipeline.build_parts(
            mode=mode, force_refresh=force_refresh, adjust=adj_norm, backend=backend, debug=debug,
            meta_map=meta_map, sec_map=sec_map, universe_asof=universe_asof
        )

        # Materialize df for downstream analysis/backtest compatibility
        panel_df = self.panel_pipeline.materialize_panel_df(part_paths)
        panel_df.attrs["adjust"] = adj_norm
        panel_df.attrs["mode"] = mode
        panel_df.attrs["fingerprint"] = fp
        panel_df.attrs["universe_asof"] = str(pd.Timestamp(universe_asof).date())
        panel_df.attrs["created_by"] = VERSION
        panel_df.attrs["parts_dir"] = parts_dir
        return panel_df, feat_cols

    def make_dataset(self, panel_df: pd.DataFrame, feature_cols: List[str]):
        parts_dir = str(panel_df.attrs.get("parts_dir", "") or "").strip()

        if parts_dir and os.path.exists(os.path.join(parts_dir, "panel_meta.json")):
            ds = self.dataset_pipeline.build_dataset_from_parts(parts_dir)
        else:
            ds = self.dataset_pipeline.build_dataset(panel_df, feature_cols)

        # ✅ 新增：给 HF load_from_disk 的落盘（默认打开，可加 cfg 开关）
        if parts_dir and bool(self.cfg.get("SAVE_HF_DATASET_TO_DISK", True)):
            out_dir = os.path.join(parts_dir, "hf_dataset")
            meta = {
                "adjust": panel_df.attrs.get("adjust"),
                "mode": panel_df.attrs.get("mode"),
                "fingerprint": panel_df.attrs.get("fingerprint"),
                "universe_asof": panel_df.attrs.get("universe_asof"),
                "created_by": panel_df.attrs.get("created_by"),
                "feature_cols": feature_cols,
            }
            self.dataset_pipeline.save_to_disk_atomic(ds, out_dir, meta=meta)
            self.logger.info(f"------save hf dataset tot {out_dir}------")
        return ds

def get_dataset(force_refresh: bool = False, adjust: str = "qfq"):
    """Backward-compatible helper: returns (DatasetDict, fdim)."""
    from ...config import Config  # type: ignore
    from ...alpha_lib import AlphaFactory  # type: ignore
    try:
        from src.data_provider.utils.vpn_rotator import vpn_rotator  # type: ignore
    except Exception:
        vpn_rotator = None

    dp = DataProvider(Config=Config, AlphaFactory=AlphaFactory, vpn_rotator=vpn_rotator)
    panel_df, feature_cols = dp.load_and_process_panel(mode="train", force_refresh=force_refresh, adjust=adjust)
    return dp.make_dataset(panel_df, feature_cols)
