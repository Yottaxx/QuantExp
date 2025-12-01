# -*- coding: utf-8 -*-
from __future__ import annotations

import concurrent.futures
import hashlib
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..core.config import DPConfig
from ..core.qc import RejectReport, validate_price_frame
from ..core.version import VERSION
from ..rules.normalize import coerce_cached_price_df
from ..rules.trading_rules import (
    TradingParams,
    add_liquidity_features,
    add_trade_masks,
    compute_hold_all_days_mask,
)
from ..rules.split_policy import SplitPolicy
from ..stores.paths import norm_adjust
from ..stores.panel_store import PanelMeta, PanelStore
from ..stores.price_store import price_glob, read_price
from ..stores.info_store import InfoStore
from ..stores.fundamental_store import FundamentalStore
from ..utils.io import atomic_save_parquet, atomic_save_json
from ..utils.code import normalize_code, norm_code_series

# external deps (kept injectable in facade)
from .. import __init__ as _noop  # noqa: F401


# =============================================================================
# Helpers
# =============================================================================
def _fingerprint(cfg: DPConfig, AlphaFactory: Any) -> str:
    """
    Fingerprint for panel cache.

    NOTE: USE_CROSS_SECTIONAL removed to avoid ambiguity.
    """
    keys = cfg.get("PANEL_CACHE_FINGERPRINT_KEYS", None)
    if not keys:
        keys = [
            "CONTEXT_LEN", "PRED_LEN", "STRIDE", "PRICE_SCHEMA_VER",
            "FEATURE_PREFIXES", "ALPHA_BACKEND", "USE_FUNDAMENTAL",
            "UNIVERSE_MIN_PRICE", "UNIVERSE_MIN_LIST_DAYS",
            "MIN_LIST_DAYS", "MIN_DOLLAR_VOL_FOR_TRADE", "MIN_PRICE",
            "LIMIT_RATE_MAIN", "LIMIT_RATE_ST", "LIMIT_RATE_GEM", "LIMIT_RATE_STAR", "LIMIT_RATE_BSE", "LIMIT_RATE_BSHARE",
            "LIMIT_EPS", "DATASET_BACKEND", "INCLUDE_BSE", "INCLUDE_BSHARE",
            "GATE_TARGET_WITH_ENTRY", "GATE_TARGET_WITH_EXIT", "ALIGN_TO_CALENDAR",
            "ENTRY_PRICE_MODE", "DELIST_GAP_DAYS", "GATE_TARGET_WITH_HOLD_ALL_DAYS",
        ]
    payload = {"dp_ver": VERSION}
    for k in keys:
        payload[k] = cfg.get(k, None)
    payload["alpha_factory"] = getattr(AlphaFactory, "VERSION", None) or getattr(AlphaFactory, "__name__", "AlphaFactory")
    raw = str(sorted(payload.items())).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _mask_ok(s: pd.Series) -> pd.Series:
    """True if mask value > 0.5; fixes the classic `.notna()` bug."""
    return pd.to_numeric(s, errors="coerce").fillna(0.0) > 0.5


def _align_to_calendar(df: pd.DataFrame, trade_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Align per-code to official trade calendar.

    FIX:
    - Reindex will create NaNs; naive validate_price_frame() will reject.
    - We fill missing trade-days as "suspension-like":
      * prices forward-filled
      * volume/amount/turnover set to 0
      * cal_filled flag for debugging/observability
    """
    if df is None or df.empty:
        return df

    df = df.sort_values(["code", "date"]).copy()
    out_parts: List[pd.DataFrame] = []

    for code, g in df.groupby("code", sort=False):
        g = g.copy()
        g["date"] = pd.to_datetime(g["date"], errors="coerce")
        g = g.dropna(subset=["date"])
        if g.empty:
            continue

        dmin, dmax = g["date"].min(), g["date"].max()
        idx = trade_dates[(trade_dates >= dmin) & (trade_dates <= dmax)]
        if len(idx) == 0:
            continue

        base = g.set_index("date")
        base.index = pd.to_datetime(base.index, errors="coerce")
        base = base[~base.index.isna()]
        base = base[~base.index.duplicated(keep="last")].sort_index()

        # reindex then fill as suspension-like
        base = base.reindex(idx)

        px_cols = [c for c in ["open", "high", "low", "close"] if c in base.columns]
        if px_cols:
            miss = base[px_cols].isna().all(axis=1)
        else:
            miss = base.isna().all(axis=1)

        base["cal_filled"] = miss.astype(np.int8)

        # forward-fill prices
        if "close" in base.columns:
            base["close"] = pd.to_numeric(base["close"], errors="coerce").ffill()
        for c in ["open", "high", "low"]:
            if c in base.columns:
                base[c] = pd.to_numeric(base[c], errors="coerce").ffill()
                if "close" in base.columns:
                    base.loc[miss, c] = base.loc[miss, "close"]

        # clear volumes on filled rows
        for c in ["volume", "amount", "turnover"]:
            if c in base.columns:
                base[c] = pd.to_numeric(base[c], errors="coerce")
                base.loc[miss, c] = 0.0

        base["code"] = code
        base = base.reset_index().rename(columns={"index": "date"})
        out_parts.append(base)

    return pd.concat(out_parts, ignore_index=True) if out_parts else df


def _attach_code_meta(df: pd.DataFrame, meta_map: Dict[str, Dict[str, Any]], sec_map: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    code = normalize_code(df["code"].iloc[0]) or str(df["code"].iloc[0])
    m = meta_map.get(code, {})
    s = sec_map.get(code, {})
    out = df.copy()
    out["code"] = norm_code_series(out["code"])
    out = out.dropna(subset=["code"]).copy()
    out["code"] = out["code"].astype(str)

    out["board"] = str(m.get("board", s.get("board", "")))
    out["is_st"] = bool(m.get("is_st", False))
    out["limit_rate"] = float(m.get("limit_rate", np.nan))
    out["name"] = str(m.get("name", s.get("name", "")))

    out["first_date_hint"] = s.get("first_date", pd.NaT)
    out["last_date_hint"] = s.get("last_date", pd.NaT)
    out["is_delisted_guess"] = bool(s.get("is_delisted_guess", False))
    return out


def _attach_static_info(df: pd.DataFrame, info_map: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    FIX:
    - int(np.nan) will crash
    - float(x or np.nan) mistakenly treats 0 as missing
    """
    if df is None or df.empty:
        return df

    code = normalize_code(df["code"].iloc[0]) or str(df["code"].iloc[0])
    m = info_map.get(code, {})
    out = df.copy()

    ind = pd.to_numeric(m.get("industry_cat", 0), errors="coerce")
    out["industry_cat"] = int(0 if pd.isna(ind) else ind)
    out["industry"] = out["industry_cat"].astype(int, copy=False)

    out["list_date"] = str(m.get("list_date", "") or "")

    cap = pd.to_numeric(m.get("total_mkt_cap", np.nan), errors="coerce")
    out["total_mkt_cap"] = float(cap) if np.isfinite(cap) else np.nan

    nm = str(m.get("name", "") or "")
    if "name" in out.columns:
        mask_empty = out["name"].astype(str).str.len() == 0
        if nm:
            out.loc[mask_empty, "name"] = nm
    else:
        out["name"] = nm
    return out


def _attach_fundamentals_pit(df: pd.DataFrame, fund_store: FundamentalStore, code: str, cfg: DPConfig) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if not bool(cfg.get("USE_FUNDAMENTAL", False)):
        return df

    f = fund_store.load_one(code)
    need_cols = ["roe", "rev_growth", "profit_growth", "debt_ratio", "eps", "bps", "pe_ttm", "pb"]

    if f is None:
        for c in need_cols:
            if c not in df.columns:
                df[c] = np.nan
        return df

    pit = fund_store.to_pit(f)
    if pit is None or pit.empty:
        for c in need_cols:
            if c not in df.columns:
                df[c] = np.nan
        return df

    x = df.sort_values("date").copy()
    pit = pit.sort_values("date").copy()
    merged = pd.merge_asof(x, pit, on="date", direction="backward")

    limit_eps = float(cfg.get("LIMIT_EPS", 0.002) or 0.002)
    if "pe_ttm" not in merged.columns:
        eps = pd.to_numeric(merged.get("eps", np.nan), errors="coerce")
        pe = np.where(eps > limit_eps, merged["close"].astype(float) / eps, np.nan)
        merged["pe_ttm"] = pe.astype(np.float32)
    if "pb" not in merged.columns:
        bps = pd.to_numeric(merged.get("bps", np.nan), errors="coerce")
        pb = np.where(bps > limit_eps, merged["close"].astype(float) / bps, np.nan)
        merged["pb"] = pb.astype(np.float32)

    for c in need_cols:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").astype(np.float32)
        else:
            merged[c] = np.nan
    return merged


def _tag_universe(df: pd.DataFrame, cfg: DPConfig) -> pd.DataFrame:
    df = df.sort_values(["code", "date"]).copy()
    df["list_days_count"] = df.groupby("code")["date"].cumcount() + 1

    cond_vol = df.get("volume", 0.0).fillna(0.0) > 0
    cond_price = df["close"].fillna(0.0) >= float(cfg.get("UNIVERSE_MIN_PRICE", 2.0) or 2.0)
    cond_list = df["list_days_count"] > int(cfg.get("UNIVERSE_MIN_LIST_DAYS", 60) or 60)

    df["is_universe"] = (cond_vol & cond_price & cond_list).astype(bool)
    df.drop(columns=["list_days_count"], inplace=True)
    return df


# =============================================================================
# PanelPipeline
# =============================================================================
class PanelPipeline:
    def __init__(self, cfg: DPConfig, logger, calendar_store, universe_store, secmaster_store, AlphaFactory):
        self.cfg = cfg
        self.logger = logger
        self.calendar = calendar_store
        self.universe = universe_store
        self.secmaster = secmaster_store
        self.AlphaFactory = AlphaFactory
        self.info_store = InfoStore(cfg, logger)
        self.fund_store = FundamentalStore(cfg, logger)

    def _process_one_code_file(
        self,
        price_file: str,
        mode: str,
        adj_norm: str,
        meta_map: Dict[str, Dict[str, Any]],
        sec_map: Dict[str, Dict[str, Any]],
        info_map: Dict[str, Dict[str, Any]],
        trade_dates: Optional[pd.DatetimeIndex],
    ) -> Tuple[Optional[pd.DataFrame], str]:
        df = read_price(self.cfg, price_file, expected_adj=adj_norm, logger=self.logger)
        if df is None or df.empty:
            return None, "EmptyPrice"

        df = coerce_cached_price_df(df)

        code = normalize_code(df.get("code", "").iloc[0] if "code" in df.columns else None) or normalize_code(os.path.basename(price_file)) or ""
        if not code:
            return None, "BadCode"
        df["code"] = code

        df = _attach_code_meta(df, meta_map, sec_map)

        if bool(self.cfg.get("ALIGN_TO_CALENDAR", False)) and trade_dates is not None:
            df = _align_to_calendar(df, trade_dates)

        ok, reason = validate_price_frame(df)
        if not ok:
            return None, reason

        context_len = int(self.cfg.get("CONTEXT_LEN", 60) or 60)
        if len(df) <= context_len:
            return None, "TooShort"

        # liquidity + masks
        df = add_liquidity_features(df)
        tp = TradingParams(
            min_list_days=int(self.cfg.get("MIN_LIST_DAYS", 60) or 60),
            min_dollar_vol=float(self.cfg.get("MIN_DOLLAR_VOL_FOR_TRADE", 1e6) or 1e6),
            min_price=float(self.cfg.get("MIN_PRICE", 1.0) or 1.0),
            limit_eps=float(self.cfg.get("LIMIT_EPS", 0.002) or 0.002),
        )
        df = add_trade_masks(df, tp)

        # static info & PIT fundamentals
        df = _attach_static_info(df, info_map)
        df = _attach_fundamentals_pit(df, self.fund_store, code, self.cfg)

        # factors
        try:
            af = self.AlphaFactory(df)
            df = af.make_factors()  # type: ignore
        except Exception as e:
            self.logger.exception(f"[AlphaFactory] failed for code={code} file={price_file}: {e}")
            return None, f"AlphaFail({type(e).__name__})"

        # ---- labels ----
        pred_len = int(self.cfg.get("PRED_LEN", 5) or 5)
        g = df.groupby("code", sort=False)

        entry_price_mode = str(self.cfg.get("ENTRY_PRICE_MODE", "open") or "open").lower().strip()
        if entry_price_mode == "close":
            df["entry_price"] = g["close"].shift(-1)
        else:
            df["entry_price"] = g["open"].shift(-1)

        df["future_close"] = g["close"].shift(-pred_len)
        df["target"] = df["future_close"] / (df["entry_price"] + 1e-9) - 1.0

        gate_entry = bool(self.cfg.get("GATE_TARGET_WITH_ENTRY", True))
        gate_exit = bool(self.cfg.get("GATE_TARGET_WITH_EXIT", False))
        gate_hold_all = bool(self.cfg.get("GATE_TARGET_WITH_HOLD_ALL_DAYS", False))

        if gate_entry:
            entry_ok = _mask_ok(g["buyable_mask"].shift(-1))
            df.loc[~entry_ok, "target"] = np.nan

        if gate_exit and pred_len > 0:
            exit_ok = _mask_ok(g["sellable_mask"].shift(-pred_len))
            df.loc[~exit_ok, "target"] = np.nan

        if gate_hold_all and pred_len > 0:
            hold_ok = compute_hold_all_days_mask(df, pred_len=pred_len)
            hold_ok = pd.to_numeric(hold_ok, errors="coerce").fillna(0.0).astype(bool)
            df.loc[~hold_ok, "target"] = np.nan

        df.loc[~np.isfinite(pd.to_numeric(df["target"], errors="coerce")), "target"] = np.nan
        if mode == "train":
            df = df.dropna(subset=["target"]).reset_index(drop=True)

        df = _tag_universe(df, self.cfg)
        return df, "OK"

    def build_parts(
        self,
        mode: str,
        force_refresh: bool,
        adjust: str,
        backend: Optional[str],
        debug: bool,
        meta_map: Dict[str, Dict[str, Any]],
        sec_map: Dict[str, Dict[str, Any]],
        universe_asof: pd.Timestamp,
    ) -> Tuple[List[str], List[str], RejectReport, str, str]:
        adj_norm = norm_adjust(adjust)
        fp = _fingerprint(self.cfg, self.AlphaFactory)
        store = PanelStore(self.cfg)
        part_dir = store.parts_dir(mode, adj_norm, fp)
        os.makedirs(part_dir, exist_ok=True)

        price_files = price_glob(self.cfg, adj_norm, self.logger)
        if not price_files:
            raise RuntimeError(f"No price data found for adjust={adj_norm}. Run download_data() first.")

        info_df = self.info_store.load_master(force=False)
        info_map = self.info_store.to_map(info_df)

        if debug or bool(self.cfg.get("DEBUG", False)):
            backend = "serial"
            price_files = price_files[: int(self.cfg.get("DEBUG_MAX_FILES", 10) or 10)]

        backend = (backend or str(self.cfg.get("ALPHA_BACKEND", "process") or "process")).lower().strip()

        trade_dates = None
        if bool(self.cfg.get("ALIGN_TO_CALENDAR", False)):
            trade_dates = self.calendar.get_trade_dates()

        flush_n = int(self.cfg.get("PANEL_FLUSH_N", 200) or 200)
        buffer: List[pd.DataFrame] = []
        part_paths: List[str] = []
        reject = RejectReport(max_samples_per_reason=30)

        def flush() -> None:
            if not buffer:
                return
            p = os.path.join(part_dir, f"part_{len(part_paths):05d}.parquet")
            tmp = pd.concat(buffer, ignore_index=True)
            atomic_save_parquet(tmp, p, index=False, compression=str(self.cfg.get("PARQUET_COMPRESSION", "zstd") or "zstd"))
            part_paths.append(p)
            buffer.clear()

        fail_fast = bool(self.cfg.get("FAIL_FAST", True))

        if backend == "serial":
            for pf in tqdm(price_files, desc="Panel(serial)"):
                try:
                    out, reason = self._process_one_code_file(pf, mode, adj_norm, meta_map, sec_map, info_map, trade_dates)
                    if out is None or out.empty:
                        reject.add(reason, normalize_code(os.path.basename(pf)) or pf[-10:])
                        continue
                    buffer.append(out)
                except Exception as e:
                    reject.add(f"Exc({type(e).__name__})", normalize_code(os.path.basename(pf)) or pf[-10:])
                    self.logger.exception(f"[serial] Failed on file={pf}: {e}")
                    if fail_fast:
                        raise
                if len(buffer) >= flush_n:
                    flush()
            flush()

        elif backend in ("thread", "threads"):
            workers = int(self.cfg.get("ALPHA_WORKERS", min(32, (os.cpu_count() or 8) + 8)) or 16)
            self.logger.info(f"[Panel] backend=threads workers={workers}")
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(self._process_one_code_file, pf, mode, adj_norm, meta_map, sec_map, info_map, trade_dates) for pf in price_files]
                for pf, fut in tqdm(list(zip(price_files, futs)), total=len(futs), desc="Panel(threads)"):
                    try:
                        out, reason = fut.result()
                        if out is None or out.empty:
                            reject.add(reason, normalize_code(os.path.basename(pf)) or pf[-10:])
                            continue
                        buffer.append(out)
                    except Exception as e:
                        reject.add(f"Exc({type(e).__name__})", normalize_code(os.path.basename(pf)) or pf[-10:])
                        self.logger.exception(f"[threads] Failed on file={pf}: {e}")
                        if fail_fast:
                            raise
                    if len(buffer) >= flush_n:
                        flush()
            flush()

        elif backend == "process":
            self.logger.warning("backend=process is not enabled in this sample; falling back to threads.")
            return self.build_parts(
                mode, force_refresh, adjust, backend="threads", debug=debug,
                meta_map=meta_map, sec_map=sec_map, universe_asof=universe_asof
            )
        else:
            raise ValueError(f"Unknown backend={backend}")

        if not part_paths:
            raise ValueError("Not enough valid price/factor data to build panel parts (no parts).")

        # =============================================================================
        # Feature columns selection
        # - parts: only base factors (no cs_)
        # - meta.feature_cols: base factors + derived cs_ factors (institutional CS)
        # =============================================================================
        user_prefixes = self.cfg.get("FEATURE_PREFIXES", None)
        if user_prefixes:
            prefixes = list(user_prefixes)
        else:
            prefixes = []

        # Keep cs_ in prefixes so downstream knows it's a feature,
        # but parts won't contain them (they will be created on materialize()).
        fallback_prefixes = [
            "raw_", "style_", "ind_", "ccf_", "adv_", "int_", "time_", "meta_", "fund_",
            "alpha_", "fac_", "cs_",
        ]
        prefixes = list(dict.fromkeys(prefixes + fallback_prefixes))

        probe = pd.read_parquet(part_paths[0], nrows=5)
        base_feat_cols = [c for c in probe.columns if any(str(c).startswith(p) for p in prefixes if p != "cs_")]

        # What AlphaFactory.add_cross_sectional_factors will transform into cs_*
        cs_src_prefixes = ["style_", "ind_", "fund_", "adv_", "int_", "ccf_"]
        cs_src_cols = [c for c in base_feat_cols if any(str(c).startswith(p) for p in cs_src_prefixes)]
        cs_feat_cols = [f"cs_{c}" for c in cs_src_cols]

        feat_cols = list(dict.fromkeys(base_feat_cols + cs_feat_cols))

        # Persist reject report
        rej_path = os.path.join(part_dir, f"rejects_{mode}_{adj_norm}_{fp}.json")
        atomic_save_json(reject.to_dict(), rej_path)

        # Write meta
        split_gap = int(self.cfg.get("SPLIT_GAP", int(self.cfg.get("CONTEXT_LEN", 60) or 60)) or 60)
        seq_len = int(self.cfg.get("CONTEXT_LEN", 60) or 60)
        stride = int(self.cfg.get("STRIDE", 1) or 1)

        dates = []
        for ddf in PanelStore(self.cfg).iter_parts(part_paths, columns=["date"]):
            d = pd.to_datetime(ddf["date"], errors="coerce").dropna().unique()
            if len(d):
                dates.append(d)
        unique_dates = np.unique(np.concatenate(dates)) if dates else np.array([], dtype="datetime64[ns]")

        split_manifest = SplitPolicy.compute(
            unique_dates,
            seq_len=seq_len, stride=stride, gap=split_gap,
            train_ratio=float(self.cfg.get("TRAIN_RATIO", 0.7) or 0.7),
            val_ratio=float(self.cfg.get("VAL_RATIO", 0.15) or 0.15),
            test_ratio=float(self.cfg.get("TEST_RATIO", None)) if self.cfg.get("TEST_RATIO", None) is not None else None,
            universe_asof=str(pd.Timestamp(universe_asof).date()),
            fingerprint=fp,
            adjust=adj_norm,
        ).to_dict()

        meta = PanelMeta(
            mode=mode,
            adjust=adj_norm,
            fingerprint=fp,
            universe_asof=str(pd.Timestamp(universe_asof).date()),
            feature_cols=feat_cols,          # includes derived cs_ cols
            part_paths=part_paths,
            created_by=VERSION,
            split_manifest=split_manifest,
        )
        store.write_meta(part_dir, meta)

        return part_paths, feat_cols, reject, part_dir, fp

    def materialize_panel_df(self, part_paths: List[str]) -> pd.DataFrame:
        """
        Materialize all parts into a single panel df, then apply institutional CS factors.

        NOTE:
        - cs_ columns do NOT exist in part parquet.
        - we create them here using AlphaFactory.add_cross_sectional_factors(panel_df).
        """
        merge_batch = int(self.cfg.get("PANEL_MERGE_BATCH", 24) or 24)
        merged_chunks: List[pd.DataFrame] = []

        for i in range(0, len(part_paths), merge_batch):
            batch = [pd.read_parquet(p) for p in part_paths[i:i + merge_batch]]
            merged_chunks.append(pd.concat(batch, ignore_index=True))

        panel_df = pd.concat(merged_chunks, ignore_index=True) if len(merged_chunks) > 1 else merged_chunks[0]
        panel_df["date"] = pd.to_datetime(panel_df["date"], errors="coerce")
        panel_df = panel_df.dropna(subset=["date"]).reset_index(drop=True)

        # Apply institutional CS once (mask-aware / neutralization / gaussrank)
        if hasattr(self.AlphaFactory, "add_cross_sectional_factors"):
            # Prevent accidental duplicates if upstream already created cs_
            cs_cols = [c for c in panel_df.columns if str(c).startswith("cs_")]
            if cs_cols:
                panel_df = panel_df.drop(columns=cs_cols)
            panel_df = self.AlphaFactory.add_cross_sectional_factors(panel_df)  # type: ignore

        return panel_df
