# -*- coding: utf-8 -*-
"""
price_pipeline.py

Institutional-grade price ETL:
- Explicit skip logic (no silent "always True")
- Incremental fetch with overlap
- Merge/dedupe on (code, date)
- Atomic parquet + sidecar meta.json
- Manifest upsert (only on real fetch results)
- Real-time progress (submit + wait(FIRST_COMPLETED))
- Progress stats: ok/bad/skip/cache/nonew + top fail reasons + recent fails
"""

from __future__ import annotations

import concurrent.futures
import datetime as _dt
import os
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import akshare as ak
import pandas as pd
from tqdm.auto import tqdm

from ..core.config import DPConfig
from ..core.version import VERSION
from ..rules.normalize import normalize_ak_hist_df
from ..stores.manifest_store import ManifestRow, ManifestStore
from ..stores.paths import meta_path, norm_adjust, price_path
from ..utils.code import normalize_code
from ..utils.io import atomic_save_json, atomic_save_parquet


def yyyymmdd(dt: pd.Timestamp) -> str:
    return pd.to_datetime(dt).strftime("%Y%m%d")


def should_skip_fetch(
    cfg: DPConfig,
    last_dt: Optional[pd.Timestamp],
    updated_at: Optional[pd.Timestamp],
    target_dt: pd.Timestamp,
) -> bool:
    """
    Skip if:
      - SKIP_IF_FRESH enabled
      - last_dt exists and is close enough to target_dt given lag tolerance
    Note: updated_at reserved for future policy; currently if last_dt is fresh enough we allow skipping.
    """
    if not bool(cfg.get("SKIP_IF_FRESH", True)):
        return False
    if last_dt is None:
        return False

    lag_days = int(cfg.get("FRESH_LAG_DAYS", 0) or 0)
    last_n = pd.Timestamp(last_dt).normalize()
    tgt_n = pd.Timestamp(target_dt).normalize()

    if last_n < (tgt_n - pd.Timedelta(days=lag_days)):
        return False

    _ = updated_at
    return True


def _short_reason(msg: str) -> str:
    """
    Compress a verbose msg into a stable counter key.
    Examples:
      FetchFail(TimeoutError:xxx) -> FetchFail:TimeoutError
      UsingCache(TimeoutError) -> UsingCache:TimeoutError
    """
    if not msg:
        return "Unknown"
    if msg.startswith("FetchFail(") and msg.endswith(")"):
        inner = msg[len("FetchFail(") : -1]
        typ = inner.split(":", 1)[0].strip() or "Error"
        return f"FetchFail:{typ}"
    if msg.startswith("UsingCache(") and msg.endswith(")"):
        inner = msg[len("UsingCache(") : -1].strip() or "Error"
        return f"UsingCache:{inner}"
    if msg in ("OK", "SkipFresh", "NoNewRows"):
        return msg
    return (msg[:80] + "…") if len(msg) > 80 else msg


@dataclass
class ProgressStats:
    ok: int = 0
    bad: int = 0
    skip: int = 0
    cache: int = 0
    nonew: int = 0
    reasons: Counter = field(default_factory=Counter)
    recent_fails: Deque[str] = field(default_factory=lambda: deque(maxlen=10))

    def observe(self, code: str, adj: str, succ: bool, msg: str) -> None:
        if succ:
            if msg == "SkipFresh":
                self.skip += 1
            elif msg == "NoNewRows":
                self.nonew += 1
            elif msg.startswith("UsingCache("):
                self.cache += 1
            else:
                self.ok += 1
            return

        self.bad += 1
        r = _short_reason(msg)
        self.reasons[r] += 1
        self.recent_fails.append(f"{code}/{adj}: {r}")

    def top_reason(self) -> str:
        if not self.reasons:
            return "-"
        k, v = self.reasons.most_common(1)[0]
        return f"{k}×{v}"


class PricePipeline:
    def __init__(self, cfg: DPConfig, ak_client, calendar_store, universe_store, logger):
        self.cfg = cfg
        self.ak_client = ak_client
        self.calendar = calendar_store
        self.universe = universe_store
        self.logger = logger

    def _ak_hist(self, code: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp, adj_norm: str) -> pd.DataFrame:
        start = yyyymmdd(start_dt)
        end = yyyymmdd(end_dt)
        ak_adjust = "" if adj_norm == "raw" else adj_norm
        return self.ak_client.call(
            ak.stock_zh_a_hist,
            symbol=code,
            period="daily",
            start_date=start,
            end_date=end,
            adjust=ak_adjust,
        )

    def _download_one(
        self,
        code: str,
        adjust: str,
        target_dt: pd.Timestamp,
        manifest_map: Dict[Tuple[str, str], Tuple[Optional[pd.Timestamp], int, Optional[pd.Timestamp]]],
    ):
        code = normalize_code(code) or str(code)
        adj_norm = norm_adjust(adjust)
        schema_ver = int(self.cfg.get("PRICE_SCHEMA_VER", 2) or 2)
        path = price_path(self.cfg, adj_norm, code)

        last_dt, rows_hint, updated_at = (None, 0, None)
        key = (code, adj_norm)
        if key in manifest_map:
            last_dt, rows_hint, updated_at = manifest_map[key]
            last_dt = pd.to_datetime(last_dt, errors="coerce") if last_dt is not None else None

        if should_skip_fetch(self.cfg, last_dt, updated_at, target_dt):
            return code, adj_norm, True, "SkipFresh", last_dt, int(rows_hint)

        # Fallback: infer last_dt from local parquet if manifest missing
        if last_dt is None and os.path.exists(path) and os.path.getsize(path) > 512:
            try:
                old = pd.read_parquet(path, columns=["date"])
                dd = pd.to_datetime(old["date"], errors="coerce").dropna()
                if not dd.empty:
                    last_dt = dd.max()
            except Exception:
                pass

        # Determine fetch start
        if last_dt is None:
            years = int(self.cfg.get("PRICE_BACKFILL_YEARS", 10) or 10)
            start_dt = pd.Timestamp(target_dt) - pd.Timedelta(days=365 * years)
        else:
            overlap = int(self.cfg.get("PRICE_OVERLAP_DAYS", 3) or 3)
            start_dt = pd.Timestamp(last_dt) - pd.Timedelta(days=overlap)

        # Fetch
        try:
            raw = self._ak_hist(code, start_dt, pd.Timestamp(target_dt), adj_norm=adj_norm)
        except Exception as e:
            if os.path.exists(path) and os.path.getsize(path) > 512:
                return code, adj_norm, True, f"UsingCache({type(e).__name__})", last_dt, int(rows_hint)
            return code, adj_norm, False, f"FetchFail({type(e).__name__}:{e})", last_dt, 0

        if raw is None or raw.empty:
            return code, adj_norm, True, "NoNewRows", last_dt, int(rows_hint)

        df_new = normalize_ak_hist_df(raw)
        df_new["code"] = code

        df_old = None
        if os.path.exists(path) and os.path.getsize(path) > 512:
            try:
                df_old = pd.read_parquet(path)
            except Exception:
                df_old = None

        if df_old is not None and not df_old.empty:
            merged = pd.concat([df_old, df_new], ignore_index=True)
            merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
            merged = merged.dropna(subset=["date"]).sort_values(["code", "date"])
            merged = merged.drop_duplicates(["code", "date"], keep="last").reset_index(drop=True)
        else:
            merged = df_new.drop_duplicates(["code", "date"], keep="last").reset_index(drop=True)

        atomic_save_parquet(
            merged,
            path,
            index=False,
            compression=str(self.cfg.get("PARQUET_COMPRESSION", "zstd") or "zstd"),
        )
        atomic_save_json(
            {
                "schema_ver": schema_ver,
                "created_by": VERSION,
                "code": code,
                "stored_adjust": adj_norm,
                "updated_at_utc": _dt.datetime.utcnow().isoformat(),
                "rows": int(len(merged)),
                "last_date": pd.to_datetime(merged["date"]).max().isoformat() if len(merged) else None,
            },
            meta_path(path),
        )

        last = pd.to_datetime(merged["date"]).max() if len(merged) else last_dt
        return code, adj_norm, True, "OK", last, int(len(merged))

    def download_data(self, adjusts: Optional[List[str]] = None) -> None:
        self.logger.info(f"\n{'='*60}\n>>> [ETL] Price Pipeline Initiated ({VERSION})\n{'='*60}")

        target_dt = self.calendar.latest_trade_date()
        self.logger.info(f"📅 Target trading date = {target_dt.date()}")

        if adjusts is None:
            adjusts = self.cfg.get("PRICE_ADJUSTS", ["qfq"]) or ["qfq"]
        adjusts = [norm_adjust(a) for a in adjusts]

        snap = self.universe.get_snapshot(
            target_dt, force_refresh=bool(self.cfg.get("FORCE_UNIVERSE_REFRESH", False))
        )
        codes = snap["code"].astype(str).tolist() if snap is not None and not snap.empty else []
        if not codes:
            self.logger.critical("❌ Universe is empty. Abort.")
            return
        self.logger.info(f"✅ Universe size = {len(codes)}")

        mstore = ManifestStore(
            os.path.join(str(self.cfg.get("DATA_DIR", "./data") or "./data"), "manifest", "price_manifest.parquet"),
            parquet_compression=str(self.cfg.get("PARQUET_COMPRESSION", "zstd") or "zstd"),
        )
        manifest_map = ManifestStore.to_map(mstore.load())

        todo = [(c, a) for c in codes for a in adjusts]
        workers = int(self.cfg.get("PRICE_WORKERS", min(8, (os.cpu_count() or 8) + 8)) or 16)
        max_inflight = int(self.cfg.get("PRICE_MAX_INFLIGHT", workers * 4) or (workers * 4))
        log_every = int(self.cfg.get("PRICE_PROGRESS_LOG_EVERY", 200) or 200)

        updates: List[ManifestRow] = []
        stats = ProgressStats()
        t0 = time.time()

        def _task(args):
            c, a = args
            return self._download_one(c, a, target_dt=target_dt, manifest_map=manifest_map)

        def _submit_more(
            ex: concurrent.futures.Executor,
            q: deque,
            inflight: Dict[concurrent.futures.Future, Tuple[str, str]],
        ) -> None:
            while q and len(inflight) < max_inflight:
                args = q.popleft()
                inflight[ex.submit(_task, args)] = args

        self.logger.info(f"🚀 Starting price download with {workers} workers... (todo={len(todo)})")

        q = deque(todo)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            inflight: Dict[concurrent.futures.Future, Tuple[str, str]] = {}
            _submit_more(ex, q, inflight)

            with tqdm(
                total=len(todo),
                dynamic_ncols=True,
                desc="📥 Download Prices",
                unit="task",
                smoothing=0.1,
            ) as pbar:
                done_cnt = 0
                while inflight:
                    done, _ = concurrent.futures.wait(
                        inflight.keys(),
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )

                    for fut in done:
                        c, a = inflight.pop(fut)

                        try:
                            code, adj, succ, msg, last_dt, rows = fut.result()
                        except Exception as e:
                            succ = False
                            code, adj = c, a
                            msg = f"Crash({type(e).__name__}:{e})"
                            last_dt, rows = None, 0

                        stats.observe(code, adj, succ=succ, msg=str(msg))

                        # Manifest policy: only refresh updated_at when we truly fetched successfully.
                        # - OK / NoNewRows: fetched (or confirmed) -> update
                        # - SkipFresh: do NOT touch updated_at
                        # - UsingCache: do NOT touch updated_at (still old data)
                        if succ and str(msg) in ("OK", "NoNewRows"):
                            updates.append(
                                ManifestRow(
                                    code=code,
                                    adjust=adj,
                                    last_date=last_dt,
                                    rows=int(rows),
                                    updated_at=pd.Timestamp.utcnow(),
                                    schema_ver=int(self.cfg.get("PRICE_SCHEMA_VER", 2) or 2),
                                )
                            )

                        if not succ:
                            tqdm.write(f"[Price] {code}/{adj} failed: {msg}")

                        done_cnt += 1
                        pbar.update(1)

                        # lightweight postfix (fast)
                        pbar.set_postfix(
                            ok=stats.ok,
                            bad=stats.bad,
                            skip=stats.skip,
                            cache=stats.cache,
                            nonew=stats.nonew,
                            topfail=stats.top_reason(),
                            last=f"{code}/{adj}",
                        )

                        # periodic richer logs (avoid flooding)
                        if log_every > 0 and done_cnt % log_every == 0:
                            elapsed = max(1e-6, time.time() - t0)
                            rate = done_cnt / elapsed
                            top3 = stats.reasons.most_common(3)
                            top3_str = ", ".join([f"{k}×{v}" for k, v in top3]) if top3 else "-"
                            recent = " | ".join(list(stats.recent_fails)[-5:]) if stats.recent_fails else "-"
                            tqdm.write(
                                f"[Price][Progress] done={done_cnt}/{len(todo)} rate={rate:.2f}/s "
                                f"ok={stats.ok} bad={stats.bad} skip={stats.skip} cache={stats.cache} nonew={stats.nonew} "
                                f"top3=[{top3_str}] recent=[{recent}]"
                            )

                    _submit_more(ex, q, inflight)

        if updates:
            mstore.upsert_many(updates)

        self.logger.info(
            f"✅ Price update done. OK={stats.ok} Bad={stats.bad} Skip={stats.skip} "
            f"Cache={stats.cache} NoNew={stats.nonew}"
        )
        if stats.reasons:
            top5 = stats.reasons.most_common(5)
            self.logger.warning("⚠️ Top fail reasons: " + ", ".join([f"{k}×{v}" for k, v in top5]))
        if stats.recent_fails:
            self.logger.warning("⚠️ Recent fails: " + " | ".join(list(stats.recent_fails)))
