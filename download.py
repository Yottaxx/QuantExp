#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download.py (project root)

What it does
- Download & cache raw streams through your DataProvider pipelines:
  * prices (required)
  * info/fundamental (optional, batch by latest universe snapshot if available)

Optional
- --build-panel: build parts + materialize panel_df (for analysis/backtest compatibility)
- --save-dataset: build HF Dataset and save to disk (parts_dir/hf_dataset)

Examples
  python download.py --adjusts qfq
  python download.py --adjusts qfq,hfq --build-panel --mode train
  python download.py --adjusts qfq --build-panel --save-dataset

  如果你希望它“只下载行情/静态/财务”分别可控（例如 --no-info / --no-fundamental），我也可以把这几个开关加上——但你当前 DataProvider.download_data() 已经是“price 主依赖 + 其余按 snapshot 可选”，这个脚本会最大程度复用你现有工程闭环。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from typing import List, Optional


def _ensure_sys_path() -> None:
    """Ensure project root is in sys.path so `import src.*` works when run as a script."""
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)


def _parse_adjusts(s: str) -> Optional[List[str]]:
    s = (s or "").strip()
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _import_dataprovider():
        from src.data_provider import DataProvider  # type: ignore
        return DataProvider


def _load_vpn_rotator():
    from src.data_provider.utils.vpn_rotator import vpn_rotator  # type: ignore
    return vpn_rotator



def main() -> int:
    _ensure_sys_path()

    p = argparse.ArgumentParser(description="Download & cache data via DataProvider.")
    p.add_argument("--adjusts", type=str, default="qfq", help="Comma-separated adjusts, e.g. qfq,hfq,raw. Empty => DataProvider default.")
    p.add_argument("--build-panel", action="store_true", help="Also build parts and materialize panel_df after download.")
    p.add_argument("--save-dataset", action="store_true", help="Also build HF Dataset and save_to_disk (requires --build-panel).")

    p.add_argument("--mode", type=str, default="train", choices=["train", "predict", "test"], help="Panel mode for build_panel.")
    p.add_argument("--force-refresh", action="store_true", help="Force refresh panel building caches (build_panel path).")
    p.add_argument("--backend", type=str, default=None, help="Optional backend selector for panel pipeline.")
    p.add_argument("--debug", action="store_true", help="Enable debug mode for panel pipeline.")

    p.add_argument("--run-manifest", type=str, default="runs", help="Where to write a small run manifest json (default: ./runs).")

    args = p.parse_args()
    adjusts = _parse_adjusts(args.adjusts)

    if args.save_dataset and not args.build_panel:
        print("❌ --save-dataset requires --build-panel")
        return 2

    # ---- Imports from your project ----
    try:
        from src.config import Config  # type: ignore
        from src.alpha_lib import AlphaFactory  # type: ignore
    except Exception as e:
        print("❌ Failed to import src.config / src.alpha_lib. Are you running from project root?")
        print(f"   Error: {e}")
        return 2

    try:
        DataProvider = _import_dataprovider()
        vpn_rotator = _load_vpn_rotator()
        dp = DataProvider(Config=Config, AlphaFactory=AlphaFactory, vpn_rotator=vpn_rotator)
    except Exception as e:
        print("❌ Failed to construct DataProvider")
        print(f"   Error: {e}")
        traceback.print_exc()
        return 2

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = {
        "ts": run_ts,
        "adjusts": adjusts,
        "build_panel": bool(args.build_panel),
        "save_dataset": bool(args.save_dataset),
        "mode": args.mode,
        "force_refresh": bool(args.force_refresh),
        "backend": args.backend,
        "debug": bool(args.debug),
    }

    try:
        print(f"\n=== [download.py] Downloading data (adjusts={adjusts}) ===")
        dp.download_data(adjusts=adjusts)
        print("✅ download_data finished (cached by pipelines).")

        if args.build_panel:
            # For panel build, pick the first adjust as the main one (most common usage).
            adj_main = (adjusts[0] if adjusts else "qfq")
            print(f"\n=== [download.py] Building panel (adjust={adj_main}, mode={args.mode}) ===")
            panel_df, feature_cols = dp.load_and_process_panel(
                mode=args.mode,
                force_refresh=bool(args.force_refresh),
                adjust=adj_main,
                backend=args.backend,
                debug=bool(args.debug),
            )
            manifest.update(
                {
                    "panel": {
                        "rows": int(len(panel_df)),
                        "cols": int(panel_df.shape[1]),
                        "feature_cols": int(len(feature_cols)),
                        "attrs": {k: panel_df.attrs.get(k) for k in ["adjust", "mode", "fingerprint", "universe_asof", "created_by", "parts_dir"]},
                    }
                }
            )
            print("✅ panel_df materialized.")
            if isinstance(panel_df.attrs.get("parts_dir"), str) and panel_df.attrs.get("parts_dir"):
                print(f"   parts_dir: {panel_df.attrs.get('parts_dir')}")

            if args.save_dataset:
                print("\n=== [download.py] Building dataset (and saving to disk if enabled in cfg) ===")
                ds = dp.make_dataset(panel_df, feature_cols)
                # dp.make_dataset already saves to parts_dir/hf_dataset when configured.
                manifest["dataset"] = {"type": str(type(ds))}

                parts_dir = str(panel_df.attrs.get("parts_dir", "") or "").strip()
                if parts_dir:
                    hf_dir = os.path.join(parts_dir, "hf_dataset")
                    manifest["dataset"]["saved_to"] = hf_dir
                    print(f"✅ dataset built. (expected save_to_disk: {hf_dir})")

        # write manifest
        out_dir = str(args.run_manifest or "").strip()
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"download_run_{run_ts}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            print(f"\n🧾 run manifest: {out_path}")

        print("\n🎉 Done.")
        return 0

    except KeyboardInterrupt:
        print("\n🟡 Interrupted by user.")
        return 130
    except Exception as e:
        print("\n❌ Failed.")
        print(f"   Error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
