# utils/utils_func.py
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


def smart_cast(v: str) -> Any:
    """
    Try to interpret a CLI string as int / float / bool / JSON / None.
    Keeps original string when parsing fails.
    """
    s = str(v).strip()
    if s.lower() in {"none", "null"}:
        return None
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"

    # int
    if re.fullmatch(r"[+-]?\d+", s):
        try:
            return int(s)
        except Exception:
            pass

    # float
    if re.fullmatch(r"[+-]?\d+\.\d*", s) or re.fullmatch(r"[+-]?\d*\.\d+", s):
        try:
            return float(s)
        except Exception:
            pass

    # JSON-like
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            return json.loads(s)
        except Exception:
            return s

    return s


def parse_kv_pairs(kv_list: Optional[List[str]]) -> Dict[str, Any]:
    """
    Parse CLI overrides: ["KEY=VAL", ...] -> dict.
    """
    out: Dict[str, Any] = {}
    if not kv_list:
        return out
    for item in kv_list:
        if "=" not in item:
            raise ValueError(f"Bad --set item: {item!r}, expected KEY=VAL")
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"Bad --set item: {item!r}, empty KEY")
        out[k] = smart_cast(v)
    return out


def apply_config_overrides(
    config_cls: Any,
    overrides: Dict[str, Any],
    strict: bool = False,
) -> List[Tuple[str, Any, Any]]:
    """
    Set attributes on Config class. Return list of (key, old, new).
    """
    changes: List[Tuple[str, Any, Any]] = []
    for k, new_v in overrides.items():
        if strict and not hasattr(config_cls, k):
            raise AttributeError(f"Config has no attribute {k!r}")
        old_v = getattr(config_cls, k, None)
        setattr(config_cls, k, new_v)
        changes.append((k, old_v, new_v))
    return changes


def ensure_dirs(config_cls: Any) -> None:
    """
    Ensure directories exist after Config.DATA_DIR / OUTPUT_DIR updates.
    """
    data_dir = getattr(config_cls, "DATA_DIR", "./data")
    out_dir = getattr(config_cls, "OUTPUT_DIR", "./output")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    # common subdirs used by DataProvider
    os.makedirs(os.path.join(data_dir, "meta"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "fundamental"), exist_ok=True)


def setup_debug_mode(config_cls: Any) -> None:
    """
    Debug mode: make everything breakpoint-friendly & lightweight.
    """
    setattr(config_cls, "DEBUG", True)

    # IO/thread pools
    for k in ["DL_WORKERS", "FIN_WORKERS", "READ_WORKERS"]:
        if hasattr(config_cls, k):
            setattr(config_cls, k, 1)

    # alpha backend hint
    if hasattr(config_cls, "ALPHA_BACKEND"):
        setattr(config_cls, "ALPHA_BACKEND", "serial")
    if hasattr(config_cls, "DEBUG_MAX_FILES"):
        setattr(
            config_cls,
            "DEBUG_MAX_FILES",
            int(getattr(config_cls, "DEBUG_MAX_FILES", 10)),
        )
    if hasattr(config_cls, "FAIL_FAST"):
        setattr(config_cls, "FAIL_FAST", True)

    # reduce training / inference batch for debug (optional)
    if hasattr(config_cls, "BATCH_SIZE"):
        setattr(
            config_cls,
            "BATCH_SIZE",
            max(1, int(getattr(config_cls, "BATCH_SIZE", 128) // 4)),
        )
    if hasattr(config_cls, "INFERENCE_BATCH_SIZE"):
        setattr(
            config_cls,
            "INFERENCE_BATCH_SIZE",
            max(1, int(getattr(config_cls, "INFERENCE_BATCH_SIZE", 256) // 2)),
        )


def debug_print_config(
    config_cls: Any,
    only: Optional[Iterable[str]] = None,
    exclude_prefixes: Tuple[str, ...] = ("__",),
) -> None:
    """
    Print Config.* values (skip callables & dunder).
    """
    keys: List[str] = []
    for k in dir(config_cls):
        if exclude_prefixes and any(k.startswith(p) for p in exclude_prefixes):
            continue
        v = getattr(config_cls, k)
        if callable(v):
            continue
        keys.append(k)

    if only is not None:
        only_set = set(only)
        keys = [k for k in keys if k in only_set]

    keys.sort()

    print("\n" + "=" * 92)
    print("🧩 Effective Config")
    print("=" * 92)
    for k in keys:
        v = getattr(config_cls, k)
        if isinstance(v, (dict, list, tuple)):
            try:
                vv = json.dumps(v, ensure_ascii=False)
            except Exception:
                vv = str(v)
        else:
            vv = str(v)
        print(f"{k:<30} = {vv}")
    print("=" * 92 + "\n")


def patch_dataprovider_defaults(
    DataProvider: Any,
    *,
    adjust: Optional[str] = None,
    force_refresh: Optional[bool] = None,
    debug_flag: Optional[bool] = None,
) -> None:
    """
    Patch DataProvider.load_and_process_panel(mode='train') calls inside other modules
    so they respect CLI adjust/force/DEBUG without changing their source code.

    Works with current signature:
        load_and_process_panel(mode='train', force_refresh=False, adjust='qfq', debug=False)
    """
    if not hasattr(DataProvider, "load_and_process_panel"):
        return

    orig = DataProvider.load_and_process_panel

    # avoid double patch
    if getattr(orig, "__patched_by_main__", False):
        return

    import inspect

    sig = inspect.signature(orig)

    def wrapped(*args, **kwargs):
        # CLI-level override only when the caller didn't specify explicitly
        if adjust is not None and "adjust" in sig.parameters and "adjust" not in kwargs:
            kwargs["adjust"] = adjust
        if (
            force_refresh is not None
            and "force_refresh" in sig.parameters
            and "force_refresh" not in kwargs
        ):
            kwargs["force_refresh"] = force_refresh
        if (
            debug_flag is not None
            and "debug" in sig.parameters
            and "debug" not in kwargs
        ):
            kwargs["debug"] = debug_flag
        return orig(*args, **kwargs)

    wrapped.__patched_by_main__ = True  # type: ignore[attr-defined]

    try:
        # keep staticmethod semantics if originally defined that way
        DataProvider.load_and_process_panel = staticmethod(wrapped)  # type: ignore[assignment]
    except Exception:
        DataProvider.load_and_process_panel = wrapped  # type: ignore[assignment]


def parse_codes_arg(codes: str) -> List[str]:
    """
    Parse comma separated codes like "000001.SZ, 600000.SH" into a clean list.
    """
    if not codes:
        return []
    return [c.strip() for c in codes.split(",") if c.strip()]
