from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

@dataclass(frozen=True)
class DPConfig:
    """Frozen config snapshot for reproducibility & testability."""
    values: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    @classmethod
    def from_module(cls, Config: Any, keys: Optional[Sequence[str]] = None) -> "DPConfig":
        if keys is None:
            # Keep this list centralized to avoid hidden global reads.
            keys = [
                "LOG_LEVEL",
                "DATA_DIR",
                "OUTPUT_DIR",
                "PROXY_URL",
                "USE_VPN_ROTATOR",
                "VPN_ROTATE_COOLDOWN_SEC",
                "VPN_POST_ROTATE_SLEEP_SEC",
                "AK_RETRIES",
                "AK_RETRY_BASE_SLEEP",
                "AK_RETRY_MAX_SLEEP",
                "PRICE_SCHEMA_VER",
                "PARQUET_COMPRESSION",
                "STRICT_PRICE_META",
                "ALLOW_LEGACY_PRICE_CACHE",
                "SKIP_IF_FRESH",
                "FRESH_LAG_DAYS",
                "PRICE_BACKFILL_YEARS",
                "PRICE_OVERLAP_DAYS",
                "PRICE_ADJUSTS",
                "PRICE_WORKERS",
                "SYNC_FUNDAMENTAL",
                "SYNC_INFO",
                "INFO_TTL_DAYS",
                "FUNDAMENTAL_START_YEAR",
                "FUND_TTL_DAYS",
                "FUND_FALLBACK_LAG_DAYS",
                "FIN_WORKERS",
                "CALENDAR_TTL_SEC",
                "MARKET_INDEX_SYMBOL",
                "UNIVERSE_TTL_SEC",
                "FORCE_UNIVERSE_REFRESH",
                "INCLUDE_BSE",
                "INCLUDE_BSHARE",
                "UNIVERSE_ALLOW_PREFIXES",
                "UNIVERSE_EXTRA_CODES_FILE",
                "SECMASTER_TTL_SEC",
                "SECMASTER_SCAN_PRICE",
                "SECMASTER_MAX_SCAN_FILES",
                "FORCE_SECMASTER_REFRESH",
                "DELIST_GAP_DAYS",
                "ALIGN_TO_CALENDAR",
                "USE_FUNDAMENTAL",
                "USE_CROSS_SECTIONAL",
                "FEATURE_PREFIXES",
                "CONTEXT_LEN",
                "PRED_LEN",
                "STRIDE",
                "TRAIN_RATIO",
                "VAL_RATIO",
                "TEST_RATIO",
                "SPLIT_GAP",
                "ALPHA_BACKEND",
                "DEBUG",
                "DEBUG_MAX_FILES",
                "PANEL_FLUSH_N",
                "PANEL_MERGE_BATCH",
                "ALPHA_WORKERS",
                "ALPHA_MAP_CHUNKSIZE",
                "FAIL_FAST",
                "MEMMAP_WRITE_CHUNK",
                "DATASET_BACKEND",
                "FORCE_DATASET_REBUILD",
                "INCLUDE_CS_IN_FEATURES",
                "MIN_LIST_DAYS",
                "MIN_DOLLAR_VOL_FOR_TRADE",
                "MIN_PRICE",
                "LIMIT_RATE_MAIN",
                "LIMIT_RATE_ST",
                "LIMIT_RATE_GEM",
                "LIMIT_RATE_STAR",
                "LIMIT_RATE_BSE",
                "LIMIT_RATE_BSHARE",
                "LIMIT_EPS",
                "UNIVERSE_MIN_PRICE",
                "UNIVERSE_MIN_LIST_DAYS",
                "GATE_TARGET_WITH_ENTRY",
                "GATE_TARGET_WITH_EXIT",
                "GATE_TARGET_WITH_HOLD_ALL_DAYS",
                "ENTRY_PRICE_MODE",
            ]
        vals: Dict[str, Any] = {}
        for k in keys:
            vals[k] = getattr(Config, k, None)
        return cls(vals)

