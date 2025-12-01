from __future__ import annotations
import random, re, threading, time
from typing import Any, Callable, Optional
from ..core.config import DPConfig

class VPNCoordinator:
    def __init__(self, cfg: DPConfig, vpn_rotator: Optional[Callable[[], Any]]):
        self.cfg = cfg
        self.vpn_rotator = vpn_rotator
        self._lock = threading.Lock()
        self._last_ts = 0.0

    def maybe_rotate(self) -> bool:
        if not bool(self.cfg.get("USE_VPN_ROTATOR", False)):
            return False
        if not self.vpn_rotator:
            return False
        cooldown = int(self.cfg.get("VPN_ROTATE_COOLDOWN_SEC", 60) or 60)
        now = time.time()
        with self._lock:
            if now - self._last_ts < cooldown:
                return False
            self.vpn_rotator()
            self._last_ts = time.time()
        time.sleep(float(self.cfg.get("VPN_POST_ROTATE_SLEEP_SEC", 3.0) or 3.0))
        return True

class AkErrorClassifier:
    _compiled = re.compile(
        "|".join(
            [
                r"timeout", r"timed out", r"connection", r"connection reset",
                r"remote end closed", r"proxy", r"tunnel", r"ssl", r"certificate",
                r"max retries exceeded", r"\b429\b", r"too many requests", r"\b403\b",
                r"forbidden", r"blocked", r"captcha", r"频繁", r"访问受限", r"拒绝",
            ]
        ),
        re.IGNORECASE,
    )

    @classmethod
    def should_rotate(cls, e: Exception) -> bool:
        return bool(cls._compiled.search(f"{type(e).__name__}: {e}"))

class AkClient:
    def __init__(self, cfg: DPConfig, vpn: VPNCoordinator, logger):
        self.cfg = cfg
        self.vpn = vpn
        self.logger = logger

    def call(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        retries = int(self.cfg.get("AK_RETRIES", 10) or 10)
        base_sleep = float(self.cfg.get("AK_RETRY_BASE_SLEEP", 1.0) or 1.0)
        max_sleep = float(self.cfg.get("AK_RETRY_MAX_SLEEP", 15.0) or 15.0)
        last_e: Optional[Exception] = None

        for i in range(retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_e = e
                self.logger.warning(f"⚠️ [AkRetry {i+1}/{retries}] {fn.__name__} failed: {e}")
                if AkErrorClassifier.should_rotate(e):
                    try:
                        self.vpn.maybe_rotate()
                    except Exception:
                        pass
                sleep_time = base_sleep * (2 ** i) * (0.8 + random.random() * 0.4)
                time.sleep(min(sleep_time, max_sleep))
        self.logger.critical(f"❌ All {retries} retries failed for {fn.__name__}. Last={last_e}")
        raise last_e  # type: ignore
