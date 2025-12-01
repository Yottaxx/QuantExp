# -*- coding: utf-8 -*-
from __future__ import annotations

import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..core.config import DPConfig


class VPNCoordinator:
    """
    Wrap a vpn_rotator (callable). Supports force rotate to bypass cooldown.
    """
    def __init__(self, cfg: DPConfig, vpn_rotator: Optional[Callable[[], Any]]):
        self.cfg = cfg
        self.vpn_rotator = vpn_rotator
        self._lock = threading.Lock()
        self._last_ts = 0.0

    def maybe_rotate(self, *, force: bool = False) -> bool:
        if not bool(self.cfg.get("USE_VPN_ROTATOR", False)):
            return False
        if not self.vpn_rotator:
            return False

        cooldown = int(self.cfg.get("VPN_ROTATE_COOLDOWN_SEC", 10) or 10)
        post_sleep = float(self.cfg.get("VPN_POST_ROTATE_SLEEP_SEC", 2.5) or 2.5)

        now = time.time()
        with self._lock:
            if (not force) and (now - self._last_ts < cooldown):
                return False
            self.vpn_rotator()
            self._last_ts = time.time()

        time.sleep(post_sleep)
        return True


class AkErrorClassifier:
    """
    分类 + 决策：
    - proxy / 429 / 403 / captcha：倾向换线
    - net：只有“被掐连接/ECONNRESET/remote closed”子集才倾向换线；纯 timeout 更倾向降速/冷却
    """
    _proxy = re.compile(r"(proxy|tunnel|Unable to connect to proxy|ProxyError)", re.IGNORECASE)
    _ratelimit = re.compile(r"(\b429\b|too many requests|频繁|访问受限)", re.IGNORECASE)
    _blocked = re.compile(r"(\b403\b|forbidden|blocked|captcha|验证码)", re.IGNORECASE)

    # broad net
    _net = re.compile(
        r"(timeout|timed out|readtimeout|connecttimeout|"
        r"connection|ssl|certificate|max retries exceeded|"
        r"remote end closed|RemoteDisconnected|"
        r"ConnectionResetError|ECONNRESET|reset by peer|errno\s*54)",
        re.IGNORECASE,
    )

    # ✅ net 中“更像被掐/被限”的子集：允许 rotate
    _net_rotate = re.compile(
        r"(ConnectionResetError|ECONNRESET|reset by peer|exceeded|errno\s*54|remote end closed|RemoteDisconnected)",
        re.IGNORECASE,
    )

    @classmethod
    def classify(cls, e: Exception) -> str:
        s = f"{type(e).__name__}: {e}"
        if cls._proxy.search(s):
            return "proxy"
        if cls._ratelimit.search(s):
            return "ratelimit"
        if cls._blocked.search(s):
            return "blocked"
        if cls._net.search(s):
            return "net"
        return "other"

    @classmethod
    def net_should_rotate(cls, e: Exception) -> bool:
        s = f"{type(e).__name__}: {e}"
        return bool(cls._net_rotate.search(s))

    @classmethod
    def should_rotate(cls, e: Exception) -> bool:
        k = cls.classify(e)
        if k in ("proxy", "ratelimit", "blocked"):
            return True
        if k == "net":
            return cls.net_should_rotate(e)
        return False


class AdaptiveRateLimiter:
    """
    Cross-thread global QPS limiter + adaptive:
    - on net/ratelimit failure: reduce QPS
    - on sustained success: slowly increase QPS
    """
    def __init__(self, init_qps: float, min_qps: float, max_qps: float):
        self._lock = threading.Lock()
        self.min_qps = max(0.1, float(min_qps))
        self.max_qps = max(self.min_qps, float(max_qps))
        self.qps = min(self.max_qps, max(self.min_qps, float(init_qps)))
        self._next_ts = 0.0
        self._succ = 0

    def acquire(self) -> None:
        with self._lock:
            min_interval = 1.0 / self.qps
            now = time.time()
            wait = max(0.0, self._next_ts - now)
            self._next_ts = max(self._next_ts, now) + min_interval
        if wait > 0:
            time.sleep(wait)

    def on_success(self) -> None:
        with self._lock:
            self._succ += 1
            # 每 80 次成功，小幅提速（很保守，避免再打爆）
            if self._succ % 80 == 0:
                self.qps = min(self.max_qps, self.qps * 1.10)

    def on_failure(self, kind: str) -> None:
        if kind in ("net", "ratelimit"):
            with self._lock:
                self.qps = max(self.min_qps, self.qps * 0.75)
                self._succ = 0


@dataclass
class _AkCircuit:
    consecutive_proxy_fail: int = 0
    consecutive_net_rotate_fail: int = 0
    last_rotate_ts: float = 0.0


class AkClient:
    def __init__(self, cfg: DPConfig, vpn: VPNCoordinator, logger):
        self.cfg = cfg
        self.vpn = vpn
        self.logger = logger

        # 1) 并发闸门
        max_parallel = int(self.cfg.get("AK_MAX_PARALLEL", 3) or 3)
        self._sem = threading.Semaphore(max(1, max_parallel))

        # 2) 全局 QPS（自适应）
        init_qps = float(self.cfg.get("AK_QPS", 2.0) or 2.0)
        min_qps = float(self.cfg.get("AK_QPS_MIN", 0.6) or 0.6)
        max_qps = float(self.cfg.get("AK_QPS_MAX", 3.0) or 3.0)
        self._rl = AdaptiveRateLimiter(init_qps=init_qps, min_qps=min_qps, max_qps=max_qps)

        # 3) 全局冷却窗口（跨线程）
        self._cool_lock = threading.Lock()
        self._cool_until = 0.0

        # circuits
        self._circuit = _AkCircuit()
        self._circuit_lock = threading.Lock()

    def _maybe_global_cooldown(self) -> None:
        now = time.time()
        with self._cool_lock:
            until = self._cool_until
        if until > now:
            time.sleep(until - now)

    def _trigger_global_cooldown(self, seconds: float) -> None:
        now = time.time()
        until = now + max(0.0, seconds)
        with self._cool_lock:
            if until > self._cool_until:
                self._cool_until = until

    def call(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        retries = int(self.cfg.get("AK_RETRIES", 6) or 6)
        base_sleep = float(self.cfg.get("AK_RETRY_BASE_SLEEP", 0.8) or 0.8)
        max_sleep = float(self.cfg.get("AK_RETRY_MAX_SLEEP", 12.0) or 12.0)

        proxy_force_n = int(self.cfg.get("AK_PROXY_FORCE_ROTATE_N", 2) or 2)
        net_force_n = int(self.cfg.get("AK_NET_FORCE_ROTATE_N", 5) or 5)  # ✅ reset/remote-closed 连续N次强制换线

        # 防止“瞎转”影响效率：两次 rotate 至少间隔这么久（force 不受影响）
        min_rotate_interval = float(self.cfg.get("AK_MIN_ROTATE_INTERVAL_SEC", 8.0) or 8.0)

        # 全局熔断冷却
        cool_base = float(self.cfg.get("AK_GLOBAL_COOLDOWN_BASE", 1.5) or 1.5)
        cool_max = float(self.cfg.get("AK_GLOBAL_COOLDOWN_MAX", 20.0) or 20.0)

        last_e: Optional[Exception] = None

        for i in range(retries):
            try:
                self._maybe_global_cooldown()

                with self._sem:
                    self._rl.acquire()
                    res = fn(*args, **kwargs)

                self._rl.on_success()
                return res

            except Exception as e:
                last_e = e
                kind = AkErrorClassifier.classify(e)
                self._rl.on_failure(kind)

                # net/ratelimit：全局冷却（所有线程一起刹车）
                if kind in ("net", "ratelimit"):
                    penalty = min(
                        cool_max,
                        cool_base * (1.6 ** min(i, 6)) * (0.8 + random.random() * 0.4),
                    )
                    self._trigger_global_cooldown(penalty)

                want_rotate = AkErrorClassifier.should_rotate(e)
                force_rotate = False

                with self._circuit_lock:
                    now = time.time()

                    # proxy 连续失败 -> force rotate
                    if kind == "proxy":
                        self._circuit.consecutive_proxy_fail += 1
                    else:
                        self._circuit.consecutive_proxy_fail = 0

                    if self._circuit.consecutive_proxy_fail >= proxy_force_n:
                        force_rotate = True
                        self._circuit.consecutive_proxy_fail = 0

                    # net(可换线子集) 连续失败 -> force rotate
                    if kind == "net" and AkErrorClassifier.net_should_rotate(e):
                        self._circuit.consecutive_net_rotate_fail += 1
                        if self._circuit.consecutive_net_rotate_fail >= net_force_n:
                            force_rotate = True
                            self._circuit.consecutive_net_rotate_fail = 0
                    else:
                        self._circuit.consecutive_net_rotate_fail = 0

                    # 防抖：非 force 情况下，rotate 间隔不足则不 rotate（提升效率）
                    if (not force_rotate) and (now - self._circuit.last_rotate_ts < min_rotate_interval):
                        want_rotate = False

                self.logger.warning(f"⚠️ [AkRetry {i+1}/{retries}] {fn.__name__} failed ({kind}): {e}")

                if want_rotate or force_rotate:
                    try:
                        rotated = self.vpn.maybe_rotate(force=force_rotate)
                        if rotated:
                            with self._circuit_lock:
                                self._circuit.last_rotate_ts = time.time()
                        self.logger.warning(
                            f"⚠️ [VPN {'FORCE ' if force_rotate else ''}{'OK' if rotated else 'SKIP'}] "
                            f"on {fn.__name__} ({kind})"
                        )
                    except Exception as rexc:
                        self.logger.warning(f"⚠️ [VPN rotate error] {type(rexc).__name__}: {rexc}")

                # 单线程退避
                sleep_time = base_sleep * (1.8 ** i) * (0.8 + random.random() * 0.4)
                time.sleep(min(sleep_time, max_sleep))

        self.logger.critical(f"❌ All {retries} retries failed for {fn.__name__}. Last={last_e}")
        raise last_e  # type: ignore
