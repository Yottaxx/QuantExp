# -*- coding: utf-8 -*-
from __future__ import annotations

import random
import time
from urllib.parse import quote
from typing import Dict, Optional, List

import requests
from src.config import Config
from utils.logging_utils import get_logger

logger = get_logger()


class ClashRotator:
    """
    Clash 代理控制器（增强版）：
    - 通过 Clash API 随机切换节点
    - 切换后 probe 东财同域（不通就继续换）
    - 坏节点短期拉黑 TTL，避免反复切到坏节点
    """

    def __init__(self, controller_url=None, secret=None):
        self.base_url = controller_url or getattr(Config, "CLASH_API_URL", "http://127.0.0.1:9090")
        _secret = secret or getattr(Config, "CLASH_SECRET", "")

        self.headers = {
            "Authorization": f"Bearer {_secret}",
            "Content-Type": "application/json",
        }

        self.selector_name: Optional[str] = None
        self.node_list: List[str] = []
        self.fallback_selectors = ["GLOBAL", "Proxy", "节点选择", "国外流量", "Global", "PROXY"]

        # Clash API session：不走系统代理
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update(self.headers)

        # Probe session：默认 trust_env=True（走系统代理/环境变量，需与你实际数据请求的代理路径一致）
        self.probe_session = requests.Session()
        self.probe_session.headers.update({"User-Agent": "Mozilla/5.0", "Connection": "close"})

        self.probe_timeout = float(getattr(Config, "CLASH_PROBE_TIMEOUT", 3.0) or 3.0)
        self.bad_ttl = int(getattr(Config, "CLASH_BAD_NODE_TTL_SEC", 180) or 180)
        self.max_switch_tries = int(getattr(Config, "CLASH_SWITCH_TRIES", 6) or 6)

        # 东财同域探测：尽量贴近 stock_zh_a_hist 的链路
        self.probe_url = getattr(
            Config,
            "CLASH_PROBE_URL",
            "https://push2his.eastmoney.com/api/qt/stock/kline/get"
            "?fields1=f1&fields2=f51&ut=7eea3edcaed734bea9cbfc24409ed989"
            "&klt=101&fqt=1&secid=1.000001&beg=20250101&end=20250102",
        )

        self._bad_until: Dict[str, float] = {}

    def _refresh_metadata(self) -> None:
        try:
            url = f"{self.base_url}/proxies"
            resp = self.session.get(url, timeout=2)
            if resp.status_code != 200:
                logger.warning(f"❌ Clash API 连接失败: {resp.status_code}")
                self.node_list = []
                return

            proxies = resp.json().get("proxies", {})
            best_group = None
            max_nodes = 0
            best_nodes: List[str] = []

            for name, info in proxies.items():
                if info.get("type") != "Selector":
                    continue
                nodes = info.get("all", []) or []
                real_nodes = [
                    n for n in nodes
                    if n not in ["DIRECT", "REJECT", "PASS", "自动选择", "故障转移", "Compatible"]
                ]
                if len(real_nodes) > max_nodes:
                    max_nodes = len(real_nodes)
                    best_group = name
                    best_nodes = real_nodes

            self.selector_name = best_group
            self.node_list = best_nodes
        except Exception as e:
            logger.warning(f"❌ 获取 Clash 元数据失败: {e}")
            self.node_list = []

    def _current_node(self) -> Optional[str]:
        if not self.selector_name:
            return None
        try:
            url = f"{self.base_url}/proxies"
            resp = self.session.get(url, timeout=2)
            if resp.status_code != 200:
                return None
            proxies = resp.json().get("proxies", {})
            info = proxies.get(self.selector_name, {}) or {}
            return info.get("now")
        except Exception:
            return None

    def _is_bad(self, node: str) -> bool:
        until = self._bad_until.get(node)
        return bool(until and until > time.time())

    def _mark_bad(self, node: str) -> None:
        self._bad_until[node] = time.time() + self.bad_ttl

    def _probe(self) -> bool:
        """
        关键：probe 走“系统代理/环境变量”路径，必须与你 AkShare 实际走的代理路径一致。
        """
        try:
            r = self.probe_session.get(self.probe_url, timeout=self.probe_timeout)
            if r.status_code != 200:
                return False
            _ = r.json()
            return True
        except Exception:
            return False

    def switch_random(self) -> bool:
        if not self.selector_name or not self.node_list:
            self._refresh_metadata()
            if not self.node_list:
                logger.error("❌ 未找到可用的 Clash 代理节点列表")
                return False

        current = self._current_node()

        # candidates: not current & not bad
        candidates = [n for n in self.node_list if n != current and not self._is_bad(n)]
        if not candidates:
            # cleanup expired badlist and retry
            now = time.time()
            self._bad_until = {k: v for k, v in self._bad_until.items() if v > now}
            candidates = [n for n in self.node_list if n != current and not self._is_bad(n)]

        random.shuffle(candidates)
        candidates = candidates[: max(1, self.max_switch_tries)]

        attempt_selectors = [self.selector_name] + [s for s in self.fallback_selectors if s != self.selector_name]

        for target_node in candidates:
            success = False

            for selector in attempt_selectors:
                if not selector:
                    continue
                try:
                    safe_group = quote(selector)
                    url = f"{self.base_url}/proxies/{safe_group}"
                    payload = {"name": target_node}
                    resp = self.session.put(url, json=payload, timeout=3)
                    if resp.status_code == 204:
                        success = True
                        break
                except Exception:
                    continue

            if not success:
                self._mark_bad(target_node)
                continue

            # ✅ rotate 成功后必须 probe：对东财不通就继续换
            if self._probe():
                logger.info(f"🔄 Clash 节点切换成功且可用: 【{target_node}】 (group={self.selector_name})")
                return True

            logger.warning(f"⚠️ 节点可切但 probe 失败: 【{target_node}】 -> 拉黑 {self.bad_ttl}s")
            self._mark_bad(target_node)

        logger.error("❌ 多次切换后仍无法找到可用节点（ProbeFail/AllBad）")
        return False

    def __call__(self):
        return self.switch_random()


vpn_rotator = ClashRotator()
