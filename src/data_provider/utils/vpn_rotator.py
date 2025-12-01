# -*- coding: utf-8 -*-
import requests
import random
import logging
from urllib.parse import quote
from src.config import Config

# 配置日志
logger = logging.getLogger(__name__)


class ClashRotator:
    """
    【Clash 代理控制器】
    用于通过 Clash API 自动切换代理节点
    """

    def __init__(self, controller_url=None, secret=None):
        """
        初始化：参数若为空则从 Config 读取
        """
        self.base_url = controller_url or getattr(Config, "CLASH_API_URL", "http://127.0.0.1:9090")
        _secret = secret or getattr(Config, "CLASH_SECRET", "")

        self.headers = {
            "Authorization": f"Bearer {_secret}",
            "Content-Type": "application/json"
        }
        self.selector_name = None
        self.node_list = []
        # 常见的分流组名称，程序会自动尝试寻找这些组
        self.fallback_selectors = ['GLOBAL', 'Proxy', '节点选择', '国外流量', 'Global', 'PROXY']

        self.session = requests.Session()
        self.session.trust_env = False  # 访问 API 时不走系统代理
        self.session.headers.update(self.headers)

    def _refresh_metadata(self):
        """刷新获取 Clash 中的策略组和节点列表"""
        try:
            url = f"{self.base_url}/proxies"
            resp = self.session.get(url, timeout=2)
            if resp.status_code != 200:
                logger.warning(f"❌ Clash API 连接失败: {resp.status_code}")
                return

            proxies = resp.json().get('proxies', {})
            best_group = None
            max_nodes = 0

            # 自动寻找包含节点最多的策略组（通常就是我们需要的选择节点的组）
            for name, info in proxies.items():
                if info['type'] == 'Selector':
                    nodes = info.get('all', [])
                    # 排除掉特殊的内置节点
                    real_nodes = [n for n in nodes if
                                  n not in ['DIRECT', 'REJECT', 'PASS', '自动选择', '故障转移', 'Compatible']]
                    if len(real_nodes) > max_nodes:
                        max_nodes = len(real_nodes)
                        best_group = name
                        self.node_list = real_nodes

            if best_group:
                self.selector_name = best_group
                # logger.info(f"✅ 锁定 Clash 策略组: 【{best_group}】 (节点数: {max_nodes})")
            else:
                self.node_list = []
        except Exception as e:
            logger.warning(f"❌ 获取 Clash 元数据失败: {e}")

    def switch_random(self) -> bool:
        """随机切换到一个新节点"""
        # 如果还没初始化或者没节点，先刷新
        if not self.selector_name or not self.node_list:
            self._refresh_metadata()
            if not self.node_list:
                logger.error("❌ 未找到可用的 Clash 代理节点列表")
                return False

        # 随机选一个节点
        target_node = random.choice(self.node_list)

        # 尝试通过已知的策略组名称去设置
        attempt_selectors = [self.selector_name] + [s for s in self.fallback_selectors if s != self.selector_name]

        success = False
        for selector in attempt_selectors:
            if not selector: continue
            try:
                safe_group = quote(selector)
                url = f"{self.base_url}/proxies/{safe_group}"
                payload = {"name": target_node}
                resp = self.session.put(url, json=payload, timeout=3)
                if resp.status_code == 204:
                    logger.info(f"🔄 VPN 已切换至节点: 【{target_node}】 (策略组: {selector})")
                    success = True
                    break  # 成功即退出
            except Exception:
                continue

        return success

    def __call__(self):
        """
        关键修复：实现 __call__ 方法，使实例可以像函数一样被调用。
        例如: vpn_rotator() 实际上会执行 vpn_rotator.switch_random()
        """
        return self.switch_random()


# 实例化并导出
# 这样外部 import vpn_rotator 后，既可以直接 vpn_rotator() 调用，也可以访问其属性
vpn_rotator = ClashRotator()