import requests
import random
import json
from urllib.parse import quote
from .config import Config


class ClashRotator:
    """
    【Clash 代理控制器】
    """

    def __init__(self, controller_url=Config.CLASH_API_URL,
                 secret=Config.CLASH_SECRET):
        """
        初始化：参数默认值从 Config 读取
        """
        self.base_url = controller_url
        self.headers = {
            "Authorization": f"Bearer {secret}",
            "Content-Type": "application/json"
        }
        self.selector_name = None
        self.node_list = []
        self.fallback_selectors = ['GLOBAL', 'Proxy', '节点选择', '国外流量', 'Global']

        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update(self.headers)

    def _refresh_metadata(self):
        try:
            url = f"{self.base_url}/proxies"
            resp = self.session.get(url, timeout=2)

            if resp.status_code != 200:
                print(f"❌ Clash API 连接失败: {resp.status_code}")
                return

            proxies = resp.json().get('proxies', {})
            best_group = None
            max_nodes = 0

            for name, info in proxies.items():
                if info['type'] == 'Selector':
                    nodes = info.get('all', [])
                    real_nodes = [n for n in nodes if
                                  n not in ['DIRECT', 'REJECT', 'PASS', '自动选择', '故障转移', 'Compatible']]

                    if len(real_nodes) > max_nodes:
                        max_nodes = len(real_nodes)
                        best_group = name
                        self.node_list = real_nodes

            if best_group:
                self.selector_name = best_group
                print(f"✅ 锁定 Clash 策略组: 【{best_group}】，可用节点: {len(self.node_list)} 个")
            else:
                print("❌ 未找到有效策略组")
                self.node_list = []

        except Exception as e:
            print(f"❌ Clash API 异常: {e}")

    def switch_random(self):
        if not self.selector_name or not self.node_list:
            self._refresh_metadata()
            if not self.node_list: return False

        target_node = random.choice(self.node_list)
        attempt_selectors = [self.selector_name] + [s for s in self.fallback_selectors if s != self.selector_name]

        for selector in attempt_selectors:
            if not selector: continue
            try:
                safe_group = quote(selector)
                url = f"{self.base_url}/proxies/{safe_group}"
                payload = {"name": target_node}
                resp = self.session.put(url, json=payload, timeout=3)

                if resp.status_code == 204:
                    print(f"🔄 [VPN] 已切换至: {target_node}")
                    self.selector_name = selector
                    return True
            except:
                continue
        return False


vpn_rotator = ClashRotator()