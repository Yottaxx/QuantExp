import requests
import random
import json
from urllib.parse import quote


class ClashRotator:
    def __init__(self, controller_url="http://127.0.0.1:49812",
                 secret="b342ba26-2ae3-47bb-a057-6624e171d5c6"):
        """
        初始化 Clash 控制器
        """
        self.base_url = controller_url
        self.headers = {
            "Authorization": f"Bearer {secret}",
            "Content-Type": "application/json"
        }
        # 自动探测到的主策略组名称
        self.selector_name = None
        # 可用的节点列表
        self.node_list = []

        # 初始化一个专用 Session
        # 关键点：trust_env=False 让它无视 data_provider 设置的 http_proxy 环境变量
        # 从而确保它直连 49812 端口，不会走代理
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.headers.update(self.headers)

    def _refresh_metadata(self):
        """
        智能探测：复用您 debug 代码中的成功逻辑
        找到包含节点最多的那个策略组
        """
        # print(f"[Clash] 正在连接 API: {self.base_url} ...")
        try:
            url = f"{self.base_url}/proxies"

            # 使用隔离的 session 发送请求
            resp = self.session.get(url, timeout=3)

            if resp.status_code == 401:
                print(f"❌ Clash API 鉴权失败 (401)。请检查 Secret 密码。")
                return

            if resp.status_code != 200:
                print(f"❌ Clash API 连接失败: Status {resp.status_code} | {resp.text}")
                return

            data = resp.json()
            proxies = data.get('proxies', {})

            # 寻找最佳策略组
            best_group = None
            max_nodes = 0

            # 遍历寻找哪个组里有节点列表 (逻辑完全复用您的 check 代码)
            for name, info in proxies.items():
                if info['type'] == 'Selector':
                    nodes = info.get('all', [])
                    # 过滤掉非 VPN 节点
                    real_nodes = [n for n in nodes if
                                  n not in ['DIRECT', 'REJECT', 'PASS', '自动选择', '故障转移', 'Compatible']]

                    # 优先找包含节点多且不是 GLOBAL(如果GLOBAL没节点) 的组
                    # 通常我们选节点最多的那个组
                    if len(real_nodes) > max_nodes:
                        max_nodes = len(real_nodes)
                        best_group = name
                        self.node_list = real_nodes

            if best_group:
                self.selector_name = best_group
                print(f"✅ 锁定 Clash 策略组: 【{best_group}】，可用节点: {len(self.node_list)} 个")
            else:
                print("❌ 未找到有效的 VPN 策略组，请检查订阅。")
                self.node_list = []

        except Exception as e:
            print(f"❌ Clash API 异常 (Port 49812): {e}")

    def switch_random(self):
        """执行切换"""
        # 懒加载：第一次调用切换时才去获取列表
        if not self.selector_name or not self.node_list:
            self._refresh_metadata()
            if not self.node_list:
                return False

        target_node = random.choice(self.node_list)

        try:
            # URL 编码 (处理中文策略组名)
            safe_group = quote(self.selector_name)
            url = f"{self.base_url}/proxies/{safe_group}"
            payload = {"name": target_node}

            # 发送切换指令
            resp = self.session.put(url, json=payload, timeout=3)

            if resp.status_code == 204:
                print(f"🔄 [VPN] 已切换至: {target_node}")
                return True
            else:
                # 如果切换失败（比如策略组改名了），强制刷新一次元数据重试
                print(f"⚠️ 切换失败 ({resp.status_code})，尝试刷新列表...")
                self._refresh_metadata()
                return False

        except Exception as e:
            print(f"⚠️ 切换异常: {e}")
            return False


# 单例模式
vpn_rotator = ClashRotator()