import requests
import random
import os
import time
from concurrent.futures import ThreadPoolExecutor


class ProxyService:
    def __init__(self):
        # 这里汇集了几个质量相对较好的 GitHub 免费源
        # 重点：我们必须混合 HTTP 和 HTTPS 源，然后自己去验证是否支持 HTTPS
        self.sources = [
            "https://raw.githubusercontent.com/jetkai/proxy-list/main/online-proxies/txt/proxies-https.txt",
            "https://raw.githubusercontent.com/hookzof/socks5_list/master/proxy.txt",
            "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt"
        ]
        self.proxy_pool = []
        self.current_proxy = None
        self.black_list = set()

        # 初始化时先不抓，等第一次调用时抓，或者手动触发
        self.is_initialized = False

    def fetch_proxies(self):
        """从网络源拉取代理"""
        print("正在初始化代理池 (可能需要 1-2 分钟验证有效性)...")
        raw_proxies = set()

        for url in self.sources:
            try:
                print(f"正在拉取源: {url} ...")
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    for line in resp.text.splitlines():
                        line = line.strip()
                        if ":" in line and line not in self.black_list:
                            raw_proxies.add(line)
            except:
                continue

        print(f"共获取 {len(raw_proxies)} 个原始 IP，开始并发验证 HTTPS 支持情况...")

        # 验证代理 (只保留能连通 HTTPS 的)
        # 免费代理 90% 都是垃圾，必须清洗
        self.proxy_pool = self._batch_validate(list(raw_proxies))
        print(f"代理池清洗完成！有效 HTTPS 代理数: {len(self.proxy_pool)}")
        self.is_initialized = True

    def _validate_single(self, proxy):
        """验证单个代理是否支持 HTTPS"""
        proxies = {
            "http": f"http://{proxy}",
            "https": f"http://{proxy}"  # 注意：requests 配置 key 为 https，但 value 协议头通常仍写 http
        }
        try:
            # 访问百度或东财验证，超时设置极短 (3秒)，只留快马
            resp = requests.get("https://www.eastmoney.com", proxies=proxies, timeout=3)
            if resp.status_code == 200:
                return proxy
        except:
            return None
        return None

    def _batch_validate(self, proxy_list):
        """多线程并发验证，提高效率"""
        valid_proxies = []
        # 仅验证前 500 个，避免太久
        candidates = list(proxy_list)[:500]

        with ThreadPoolExecutor(max_workers=50) as executor:
            results = executor.map(self._validate_single, candidates)

        for p in results:
            if p: valid_proxies.append(p)

        return valid_proxies

    def rotate_proxy(self):
        """切换下一个代理，并应用到环境变量"""
        if not self.is_initialized or len(self.proxy_pool) < 2:
            self.fetch_proxies()

        if not self.proxy_pool:
            print("警告：无可用代理，将尝试直连...")
            self._clear_env()
            return False

        # 随机选一个
        self.current_proxy = random.choice(self.proxy_pool)

        # 设置环境变量 (AkShare/Requests 会自动读取)
        # 格式核心：http://IP:PORT
        p_str = f"http://{self.current_proxy}"
        os.environ['http_proxy'] = p_str
        os.environ['https_proxy'] = p_str
        # print(f"已切换代理: {self.current_proxy}")
        return True

    def remove_current_proxy(self):
        """当前代理失效，剔除"""
        if self.current_proxy in self.proxy_pool:
            self.proxy_pool.remove(self.current_proxy)
            self.black_list.add(self.current_proxy)
        # 立即切换
        self.rotate_proxy()

    def _clear_env(self):
        for k in ['http_proxy', 'https_proxy', 'all_proxy']:
            if k in os.environ: del os.environ[k]


# 单例实例
proxy_service = ProxyService()