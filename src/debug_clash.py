import requests
import json
import os
API_URL = "http://127.0.0.1:49812"
PROXY_URL = "http://127.0.0.1:7890"
# 👇 填入您的密钥
SECRET = "b342ba26-2ae3-47bb-a057-6624e171d5c6"


def check_clash_status():
    # ...
    headers = {"Authorization": f"Bearer {SECRET}"}  # <--- 加上 Header

    print(f"[1/3] 尝试连接 Clash 控制接口...")
    try:
        # 请求时带上 headers
        resp = requests.get(f"{API_URL}/proxies", headers=headers, timeout=3)

        if resp.status_code == 200:
            print(f"✅ API 连接成功！")
            data = resp.json()
            proxies = data.get('proxies', {})

            # 寻找包含节点的策略组
            # ClashX 通常主策略组叫 'Proxy', 'GLOBAL', 或者中文 '节点选择'
            print(f"    检测到 {len(proxies)} 个代理组/节点。")

            found_selector = None
            # 遍历寻找哪个组里有节点列表
            for name, info in proxies.items():
                if info['type'] == 'Selector':
                    print(f"    发现策略组: 【{name}】 - 当前选中: {info.get('now')}")
                    # 优先找包含 'all' 列表且比较大的组
                    if len(info.get('all', [])) > 2 and not found_selector:
                        found_selector = name

            if found_selector:
                print(f"✅ 锁定主策略组名称为: 【{found_selector}】")
                print(f"    (请确保 vpn_rotator.py 里使用这个名字)")
            else:
                print(f"❌ 警告: 未找到明显的选择器组。请检查您的 Clash 订阅配置。")

        elif resp.status_code == 401:
            print(f"❌ 失败: 401 Unauthorized。您设置了 Secret 密码，请在代码里填入。")
        else:
            print(f"❌ 失败: 状态码 {resp.status_code}。")

    except Exception as e:
        print(f"❌ 严重错误: 无法连接到 API。请确认 ClashX 正在运行且端口确实是 49812。")
        print(f"   错误详情: {e}")
        return

    # 2. 测试代理连通性
    print(f"\n[2/3] 尝试通过代理访问网络 (百度)...")
    try:
        proxies = {"http": PROXY_URL, "https": PROXY_URL}
        resp = requests.get("https://www.baidu.com", proxies=proxies, timeout=5)
        if resp.status_code == 200:
            print(f"✅ 代理通道畅通 (Port 7890)。")
        else:
            print(f"❌ 代理连接异常，状态码: {resp.status_code}")
    except Exception as e:
        print(f"❌ 代理连接失败: {e}")


if __name__ == "__main__":
    check_clash_status()