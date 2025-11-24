"""
Convenience launcher for the lightweight pytest suite with enhanced logging.
"""
import subprocess
import sys
import os
from datetime import datetime


def main():
    # 1. 准备日志目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # 2. 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"test_run_{timestamp}.log")

    print(f"🚀 启动测试套件...")
    print(f"📂 日志文件将保存至: {log_file}")
    print("-" * 60)

    # 3. 构建 Pytest 命令
    # -v: 详细模式 (Verbose)，显示每个测试函数的名称和结果
    # -s:不仅捕获输出 (Show output)，允许 print() 语句直接输出到控制台
    # --log-cli-level=INFO: 在控制台显示 INFO 及以上级别的日志 (如果代码使用了 logging 模块)
    # --log-file=...: 将日志输出到指定文件
    # --log-file-level=DEBUG: 在文件中记录所有 DEBUG 级别的详细信息
    # -q 极简模式
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",
        "-s",
        f"--log-file={log_file}",
        "--log-file-level=DEBUG"
    ]

    # 4. 执行命令
    try:
        # 使用 subprocess.call 执行，保持当前进程等待测试结束
        result = subprocess.call(cmd)
    except KeyboardInterrupt:
        print("\n⚠️ 测试被用户中断")
        result = 1

    print("-" * 60)
    if result == 0:
        print(f"✅ 所有测试通过！完整日志已保存: {os.path.basename(log_file)}")
    else:
        print(f"❌ 测试失败 (Exit Code: {result})。请查看日志文件排查问题: {os.path.basename(log_file)}")

    raise SystemExit(result)


if __name__ == "__main__":
    main()