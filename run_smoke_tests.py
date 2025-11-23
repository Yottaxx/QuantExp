"""
Convenience launcher for the lightweight pytest suite.
"""
import subprocess
import sys


def main():
    cmd = [sys.executable, "-m", "pytest", "-q"]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
