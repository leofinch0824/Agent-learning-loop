#!/usr/bin/env python3
"""监视仓库源文件，变更后自动运行 sync-site-src.py（配合 zensical serve 热重载）。

zensical serve 的 watcher 只监视 site-src/ 内真实文件的内容变化，而课程
源文件（lessons/、docs/、lib/ 等）在监视树之外，直接编辑不会触发重建。
本脚本补上这段：轮询源文件的 mtime/size，发现变化就调用
scripts/sync-site-src.py 把新内容发布进 site-src/，serve 随即自动重建并
刷新浏览器。零第三方依赖（纯 os.stat 轮询，默认 1s）。

用法（开两个终端，或都放后台）：

    poetry run zensical serve          # 终端 1：站点 + 热重载
    python scripts/watch-sync.py       # 终端 2：源文件 → site-src 自动发布

停止：Ctrl-C（或 pkill -f watch-sync）。
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNC_SCRIPT = REPO_ROOT / "scripts" / "sync-site-src.py"
INTERVAL_SECONDS = 1.0

# 与 sync-site-src.py 的 TREES/ROOT_FILES 保持一致
WATCH_DIRS = ["lessons", "docs", "lib"]
WATCH_ROOT_FILES = ["README.md", "CONTEXT.md", "CLAUDE.md"]
WATCH_EXTS = {".md", ".py"}
SKIP_PARTS = {"__pycache__"}


def snapshot() -> dict[Path, tuple[int, int]]:
    """收集所有受监视文件的 (mtime_ns, size)。"""
    state: dict[Path, tuple[int, int]] = {}
    for name in WATCH_DIRS:
        for p in (REPO_ROOT / name).rglob("*"):
            if p.is_file() and p.suffix in WATCH_EXTS and not SKIP_PARTS & set(p.parts):
                st = p.stat()
                state[p] = (st.st_mtime_ns, st.st_size)
    for name in WATCH_ROOT_FILES:
        p = REPO_ROOT / name
        if p.is_file():
            st = p.stat()
            state[p] = (st.st_mtime_ns, st.st_size)
    return state


def run_sync() -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] 源文件有变更，执行 sync …", flush=True)
    proc = subprocess.run(
        [sys.executable, str(SYNC_SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout or proc.stderr, flush=True)


def main() -> int:
    print(f"监视中：{', '.join(WATCH_DIRS)} 与 {', '.join(WATCH_ROOT_FILES)}"
          f"（每 {INTERVAL_SECONDS:.0f}s 轮询，Ctrl-C 退出）", flush=True)
    state = snapshot()
    try:
        while True:
            time.sleep(INTERVAL_SECONDS)
            current = snapshot()
            if current != state:
                # 等写入稳定后再同步，避免编辑器分多次写文件触发半成品发布
                time.sleep(0.4)
                current = snapshot()
                run_sync()
                state = snapshot()
    except KeyboardInterrupt:
        print("已停止监视", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
