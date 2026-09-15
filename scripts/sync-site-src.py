#!/usr/bin/env python3
"""把仓库内的课程文档"发布"到 site-src/（zensical 站点的文档树）。

背景：zensical 0.0.62 的 exclude_docs 不拦截静态文件拷贝，直接用
docs_dir="." 会把 .venv、mlflow-data（23MB）、lock 文件全拖进 site/；
而 serve 的 watcher 只监视 site-src/ 内真实文件的内容变化——仓库源
文件在监视树之外，改了也不会触发重建（symlink 指向的目标变更同样
不可见，touch 改 mtime 也不算）。因此本脚本采用"发布"模型：

- .md  → 拷贝为 site-src 下的真实文件（内容有变化才重写）
- .py  → 生成同名 .py.md 包裹页，内嵌源码 sha256 短哈希；
         源码变化 → 哈希变化 → 包裹页重写 → serve 自动重建刷新
- 目录级 symlink zensical 不跟随；.py 本体不能作为静态文件链接
  （与包裹页路由 main.py/ 冲突，build 报 File exists）

用法：改完任意源文件后执行一次，serve 中的站点即自动刷新：

    python scripts/sync-site-src.py

幂等：重复执行安全；源树中已删除的旧文件会被清理。
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 发布的仓库内文档树：(源目录, 站内目录, 允许的扩展名)
TREES: list[tuple[str, str, set[str]]] = [
    ("lessons", "site-src/lessons", {".md", ".py"}),
    ("docs", "site-src/docs", {".md"}),
    ("lib", "site-src/lib", {".py"}),
]

# 站点根部的单文件：(源文件, 站内路径)
ROOT_FILES: list[tuple[str, str]] = [
    ("README.md", "site-src/index.md"),
    ("CONTEXT.md", "site-src/CONTEXT.md"),
    ("CLAUDE.md", "site-src/CLAUDE.md"),
]

SKIP_PARTS = {"__pycache__"}
# 内容为空、只有课程包结构的文件，不值得生成代码页
SKIP_FILES = {"__init__.py"}

WRAPPER_TEMPLATE = """# {name}

> 源文件：`{src}`（本页由 scripts/sync-site-src.py 生成）

<!-- src sha256: {digest} -->

```python
--8<-- "{src}"
```
"""


def publish_file(src: Path, dest: Path) -> bool:
    """拷贝源文件内容到 site-src（内容有变化才重写），返回是否有变更。"""
    content = src.read_text(encoding="utf-8")
    if dest.exists() and not dest.is_symlink() and dest.read_text(encoding="utf-8") == content:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    dest.write_text(content, encoding="utf-8")
    return True


def write_wrapper(src_root: Path, rel: Path, link_path: Path) -> bool:
    """为一个 .py 生成 .md 包裹页（经 snippets 渲染成高亮代码）。

    哈希随源码变化，serve 的内容 watcher 因此能感知到源码更新。
    """
    src = src_root / rel
    digest = hashlib.sha256(src.read_bytes()).hexdigest()[:12]
    content = WRAPPER_TEMPLATE.format(name=link_path.name, src=src.as_posix(), digest=digest)
    wrapper = link_path.with_suffix(".py.md")
    if wrapper.exists() and wrapper.read_text(encoding="utf-8") == content:
        return False
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(content, encoding="utf-8")
    return True


def main() -> int:
    changed = 0

    for src_name, dest_name, exts in TREES:
        src_root = REPO_ROOT / src_name
        dest_root = REPO_ROOT / dest_name
        wanted: set[Path] = set()
        wanted_wrappers: set[Path] = set()
        for p in src_root.rglob("*"):
            if p.is_dir() or SKIP_PARTS & set(p.parts) or p.suffix not in exts:
                continue
            rel = p.relative_to(src_root)
            dest_path = dest_root / rel
            if p.suffix == ".py":
                # .py 只生成代码页，不作为静态文件发布（避免与包裹页路由冲突）
                if p.name not in SKIP_FILES:
                    wanted_wrappers.add(dest_path.with_suffix(".py.md"))
                    if write_wrapper(src_root, rel, dest_path):
                        print(f"wrapped {dest_path.with_suffix('.py.md').relative_to(REPO_ROOT)}")
                        changed += 1
                continue
            wanted.add(dest_path)
            if publish_file(src_root / rel, dest_path):
                print(f"published {dest_path.relative_to(REPO_ROOT)}")
                changed += 1
        # 清理源树中已不存在的旧发布文件
        for old in dest_root.rglob("*"):
            if old.is_dir():
                continue
            is_wrapper = old.name.endswith(".py.md")
            if (is_wrapper and old not in wanted_wrappers) or (
                not is_wrapper and old not in wanted
            ):
                old.unlink()
                print(f"removed stale {old.relative_to(REPO_ROOT)}")
                changed += 1

    for src_name, dest_name in ROOT_FILES:
        if publish_file(REPO_ROOT / src_name, REPO_ROOT / dest_name):
            print(f"published {dest_name}")
            changed += 1

    print("site-src 已同步" if changed else "site-src 已是最新（无变更）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
