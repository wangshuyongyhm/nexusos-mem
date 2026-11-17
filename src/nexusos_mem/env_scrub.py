"""env_scrub.py — subprocess 环境净化（Task596 池#50 建，Task600 迁正源于 nexus/）。

正源在 nexus/ 包（更底层，nexusos_xdr 依赖 nexus 而非反向）；nexusos_xdr/env_scrub.py
是 re-export shim（Task596 的 5 处调用点零改动）。

问题：subprocess spawn 零 scrub 全继承父 env——子进程（shell 工具/脚本钩子）
可见父进程的 API_KEY/TOKEN 等敏感变量。

用法（spawn 点机械替换）：
    from nexus.env_scrub import scrubbed_env
    subprocess.run([...], env=scrubbed_env())

语义：
  - 默认黑名单式：copy 整个 env，丢弃名字含 KEY/SECRET/TOKEN/PASSWORD/CREDENTIAL
    （大小写不敏感）的项——PATH/HOME/LANG 等天然保留（缺了会废 shell）。
  - whitelist_only=True 白名单式（更严）：只保留进程存活必需的基底变量。
  - extra_keep：显式补充保留（调用方确知某变量对子进程必需且安全）。

不适用：os.environ 直接读取的配置路径（envref/NEXUS_* 开关等）不是 spawn，不接；
tmux send-keys 文本同理。
"""
import os
import re
from typing import Iterable

# 敏感词命中即丢（子进程无理由见到它们；KEYBOARD 之类误伤无害）
_SENSITIVE_RE = re.compile(r"KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL", re.IGNORECASE)

# 白名单式的基底（PATH 缺失废 shell；HOME 缺失废 expanduser；LANG/TZ 缺失乱码乱时区）
_BASELINE = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TZ", "USER", "SHELL")


def scrubbed_env(extra_keep: Iterable[str] = (), whitelist_only: bool = False) -> dict:
    """返回净化后的 env 副本（不动 os.environ 本身）。"""
    if whitelist_only:
        env = {k: os.environ[k] for k in _BASELINE if k in os.environ}
    else:
        env = {k: v for k, v in os.environ.items()
               if not _SENSITIVE_RE.search(k)}
    for k in extra_keep:
        if k in os.environ:
            env[k] = os.environ[k]
    return env


def is_scrubbed_key(key: str) -> bool:
    """测试/审计用：判定某键是否会被净化丢弃。"""
    return bool(_SENSITIVE_RE.search(key))
