#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nexus_mem_utils.py — NEXUS-MEM 共享工具模块

提供：
  - atomic_write_json: 原子写入 JSON（temp + os.replace）
  - safe_read_json:    安全读取 JSON（失败时记录日志而非静默吞咽）
  - run_micro_check:   共享微检查（避免 init/persist/archive 三处重复）
  - compute_entropy_score: 熵预算核心公式（避免 shadow-sync/entropy-budget 双份）

设计原则：
  - 最小化依赖（仅标准库）
  - 不修改全局状态
  - 所有函数可独立测试

引入背景（task 224）：
  - P1 #2: 1.8MB knowledge-evolution.json 全量加载 → 改读 nexus-stats.json (~100B)
  - P1 #3: 全体系无原子写 → atomic_write_json
  - P1 #4: stale JSON 损毁静默吞咽 → safe_read_json 自动 log
  - 审计 P1 #1: run_micro_check 重复 49 行 ×3 → 共享函数
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple


# ---------------------------------------------------------------------------
# Task 298 P0-4: DB 连接统一（DSN helper）
# ---------------------------------------------------------------------------
def get_dsn(*, explicit_user: Optional[str] = None) -> str:
    """返回统一的 PostgreSQL DSN 字符串。

    Task 298 P0-4：体系文档/任务书到处写 `nexus_mem` user，但实际 PG 只有
    若干业务 user（nexusos_mem role 不存在）。代码能跑是因为
    psycopg2 不带 user 时走 default user（PG env / 当前系统用户）。

    本 helper 把"依赖 default user"显式化——所有 psycopg2.connect 都调它，
    统一 DSN 格式，避免未来某天 default user 变了全线崩。

    Args:
        explicit_user: 显式覆盖（测试用）；默认从 NEXUS_PG_USER 环境变量读，
                       再 fallback 到 "wang"（当前实际能连的 user）

    Returns:
        libpq 连接串，如 "dbname=nexus_mem user=wang host=localhost"

    用法：
        conn = psycopg2.connect(get_dsn())
    """
    user = explicit_user or os.environ.get("NEXUS_PG_USER", "wang")
    dbname = os.environ.get("NEXUS_PG_DB", "nexus_mem")
    host = os.environ.get("NEXUS_PG_HOST", "localhost")
    # 1822: connect_timeout=5——半开网络下 psycopg2.connect 可无限阻塞（TCP 重传窗）。
    # 本 helper 是直连单一真源，注入一次全部调用点受益。
    return f"dbname={dbname} user={user} host={host} connect_timeout=5"



# ---------------------------------------------------------------------------
# 原子写
# ---------------------------------------------------------------------------

def atomic_write_json(
    path: Path,
    data: Any,
    indent: int = 2,
    ensure_ascii: bool = False,
    parent_mkdir: bool = True,
) -> None:
    """原子写入 JSON 文件。

    实现：写入 `{path}.tmp.{pid}` → `os.replace(tmp, path)`。
    POSIX 保证同文件系统 rename 是原子操作，即使进程被 SIGKILL 也不会留下撕裂文件。

    Args:
        path: 目标文件路径
        data: 可 JSON 序列化的数据
        indent: JSON 缩进（默认 2）
        ensure_ascii: 是否转义非 ASCII（默认 False，保留中文可读）
        parent_mkdir: 是否自动创建父目录（默认 True）

    Raises:
        原样抛出底层异常（IOError/TypeError 等），调用方自行处理。
    """
    path = Path(path)
    if parent_mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    # 用 pid 后缀避免多进程并发写时 tmp 文件冲突
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        tmp.write_text(
            json.dumps(data, ensure_ascii=ensure_ascii, indent=indent),
            encoding="utf-8",
        )
        os.replace(tmp, path)  # 原子替换（同文件系统）
    except Exception:
        # 失败时清理 tmp 文件，避免堆积
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        raise


# ---------------------------------------------------------------------------
# 安全读
# ---------------------------------------------------------------------------

def safe_read_json(
    path: Path,
    default: Any = None,
    log_warnings: bool = True,
    warn_prefix: str = "[WARN]",
) -> Any:
    """安全读取 JSON 文件。

    与裸 json.loads 的区别：
      1. 文件不存在时返回 default（不抛 FileNotFoundError）
      2. JSON 解析失败时返回 default **并打印警告到 stderr**（不静默吞咽）
      3. 默认值支持任意类型（dict/list/None/0 等）

    Args:
        path: 文件路径
        default: 解析失败时的返回值（默认 None）
        log_warnings: 是否打印警告（默认 True，测试时可关闭）
        warn_prefix: 警告前缀（默认 "[WARN]"）

    Returns:
        解析后的数据，或 default。
    """
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        if log_warnings:
            print(
                f"{warn_prefix} {path} 解析失败，使用默认值 {default!r}: {type(e).__name__}: {e}",
                file=sys.stderr,
            )
        return default


# ---------------------------------------------------------------------------
# 共享微检查
# ---------------------------------------------------------------------------

def run_micro_check(report: Any, script_dir: Path) -> None:
    """共享微检查：读取状态文件，发现异常时追加 WARN 到报告。

    替代 init.py / persist.py / archive.py 中三处几乎相同的实现（共 ~150 行重复）。

    优化点（vs 旧实现）：
      1. 不再加载 1.8MB 的 knowledge-evolution.json —— 改读 nexus-stats.json（~100B）
         nexus-stats.json 由 knowledge-evolution.py 在扫描后写入。
         如果 nexus-stats.json 不存在（旧版兼容），fallback 到 knowledge-evolution.json
         并打印 deprecation 警告。
      2. 使用 safe_read_json —— 文件损毁时记录日志而非静默吞咽。
      3. 阈值集中定义，避免三处不同步。

    Args:
        report: 报告对象，必须有 .add(name, status, detail) 方法
                (InitReport / PersistReport / 任何带 add 方法的对象)
        script_dir: 05-System/自动化脚本/ 目录路径
    """
    # 阈值集中定义（原 init/persist/archive 各自硬编码，易漂移）
    STALE_WARN = 50
    CONFLICTS_WARN = 500
    TOKEN_PCT_WARN = 80
    TOKEN_PCT_CRIT = 90

    issues = []

    # 1. session-health 状态（tiny 文件 ~8KB，安全加载）
    health = safe_read_json(script_dir / "session-health.json", default={})
    if health:
        # session-health.json 实际无 token_usage_pct 字段（需计算），用 baseline+session 估算
        baseline = health.get("baseline_shadow_tokens", 867)
        rounds = health.get("round_count", 0)
        tools = health.get("tool_calls_total", 0)
        est_total = baseline + rounds * 200 + tools * 100
        token_pct = int(est_total / 4000 * 100) if est_total else 0
        if token_pct >= TOKEN_PCT_CRIT:
            issues.append(
                f"[健康] 🔴 会话上下文占用率 ~{token_pct}% >= {TOKEN_PCT_CRIT}%，建议立即压缩"
            )
        elif token_pct >= TOKEN_PCT_WARN:
            issues.append(
                f"[健康] 🟡 会话上下文占用率 ~{token_pct}% >= {TOKEN_PCT_WARN}%，建议归档"
            )

    # 2. nexus-stats.json（tiny 状态文件，~100B，由 knowledge-evolution.py 写入）
    stats_path = script_dir / "nexus-stats.json"
    stats = safe_read_json(stats_path, default=None)
    if stats is None:
        # Fallback: 加载完整 knowledge-evolution.json（1.8MB，慢但兼容旧版）
        # 这种情况只在 knowledge-evolution.py 未升级（未写 nexus-stats.json）时发生
        ke_path = script_dir / "knowledge-evolution.json"
        ke = safe_read_json(ke_path, default={})
        if ke:
            stale = ke.get("stale_count", 0)
            # 兼容：旧版字段 conflict_count 不存在时用 len(conflicts)
            conflicts = ke.get("conflict_count")
            if conflicts is None:
                conflicts = len(ke.get("conflicts", []))
        else:
            stale = conflicts = 0
    else:
        stale = stats.get("stale_count", 0)
        conflicts = stats.get("conflict_count", 0)

    if stale >= STALE_WARN:
        issues.append(
            f"[健康] 🟡 过期知识 {stale} >= {STALE_WARN}，建议运行 knowledge-evolution.py --scan"
        )
    if conflicts >= CONFLICTS_WARN:
        issues.append(
            f"[健康] 🟡 知识冲突 {conflicts} >= {CONFLICTS_WARN}，建议人工审查"
        )

    if issues:
        print("\n[微检查] 发现异常：")
        for i in issues:
            print(f"  {i}")
        report.add("微检查", "WARN", "\n".join(issues))


def write_nexus_stats(script_dir: Path, stale_count: int, conflict_count: int,
                       total_notes: int = 0, **extra: Any) -> None:
    """写入 nexus-stats.json（tiny 状态文件）。

    由 knowledge-evolution.py 在 --scan 完成后调用，避免下游脚本加载 1.8MB 主文件。

    Args:
        script_dir: 05-System/自动化脚本/ 目录
        stale_count: 过期知识数
        conflict_count: 冲突数
        total_notes: 笔记总数
        **extra: 其他需要快速访问的指标
    """
    stats = {
        "stale_count": stale_count,
        "conflict_count": conflict_count,
        "total_notes": total_notes,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    atomic_write_json(script_dir / "nexus-stats.json", stats)


# ---------------------------------------------------------------------------
# 熵预算核心公式（P2 dedup）
# ---------------------------------------------------------------------------

# 默认参数（与 shadow-sync.py / entropy-budget.py 保持一致）
DEFAULT_T_TOTAL = 1000.0
DEFAULT_LAMBDA = 0.01


def compute_entropy_score(
    R: float,
    W: float,
    delta_hours: float,
    lam: float = DEFAULT_LAMBDA,
) -> float:
    """熵预算核心公式：score = W × R^1.5 × exp(-λ × Δt)。

    被 shadow-sync.compute_entropy_budget 和 entropy-budget.compute_entropy_budget
    共同调用，避免公式系数（1.5 / λ）在两处不同步。

    Args:
        R: 引用计数（自动 clamp 到 ≥0）
        W: 权重
        delta_hours: 距上次使用的小时数（自动 clamp 到 ≥0）
        lam: 衰减系数（默认 0.01）

    Returns:
        原始分数（未归一化）
    """
    R = max(R, 0)
    delta_hours = max(delta_hours, 0)
    if R == 0:
        return 0.0  # 0^1.5 = 0，短路避免数值抖动
    return W * (R ** 1.5) * math.exp(-lam * delta_hours)


def allocate_entropy_budget(
    scores: dict,
    t_total: float = DEFAULT_T_TOTAL,
) -> Tuple[dict, float]:
    """将原始 scores 归一化为预算分配。

    Args:
        scores: {code: raw_score}
        t_total: 总预算（默认 1000）

    Returns:
        (budget_dict, denom): budget = {code: B_i}，denom = Σ scores
        全零时均分。
    """
    denom = sum(scores.values())
    if denom == 0:
        n = len(scores)
        if n == 0:
            return {}, 0.0
        return {c: t_total / n for c in scores}, t_total
    return {c: t_total * (s / denom) for c, s in scores.items()}, denom
