#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_honesty.py — 认知诚实体系 Layer 0：套用诚实标注 + trace 文件生成

设计依据（Task 258 任务书 §Layer 0 + Experience-Context-Coord-Template.md §2）：
  触发：LLM 每次调 search 命中经验，准备套用前
  机制：
    1. 读 knowledge.context_coord（源坐标）vs 当前坐标
    2. compute_sequence 算有序序列
    3. 写 trace md 文件（Obsidian 图谱节点，含 wikilink 占位）
    4. 生成标注字符串
    5. 写 agent_activity(action='apply_attempt', detail={seq, trace_path, uncertainty, priority_order})

公共 API：
  annotate_apply_attempt(kb, exp_id, current_coord, priority_order,
                         uncertainty=None, wikilink_hints=None) -> dict
      完成全流程，返回 {annotation, trace_path, trace_id, seq, uncertainty, explanations}

  write_trace_file(trace_id, exp_id, seq, uncertainty, explanations,
                   wikilink_hints, trace_dir, exp_title=None) -> Path

关键设计：
  - wikilink 占位由模块生成骨架，LLM 填具体依据文件（模块不知道 Vault 里有哪些文件）
  - uncertainty：LLM 显式传；不传则 suggest_uncertainty 启发式参考（LLM 最终自声明）
  - 存量经验无 context_coord → 序列对应位 '-'，不阻塞但记录到 explanations
"""

from __future__ import annotations
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple  # Task 281 P2-20: 加 Tuple（L161 用）

# Task631 迁包: context_capture 同在 nexusos_mem 包内, 直接包路径 import(原平铺 sys.path hack 删)
from nexusos_mem.context_capture import compute_sequence, suggest_uncertainty  # noqa: E402

# trace 文件默认目录（Obsidian Vault 内，NEXUS_VAULT_ROOT env 可覆盖）
# 1957: 锚点单源 nexusos-common；缺 vault=None（废 /nonexistent 占位），写入口显式报错
from nexusos_mem._vault_env import VAULT_ROOT as _VR_TRACE
from nexusos_common import ConfigError, VAULT_MISSING_REMEDIATION
DEFAULT_TRACE_DIR = (
    _VR_TRACE / "05-System" / "NEXUS-MEM" / "coord-traces" if _VR_TRACE is not None else None
)
MAX_TRACE_BYTES = 50_000  # trace 文件大小上限（Task 331，防 crucible 空转膨胀）


# ---------------------------------------------------------------------------
# §1 trace_id 生成（YYYYMMDD-NNN，同日自增，v2 支持分桶扫描）
# ---------------------------------------------------------------------------

def _next_trace_id(trace_root: Path) -> str:
    """扫描 trace_root 下所有 YYYY/MM/ 子目录，生成当日自增 trace_id。

    v2 P0-3：trace 不再平铺，改 YYYY/MM 分桶。本函数递归扫描所有子目录
    里当日创建的 trace 文件，保证 NNN 跨分桶不冲突。

    task 265 修复：加文件存在性校验。即使 rglob 扫描漏（目录竞态/清理），
    也保证返回的 trace_id 对应文件不存在——避免 write_trace_file 覆盖旧 trace。
    设计决策：每次套用是独立事件，新建 trace，不覆盖历史。

    Task 288 P1-1 修复：加进程级文件锁（fcntl.flock），防止多进程并发扫描时
    同时拿到相同 ID 后写文件覆盖。task 265 的循环只能防文件已存在，不能防
    两个进程同时扫描到 max_n=5 同时返回 n=6 然后竞争创建文件。
    """
    import fcntl
    today = datetime.now().strftime("%Y%m%d")
    pattern = re.compile(rf"trace-{today}-(\d{{3}})\.md$")
    # Task 288 P1-1: 进程级锁——lockfile 放 trace_root 根下
    trace_root.mkdir(parents=True, exist_ok=True)
    lockfile = trace_root / ".trace_id.lock"
    with open(lockfile, "w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)  # 阻塞获取独占锁
        try:
            max_n = 0
            if trace_root.exists():
                # rglob 递归所有 YYYY/MM 子目录
                for f in trace_root.rglob(f"trace-{today}-*.md"):
                    m = pattern.search(f.name)
                    if m:
                        max_n = max(max_n, int(m.group(1)))
            # task 265: 防覆盖——递增直到文件不存在（处理扫描漏/竞态）
            n = max_n + 1
            while n < 999:
                candidate = f"{today}-{n:03d}"
                candidate_path = _bucket_dir(trace_root, candidate) / f"trace-{candidate}.md"
                if not candidate_path.exists():
                    return candidate
                n += 1
            # 兜底（同日 999 次，极端情况）
            return f"{today}-999"
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def _bucket_dir(trace_root: Path, trace_id: str) -> Path:
    """根据 trace_id 解析分桶目录 YYYY/MM（v2 P0-3）。

    trace_id 格式 YYYYMMDD-NNN → 分桶 YYYY/MM
    """
    ymd = trace_id.split("-")[0]  # YYYYMMDD
    year = ymd[:4]
    month = ymd[4:6]
    return trace_root / year / month


# ---------------------------------------------------------------------------
# §1b wikilink 校验（v2 P0-3：不静默接受死链）
# ---------------------------------------------------------------------------

# Obsidian Vault 根（wikilink [[X]] 相对 Vault 解析）
# Task 282 P2-1: 改用 constants.VAULT_ROOT 单源（原硬编码本机绝对路径）
from nexusos_mem._vault_env import VAULT_ROOT  # Task668: 包内自足

# 已知无源文件的占位标识（模块自己生成的不算死链）
_PLACEHOLDER_PATTERNS = (
    "待 LLM 填",
    "counterexample-XXX",
    "经验-{",  # 模板占位
)

_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def _resolve_wikilink_target(target: str) -> Optional[Path]:
    """解析 wikilink 目标到 Vault 内文件路径（含 .md 隐含扩展）。

    Obsidian wikilink [[X]] 规则：
      - target 可能含 alias：[[X|显示名]] → 取 X
      - target 可能含路径：[[folder/X]] → 相对 Vault 根
      - target 无扩展 → 尝试 .md，再尝试同名目录
    """
    target = target.split("|")[0].strip()  # 去 alias
    if not target or target.startswith("broken:"):
        return None
    if VAULT_ROOT is None:
        raise ConfigError("vault 不可用，wikilink 校验无法进行。" + VAULT_MISSING_REMEDIATION)
    # 跳过明显占位
    if any(p in target for p in _PLACEHOLDER_PATTERNS):
        return None

    # 相对 Vault 根解析
    candidate = VAULT_ROOT / target
    if candidate.is_file():
        return candidate
    if (candidate.with_suffix(".md")).is_file():
        return candidate.with_suffix(".md")
    # 同名目录也算存在（Obsidian 文件夹节点）
    if candidate.is_dir():
        return candidate
    return None


def validate_wikilinks(text: str) -> List[Dict[str, str]]:
    """扫描文本里所有 [[...]]，校验目标是否存在。

    Returns:
        [{"raw": "[[xxx]]", "target": "xxx", "exists": bool}, ...]
        不修改文本，只报告。调用方决定如何处理（标 broken 或保留）
    """
    results = []
    for m in _WIKILINK_RE.finditer(text):
        raw = m.group(0)
        target = m.group(1).split("|")[0].strip()
        # 跳过占位和已标 broken 的
        if any(p in target for p in _PLACEHOLDER_PATTERNS):
            continue
        if target.startswith("broken:"):
            continue
        exists = _resolve_wikilink_target(target) is not None
        results.append({"raw": raw, "target": target, "exists": exists})
    return results


def _mark_broken_wikilinks(text: str) -> Tuple[str, List[str]]:
    """把死链 [[X]] 改成 [[broken:X]] 并返回 broken 列表。

    v2 P0-3 #3：写入前校验，不存在 → 标 broken，不静默接受死链。
    占位标识（模块自生成的"待 LLM 填"等）跳过不标。
    """
    broken = []

    def _replace(m: re.Match) -> str:
        raw = m.group(0)
        target = m.group(1).split("|")[0].strip()
        # 跳过占位
        if any(p in target for p in _PLACEHOLDER_PATTERNS):
            return raw
        if target.startswith("broken:"):
            return raw
        if _resolve_wikilink_target(target) is None:
            broken.append(target)
            # 保留 alias（如有）
            if "|" in m.group(1):
                alias = m.group(1).split("|", 1)[1]
                return f"[[broken:{target}|{alias}]]"
            return f"[[broken:{target}]]"
        return raw

    new_text = _WIKILINK_RE.sub(_replace, text)
    return new_text, broken


# ---------------------------------------------------------------------------
# §1c trace 索引（v2 P0-3 #2：index.jsonl，程序查找走索引）
# ---------------------------------------------------------------------------

def _index_path(trace_root: Path) -> Path:
    return trace_root / "index.jsonl"


def append_trace_index(
    trace_root: Path,
    trace_id: str,
    exp_id: int,
    seq: str,
    trace_path: Path,
    created: Optional[str] = None,
) -> None:
    """往 coord-traces/index.jsonl 追加一行（程序查找用）。

    格式：{"trace_id","exp_id","seq","path","created"}
    path 用相对 trace_root 的路径（分桶后 YYYY/MM/trace-...md）
    """
    trace_root.mkdir(parents=True, exist_ok=True)
    idx = _index_path(trace_root)
    try:
        rel = trace_path.relative_to(trace_root)
    except ValueError:
        rel = trace_path  # 兜底用绝对路径
    entry = {
        "trace_id": trace_id,
        "exp_id": exp_id,
        "seq": seq,
        "path": str(rel),
        "created": created or datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    # Task 288 P1-2: 加文件锁防止多进程并发追加交错
    import fcntl
    with open(idx, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def lookup_trace_by_exp(trace_root: Path, exp_id: int, limit: int = 20) -> List[Dict[str, Any]]:
    """从索引查某经验最近的 N 条 trace（程序查找入口）。"""
    idx = _index_path(trace_root)
    if not idx.exists():
        return []
    results = []
    # 从尾部读（最新在前）
    with open(idx, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("exp_id") == exp_id:
            results.append(entry)
            if len(results) >= limit:
                break
    return results


# ---------------------------------------------------------------------------
# §2 trace md 文件骨架（wikilink 占位由 LLM 填）
# ---------------------------------------------------------------------------

def _build_trace_md(
    trace_id: str,
    exp_id: int,
    seq: str,
    uncertainty: str,
    explanations: List[Dict[str, Any]],
    priority_order: List[str],
    wikilink_hints: Optional[Dict[str, str]],
    exp_title: Optional[str],
) -> str:
    """构造 trace 文件内容。wikilink 由 LLM 填，模块只生成骨架。"""
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    wikilink_hints = wikilink_hints or {}

    # 经验源 wikilink：优先用 hints 给的，否则占位（LLM 后填）
    # Task 288 P2-1: 占位符用 `经验-{exp_id}` 模板格式（带 `{`），匹配 _PLACEHOLDER_PATTERNS
    # 的 `"经验-{"` 模式，避免被 _mark_broken_wikilinks 误标死链。
    # Task 567: 占位符不再用 [[...]] 包裹——未填的占位不是链接，[[...]] 语法会被
    # Obsidian 解析为断链（用户图谱灰色断链点最大来源，866+864×2+615+23 处实锤）。
    # 有真实 hint 时仍输出 [[...]]（真链）；占位时输出纯文本。
    _hint_source = wikilink_hints.get("exp_source")
    exp_wikilink = (
        f"[[{_hint_source}]]" if _hint_source
        else f"`经验-{{{exp_id}}}`（待 LLM 填源文件 wikilink）"
    )
    exp_title_md = exp_title or f"经验 #{exp_id}"

    # 优先级解释表
    rows = []
    for expl in explanations:
        pos = expl["pos"]
        letter = expl["letter"]
        dim = expl["dim"]
        reason = expl["reason"]
        # 每位的 wikilink 依据：hints 里可用 dim 作 key
        wl = wikilink_hints.get(f"dim_{pos}") or wikilink_hints.get(f"dim_{dim}") or "—"
        rows.append(f"| {pos} | {letter} | {dim} | {reason} | {wl} |")
    table = "\n".join(rows) if rows else "| - | - | - | - | - |"

    return f"""---
title: 套用溯源 - {trace_id}
trace_id: {trace_id}
exp_id: {exp_id}
seq: {seq}
uncertainty: {uncertainty}
created: {now_iso}
tags: [coord-trace, apply_attempt]
---

## 套用溯源

- **套用经验**: {exp_wikilink}（{exp_title_md}）
- **坐标序列**: {seq}
- **LLM 自评**: uncertainty={uncertainty}

## 优先级与维度解释（wikilink 溯源）

> 位置=优先级（第一位最重要）。LLM 当场定的优先级顺序：{priority_order}
> wikilink 列为依据文件占位，由 LLM 填（填 `-` 表示无外部依据，依据即场景本身）。

| 位 | 字母 | 优先级维度 | 解释 | 依据(wikilink) |
|----|------|-----------|------|----------------|
{table}

## 熔炉探寻（如 uncertainty=H 触发）

> 此段由 crucible_trigger.py 追加。触发后各 Agent verdict + 收敛/分歧判定写在此处。

## 结论

> 套用前由 LLM 填：套用 / 条件套用 / 不适用。
> 如不适用：反例 wikilink `counterexample-XXX`（Task 567：占位不再用 [[]] 语法防断链）
"""


def write_trace_file(
    trace_id: str,
    exp_id: int,
    seq: str,
    uncertainty: str,
    explanations: List[Dict[str, Any]],
    priority_order: List[str],
    trace_dir: Path = DEFAULT_TRACE_DIR,
    wikilink_hints: Optional[Dict[str, str]] = None,
    exp_title: Optional[str] = None,
) -> Path:
    """写 trace md 文件（v2 P0-3：分桶 YYYY/MM + 索引 + wikilink 校验）。

    v2 变更：
      - 分桶：写到 trace_dir/YYYY/MM/trace-{trace_id}.md（不再平铺）
      - 索引：写完后追加一行到 trace_dir/index.jsonl（程序查找入口）
      - wikilink 校验：写入前扫 [[...]]，死链标 [[broken:...]]
    """
    if trace_dir is None:
        raise ConfigError("vault 不可用，trace 写入无法进行。" + VAULT_MISSING_REMEDIATION)
    bucket = _bucket_dir(trace_dir, trace_id)
    bucket.mkdir(parents=True, exist_ok=True)
    content = _build_trace_md(
        trace_id, exp_id, seq, uncertainty, explanations,
        priority_order, wikilink_hints, exp_title,
    )
    # v2 P0-3 #3: wikilink 校验，死链标 broken
    content, broken = _mark_broken_wikilinks(content)
    if broken:
        # 把 broken 列表记到 trace 末尾（透明告知）
        broken_md = "\n".join(f"- [[broken:{b}]]" for b in broken)
        content += (
            f"\n<!-- wikilink 校验：以下依据文件在 Vault 内未找到，已标 broken。\n"
            f"     LLM 若确认依据存在，请核对文件名/路径后去 broken: 前缀 -->\n"
            f"{broken_md}\n"
        )
    path = bucket / f"trace-{trace_id}.md"
    # v2 P0-3 #4: 大小兜底（防 crucible 追加段失控，2026-07-28 Task 331）
    if len(content.encode("utf-8")) > MAX_TRACE_BYTES:
        # 保留 frontmatter + 前 N 行正文 + 截断标记
        lines = content.split("\n")
        # frontmatter 是前导 ---...--- 块（约 8-10 行）
        fm_end = 0
        fm_count = 0
        for idx, ln in enumerate(lines):
            if ln.strip() == "---":
                fm_count += 1
                if fm_count == 2:
                    fm_end = idx + 1
                    break
        head = "\n".join(lines[:fm_end]) if fm_end else "\n".join(lines[:15])
        content = (
            head + "\n\n"
            + f"<!-- ⚠️ TRACE TRUNCATED: 原文 {len(content)} 字节超 {MAX_TRACE_BYTES} 上限，"
            f"疑 crucible 追加段失控。保留 frontmatter + 本截断标记，正文已截断。"
            f"完整原文见 process/full-log.md 或 git 历史。Task 331 -->\n"
        )
    path.write_text(content, encoding="utf-8")

    # v2 P0-3 #2: 追加索引
    try:
        append_trace_index(
            trace_root=trace_dir,
            trace_id=trace_id,
            exp_id=exp_id,
            seq=seq,
            trace_path=path,
        )
    except Exception as e:
        print(f"[WARN] 追加 trace 索引失败（不阻塞）: {e}", file=sys.stderr)

    return path


# ---------------------------------------------------------------------------
# §3 annotate_apply_attempt：Layer 0 主入口
# ---------------------------------------------------------------------------

def annotate_apply_attempt(
    kb,
    exp_id: int,
    current_coord: Dict[str, Any],
    priority_order: Optional[List[str]] = None,
    *,
    priority_override: Optional[List[str]] = None,
    priority_change_reason: Optional[str] = None,
    uncertainty: Optional[str] = None,
    wikilink_hints: Optional[Dict[str, str]] = None,
    trace_dir: Path = DEFAULT_TRACE_DIR,
    log_activity: bool = True,
    task_id: Optional[str] = None,  # Task396 Step4: 写入 agent_activity.task_id 列
) -> Dict[str, Any]:
    """Layer 0：套用诚实标注全流程（v2 P0-1/P0-3 修复版）。

    v2 变更：
      - priority_order 改可选（None → 走锚定继承/默认核心4维）
      - 新增 priority_override（显式覆盖，需配 priority_change_reason）
      - 返回值加 priority_source（inherited/explicit/override/default）+ broken_links

    Args:
        kb: SharedKnowledge 实例。测试时传 None 跳过 DB 操作。
        exp_id: 准备套用的经验 ID
        current_coord: 当前实时捕获的场景坐标
        priority_order: 当场定的优先级。None → 继承 source_coord.priority_order
        priority_override: 显式覆盖（最高优先级）。配 priority_change_reason 写进 trace
        priority_change_reason: override 时必填的变更理由（写进 trace 注释）
        uncertainty: LLM 显式自评 H/M/L。None → suggest_uncertainty 启发式
        wikilink_hints: wikilink 依据（v2 会校验存在性，死链标 broken）
        trace_dir: trace 根目录（v2 内部分桶到 YYYY/MM）
        log_activity: 是否写 agent_activity

    Returns:
        {
          "annotation": "[NEXUS:apply_attempt | ...]",
          "trace_path": "/.../YYYY/MM/trace-...md",
          "trace_id": "YYYYMMDD-NNN",
          "seq": "ABAA",
          "uncertainty": "M",
          "explanations": [...],
          "source_coord_missing": bool,
          "priority_source": "inherited|explicit|override|default",
          "priority_order_used": [...],
          "broken_links": ["..."],   # v2 wikilink 校验发现的死链
        }
    """
    if trace_dir is None:
        raise ConfigError("vault 不可用，trace 写入无法进行。" + VAULT_MISSING_REMEDIATION)
    # 1. 读源坐标
    source_coord = None
    source_missing = False
    exp_title = None
    if kb is not None:
        try:
            entry = kb.get_knowledge(exp_id)
            if entry:
                source_coord = entry.get("context_coord")
                exp_title = entry.get("title")
            if source_coord is None:
                source_missing = True
        except Exception as e:
            print(f"[WARN] 读经验 #{exp_id} context_coord 失败（按缺失处理）: {e}",
                  file=sys.stderr)
            source_missing = True
    else:
        source_missing = True

    # 2. v2 P0-1: 解析优先级（继承/显式/override/默认）
    from nexusos_mem.context_capture import resolve_priority_order, CORE_DIMENSIONS
    resolved_order, priority_source = resolve_priority_order(
        source_coord, priority_order, priority_override,
    )
    # override 必须带变更理由
    if priority_source == "override" and not priority_change_reason:
        print("[WARN] priority_override 使用但未提供 priority_change_reason "
              "（应写进 trace 说明为何偏离继承锚）", file=sys.stderr)

    # 3. 算有序序列
    seq, explanations = compute_sequence(
        source_coord, current_coord, resolved_order,
    )

    # 4. uncertainty：LLM 显式优先，否则启发式
    if uncertainty not in ("H", "M", "L"):
        uncertainty = suggest_uncertainty(seq)

    # 5. 写 trace md 文件（v2: 分桶 + 索引 + wikilink 校验内部完成）
    trace_id = _next_trace_id(trace_dir)
    trace_path = write_trace_file(
        trace_id, exp_id, seq, uncertainty, explanations, resolved_order,
        trace_dir=trace_dir, wikilink_hints=wikilink_hints, exp_title=exp_title,
    )
    # 读回 trace 文件，统计 broken 链（write_trace_file 已标 [[broken:...]]）
    # 直接正则提取 [[broken:...]]，因为 validate_wikilinks 会跳过已标 broken 的
    broken_links = []
    try:
        trace_text = Path(trace_path).read_text(encoding="utf-8")
        for m in re.finditer(r"\[\[broken:([^\]|]+)(?:\|([^\]]+))?\]\]", trace_text):
            broken_links.append(m.group(1))
    except Exception as e:
        # Task 288 P1-3: 原静默 except Exception: pass——调用方无法区分"无死链"
        # 和"读取失败"。改为 WARN 日志（不阻塞，但留下诊断痕迹）。
        print(f"[WARN] 读 trace 文件提取 broken_links 失败（按空列表处理）: {e}",
              file=sys.stderr)

    # v2 P0-1: override 时把变更理由追加到 trace
    if priority_source == "override" and priority_change_reason:
        try:
            tp = Path(trace_path)
            txt = tp.read_text(encoding="utf-8")
            note = (
                f"\n<!-- 优先级变更：本次使用 override 而非继承锚。\n"
                f"     理由：{priority_change_reason} -->\n"
            )
            tp.write_text(txt + note, encoding="utf-8")
        except Exception as e:
            print(f"[WARN] 写优先级变更理由失败（不阻塞）: {e}", file=sys.stderr)

    # 6. 生成标注字符串
    trace_file_name = trace_path.name
    annotation = (
        f"[NEXUS:apply_attempt | exp_id={exp_id} | seq={seq} | "
        f"trace={trace_file_name} | uncertainty={uncertainty} | "
        f"prio={priority_source}]"
    )

    # 7. 写 agent_activity
    # Task 288 P1-4: DB 写入失败原静默 print WARN——调用方无法区分"写入成功"和"失败"，
    # apply_attempt 数据静默丢失。改为 ERROR 日志 + 返回 error 字段（不 raise，避免
    # 阻塞主流程；调用方可检查返回值 error 字段决定是否重试）。
    activity_error = None
    _activity_id = None  # Task400: 接收 log_activity 返回的 id
    if log_activity and kb is not None:
        try:
            _activity_id = kb.log_activity(
                action="apply_attempt",
                target_type="knowledge",
                target_id=exp_id,
                detail={
                    "tag": "apply_attempt",  # Task396 Step3: 必填, 防 A1 去重失效
                    "seq": seq,
                    "trace_path": str(trace_path),
                    "trace_id": trace_id,
                    "uncertainty": uncertainty,
                    "priority_order": resolved_order,
                    "priority_source": priority_source,
                    "source_coord_missing": source_missing,
                    "broken_links": broken_links,
                    "annotation": annotation,
                },
                task_id=task_id,  # Task396 Step4: 写入 task_id 列（通电反查用）
            )
        except Exception as e:
            activity_error = str(e)
            print(f"[ERROR] 写 agent_activity(apply_attempt) 失败（apply_attempt 数据丢失）: {e}",
                  file=sys.stderr)

    return {
        "annotation": annotation,
        "trace_path": str(trace_path),
        "trace_id": trace_id,
        "seq": seq,
        "uncertainty": uncertainty,
        "explanations": explanations,
        "source_coord_missing": source_missing,
        "priority_source": priority_source,
        "priority_order_used": resolved_order,
        "broken_links": broken_links,
        # Task 288 P1-4: 让调用方能区分 agent_activity 写入成功/失败
        "activity_error": activity_error,
        "activity_id": _activity_id,  # Task400: log_activity 返回的 id（record_apply_outcome 关联用）
    }


# ---------------------------------------------------------------------------
# §4 追加熔炉段（供 crucible_trigger.py 调用，集中在本文件避免重复实现）
# ---------------------------------------------------------------------------

def append_crucible_section(
    trace_path: Path,
    trigger_uncertainty: str,
    distant_dims: List[str],
    agent_verdicts: List[Dict[str, Any]],
    consensus_points: List[str],
    dissent_points: List[str],
    final_verdict: str,
    conditions: Optional[List[str]] = None,
) -> None:
    """把熔炉探寻段追加到 trace 文件（不新建文件）。

    Args:
        trace_path: trace md 文件路径
        trigger_uncertainty: 触发时的 uncertainty（通常是 'H'）
        distant_dims: LLM 标注的距离远的维度（如 ['time:D', 'data_state:G']）
        agent_verdicts: 各 Agent 立场
            [{"agent": "agent-a", "position": "可套用", "reason": "...",
              "confidence": 0.8, "wikilink": "[[xxx]]"}, ...]
        consensus_points: 收敛点（共识）→ 高置信
        dissent_points: 分歧点（探寻区）→ 待 95% 探寻
        final_verdict: apply / conditional_apply / not_applicable
        conditions: 套用条件（conditional_apply 时）
    """
    trace_path = Path(trace_path)
    if not trace_path.exists():
        print(f"[WARN] trace 文件不存在，跳过追加: {trace_path}", file=sys.stderr)
        return

    distant_str = ", ".join(distant_dims) if distant_dims else "（无标注远维度）"
    rows = []
    for v in agent_verdicts:
        wl = v.get("wikilink", "—")
        conf = v.get("confidence", "")
        conf_str = f"（conf={conf}）" if conf != "" else ""
        rows.append(
            f"| {v['agent']} | {v['position']} | {v.get('reason', '')} | {conf_str} | {wl} |"
        )
    table = "\n".join(rows) if rows else "| - | - | - | - | - |"

    consensus_md = "\n".join(f"- {p}" for p in consensus_points) or "- （无收敛点）"
    dissent_md = "\n".join(f"- {p}" for p in dissent_points) or "- （无分歧点）"
    conditions_md = ""
    if conditions:
        conditions_md = "\n\n**套用条件：**\n" + "\n".join(f"- {c}" for c in conditions)

    section = f"""
### 熔炉探寻记录（{datetime.now(timezone.utc).isoformat(timespec="seconds")}）

- **触发**: uncertainty={trigger_uncertainty}，远维度: {distant_str}

#### Agent verdicts

| Agent | 立场 | 理由 | confidence | 依据 |
|-------|------|------|------------|------|
{table}

#### 收敛与分歧

**收敛点（共识，高置信，直接采信）：**
{consensus_md}

**分歧点（探寻区，标记待 95% 探寻）：**
{dissent_md}

#### 最终判定

{final_verdict}{conditions_md}
"""

    # Task 335: 写盘前大小守卫（防 crucible 追加段失控，补 Task 331 漏洞）
    # 复用 write_trace_file 的 MAX_TRACE_BYTES（同模块常量，L48）
    # 检查写盘前的文件大小，超限直接 return（crucible 探寻是增强信息非必需）
    current_size = trace_path.stat().st_size
    if current_size > MAX_TRACE_BYTES:
        print(
            f"[WARN] trace 文件已超 {MAX_TRACE_BYTES}B 上限（当前 {current_size}B），"
            f"跳过 crucible 段追加防膨胀: {trace_path}",
            file=sys.stderr,
        )
        return
    # 替换 trace 文件里"## 熔炉探寻"段下的占位
    content = trace_path.read_text(encoding="utf-8")
    # Task 288 P2-3: marker 用正则匹配，兼容 ## 结论 / ## 结论（已填）等变体
    # 原硬编码 "\n## 结论" 用 in 判断，LLM 改过标题后会找不到，熔炉段追加到文件末尾
    import re as _re
    marker_match = _re.search(r"\n## 结论[^\n]*\n", content)
    if marker_match:
        # 在 "## 结论 ..." 段前插入熔炉段
        idx = marker_match.start()
        new_content = content[:idx] + section + content[idx:]
    else:
        new_content = content + "\n" + section
    trace_path.write_text(new_content, encoding="utf-8")


# ---------------------------------------------------------------------------
# §5 自测（无 DB，用 mock）
# ---------------------------------------------------------------------------

def _self_test() -> None:
    import tempfile

    # mock kb：get_knowledge 返回带 context_coord 的条目
    class MockKB:
        def __init__(self, ctx):
            self.ctx = ctx
            self.logged = []
        def get_knowledge(self, kid):
            return {"id": kid, "title": f"测试经验 #{kid}", "context_coord": self.ctx}
        def log_activity(self, **kw):
            self.logged.append(kw)

    src = {
        "task_type": "protocol_analysis",
        "agent": {"name": "agent-a"},
        "env": {"os": "Darwin"},
    }
    cur = {
        "task_type": "protocol_analysis",
        "agent": {"name": "agent-b"},
        "env": {"os": "Darwin"},
    }

    with tempfile.TemporaryDirectory() as td:
        trace_dir = Path(td)
        kb = MockKB(src)
        result = annotate_apply_attempt(
            kb, exp_id=288, current_coord=cur,
            priority_order=["task_type", "agent.name", "env.os"],
            uncertainty="M",
            trace_dir=trace_dir,
        )
        # task_type=A, agent.name=非A(oc vs ac 有距离), env.os=A
        # 注意：A-Z 是梯度不是好坏，oc/ac 单字符差异算出 N(中距)是有效信号
        assert result["seq"][0] == "A" and result["seq"][1] != "A" and result["seq"][2] == "A", \
            f"期望 A?A（中位非A）实际 {result['seq']}"
        assert "exp_id=288" in result["annotation"]
        assert Path(result["trace_path"]).exists()
        assert len(kb.logged) == 1
        assert kb.logged[0]["action"] == "apply_attempt"
        # v2 P0-3 #1: 分桶路径 YYYY/MM
        tp = Path(result["trace_path"])
        assert re.search(r"\d{4}/\d{2}/trace-\d{8}-\d{3}\.md$", str(tp)), \
            f"trace 应分桶到 YYYY/MM/，实际 {tp}"
        # v2 P0-1: priority_source
        assert result["priority_source"] == "explicit", \
            f"显式传 priority_order 应得 explicit，实际 {result['priority_source']}"
        print(f"[OK] apply_honesty trace 分桶写入: {result['trace_path']}")
        print(f"[OK] 标注: {result['annotation']}")

        # v2 P0-3 #2: 索引 index.jsonl
        idx_file = trace_dir / "index.jsonl"
        assert idx_file.exists(), "index.jsonl 应存在"
        idx_lines = idx_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(idx_lines) >= 1
        # Task 288 P2-2: 文件顶部已 import json，删重复 import 直接用 json.loads
        idx_entry = json.loads(idx_lines[0])
        assert idx_entry["exp_id"] == 288
        assert idx_entry["trace_id"] == result["trace_id"]
        print(f"[OK] 索引写入: {idx_entry}")

        # v2 P0-3 #2: lookup_trace_by_exp 能查到
        looked = lookup_trace_by_exp(trace_dir, 288)
        assert len(looked) == 1 and looked[0]["trace_id"] == result["trace_id"]
        print(f"[OK] 索引查找: {looked[0]['trace_id']}")

        # v2 P0-1: 优先级继承（source_coord 有 priority_order）
        kb_inh = MockKB({**src, "priority_order": ["task_type", "env.os"]})
        r_inh = annotate_apply_attempt(
            kb_inh, exp_id=289, current_coord=cur,
            trace_dir=trace_dir,  # 不传 priority_order → 走继承
        )
        assert r_inh["priority_source"] == "inherited", \
            f"应继承，实际 {r_inh['priority_source']}"
        assert r_inh["priority_order_used"] == ["task_type", "env.os"]
        print(f"[OK] 优先级继承: {r_inh['priority_source']}")

        # v2 P0-1: override（需配变更理由）
        kb_ov = MockKB({**src, "priority_order": ["task_type", "env.os"]})
        r_ov = annotate_apply_attempt(
            kb_ov, exp_id=290, current_coord=cur,
            priority_override=["task_type"],
            priority_change_reason="测试 override",
            trace_dir=trace_dir,
        )
        assert r_ov["priority_source"] == "override"
        assert r_ov["priority_order_used"] == ["task_type"]
        # trace 文件应含变更理由注释
        ov_text = Path(r_ov["trace_path"]).read_text(encoding="utf-8")
        assert "优先级变更" in ov_text and "测试 override" in ov_text
        print(f"[OK] 优先级 override: {r_ov['priority_source']}")

        # v2 P0-3 #3: wikilink 校验（死链标 broken）
        kb_wl = MockKB(src)
        r_wl = annotate_apply_attempt(
            kb_wl, exp_id=291, current_coord=cur,
            priority_order=["task_type"],
            wikilink_hints={"exp_source": "不存在的不存在的文件-XYZABC"},
            trace_dir=trace_dir,
        )
        # 死链应在 broken_links 列表里（exp_source 指向不存在的文件）
        # 注意：exp_source 写进 trace 的 [[...]]，校验后标 broken
        wl_text = Path(r_wl["trace_path"]).read_text(encoding="utf-8")
        assert "broken:" in wl_text, f"死链应被标 broken，trace 内容: {wl_text[:300]}"
        print(f"[OK] wikilink 死链校验: broken_links={r_wl['broken_links']}")

        # 存量经验无 context_coord → '-' + priority_source=default
        kb2 = MockKB(None)
        r2 = annotate_apply_attempt(
            kb2, exp_id=999, current_coord=cur,
            priority_order=["task_type", "agent.name"],
            trace_dir=trace_dir,
        )
        assert r2["seq"] == "--", f"源缺维度应得 '--' 实际 {r2['seq']}"
        assert r2["source_coord_missing"] is True
        print(f"[OK] 存量经验降级: seq={r2['seq']}")

        # 熔炉段追加（v1 函数保留，Layer 1 暂缓但函数可用）
        append_crucible_section(
            Path(result["trace_path"]),
            trigger_uncertainty="H",
            distant_dims=["data_state:G"],
            agent_verdicts=[
                {"agent": "agent-a", "position": "可套用", "reason": "测试",
                 "confidence": 0.8, "wikilink": "[[test]]"},
            ],
            consensus_points=["c1"],
            dissent_points=["d1"],
            final_verdict="conditional_apply",
            conditions=["需验证 X"],
        )
        after = Path(result["trace_path"]).read_text(encoding="utf-8")
        assert "熔炉探寻记录" in after
        assert "conditional_apply" in after
        print("[OK] 熔炉段追加成功（v1 函数保留）")


# ============================================================================
# Task546 U8-b: 真套用判定 + 统计分母修正（用户定向方案 b）
# ============================================================================

# 固定 cutoff：本修正部署时刻（2026-08-14 17:30 UTC+8）前的存量 pending 整体排除
# 出新分母——历史普通查询稀释（zcode 85/omp 31/oc 8/ac 4/subagent 3）不追溯。
REAL_APPLY_CUTOFF = datetime(2026, 8, 14, 9, 30, 0, tzinfo=timezone.utc)

# 引用检测窗：attempt.created 起 24h
REAL_APPLY_REF_WINDOW = timedelta(hours=24)


def _exp_referenced_in_reports(exp_id: int, since: datetime, until: datetime) -> bool:
    """检测 exp_id 在 [since, until] 时间窗内是否被 task 产物引用（MEM#<id> 格式）。

    Task546: 判定锚在产物证据——report.md 里的 "MEM#49179" 式引用是真套用的
    强信号（小弟真的把经验用进了产出）。正则 \b 防前缀误伤（MEM#4919 vs 49197）。
    实现从简：只扫 task/*/output/report.md（mtime 或内容时间窗）。
    """
    import glob as _glob
    from nexusos_mem._vault_env import VAULT_ROOT as _VR
    if _VR is None:
        raise ConfigError("vault 不可用，产物引用扫描无法进行。" + VAULT_MISSING_REMEDIATION)
    vault = str(_VR)
    pattern = re.compile(rf"MEM#?{exp_id}\b")
    for report in _glob.glob(os.path.join(vault, "task", "*", "output", "report.md")):
        try:
            st = os.stat(report)
            mt = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
            # report mtime 在窗口后才可能引用了该 attempt 之后的经验
            if mt < since:
                continue
            content = open(report, encoding="utf-8", errors="replace").read()
            if pattern.search(content) and mt <= until + timedelta(days=1):
                return True
        except (OSError, ValueError):
            continue
    return False


def is_real_apply(attempt: Dict[str, Any]) -> bool:
    """Task546: 判定 apply_attempt 是否为真套用（产物证据锚定，非 Agent 声明）。

    三态：
    1. 有非 pending outcome 回填（显式 apply_outcome 行关联）→ True
       （Task581: pending 是 Task412 通电占位非真套用结果，显式排除防分母虚高）
    2. 无 outcome 但 attempt.created 起 24h 内 report 引用 MEM#<target_id> → True
    3. pending 占位/无引用（无回填）→ False（普通查询稀释，不是真套用）

    Args:
        attempt: agent_activity 行（action=apply_attempt），需含 id/target_id/outcome/created
    """
    if not attempt:
        return False
    # 态1: outcome 非空且非 pending（显式回填——含 apply_outcome 行回写的推断值；
    # Task581: pending 也是非空值，必须显式 != 'pending'，不能用「非空」语义）
    outcome = attempt.get("outcome")
    # Task984 后续：not_applicable = 评审确认非真套用（U8 仅查证/测试数据/自指引用），
    # 显式排除出分母——不回填率统计把裁定豁免项当已回填虚高
    if outcome == "not_applicable":
        return False
    if outcome and outcome != "pending":
        return True
    exp_id = attempt.get("target_id")
    created = attempt.get("created")
    if not exp_id or not created:
        return False
    # created 可能是 datetime 或 str（RealDictCursor vs 手构 dict）
    if isinstance(created, str):
        try:
            created = datetime.fromisoformat(created.replace("Z", "+00:00"))
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
        except ValueError:
            return False
    # 态2: 24h 窗内产物引用
    return _exp_referenced_in_reports(exp_id, created, created + REAL_APPLY_REF_WINDOW)


def fetch_real_apply_stats(kb) -> Dict[str, Any]:
    """Task546: 真套用分母统计（回填率/多主一致性的新分母）。

    分母 = 真套用（is_real_apply）且 created >= REAL_APPLY_CUTOFF 的 attempt 数。
    分子 = 有 outcome 回填的（显式 apply_outcome 行关联）。
    历史不追溯：cutoff 前的 pending 整体排除（固定常量勿漂移）。

    Returns:
        {denominator, numerator, rate, cutoff, legacy_pending_excluded}
    """
    from nexusos_mem.shared import SharedKnowledge  # noqa: F401
    conn = kb._get_read_conn() if hasattr(kb, "_get_read_conn") else kb._get_conn()
    import psycopg2.extras
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        # cutoff 后的全部 attempt + 有关联 outcome 行的 id 集
        cur.execute(
            """
            SELECT aa.id, aa.target_id, aa.outcome, aa.created,
                   EXISTS(SELECT 1 FROM agent_activity ao
                          WHERE ao.action = 'apply_outcome'
                            AND ao.detail->>'apply_attempt_id' = aa.id::text
                            -- Task581: pending 是 Task412 通电占位非回填，滤出分子
                            AND COALESCE(ao.detail->>'result', '') != 'pending') AS has_real_outcome_row
            FROM agent_activity aa
            WHERE aa.action = 'apply_attempt'
              AND aa.created >= %s
            """,
            (REAL_APPLY_CUTOFF,),
        )
        rows = cur.fetchall()
        # cutoff 前存量（报告用，不进分母）
        cur.execute(
            "SELECT count(*) AS n FROM agent_activity "
            "WHERE action='apply_attempt' AND created < %s",
            (REAL_APPLY_CUTOFF,),
        )
        legacy = cur.fetchone()["n"]
    finally:
        cur.close()
        if not hasattr(kb, "_get_conn"):
            conn.close()

    denominator = 0
    numerator = 0
    for r in rows:
        # Task581: has_real_outcome_row 已滤 pending（占位不进分子）
        if r.get("has_real_outcome_row"):
            numerator += 1
        # Task548: 分母判定取并集（列 OR action 行）——过渡期 action 行已写但
        # outcome 列尚未回写时（历史断裂窗口），单看列会把已回填判 False →
        # numerator > denominator 倒挂。列与行任一有值即真套用。
        # Task581: 两路均已滤 pending（列走 is_real_apply 态1，行走 SQL 过滤）
        if r.get("has_real_outcome_row") or is_real_apply(r):
            denominator += 1
    rate = round(numerator / denominator, 4) if denominator > 0 else None
    # Task1401-D5B: 全量诚实口径并列——U8 口径（上方 denominator/numerator/rate）
    # 把 66.9% 无回填裁出分母，单口径输出=统计性美化（体检报告定性）。双口径
    # 并列让报表诚实：raw=全量 attempt 数、filled=已回填（列或行任一）、
    # stale_adjudicated=D5 自动裁定数（not_applicable 是 attempt 级裁定，
    # 语义=已结案非已回填，单列透明）。两处都数（MEM#49286 双表示）。
    cur3 = conn.cursor() if hasattr(kb, "_get_conn") else None
    if cur3 is not None:
        try:
            cur3.execute(
                "SELECT count(*) FROM agent_activity WHERE action='apply_attempt'"
            )
            raw_all = cur3.fetchone()[0]
            cur3.execute(
                "SELECT count(*) FROM agent_activity "
                "WHERE action='apply_attempt' "
                "AND (outcome IS NOT NULL AND outcome != 'pending')"
            )
            filled_col = cur3.fetchone()[0]
            cur3.execute(
                "SELECT count(*) FROM agent_activity "
                "WHERE action='apply_attempt' AND outcome = 'not_applicable'"
            )
            stale_adj = cur3.fetchone()[0]
        finally:
            cur3.close()
    else:
        raw_all, filled_col, stale_adj = len(rows), numerator, 0
    return {
        "denominator": denominator,       # 真套用数（新分母，U8 口径）
        "numerator": numerator,           # 有 outcome 回填
        "rate": rate,                     # 回填率（新，U8 口径）
        "cutoff": REAL_APPLY_CUTOFF.isoformat(),
        "legacy_pending_excluded": legacy,  # cutoff 前存量（排除）
        "raw_attempt_count": len(rows),   # cutoff 后全部（旧分母口径）
        "raw": {                          # Task1401-D5B: 全量诚实口径并列
            "raw_attempt_count": raw_all,      # 全库 attempt 总数（含 cutoff 前）
            "filled": filled_col,              # 已回填（attempt.outcome 列口径）
            "stale_adjudicated": stale_adj,    # D5 自动裁定结案数（非回填）
            "rate_raw": round(filled_col / raw_all, 4) if raw_all else None,
        },
    }


def auto_adjudicate_stale(kb, cutoff_days: int = 14, limit: int = 2000) -> Dict[str, Any]:
    """Task1401-D5B（设计稿块3 T3）: 超期欠账自动裁定——诚实体系止血。

    管什么：apply_attempt 欠账（NULL/pending）只增不减的根因之一是无自动
    裁定机制。本函数把「14 天滚动窗口外 + 无回填 + 无 24h 产物引用」的
    attempt（即 is_real_apply 态3 语义：查证式命中非真套用）批量裁定为
    not_applicable——账有生命周期，不再无限堆积。

    为什么用 not_applicable 而非 failed：not_applicable 是 attempt 级裁定
    （评审确认非真套用），不污染 knowledge.outcome、不触发 counterexample、
    不计 ref_count（record_apply_outcome 既有语义）；failed 会造反例冤枉经验。

    late-override 语义：裁定不是终审——裁定后真回填到达时
    record_apply_outcome 仍可覆盖（列回写+新 action 行，不拒绝已裁定
    attempt）。设计稿反例 R2：跨周长任务的真回填不许被吞。

    回滚锚：evidence 前缀固定 'auto-adjudicated stale (D5)'——
    反查此前缀可批量撤销裁定（可逆设计，违反 B1 不物理删除）。

    Args:
        kb: SharedKnowledge 实例
        cutoff_days: 滚动窗口天数（默认 14）
        limit: 单次跑批上限（防长事务；存量清偿分多次跑）

    Returns:
        {adjudicated, remaining, window_days, evidence_prefix, rows:[{attempt_id, exp_id}]}
    """
    import psycopg2.extras
    conn = kb._get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    EVID = "auto-adjudicated stale (D5)"
    try:
        # 候选=态3 语义（NULL/pending 且无 apply_outcome 行回填）且超窗口。
        # 不在此处调 is_real_apply 逐行判 24h 引用（其内部扫全库 report 文件
        # 太慢）——候选集先 SQL 收窄到超期未回填，再对候选逐行 is_real_apply
        # 复核（态2 的 24h 引用者不裁定；态1 已在 SQL 滤除）。
        cur.execute(
            """
            SELECT aa.id, aa.target_id, aa.created, aa.outcome
            FROM agent_activity aa
            WHERE aa.action = 'apply_attempt'
              AND (aa.outcome IS NULL OR aa.outcome = 'pending')
              AND aa.created < NOW() - (%s || ' days')::interval
              AND NOT EXISTS (
                SELECT 1 FROM agent_activity ao
                WHERE ao.action = 'apply_outcome'
                  AND ao.detail->>'apply_attempt_id' = aa.id::text
                  AND COALESCE(ao.detail->>'result', '') != 'pending')
            ORDER BY aa.id
            LIMIT %s
            """,
            (cutoff_days, limit),
        )
        candidates = cur.fetchall()
    finally:
        cur.close()

    adjudicated = []
    for row in candidates:
        attempt = {
            "id": row["id"],
            "target_id": row["target_id"],
            "outcome": row["outcome"],
            "created": row["created"],
        }
        # 态2 复核：24h 内有产物引用 = 真套用，不裁定（留给人工/迟到回填）
        if _exp_referenced_in_reports(
            attempt["target_id"],
            attempt["created"],
            attempt["created"] + REAL_APPLY_REF_WINDOW,
        ):
            continue
        ret = kb.record_apply_outcome(
            apply_attempt_id=attempt["id"],
            exp_id=attempt["target_id"],
            result="not_applicable",
            evidence=EVID,
        )
        if ret.get("ok"):
            adjudicated.append({"attempt_id": attempt["id"], "exp_id": attempt["target_id"]})

    # 跑批留痕（验收 5：agent_activity 出现 action='auto_adjudicate' 行）
    try:
        kb.log_activity(
            action="auto_adjudicate",
            detail={
                "adjudicated": len(adjudicated),
                "window_days": cutoff_days,
                "evidence_prefix": EVID,
                "candidates_seen": len(candidates),
            },
        )
    except Exception:
        pass  # 留痕失败不阻塞裁定本身

    # 剩余欠账（窗内+窗外未处理完的）
    conn2 = kb._get_conn()
    cur2 = conn2.cursor()
    try:
        cur2.execute(
            "SELECT count(*) FROM agent_activity "
            "WHERE action='apply_attempt' AND (outcome IS NULL OR outcome='pending')"
        )
        remaining = cur2.fetchone()[0]
    finally:
        cur2.close()

    return {
        "adjudicated": len(adjudicated),
        "remaining": remaining,
        "window_days": cutoff_days,
        "evidence_prefix": EVID,
        "rows": adjudicated[:50],
    }


if __name__ == "__main__":
    _self_test()
