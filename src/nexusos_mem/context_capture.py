#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
context_capture.py — 认知诚实体系 Layer 0：场景坐标捕获 + 有序序列算法

设计依据（NEXUS v1.4 §十 + Task 258 任务书 + Experience-Context-Coord-Template.md）：
  - 经验是"带场景坐标的适用域"，不是扁平文本
  - A-Z 是该维度的距离梯度（A=一致, B-Z=距离递增），全有效信息（非好坏）
  - 有序序列：位置=优先级（LLM 当场定优先级顺序，模块不写死）
  - 维度无上限：priority_order 有几位，序列就有几位
  - 存量经验无 context_coord → 对应位标 '-'（不阻塞但记录）

公共 API：
  capture_current_context(agent_name, task_type, phase=None) -> dict
      实时捕获工作上下文，返回 jsonb-ready dict
  compute_sequence(source_coord, current_coord, priority_order) -> (seq_str, explanations)
      按给定优先级顺序算有序序列 + 每位解释（供 trace 文件用）

注意：
  - 模块只负责按 priority_order 编码，优先级顺序由 LLM 当场定（传入参数）
  - 不依赖 nexus.shared，纯函数，可独立测试
"""

from __future__ import annotations
import difflib
import json
import math
import os
import platform
import socket
from pathlib import Path
import threading
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# §0 BGE-M3 embedding 客户端（task 265：替代 difflib 对中文失效问题）
# ---------------------------------------------------------------------------

_EMBED_URL = "http://127.0.0.1:11436/embed"
_EMBED_TIMEOUT = 5  # 秒，超时降级 difflib
_EMBED_CACHE: Dict[str, List[float]] = {}  # text → vector 缓存（同串不重复 embed）
_EMBED_CACHE_LOCK = threading.Lock()
_EMBED_AVAILABLE: Optional[bool] = None  # None=未探测, True/False=已知状态


def _embed_text_via_service(text: str) -> Optional[List[float]]:
    """调 EMBED 服务算 BGE-M3 向量（带缓存）。

    Returns:
        1024 维向量 list，或 None（服务 DOWN / 超时 / 错误）
    """
    global _EMBED_AVAILABLE
    if not text:
        return None
    # 缓存命中
    with _EMBED_CACHE_LOCK:
        if text in _EMBED_CACHE:
            return _EMBED_CACHE[text]
    # 已知服务 DOWN → 不重试（避免每次套用都超时）
    if _EMBED_AVAILABLE is False:
        return None
    try:
        payload = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            _EMBED_URL, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=_EMBED_TIMEOUT) as resp:
            data = json.loads(resp.read())
        vecs = data.get("vectors") or []
        if not vecs:
            return None
        vec = vecs[0]
        _EMBED_AVAILABLE = True
        # 写缓存（限制缓存大小防内存泄漏）
        with _EMBED_CACHE_LOCK:
            if len(_EMBED_CACHE) < 2000:
                _EMBED_CACHE[text] = vec
        return vec
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        # 服务 DOWN / 超时 → 标记不可用，后续直接降级
        if _EMBED_AVAILABLE is not False:
            import sys
            print(f"[WARN] EMBED 服务不可用（{type(e).__name__}: {e}），降级 difflib",
                  file=sys.stderr)
        _EMBED_AVAILABLE = False
        return None


def _embed_pair_via_service(a: str, b: str) -> tuple:
    """Task582: 批量接口一次拿两个向量（对比场景 HTTP 往返减半）。

    复用 _embed_text_via_service 的缓存与 DOWN 标记语义：
    - 两条都在缓存 → 零 HTTP
    - 有 miss → 一次 POST texts:[a,b] 批量请求补齐
    Returns:
        (va, vb) 各为 1024 维 list 或 None（服务不可用时双 None，调用方降级 difflib）
    """
    global _EMBED_AVAILABLE
    with _EMBED_CACHE_LOCK:
        va = _EMBED_CACHE.get(a)
        vb = _EMBED_CACHE.get(b)
    if va is not None and vb is not None:
        return va, vb
    if _EMBED_AVAILABLE is False or not a or not b:
        # 服务已知 DOWN：仍给缓存半命中机会（上面已查），这里直接回传
        return va, vb
    try:
        payload = json.dumps({"texts": [a, b]}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            _EMBED_URL, data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=_EMBED_TIMEOUT) as resp:
            data = json.loads(resp.read())
        vecs = data.get("vectors") or []
        if len(vecs) == 2:
            _EMBED_AVAILABLE = True
            with _EMBED_CACHE_LOCK:
                if len(_EMBED_CACHE) < 2000:
                    _EMBED_CACHE[a] = vecs[0]
                    _EMBED_CACHE[b] = vecs[1]
            return vecs[0], vecs[1]
        return va, vb
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        if _EMBED_AVAILABLE is not False:
            import sys
            print(f"[WARN] EMBED 服务不可用（{type(e).__name__}: {e}），降级 difflib",
                  file=sys.stderr)
        _EMBED_AVAILABLE = False
        return va, vb


def _cosine_similarity(va: List[float], vb: List[float]) -> float:
    """余弦相似度 ∈ [-1, 1]，截断到 [0, 1]（距离算法只关心正向相关）。"""
    if not va or not vb or len(va) != len(vb):
        return 0.0
    dot = sum(a * b for a, b in zip(va, vb))
    na = math.sqrt(sum(a * a for a in va))
    nb = math.sqrt(sum(b * b for b in vb))
    if na == 0 or nb == 0:
        return 0.0
    # BGE-M3 余弦通常 [0, 1]，负值截 0
    return max(0.0, min(1.0, dot / (na * nb)))


def _reset_embed_state() -> None:
    """重置 embedding 缓存 + 可用性状态（测试用）。"""
    global _EMBED_AVAILABLE
    with _EMBED_CACHE_LOCK:
        _EMBED_CACHE.clear()
    _EMBED_AVAILABLE = None


# ---------------------------------------------------------------------------
# §1 场景坐标捕获
# ---------------------------------------------------------------------------

def capture_current_context(
    agent_name: str,
    task_type: str,
    phase: Optional[str] = None,
    *,
    extra: Optional[Dict[str, Any]] = None,
    priority_order: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """实时捕获当前工作上下文。

    维度无上限——本函数捕获一组"常见维度"，调用方可通过 extra= 注入更多
    场景特化维度（如 data_state, repo, host）。captured_dims 自动计数。

    Args:
        agent_name: Agent 标识（如 agent-a/agent-b/...）
        task_type: 任务类型（protocol_analysis/search/fix/design/...）
        phase: 阶段（reverse_engineering/test/audit/...）可空
        extra: 额外维度，合并到顶层（覆盖同名默认键）
        priority_order: 本场景的维度优先级顺序（Task 295 Step 4）。
            None → 写入 CORE_DIMENSIONS 作为锚定基准。
            显式传入 → 记录 LLM 本次决定的顺序（供 resolve_priority_order 继承）。
            **向后兼容**：老 context_coord 数据没有此字段时，下游
            resolve_priority_order 自动 fallback 到 CORE_DIMENSIONS，不破坏。

    Returns:
        jsonb-ready dict，含 task_type/agent/phase/env/captured_dims/captured_at/priority_order
    """
    ctx: Dict[str, Any] = {
        "task_type": task_type,
        "agent": {
            "name": agent_name,
            # 仅记录已知模型推断；不在本模块做 LLM 探测，调用方可显式传 extra={"agent": {...}}
        },
        "phase": phase,
        "env": {
            "os": platform.system(),
            "host": socket.gethostname(),
        },
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Task 295 Step 4: priority_order 字段（Kimi 286/288 共同推荐 Layer 1 前置）
        # None → 写 CORE_DIMENSIONS；显式传入 → 用传入值。
        # resolve_priority_order 读取此字段做"锚定继承"，使经验套用的维度顺序可审计可复现。
        "priority_order": list(priority_order) if priority_order else list(CORE_DIMENSIONS),
    }

    # 默认捕获 data_state（PG 行数等），但本模块不强制连 PG——留给调用方在 extra 里补
    if extra:
        for k, v in extra.items():
            if k == "agent" and isinstance(v, dict):
                ctx["agent"] = {**ctx["agent"], **v}
            elif k == "env" and isinstance(v, dict):
                ctx["env"] = {**ctx["env"], **v}
            else:
                ctx[k] = v

    # 自动计数捕获维度（顶层键数 + env 内键数；agent 子键算 1）
    captured_dims = (
        len(ctx)
        + (len(ctx.get("env", {})) if isinstance(ctx.get("env"), dict) else 0)
        + (len(ctx.get("data_state", {})) if isinstance(ctx.get("data_state"), dict) else 0)
    )
    ctx["captured_dims"] = captured_dims
    return ctx


# ---------------------------------------------------------------------------
# §2 单维度距离编码（A-Z 或 '-'）
# ---------------------------------------------------------------------------

def _flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    """把嵌套 dict/list 摊平为 {path: leaf}，便于按字段比对。"""
    out: Dict[str, Any] = {}
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten(v, key))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            key = f"{prefix}[{i}]" if prefix else f"[{i}]"
            out.update(_flatten(v, key))
    else:
        out[prefix] = value
    return out


def _string_similarity(a: str, b: str) -> float:
    """字符串语义相似度 ∈ [0, 1]。

    task 265 修复：优先用 BGE-M3 embedding 余弦相似度（对中文有效）。
    fallback：EMBED 服务 DOWN 时降级 difflib（对中文几乎返回 0，但保底不崩）。

    判定依据：difflib 对中文 token 完全失明（"协议分析" vs "逆向工程" → 0.000），
    embedding 能给出 0.457 的语义信号。经验大量中文，必须用 embedding。

    重映射：BGE-M3 相关但非相同的串 cosine 通常 0.4-0.7，线性映射会全打到 Z-N
    （极远段）。用平方根 cosine**0.5 拉伸：0.45→0.67(I), 0.55→0.74(G), 0.7→0.84(E)，
    让"语义相关"落到中段字母而非极远段。数值维度不走本函数，不受影响。
    """
    if not a and not b:
        return 1.0
    if a == b:
        return 1.0
    # 优先 embedding（中文有效）
    # Task582: pair 批量——原 va/vb 两次单条 HTTP 合并为一次 texts:[a,b]
    va, vb = _embed_pair_via_service(a, b)
    if va is not None and vb is not None:
        cosine = _cosine_similarity(va, vb)
        # task 265: 平方根重映射，把 embedding [0,1) 拉伸到中段字母区
        return cosine ** 0.5
    # fallback：difflib（EMBED DOWN 时保底）
    return difflib.SequenceMatcher(None, a, b).ratio()


def _scalar_similarity(src: Any, cur: Any) -> float:
    """标量相似度 ∈ [0, 1]。

    - 同类型数值：1 - |a-b|/(|a|+|b|+eps)
    - 字符串：difflib 比率
    - 同 bool：相等=1.0 else 0.0
    - 跨类型：转 str 后 difflib（弱信号）
    """
    if src == cur:
        return 1.0
    if isinstance(src, bool) or isinstance(cur, bool):
        return 1.0 if src == cur else 0.0
    if isinstance(src, (int, float)) and isinstance(cur, (int, float)):
        denom = abs(src) + abs(cur) + 1e-9
        return max(0.0, 1.0 - abs(src - cur) / denom)
    return _string_similarity(str(src), str(cur))


def _value_similarity(src: Any, cur: Any) -> float:
    """复合值相似度 ∈ [0, 1]。

    - dict/list/tuple：按摊平后公共路径的平均标量相似度 + Jaccard 加权
    - 标量：直接走 _scalar_similarity

    task 266 修复（方案 Z）：一边缺失子键不再直接计 0，改弱惩罚 0.3 + Jaccard 加权。
    判定依据：键名差异（如 env 的 vault vs host）不应等同"信息完全缺失"。
    旧逻辑 (1.0+0.0+0.0)/3=0.33→R；新逻辑 base×jaccard + 0.3×(1-jaccard)
    让共有键贡献被合理加权，缺失键保留弱信号。
    """
    if isinstance(src, (dict, list, tuple)) or isinstance(cur, (dict, list, tuple)):
        f_src = _flatten(src)
        f_cur = _flatten(cur)
        keys_src = set(f_src)
        keys_cur = set(f_cur)
        common = keys_src & keys_cur
        missing = keys_src ^ keys_cur   # 对称差（一边缺失的键）
        if not (common or missing):
            return 1.0
        # 共有键：真实相似度平均
        if common:
            base = sum(_scalar_similarity(f_src[k], f_cur[k]) for k in common) / len(common)
        else:
            base = 0.0
        # Jaccard 加权：共有键占比越高，base 权重越大
        union = keys_src | keys_cur
        jaccard = len(common) / len(union) if union else 1.0
        # 缺失键：弱惩罚 0.3（非 0），保留"信息缺失"信号但不极端
        missing_penalty = 0.3
        score = base * jaccard + missing_penalty * (1.0 - jaccard)
        return max(0.0, min(1.0, score))
    return _scalar_similarity(src, cur)


# ---------------------------------------------------------------------------
# §1d env 维度语义角色分组（task 266 方案 Y：键名高变体维度专用）
# ---------------------------------------------------------------------------

# env 维度语义角色分组（键名变体归一到语义角色）
# task 266：env 的 host/vault/hostname 等键名变体同指"机器标识"，
# 纯键名比对会过度惩罚。按语义角色分组，组内任一键存在即视为该角色已捕获。
# Task 282 P2-8: ENV_SEMANTIC_GROUPS + CORE_DIMENSIONS 外置到 config/context_dimensions.yaml
def _load_context_dimensions():
    """从 config/context_dimensions.yaml 加载语义组+核心维度，失败降级默认值。"""
    cfg = Path(__file__).parent / "config" / "context_dimensions.yaml"
    if cfg.exists():
        try:
            import yaml
            data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            return (
                data.get("env_semantic_groups", {}),
                data.get("core_dimensions", ["task_type", "env", "phase", "data_state"]),
            )
        except Exception:
            pass
    # 降级默认（保持原硬编码行为）
    return (
        {
            "machine_id": ["host", "vault", "hostname", "machine_name", "hostname_local"],
            "os": ["os", "platform", "os_version"],
            "location": ["vault_path", "disk", "mount", "path"],
        },
        ["task_type", "env", "phase", "data_state"],
    )

ENV_SEMANTIC_GROUPS: Dict[str, List[str]] = {}
CORE_DIMENSIONS: List[str] = []
ENV_SEMANTIC_GROUPS, CORE_DIMENSIONS = _load_context_dimensions()


def _first_match(d: Dict[str, Any], keys: List[str]) -> Any:
    """从 dict 中按 keys 顺序取首个存在的值（大小写不敏感），全无返回 _MISSING。"""
    for k in keys:
        for actual_key, v in d.items():
            if actual_key == k or actual_key.lower() == k.lower():
                return v
    return _MISSING


def _env_similarity(src_env: Any, cur_env: Any) -> float:
    """env 维度专用相似度：按语义角色分组比对（task 266 方案 Y）。

    组内任一键存在即视为该角色已捕获，比对两边的值（取首个匹配键的值）。
    比纯 _value_similarity 键名敏感度低，适合 env 这类键名高变体维度。

    判定依据：env 的 vault=实例根目录 和 host=本机名 同指"机器标识"角色，
    纯键名比对视为缺失（0 分）；语义分组后两者都进入 machine_id 角色比对值。
    """
    if not isinstance(src_env, dict) or not isinstance(cur_env, dict):
        return _value_similarity(src_env, cur_env)
    if not src_env and not cur_env:
        return 1.0

    role_scores: List[float] = []
    for role, aliases in ENV_SEMANTIC_GROUPS.items():
        src_v = _first_match(src_env, aliases)
        cur_v = _first_match(cur_env, aliases)
        if src_v is _MISSING and cur_v is _MISSING:
            continue  # 两边都没该角色，不计入
        if src_v is _MISSING or cur_v is _MISSING:
            role_scores.append(0.3)  # 一边有该角色，弱惩罚
        else:
            role_scores.append(_scalar_similarity(src_v, cur_v))
    if not role_scores:
        # 两边都没有任何已知语义角色 → 退回 _value_similarity
        return _value_similarity(src_env, cur_env)
    return sum(role_scores) / len(role_scores)


def _similarity_to_letter(r: float) -> str:
    """[0, 1] 相似度 → A-Z 梯度字母。

    - 1.0 → A（一致，可数一致点）
    - [0, 1) → 线性映射到 B-Z（B=最近非一致，Z=最远）

    用户原则："A-Z 全部是有效信息，不要只用 A 和 B"——所以用满 25 档。
    """
    if r >= 1.0:
        return "A"
    if r <= 0.0:
        return "Z"
    # [0, 1) → B-Z（25 档）。r 越接近 1 越靠 B，越接近 0 越靠 Z。
    # 档位索引 0..24 对应字母 B..Z
    idx = int((1.0 - r) * 25)
    if idx > 24:
        idx = 24
    return chr(ord("B") + idx)


def _dim_distance(
    dim_name: str,
    src_coord: Optional[Dict[str, Any]],
    cur_coord: Optional[Dict[str, Any]],
) -> Tuple[str, float, str]:
    """算单个维度的距离字母。

    Args:
        dim_name: 维度路径（支持点号嵌套，如 "agent.name"）
        src_coord: 源场景坐标（经验写入时）
        cur_coord: 当前场景坐标

    Returns:
        (letter, similarity, reason)
        letter ∈ {'A'..'Z', '-'}
        similarity ∈ [0, 1]（'-' 时为 -1.0）
        reason: 简短说明（供 trace 解释表用）
    """
    src_v = _dig(src_coord, dim_name)
    cur_v = _dig(cur_coord, dim_name)

    # 源坐标缺该维度 → '-'（不阻塞但记录）
    if src_v is _MISSING:
        return "-", -1.0, f"源坐标缺维度 {dim_name}（存量经验未捕获）"
    # 当前坐标缺该维度 → 仍可估，但记为远（信息缺失=不可判断匹配度）
    if cur_v is _MISSING:
        return "M", 0.3, f"当前坐标缺维度 {dim_name}（无法判定，估中距）"

    # task 266: env 维度走语义角色分组（方案 Y），其他维度走通用 _value_similarity
    if dim_name == "env":
        sim = _env_similarity(src_v, cur_v)
    else:
        sim = _value_similarity(src_v, cur_v)
    letter = _similarity_to_letter(sim)
    if letter == "A":
        reason = f"维度 {dim_name} 一致（值={_short_repr(src_v)}）"
    else:
        reason = (
            f"维度 {dim_name} 有距离：源={_short_repr(src_v)} vs "
            f"当前={_short_repr(cur_v)} → 相似度 {sim:.2f} → {letter}"
        )
    return letter, sim, reason


_MISSING = object()  # sentinel for "key absent"


def _dig(coord: Optional[Dict[str, Any]], path: str) -> Any:
    """按点号路径取值，缺失返回 _MISSING。"""
    if coord is None:
        return _MISSING
    cur: Any = coord
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return _MISSING
    return cur


def _short_repr(v: Any, maxlen: int = 40) -> str:
    s = repr(v)
    return s if len(s) <= maxlen else s[:maxlen - 1] + "…"


# ---------------------------------------------------------------------------
# §3 有序序列算法（v2: P0-1 优先级锚定 + 核心4维 + 变长可比）
# ---------------------------------------------------------------------------

# 核心维度（前 4 位，固定）：task_type → env → phase → data_state
# P0-1 修复：稳定可比性。前 4 位跨经验通用，第 5 位起为扩展维度（场景特化）。
# 比较两条经验的序列时：核心维度匹配数优先，扩展维度 bonus。
CORE_DIMENSIONS: List[str] = ["task_type", "env", "phase", "data_state"]

# 扩展维度在第 5 位起；此常量仅作文档/校验用，不限制 LLM 自由加维度
EXTENSION_DIM_START_POS = 5


def resolve_priority_order(
    source_coord: Optional[Dict[str, Any]],
    priority_order: Optional[List[str]],
    priority_override: Optional[List[str]] = None,
) -> Tuple[List[str], str]:
    """解析本次套用实际使用的优先级顺序（v2 P0-1 锚定继承）。

    解析优先级（高→低）：
      1. priority_override 非空 → 用它（显式变更，调用方需另写变更理由进 trace）
      2. priority_order 非空   → 用它（兼容 v1 显式传参）
      3. source_coord.priority_order 存在 → 继承首次套用定的锚（默认行为）
      4. 都没有 → 回落 CORE_DIMENSIONS（核心 4 维）

    Args:
        source_coord: 源场景坐标（继承锚点从此读 priority_order）
        priority_order: 调用方当场定的优先级（None 表示不重新定，读继承）
        priority_override: 显式覆盖（需写变更理由）。优先级最高

    Returns:
        (resolved_order, source)  source ∈ {"override","explicit","inherited","default"}
        供 trace 记录"本次优先级来源"，让锚定继承可审计。
    """
    if priority_override:
        return list(priority_override), "override"
    if priority_order:
        return list(priority_order), "explicit"
    inherited = None
    if isinstance(source_coord, dict):
        inherited = source_coord.get("priority_order")
    if inherited and isinstance(inherited, list) and inherited:
        return list(inherited), "inherited"
    return list(CORE_DIMENSIONS), "default"


def compute_sequence(
    source_coord: Optional[Dict[str, Any]],
    current_coord: Dict[str, Any],
    priority_order: Optional[List[str]] = None,
    *,
    priority_override: Optional[List[str]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """按优先级顺序算有序序列（v2 P0-1：支持锚定继承）。

    规矩死：序列有序，位置=优先级，第一位最重要
    LLM 活：priority_order 由 LLM 当场传入或继承首次锚定（resolve_priority_order 决定）

    v2 签名变更（向后兼容 v1）：
        v1: compute_sequence(source, current, priority_order)        # priority_order 必传
        v2: compute_sequence(source, current, priority_order=None)   # None 时走继承/默认

    Args:
        source_coord: 经验写入时的场景坐标。None/缺维度 → 对应位 '-'。
                      含 priority_order 字段时，后续套用默认继承（锚定）
        current_coord: 当前实时捕获的场景坐标
        priority_order: 当场定的优先级维度序列。None → 走继承/默认
        priority_override: 显式覆盖（优先级最高，需另写变更理由）

    Returns:
        (seq_str, explanations)
        seq_str: 有序字符串，如 "ABA" 或 "A-A"（- 表示源缺维度）
        explanations: 每位一条 dict，含 pos/letter/dim/similarity/reason，
                      v2 额外含 is_core（核心维度标记，供比较用）
    """
    resolved, _src = resolve_priority_order(
        source_coord, priority_order, priority_override,
    )
    if not resolved:
        return "", []

    seq_chars: List[str] = []
    explanations: List[Dict[str, Any]] = []
    core_set = {d for d in CORE_DIMENSIONS}
    for pos0, dim in enumerate(resolved, start=1):
        letter, sim, reason = _dim_distance(dim, source_coord, current_coord)
        seq_chars.append(letter)
        # is_core：维度名或其顶层前缀命中核心维度集合
        top = dim.split(".")[0]
        explanations.append({
            "pos": pos0,
            "letter": letter,
            "dim": dim,
            "similarity": round(sim, 3) if sim >= 0 else None,
            "reason": reason,
            "is_core": top in core_set,
        })
    return "".join(seq_chars), explanations


def compare_sequences(seq_a: str, seq_b: str) -> int:
    """变长序列横向比较（v2 P0-1：变长可比性）。

    规则（任务书 P0-1 #2）：
      核心维度匹配数优先（A 越多越靠前）；扩展维度作 bonus。

    约定：前 4 位为核心维度（与 CORE_DIMENSIONS 对齐），第 5 位起扩展。
    变长序列对齐到核心前缀比较；扩展位单独计 bonus。

    Returns:
        >0 → seq_a 更优；<0 → seq_b 更优；0 → 平
    """
    def _score(seq: str) -> Tuple[int, int]:
        if not seq:
            return (0, 0)
        core = seq[:len(CORE_DIMENSIONS)]
        ext = seq[len(CORE_DIMENSIONS):]
        core_a = sum(1 for c in core if c == "A")
        ext_a = sum(1 for c in ext if c == "A")
        return (core_a, ext_a)

    ca, ea = _score(seq_a)
    cb, eb = _score(seq_b)
    if ca != cb:
        return ca - cb
    return ea - eb


# ---------------------------------------------------------------------------
# §4 辅助：uncertainty 推断参考（非强制，LLM 最终自声明）
# ---------------------------------------------------------------------------

def suggest_uncertainty(seq: str) -> str:
    """给 LLM 一个参考 uncertainty（H/M/L），最终由 LLM 自声明。

    启发式（非死规则）：
      - 任何 '-' 源缺 → 至少 M（信息不全无法自信）
      - 首位非 A（最高优先级维度就有距离）→ H
      - 否则看非 A 位比例：<25% → L，<50% → M，>=50% → H

    Task 286+288 修订（Layer 1 前置）：
      原 H 触发阈值过严（含 `-` 时 non_a_ratio≥0.5，不含 `-` 时≥0.6），
      14 条 apply_attempt 没有一条满足——H=0 让 Layer 1 按 H 触发永远不会启动。
      调宽：含 `-` 时降到≥0.3，不含 `-` 时降到≥0.5。
      现有 14 条数据不受影响（都是 0.25 或 0.00），未来出现 0.3+ 的 seq 才会真实出 H。
    """
    if not seq:
        return "M"
    if "-" in seq:
        # 源坐标缺维度=信息不全，至少中不确定
        non_a_ratio = sum(1 for c in seq if c not in ("A", "-")) / len(seq)
        return "H" if non_a_ratio >= 0.3 else "M"  # Task 286+288: 0.5 → 0.3
    if seq[0] != "A":
        return "H"
    non_a_ratio = sum(1 for c in seq if c != "A") / len(seq)
    if non_a_ratio >= 0.5:  # Task 286+288: 0.6 → 0.5
        return "H"
    if non_a_ratio >= 0.25:
        return "M"
    return "L"


# ---------------------------------------------------------------------------
# §5 单元测试（python3 context_capture.py 自跑）
# ---------------------------------------------------------------------------

def _self_test() -> None:
    src = {
        "task_type": "protocol_analysis",
        "agent": {"name": "agent-a"},
        "env": {"os": "Darwin", "host": "mbp"},
        "data_state": {"pg_schema": "v1.3", "knowledge_rows": 21082},
    }
    cur_match = {
        "task_type": "protocol_analysis",
        "agent": {"name": "agent-a"},
        "env": {"os": "Darwin", "host": "mbp"},
        "data_state": {"pg_schema": "v1.3", "knowledge_rows": 21082},
    }
    cur_partial = {
        "task_type": "protocol_analysis",
        "agent": {"name": "agent-b"},          # 不同 agent
        "env": {"os": "Darwin", "host": "mbp"},
        "data_state": {"pg_schema": "v1.4", "knowledge_rows": 22500},
    }
    cur_missing = {"task_type": "search"}  # 大部分维度缺

    # 全一致 → AAAA
    seq, expl = compute_sequence(src, cur_match,
                                 ["task_type", "agent.name", "env.os", "data_state.pg_schema"])
    assert seq == "AAAA", f"全一致应得 AAAA，实际 {seq}"
    assert all(e["letter"] == "A" for e in expl)

    # 部分距离 → A B A C-ish
    seq2, _ = compute_sequence(src, cur_partial,
                               ["task_type", "agent.name", "env.os", "data_state.pg_schema"])
    assert seq2[0] == "A" and seq2[1] != "A" and seq2[2] == "A" and seq2[3] != "A", \
        f"部分距离序列异常: {seq2}"

    # 源缺维度 → '-'
    seq3, expl3 = compute_sequence(None, cur_match, ["task_type"])
    assert seq3 == "-", f"源坐标 None 应得 '-'，实际 {seq3}"
    assert expl3[0]["letter"] == "-"

    # 当前缺维度 → 'M'
    seq4, _ = compute_sequence(src, cur_missing, ["env.os"])
    assert seq4 == "M", f"当前缺维度应得 'M'，实际 {seq4}"

    # uncertainty 启发式
    assert suggest_uncertainty("AAAA") == "L"
    assert suggest_uncertainty("ABAA") == "M"
    assert suggest_uncertainty("DBAA") == "H"  # 首位非 A
    assert suggest_uncertainty("A-A") == "M"   # 源缺维度
    assert suggest_uncertainty("AZZZ") == "H"  # 非 A 占比高

    # similarity_to_letter 边界
    assert _similarity_to_letter(1.0) == "A"
    assert _similarity_to_letter(0.0) == "Z"
    assert _similarity_to_letter(0.999) == "B"
    assert _similarity_to_letter(0.001) == "Z"

    # === v2 P0-1: 优先级锚定继承 ===
    # 继承：source_coord 里有 priority_order → 不传也能读
    src_anchored = dict(src)
    src_anchored["priority_order"] = ["task_type", "env.os", "phase", "data_state.pg_schema"]
    seq_inh, expl_inh = compute_sequence(src_anchored, cur_match)  # priority_order=None
    # 继承了4维：task_type=A, env.os=A, phase='-'（src 缺 phase）, data_state.pg_schema=A
    assert len(seq_inh) == 4, f"继承应得4维序列，实际 {len(seq_inh)}: {seq_inh}"
    assert seq_inh[0] == "A" and seq_inh[1] == "A", f"前两位应 A，实际 {seq_inh}"
    assert seq_inh[2] == "-", f"phase 源缺应 '-'，实际 {seq_inh}"
    assert seq_inh[3] == "A", f"第4位应 A，实际 {seq_inh}"

    # 默认：source 无 priority_order、也不传 → 用 CORE_DIMENSIONS
    seq_def, expl_def = compute_sequence(src, cur_match)  # 全 None
    # src 含 task_type/env/phase(缺)/data_state → 核心4维中 phase 缺
    assert len(seq_def) == 4, f"默认核心4维长度应为4，实际 {len(seq_def)}: {seq_def}"
    assert expl_def[0]["is_core"] is True

    # override 优先级最高
    seq_ov, _ = compute_sequence(src_anchored, cur_match,
                                  priority_override=["task_type"])
    assert seq_ov == "A", f"override 应只用1维，实际 {seq_ov}"

    # resolve_priority_order 返回 source 标识
    _, s1 = resolve_priority_order(src_anchored, None, None)
    assert s1 == "inherited", f"应继承，实际 {s1}"
    _, s2 = resolve_priority_order(src, None, None)
    assert s2 == "default", f"应默认，实际 {s2}"
    _, s3 = resolve_priority_order(src, ["task_type"], None)
    assert s3 == "explicit", f"应显式，实际 {s3}"
    _, s4 = resolve_priority_order(src, None, ["env.os"])
    assert s4 == "override", f"应 override，实际 {s4}"

    # === v2 P0-1: 变长序列比较 ===
    # 核心匹配数优先：AAAA > AAAB（第4位非A → 核心3A < 核心4A）
    assert compare_sequences("AAAA", "AAAB") > 0
    # 核心相同看扩展：AAAAA（核心4A+扩展1A）> AAAA（核心4A+扩展0A）
    assert compare_sequences("AAAAA", "AAAA") > 0
    # 核心少一必输，无论扩展多少：AAAB（核心3A）< AAAA（核心4A）
    assert compare_sequences("AAAB", "AAAA") < 0
    # 核心相同扩展也相同 → 平
    assert compare_sequences("AAAA", "AAAA") == 0
    # 扩展位只数 A，非 A 不加分：AAAAZ（ext 0A）< AAAAA（ext 1A）
    assert compare_sequences("AAAAZ", "AAAAA") < 0
    # 扩展 A 数相同则平：AAAAZZ（ext 0A）== AAAAB（ext 0A）
    assert compare_sequences("AAAAZZ", "AAAAB") == 0

    print("[OK] context_capture self-test passed (含 v2 P0-1)")
    print(f"  全一致: {seq}")
    print(f"  部分距离: {seq2}")
    print(f"  源缺维度: {seq3}")
    print(f"  当前缺维度: {seq4}")


if __name__ == "__main__":
    _self_test()
