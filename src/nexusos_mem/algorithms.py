#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nexus.algorithms — NEXUS-MEM 纯函数算法库

集中所有可独立测试的算法，替代散落在各脚本的重复实现（task 225 §1.2）。

包含：
    - 熵预算公式（替代 shadow-sync.py + entropy-budget.py 各一份）
    - RL 检索概率 + 参数更新（替代 rl-feedback.py）
    - 层级升降提案（替代 shadow-sync.py）
    - sigmoid 数值稳定实现

公式来源：NEXUS-MEM-架构.md §三 §七
"""

import math
from typing import Dict, Tuple


# 默认参数（与各脚本历史默认值保持一致）
DEFAULT_T_TOTAL = 1000.0
DEFAULT_LAMBDA = 0.01
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_NUMERICAL_EPS = 1e-4

DEFAULT_RL_PARAMS = {
    "alpha": 0.40,   # 语义相似度权重
    "beta": 0.30,    # 引用频率权重
    "gamma": 0.20,   # 时间衰减权重
    "delta": 0.10,   # 层级偏好权重
    "lambda": 0.01,  # 时间衰减系数
}


# ---------------------------------------------------------------------------
# 熵预算公式
# ---------------------------------------------------------------------------

def compute_entropy_score(
    R: float,
    W: float,
    delta_hours: float,
    lam: float = DEFAULT_LAMBDA,
) -> float:
    """熵预算核心公式（task 252 修正版）。

    旧公式: score = W × R^1.5 × exp(-λ × Δt)
        问题: R=0 时短路返回 0.0 → 82/84 条 R=0 全平局，按字母序灌满 top-K，
              熵预算模型失效（task 250 审计发现）。

    新公式: score = W × R^1.5 × exp(-λ × Δt) + last_access_score
        其中 last_access_score = 0.01 × exp(-λ × Δt)  (R=0 条目的区分度来源)

    Args:
        R: 引用计数（自动 clamp 到 ≥0）
        W: 权重
        delta_hours: 距上次使用的小时数（自动 clamp 到 ≥0）
        lam: 衰减系数（默认 0.01，100h 衰减 ~63%）

    Returns:
        原始分数（未归一化）。R>0 条目受 R^1.5 主导；R=0 条目按最近访问时间区分。

    References:
        NEXUS-MEM-架构.md §三 公式 (1)
        task 252 P0-2: 加 last_access_score 修复熵预算失效
    """
    R = max(R, 0)
    delta_hours = max(delta_hours, 0)
    # 主项: 引用计数驱动 (R>0 条目)
    main_score = W * (R ** 1.5) * math.exp(-lam * delta_hours)
    # 副项: 最近访问时间 (R=0 条目间的区分度, 0.01 系数保证不喧宾夺主)
    last_access_score = 0.01 * math.exp(-lam * delta_hours)
    return main_score + last_access_score


def allocate_entropy_budget(
    scores: Dict[str, float],
    t_total: float = DEFAULT_T_TOTAL,
) -> Tuple[Dict[str, float], float]:
    """将原始 scores 归一化为预算分配。

    Args:
        scores: {code: raw_score}
        t_total: 总预算（默认 1000 tokens）

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


# ---------------------------------------------------------------------------
# [DEPRECATED Task 369, 2026-07-28] RL 检索概率 + 参数更新
# 弃用原因：RL 是给 SHADOW 编码体系用的（检索概率更新），SHADOW 于 2026-07-23
# 退役（shared.py 注释确认），RL 无消费方。rl_history 停在 07-09（19天前）。
# 接口保留做审计，rl_history/rl_params 数据保留但不再写入。
# ---------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    """数值稳定的 sigmoid 函数。

    替代 rl-feedback.py:108 的实现。避免大数溢出。
    """
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def compute_retrieval_probability(
    sim: float,
    R: float,
    R_max: float,
    delta_t: float,
    layer: str,
    params: dict,
) -> float:
    """RL 检索概率公式：P = σ(α·sim + β·(R/R_max) + γ·e^(-λΔt) + δ·1_{layer≤Z})。

    替代 rl-feedback.py:118 的实现。

    Args:
        sim: 语义相似度 [0, 1]
        R: 引用计数
        R_max: 最大引用计数（用于归一化）
        delta_t: 距上次使用的小时数
        layer: 编码层级（"L0"/"L1"/"L2"/"L3"）
        params: RL 参数 {alpha, beta, gamma, delta, lambda}

    Returns:
        检索概率 [0, 1]

    References:
        NEXUS-MEM-架构.md §七 公式 (2)
    """
    alpha = params.get("alpha", DEFAULT_RL_PARAMS["alpha"])
    beta = params.get("beta", DEFAULT_RL_PARAMS["beta"])
    gamma = params.get("gamma", DEFAULT_RL_PARAMS["gamma"])
    delta = params.get("delta", DEFAULT_RL_PARAMS["delta"])
    lam = params.get("lambda", DEFAULT_RL_PARAMS["lambda"])

    term_sim = alpha * sim
    term_R = beta * (R / R_max if R_max > 0 else 0.0)
    term_time = gamma * math.exp(-lam * delta_t)
    layer_bonus = delta * 1.0 if layer in ("L0", "L1") else 0.0

    z = term_sim + term_R + term_time + layer_bonus
    return sigmoid(z)


def update_rl_params(
    params: dict,
    sim: float,
    R: float,
    R_max: float,
    delta_t: float,
    layer: str,
    feedback: float,
    lr: float = DEFAULT_LEARNING_RATE,
) -> dict:
    """数值梯度法更新 RL 参数。

    替代 rl-feedback.py:146 的实现。

    Args:
        params: 当前参数
        sim/R/R_max/delta_t/layer: 与 compute_retrieval_probability 相同
        feedback: 反馈值 [-1, 1]，正=正纠正，负=负纠正
        lr: 学习率（默认 0.05）

    Returns:
        新参数 dict（不修改原 dict）

    Notes:
        目标概率 P_target = 0.5 + feedback × 0.5（feedback=+1 → P=1.0）
        损失 L = (P - P_target)^2
        对每个参数做数值梯度下降。
    """
    new_params = dict(params)
    target = max(0.0, min(1.0, 0.5 + feedback * 0.5))

    p = compute_retrieval_probability(sim, R, R_max, delta_t, layer, params)
    dL_dP = 2.0 * (p - target)

    eps = DEFAULT_NUMERICAL_EPS
    for pname in ("alpha", "beta", "gamma", "delta"):
        p_plus = dict(params)
        p_plus[pname] = params.get(pname, 0.0) + eps
        P_plus = compute_retrieval_probability(sim, R, R_max, delta_t, layer, p_plus)
        dP_dparam = (P_plus - p) / eps
        grad = dL_dP * dP_dparam
        new_val = params.get(pname, 0.0) - lr * grad
        new_params[pname] = max(0.0, min(1.0, new_val))

    return new_params


# ---------------------------------------------------------------------------
# 层级升降提案
# ---------------------------------------------------------------------------

def generate_proposals(
    state_entries: Dict[str, dict],
    budget: Dict[str, float],
    now=None,
    l1_slots: int = 20,
) -> dict:
    """生成层级升降提案（替代 shadow-sync.py:243）。

    Args:
        state_entries: {code: {layer, R, W, last_used, consecutive_sessions, ...}}
        budget: {code: B_i}（来自 allocate_entropy_budget）
        now: 当前 datetime（None 表示现在）
        l1_slots: L1 最大槽位（默认 20）

    Returns:
        {upgrade: [...], downgrade: [...], retain: [...], new_arrivals: [...], timestamp: ...}
    """
    from datetime import datetime, timezone
    if now is None:
        now = datetime.now(timezone.utc)

    l1_entries = {c: e for c, e in state_entries.items() if e.get("layer") == "L1"}
    l2_entries = {c: e for c, e in state_entries.items() if e.get("layer") == "L2"}

    proposals = {
        "upgrade": [],
        "downgrade": [],
        "retain": [],
        "new_arrivals": [],
        "timestamp": now.isoformat(),
    }

    # L2 升级候选：R>=5 且 consecutive>=2 且 budget>20
    l2_sorted = sorted(l2_entries.items(), key=lambda x: budget.get(x[0], 0), reverse=True)
    available_slots = max(l1_slots - len(l1_entries), 0)

    for code, ent in l2_sorted:
        b = budget.get(code, 0)
        if (
            ent.get("R", 0) >= 5
            and ent.get("consecutive_sessions", 0) >= 2
            and b > 20
        ):
            proposals["upgrade"].append({
                "code": code,
                "target": ent.get("target", ""),
                "R": ent.get("R", 0),
                "B_i": round(b, 2),
            })
        else:
            proposals["retain"].append({"code": code, "layer": "L2", "B_i": round(b, 2)})

    # L1 降级候选：R==0 闲置 >48h，或 budget<5
    for code, ent in l1_entries.items():
        b = budget.get(code, 0)
        last_used = ent.get("last_used", "")
        try:
            from datetime import datetime as _dt
            last_dt = _dt.fromisoformat(last_used.replace("Z", "+00:00")) if last_used else now
            idle_hours = (now - last_dt).total_seconds() / 3600.0
        except (ValueError, TypeError):
            idle_hours = float("inf")

        if (ent.get("R", 0) == 0 and idle_hours > 48) or b < 5:
            proposals["downgrade"].append({
                "code": code,
                "target": ent.get("target", ""),
                "R": ent.get("R", 0),
                "B_i": round(b, 2),
                "idle_hours": round(idle_hours, 1) if idle_hours != float("inf") else -1,
            })
        else:
            proposals["retain"].append({"code": code, "layer": "L1", "B_i": round(b, 2)})

    # 截断升级到可用槽位
    if len(proposals["upgrade"]) > available_slots:
        proposals["upgrade"].sort(key=lambda x: x["B_i"], reverse=True)
        proposals["upgrade"] = proposals["upgrade"][:available_slots]

    return proposals
