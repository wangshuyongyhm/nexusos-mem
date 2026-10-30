#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nexus.shadow_match — knowledge → SHADOW 编码 确定性匹配（v2）

v2 变更（2026-07-14）：
  - _path_prefix_code 从硬编码 19+ if 分支改为动态读取 SHADOW-索引.md
  - 路径映射表 {关键词: 编码} 从索引文件自动构建，文件改名/重组自动适配
  - 兜底码从 BZ/ZE 改为新方案的 E（L4 uncategorized）
  - 命名空间推断更新为 NEXUS-CONSTRAINT 体系
  - 同层级码长优先规则保留（打分逻辑不变，仅映射源改变）

设计原则（v1 保留）:
    - 确定性优先，不用模糊语义
    - 主题优先于结构
    - 动态读表 —— 关键词从 shadow_codes 表的 section/target 派生
    - 保底 100% 覆盖

v2 新增:
    - 路径映射也动态读 —— 从 SHADOW-索引.md 构建 {target_fragment: code} 反查表
    - 新增 _load_path_map() 统一路径映射逻辑

匹配优先级（分数最高者胜）:
    1. tag 精确等于某码的 section  → +5
    2. path 前缀命中结构规则        → +4
    3. section 出现在 title         → +3
    4. project 命中 target 中的项目段 → +2
    5. section 出现在 content 前段   → +1
    → 全 0 分则走 path_code → project 默认码 → E（兜底）
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple  # Task 281 P2-22: 加 Any（_path_map_cache 实际 3 元组含 Any）

# ── 路径常量 ───────────────────────────────────────────────
# 2026-07-24 统一数据源：从 constants.SHADOW_INDEX_PATH 读（不再自己拼路径）
from nexusos_mem._vault_env import SHADOW_INDEX_PATH as _SHADOW_INDEX  # Task668: 包内自足

# 结构性 section（非技术主题，仅作弱信号）
_STRUCTURAL_SECTIONS = {
    "exp", "session", "sessions", "docs", "daily", "archive", "template",
    "basic", "data", "research", "install", "tech", "system", "env",
    "v1", "cyd", "openclaw", "clawith", "hotel", "langman", "sifu",
    "ops", "tool", "deploy", "jupyter", "macos", "agent", "email", "copy",
}

# v2 兜底码：E=L4(uncategorized)，替代旧的 BZ/ZE
_FALLBACK = "E"

# 泛结构码（"杂项文件夹"）：path 命中时权重降低
# v2: 动态从 L3/L4 层级推导，不再硬编码
_GENERIC_LAYERS = {"L3", "L4", "L5"}


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


# ── 动态路径映射（v2 核心）─────────────────────────────────

_path_map_cache: Optional[List[Tuple[str, str, Any]]] = None  # Task 281 P2-22: 改 2 元组 → 3 元组（实际 append 3 元组，第 3 项是 Any 类型元数据）


def _load_path_map() -> List[Tuple[str, str, str]]:
    """从 SHADOW-索引.md 构建 [(关键词, 编码, 层级)] 反查表。

    遍历所有编码的 target，提取可用于路径匹配的关键词。
    按 target 长度降序排列（更具体的优先匹配）。

    Returns:
        [(keyword, code, layer), ...] 按 keyword 长度降序
    """
    global _path_map_cache
    if _path_map_cache is not None:
        return _path_map_cache

    # 延迟 import 避免循环依赖(Task631 迁包: parser 同包, 无需 sys.path hack)
    from nexusos_mem.parser import parse_shadow_index

    if _SHADOW_INDEX is None:
        from nexusos_common import ConfigError, VAULT_MISSING_REMEDIATION
        raise ConfigError("vault 不可用，SHADOW 索引不可读。" + VAULT_MISSING_REMEDIATION)
    entries = parse_shadow_index(_SHADOW_INDEX)

    triples: List[Tuple[str, str, str]] = []
    for code, meta in entries.items():
        target = meta.get("target", "")
        layer = meta.get("layer", "L1")
        # 提取 § 前的路径部分作为关键词
        base = target.split("§")[0].strip()
        if not base or len(base) < 2:
            continue
        triples.append((base, code, layer))

    # 按 keyword 长度降序（更具体的路径优先匹配）
    triples.sort(key=lambda t: -len(t[0]))

    _path_map_cache = triples
    return triples


def _path_prefix_code(path: Optional[str], project: Optional[str],
                      code_set: set) -> Optional[str]:
    """v2: 动态路径前缀 → 编码映射。

    遍历 _load_path_map() 的反查表，找到第一个 path 包含的 keyword。
    """
    if not path:
        return None

    pl = path.lower()
    p = path  # 保留大小写用于中文匹配

    triples = _load_path_map()
    for keyword, code, layer in triples:
        kw_lower = keyword.lower()
        # 双重匹配：小写路径包含 或 原始路径包含（中文）
        if kw_lower in pl or keyword in p:
            if code in code_set:
                return code

    return None


def load_shadow_codes(conn) -> List[dict]:
    """读 shadow_codes 表（code/layer/target/section）。"""
    import psycopg2.extras
    # Task 379: 接收外部 conn,只读 SELECT。若 conn 非 autocommit 则临时切换+恢复
    # (调用方多为 assign_shadow_code 的写路径内,不能改连接所有权,故就地保形)
    _ac = getattr(conn, "autocommit", False)
    _needs_restore = (not _ac)
    if _needs_restore:
        try: conn.autocommit = True
        except Exception: _needs_restore = False
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            cur.execute("SELECT code, layer, target, section FROM shadow_codes")
            return [dict(r) for r in cur.fetchall()]
        finally:
            try: cur.close()
            except Exception: pass
    finally:
        if _needs_restore:
            try: conn.autocommit = _ac
            except Exception: pass


def _section_of(code_row: dict) -> str:
    """取该码的主题 section：优先 section 字段，否则从 target 的 §后缀取。"""
    sec = _norm(code_row.get("section"))
    if sec:
        return sec
    target = code_row.get("target") or ""
    if "§" in target:
        return _norm(target.rsplit("§", 1)[-1])
    return ""


# 项目 → 默认兜底码（v2 更新为新编码体系）
# Task 282 P2-6: _PROJECT_DEFAULT + _PROJECT_NS 外置到 config/project_mapping.yaml
def _load_project_mapping():
    """从 config/project_mapping.yaml 加载项目映射，失败降级空字典。"""
    cfg = Path(__file__).parent / "config" / "project_mapping.yaml"
    if cfg.exists():
        try:
            import yaml
            return yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        except Exception:
            pass
    return {}

_PM_CFG = _load_project_mapping()
_PROJECT_DEFAULT: dict = _PM_CFG.get("project_default", {})
_PROJECT_NS: dict = _PM_CFG.get("project_ns", {})


def _ns_for_text(text: str) -> Optional[str]:
    """按 config/project_mapping.yaml 的 namespace_rules 推断命名空间。

    规则数据驱动（新增私项目只改 yaml 不改代码）：按序匹配 keyword（子串）
    或 keywords_any（任一命中），首条命中即返回 ns；无命中返回 None。
    """
    for rule in _PM_CFG.get("namespace_rules", []):
        if "keywords_any" in rule:
            if any(_norm(k) in _norm(text) for k in rule.get("keywords_any", [])):
                return rule.get("ns")
        keyword = rule.get("keyword")
        if keyword and _norm(keyword) in _norm(text):
            return rule.get("ns")
    return None


def _code_namespace(target_l: str) -> Optional[str]:
    """从 target 推断编码所属命名空间（跨项目误配防护）——namespace_rules 驱动。"""
    return _ns_for_text(target_l)


def assign_shadow_code(
    conn,
    title: str = "",
    content: str = "",
    tags: Optional[List[str]] = None,
    project: Optional[str] = None,
    path: Optional[str] = None,
    knowledge_type: Optional[str] = None,
    _codes_cache: Optional[List[dict]] = None,
) -> str:
    """给一条 knowledge 确定性分配 SHADOW 编码。保底返回 E，绝不返回 None。

    v2: 路径映射从 SHADOW-索引.md 动态加载。
    """
    codes = _codes_cache if _codes_cache is not None else load_shadow_codes(conn)
    code_set = {c["code"] for c in codes}

    tags_l = {_norm(t) for t in (tags or [])}
    title_l = _norm(title)
    content_l = _norm(content)[:800]

    path_code = _path_prefix_code(path, project, code_set)

    # 命名空间推断（namespace_rules 驱动；结构目录 05-system 作体系兜底信号）
    pl = _norm(path)
    my_ns = _ns_for_text(pl)
    if my_ns is None and "05-system" in pl:
        my_ns = "system"
    if my_ns is None:
        my_ns = _PROJECT_NS.get(_norm(project))

    best_code = None
    best_score = 0.0
    for c in codes:
        code = c["code"]
        layer = (c.get("layer") or "").strip()
        sec = _section_of(c)
        target_l = _norm(c.get("target"))
        code_ns = _code_namespace(target_l)
        score = 0.0

        ns_ok = (code_ns is None) or (my_ns is None) or (code_ns == my_ns)
        sec_substr_ok = bool(sec) and sec.isascii() and sec.isalpha() and len(sec) >= 3

        if ns_ok and sec and sec not in _STRUCTURAL_SECTIONS:
            if sec in tags_l:
                score += 5.0
            elif sec_substr_ok and sec in title_l:
                score += 3.0
            elif sec_substr_ok and sec in content_l:
                score += 1.0
        elif ns_ok and sec and sec in tags_l:
            score += 1.5

        if project and _norm(project) in target_l:
            score += 2.0

        # path 前缀命中：泛结构码（L3/L4）降权
        if path_code and code == path_code:
            score += 1.5 if layer in _GENERIC_LAYERS else 4.0

        if score > best_score:
            best_score = score
            best_code = code
        elif score == best_score and best_code is not None and score > 0:
            # 同分：码更长更优先（v2 新规则：A < AA < AAA）
            if len(code) > len(best_code):
                best_code = code
            elif len(code) == len(best_code) and code > best_code:
                best_code = code

    if best_score > 0 and best_code:
        return best_code

    # 无任何命中：路径码 → 项目默认 → E（兜底）
    if path_code:
        return path_code
    proj_default = _PROJECT_DEFAULT.get(_norm(project))
    if proj_default and proj_default in code_set:
        return proj_default
    return _FALLBACK if _FALLBACK in code_set else (codes[0]["code"] if codes else "E")


def backfill(conn, dry_run: bool = True, only_null: bool = True) -> dict:
    """回填 knowledge 表的 shadow_code。

    v2: 兜底码从 BZ/ZE 改为 E。
    """
    import psycopg2.extras
    codes = load_shadow_codes(conn)

    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    where = "WHERE shadow_code IS NULL" if only_null else ""
    cur.execute(f"""
        SELECT id, title, content, tags, project, path, knowledge_type
        FROM knowledge {where}
    """)
    rows = [dict(r) for r in cur.fetchall()]
    cur.close()

    by_code: Dict[str, int] = {}
    updates = []
    for r in rows:
        code = assign_shadow_code(
            conn,
            title=r.get("title") or "",
            content=r.get("content") or "",
            tags=r.get("tags") or [],
            project=r.get("project"),
            path=r.get("path"),
            knowledge_type=r.get("knowledge_type"),
            _codes_cache=codes,
        )
        by_code[code] = by_code.get(code, 0) + 1
        updates.append((code, r["id"]))

    updated = 0
    if not dry_run and updates:
        wcur = conn.cursor()
        wcur.executemany(
            "UPDATE knowledge SET shadow_code = %s, updated = NOW() WHERE id = %s",
            updates,
        )
        conn.commit()
        updated = wcur.rowcount
        wcur.close()

    return {
        "scanned": len(rows),
        "assigned": len(updates),
        "by_code": dict(sorted(by_code.items(), key=lambda x: -x[1])),
        "updated": updated,
    }


def refresh_path_map_cache() -> None:
    """清除路径映射缓存（SHADOW-索引.md 更新后调用）。"""
    global _path_map_cache
    _path_map_cache = None


if __name__ == "__main__":
    import sys
    import psycopg2
    import os

    dsn = os.environ.get("NEXUS_PG_DSN", "postgresql://localhost/nexus_mem")
    dry = "--apply" not in sys.argv
    only_null = "--all" not in sys.argv

    # 支持子命令
    if "--test-parse" in sys.argv:
        # 测试 SHADOW-索引.md 解析 + 路径映射
        triples = _load_path_map()
        print(f"路径映射表: {len(triples)} 条")
        for kw, code, layer in triples[:10]:
            print(f"  {code:4s} ({layer}) ← {kw}")
        sys.exit(0)

    conn = psycopg2.connect(dsn)
    result = backfill(conn, dry_run=dry, only_null=only_null)
    conn.close()

    print("=" * 60)
    print(f"{'[DRY RUN] ' if dry else ''}shadow_code 回填"
          f"（{'仅 NULL' if only_null else '全部'}）")
    print("=" * 60)
    print(f"扫描: {result['scanned']}  分配: {result['assigned']}  "
          f"实际写入: {result['updated']}")
    print("\n按编码分布:")
    for code, n in result["by_code"].items():
        print(f"  {code:<4} {n}")
