#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""semantic_pilot.py — Task1537 M2.5 语义边试点：候选对生成 + 复审决议落库。

管什么：embedding（BGE-M3/pgvector HNSW）预筛高相似知识对 → 输出候选 JSON
供 LLM 复审；复审决议（confidence=high）经 apply 落成 semantic 边
（evidence 带判断理由+confidence+相似度，created_by='semantic:1537'）。

在链路哪一环：位于"边框架（relation.py）"与"LLM 复审（人工/Agent 判断）"
之间——本脚本只做机械层（相似度召回/排除已有边/幂等落库），语义判断不外包
给规则（零错边红线：拿不准=不建）。

候选对过滤规则（陷阱地图对策）：
  - 排除 document 行/deprecated/无 embedding/已被取代（superseded_by 非空）
  - 排除同 task_id 对（同源任务条目天然高相似但未必"值得连"——陷阱1）
  - 排除已有任何类型边的对（双方向）
  - 相似度下界 SIM_FLOOR（BGE-M3 余弦，低于此值多为噪声）

方向语义：semantic 无方向，统一小 id 为 src（规范化存储防 (a,b)/(b,a) 双份）。

用法：
  python3 semantic_pilot.py candidates --anchors 100 --topn 3 --out candidates.json
  python3 semantic_pilot.py apply --decisions decisions.json --candidates candidates.json [--dry-run]
"""

import argparse
import json
import sys

import psycopg2
import psycopg2.extras

from nexusos_mem.relation import add_relation, ensure_schema
from nexusos_mem.shared import DEFAULT_DSN

# BGE-M3 余弦相似度下界（经验值：低于 0.55 多为"同词不同事"噪声对）
SIM_FLOOR = 0.55
# 相似度上界：>=此值疑似近重复/同文改写（留给 supersede 治理，不占语义边名额）
SIM_CEIL = 0.97
CREATED_BY = "semantic:1537"

# 核心知识类型（与 search.py core_only 同口径）——探针/影子/并发写等
# knowledge_type='memory' 的测试条目进不了池（陷阱1：同批次探针互为高相似
# 但纯属同源的机器产物，连了只会污染 riding）。
CORE_TYPES = ("experience", "finding", "decision", "pitfall")


def _connect():
    conn = psycopg2.connect(DEFAULT_DSN)
    conn.autocommit = True
    return conn


def _anchor_ids(conn, anchors: int) -> list:
    """锚点抽样：一半取最新（新知识最缺关联），一半全 id 域等距抽样（跨期多样）。

    只取非 document、未废弃、有 embedding、未被取代的条目（边端点必须"活"）。
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM knowledge "
            "WHERE knowledge_type = ANY(%s) AND deprecated = FALSE "
            "  AND embedding IS NOT NULL AND superseded_by IS NULL "
            "ORDER BY id", (list(CORE_TYPES),))
        all_ids = [r[0] for r in cur.fetchall()]
    if not all_ids:
        return []
    half = anchors // 2
    recent = all_ids[-half:] if half else []
    step = max(1, len(all_ids) // max(1, anchors - half))
    spread = all_ids[::step][: anchors - half]
    return sorted(set(recent) | set(spread), reverse=True)


def gen_candidates(conn, anchors: int, topn: int) -> dict:
    """逐锚点 HNSW top-N 近邻 → 过滤 → 规范化去重 → 候选对列表。"""
    anchor_ids = _anchor_ids(conn, anchors)
    stats = {"anchor_pool": len(anchor_ids), "raw_pairs": 0,
             "skip_sim": 0, "skip_same_task": 0, "skip_existing_edge": 0}
    pairs = {}  # (lo, hi) -> pair dict
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        for aid in anchor_ids:
            cur.execute(
                "SELECT id, title, project, tags, task_id, knowledge_type "
                "FROM knowledge WHERE id = %s", (aid,))
            a = cur.fetchone()
            if not a:
                continue
            # 近邻召回：同池过滤 + 相似度窗口 + 同源排除在 SQL 层做完
            cur.execute(
                """
                SELECT b.id, b.title, b.project, b.tags, b.task_id,
                       b.knowledge_type,
                       1 - (b.embedding <=> a.embedding) AS sim
                FROM knowledge a, knowledge b
                WHERE a.id = %s
                  AND b.id <> a.id
                  AND b.knowledge_type = ANY(%s) AND b.deprecated = FALSE
                  AND b.embedding IS NOT NULL AND b.superseded_by IS NULL
                  AND NOT (b.task_id IS NOT NULL AND a.task_id IS NOT NULL
                           AND b.task_id = a.task_id)
                ORDER BY b.embedding <=> a.embedding
                LIMIT %s
                """,
                (aid, list(CORE_TYPES), topn * 3),  # 多召回兜底相似度窗口过滤后的损耗
            )
            kept = 0
            for b in cur.fetchall():
                if kept >= topn:
                    break
                stats["raw_pairs"] += 1
                sim = float(b["sim"])
                if not (SIM_FLOOR <= sim < SIM_CEIL):
                    stats["skip_sim"] += 1
                    continue
                lo, hi = (a["id"], b["id"]) if a["id"] < b["id"] else (b["id"], a["id"])
                if (lo, hi) in pairs:
                    continue
                # 排除已有任何类型边（双方向）
                cur.execute(
                    "SELECT 1 FROM knowledge_relation "
                    "WHERE (src_id = %s AND dst_id = %s) "
                    "   OR (src_id = %s AND dst_id = %s) LIMIT 1",
                    (lo, hi, hi, lo))
                if cur.fetchone():
                    stats["skip_existing_edge"] += 1
                    continue
                pairs[(lo, hi)] = {
                    "a_id": lo, "b_id": hi, "sim": round(sim, 4),
                    "a_title": (a if a["id"] == lo else b)["title"],
                    "b_title": (b if b["id"] == hi else a)["title"],
                    "a_project": (a if a["id"] == lo else b)["project"],
                    "b_project": (b if b["id"] == hi else a)["project"],
                    "a_type": (a if a["id"] == lo else b)["knowledge_type"],
                    "b_type": (b if b["id"] == hi else a)["knowledge_type"],
                }
                kept += 1
    stats["candidates"] = len(pairs)
    return {"stats": stats, "pairs": sorted(pairs.values(), key=lambda p: -p["sim"])}


def enrich_snippets(conn, pairs: list, width: int = 120) -> None:
    """给候选对补正文摘要（LLM 复审要看内容判断，仅标题不够）。"""
    ids = sorted({p["a_id"] for p in pairs} | {p["b_id"] for p in pairs})
    if not ids:
        return
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, left(content, %s) FROM knowledge WHERE id = ANY(%s)",
            (width, ids))
        snip = dict(cur.fetchall())
    for p in pairs:
        p["a_snippet"] = (snip.get(p["a_id"]) or "").replace("\n", " ")
        p["b_snippet"] = (snip.get(p["b_id"]) or "").replace("\n", " ")


def apply_decisions(conn, decisions: list, dry_run: bool = False) -> dict:
    """复审决议落库：仅 confidence=high 建边（宁少勿滥），幂等 ON CONFLICT。"""
    ensure_schema(conn)
    stats = {"decisions": len(decisions), "high": 0, "inserted": 0,
             "existing": 0, "skip_endpoint": 0, "skip_not_high": 0}
    with conn.cursor() as cur:
        for d in decisions:
            if d.get("confidence") != "high":
                stats["skip_not_high"] += 1
                continue
            stats["high"] += 1
            lo, hi = sorted((int(d["a_id"]), int(d["b_id"])))
            cur.execute(
                "SELECT count(*) FROM knowledge "
                "WHERE id IN (%s, %s) AND deprecated = FALSE", (lo, hi))
            if cur.fetchone()[0] != 2:
                stats["skip_endpoint"] += 1
                continue
            if dry_run:
                stats["inserted"] += 1
                continue
            evidence = (f"conf=high; sim={d.get('sim')}; {d.get('reason', '')}")[:500]
            res = add_relation(conn, lo, hi, "semantic",
                               source=CREATED_BY, evidence=evidence)
            if res.get("inserted"):
                stats["inserted"] += 1
            elif res.get("ok"):
                stats["existing"] += 1
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates", help="生成候选对 JSON")
    c.add_argument("--anchors", type=int, default=100)
    c.add_argument("--topn", type=int, default=3)
    c.add_argument("--out", required=True)
    a = sub.add_parser("apply", help="复审决议落库")
    a.add_argument("--decisions", required=True)
    a.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conn = _connect()
    try:
        if args.cmd == "candidates":
            result = gen_candidates(conn, args.anchors, args.topn)
            enrich_snippets(conn, result["pairs"])
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=1)
            print(json.dumps(result["stats"], ensure_ascii=False))
        else:
            with open(args.decisions, encoding="utf-8") as f:
                decisions = json.load(f)
            stats = apply_decisions(conn, decisions, dry_run=args.dry_run)
            print(json.dumps(stats, ensure_ascii=False))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
