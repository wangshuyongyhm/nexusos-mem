#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""knowledge_dig.py — Task1538 消费固化①：知识考古（一条命令看全前世今生）。

输入 MEM#N 或 vault 相对路径 → 解析 identity → 输出：
  ① 演进链：supersede 链（双向，深度上限 10——历史脏数据防环）
  ② 血缘链：provenance 边（产生于哪个 task/report/会话——读模型统一身份空间）
  ③ 关联边：relates（citation/互链）+ semantic（1537 前瞻）
只读查询，零写入。
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import psycopg2  # noqa: E402
from nexusos_mem.nexus_mem_utils import get_dsn  # noqa: E402

MAX_DEPTH = 10


def resolve(cur, arg):
    """MEM#N / 裸数字 / vault 路径 → (identity_id, label, kind)。"""
    m = re.fullmatch(r"MEM#(\d+)", arg) or re.fullmatch(r"(\d+)", arg)
    if m:
        cur.execute(
            "SELECT i.id, k.title, i.identity_kind FROM knowledge_identity i "
            "JOIN knowledge k ON k.id=i.mem_id WHERE i.identity_kind='mem' AND i.mem_id=%s",
            (int(m.group(1)),))
    else:
        cur.execute(
            "SELECT id, vault_path, identity_kind FROM knowledge_identity "
            "WHERE vault_path=%s", (arg,))
    row = cur.fetchone()
    return row


def supersede_chain(cur, kid):
    """supersede 双向链，深度上限+防环。"""
    chain = [{"id": kid, "dir": "self"}]
    seen = {kid}
    # 向新版（我的 superseded_by）
    cur_id, depth = kid, 0
    while depth < MAX_DEPTH:
        cur.execute("SELECT superseded_by FROM knowledge WHERE id=%s", (cur_id,))
        r = cur.fetchone()
        nxt = r[0] if r else None
        if not nxt or nxt == cur_id or nxt in seen:
            break
        chain.append({"id": nxt, "dir": "superseded_by →"})
        seen.add(nxt)
        cur_id, depth = nxt, depth + 1
    # 向旧版（谁被我取代）
    cur_id, depth = kid, 0
    while depth < MAX_DEPTH:
        cur.execute("SELECT id FROM knowledge WHERE superseded_by=%s ORDER BY id LIMIT 1", (cur_id,))
        r = cur.fetchone()
        if not r or r[0] in seen:
            break
        chain.insert(0, {"id": r[0], "dir": "← 取代我"})
        seen.add(r[0])
        cur_id, depth = r[0], depth + 1
    return chain


def main():
    if len(sys.argv) != 2:
        print("用法: knowledge-dig.sh <MEM#N | vault相对路径>", file=sys.stderr)
        return 2
    conn = psycopg2.connect(get_dsn())
    cur = conn.cursor()
    row = resolve(cur, sys.argv[1])
    if not row:
        print(f"✗ 无法解析: {sys.argv[1]}", file=sys.stderr)
        return 2
    ident, label, kind = row
    print(f"◎ {sys.argv[1]} → identity#{ident} [{kind}] {label[:60] if label else ''}")

    kid = None
    if kind == "mem":
        cur.execute("SELECT mem_id FROM knowledge_identity WHERE id=%s", (ident,))
        kid = cur.fetchone()[0]
        print("\n== 演进链（supersede，深度≤10） ==")
        for c in supersede_chain(cur, kid):
            cur.execute("SELECT coalesce(title,'?'), outcome FROM knowledge WHERE id=%s", (c["id"],))
            t, oc = cur.fetchone()
            print(f"  {c['dir']:<14} #{c['id']} [{oc or '-'}] {t[:52]}")

    print("\n== 血缘链（provenance，产生于） ==")
    cur.execute(
        "SELECT e.dst_identity, n.kind, n.vault_path, n.mem_id, left(e.evidence,60) "
        "FROM readmodel_edge e JOIN readmodel_node n ON n.identity_id=e.dst_identity "
        "WHERE e.src_identity=%s AND e.edge_kind='provenance'", (ident,))
    rows = cur.fetchall()
    for r in rows or []:
        tgt = r[2] or (f"knowledge#{r[3]}" if r[3] else "?")
        print(f"  → {r[1]:<10} {tgt[:64]}  ({r[4]})")
    if not rows:
        print("  （无血缘边）")

    print("\n== 关联边（relates/semantic） ==")
    cur.execute(
        "SELECT e.dst_identity, e.edge_kind, n.kind, n.vault_path, n.mem_id "
        "FROM readmodel_edge e JOIN readmodel_node n ON n.identity_id=e.dst_identity "
        "WHERE e.src_identity=%s AND e.edge_kind IN ('relates','semantic') "
        "UNION ALL "
        "SELECT e.src_identity, e.edge_kind, n.kind, n.vault_path, n.mem_id "
        "FROM readmodel_edge e JOIN readmodel_node n ON n.identity_id=e.src_identity "
        "WHERE e.dst_identity=%s AND e.edge_kind IN ('relates','semantic')", (ident, ident))
    rows = cur.fetchall()
    for r in rows or []:
        tgt = r[3] or (f"knowledge#{r[4]}" if r[4] else "?")
        print(f"  ↔ {r[1]:<9} {tgt[:66]}")
    if not rows:
        print("  （无关联边）")

    print("\n== 被谁引用（wikilink 入边，前 8） ==")
    cur.execute(
        "SELECT s.vault_path FROM readmodel_edge e JOIN readmodel_node s ON s.identity_id=e.src_identity "
        "WHERE e.dst_identity=%s AND e.edge_kind='wikilink' LIMIT 8", (ident,))
    for r in cur.fetchall() or []:
        print(f"  ← {r[0][:70]}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
