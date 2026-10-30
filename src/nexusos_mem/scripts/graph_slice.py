#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""graph_slice.py — Task1538 消费固化②：落盘图谱切片（会话复盘可视化）。

给定 task 目录 → 找该任务的知识节点（provenance 边指向本任务 task.md/report.md
的 mem 节点 + task_id 列命中行）→ 导出一跳子图为 mermaid + wikilink 列表，
写 task/NN/output/graph-slice.md。只读。
用法：python3 graph_slice.py <task_dir相对路径> [--out 覆盖输出路径]
"""
import datetime as _dt
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import psycopg2  # noqa: E402
from nexusos_mem._vault_env import VAULT_ROOT  # noqa: E402
from nexusos_mem.nexus_mem_utils import get_dsn  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print("用法: graph_slice.py <task_dir> [--out PATH]", file=sys.stderr)
        return 2
    task_dir = sys.argv[1].strip("/")
    out_override = None
    if "--out" in sys.argv:
        out_override = sys.argv[sys.argv.index("--out") + 1]
    conn = psycopg2.connect(get_dsn())
    cur = conn.cursor()

    # 任务身份节点（task.md / report.md / 全目录下已注册 vault_doc）
    cur.execute(
        "SELECT id, vault_path FROM knowledge_identity "
        "WHERE identity_kind IN ('vault_doc','graph_card','mirror_shard') "
        "AND (vault_path LIKE %s OR vault_path LIKE %s)",
        (f"{task_dir}/%", f"{task_dir}.md"))
    task_nodes = {r[0]: r[1] for r in cur.fetchall()}
    if not task_nodes:
        print(f"✗ 任务目录无身份节点: {task_dir}", file=sys.stderr)
        return 2

    # 知识节点：provenance 边连到任务节点的 mem（读模型统一空间）
    cur.execute(
        "SELECT DISTINCT n.mem_id FROM readmodel_edge e "
        "JOIN readmodel_node n ON n.identity_id=e.src_identity AND n.kind='mem' "
        "WHERE e.edge_kind='provenance' AND e.dst_identity = ANY(%s)",
        (list(task_nodes),))
    mem_ids = [r[0] for r in cur.fetchall()]

    lines = [
        "---",
        f"task: {task_dir}",
        f"generated: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "type: graph-slice",
        "---",
        "",
        f"# 图谱切片 · {task_dir}",
        "",
        f"知识节点 {len(mem_ids)} 个（provenance 溯源）｜任务文件节点 {len(task_nodes)} 个",
        "",
    ]
    if mem_ids:
        cur.execute(
            "SELECT id, title FROM knowledge WHERE id = ANY(%s) ORDER BY id", (mem_ids,))
        rows = cur.fetchall()
        lines.append("## 知识节点（本任务产出/溯源到本任务）")
        lines.append("")
        for kid, title in rows:
            lines.append(f"- MEM#{kid} {title[:70]}")
        lines.append("")

    # 一跳边（这些节点间的边 + 它们到任务节点的边）
    lines.append("## 子图（mermaid）")
    lines.append("")
    lines.append("```mermaid")
    lines.append("graph LR")
    def q(s):
        return re.sub(r'["<>]', "", str(s))[:40]
    cur.execute(
        "SELECT e.src_identity, e.dst_identity, e.edge_kind FROM readmodel_edge e "
        "WHERE (e.src_identity = ANY(%s) OR e.dst_identity = ANY(%s)) AND e.refresh_id=(SELECT max(refresh_id) FROM readmodel_meta) LIMIT 120",
        (list(task_nodes) + [], list(task_nodes) + []))
    shown = 0
    label_of = {}
    for s, d, k in cur.fetchall():
        if s in task_nodes:
            a, b = q(task_nodes[s].split("/")[-1]), (f"M{d}" if d not in task_nodes else q(task_nodes[d].split("/")[-1]))
        elif d in task_nodes:
            a, b = (f"M{s}" if s not in task_nodes else q(task_nodes[s].split("/")[-1])), q(task_nodes[d].split("/")[-1])
        else:
            continue
        lines.append(f"    {a} -->|{k}| {b}")
        shown += 1
    if not shown:
        lines.append("    empty[（本切片无任务文件直连边）]")
    lines.append("```")
    lines.append("")

    out = Path(out_override) if out_override else VAULT_ROOT / task_dir / "output" / "graph-slice.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(out), "mem_nodes": len(mem_ids),
                      "task_nodes": len(task_nodes), "edges_shown": shown},
                     ensure_ascii=False))
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
