#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""backfill_provenance.py — Task1534 M2b：存量血缘边回填（幂等）。

经验 → 产生于 → task/会话 三路线索，结构化匹配零 LLM 判断（血缘=事实语义，
宁可少建不可错建——模糊线索只记录不建边）：

  ①a knowledge.task_id 列（最高置信：写入链 context_capture 直存 task slug）
  ①b title/content/path 的 `task/NNNN` 路径模式（锚定路径模式，非裸数字——池号歧义）
  ②  task/*/output/report.md 的 MEM#N 引用（task→经验反向；已有 citation 边的跳过）
  ③  03-Daily 会话日志的 MEM#N 引用（confidence='medium'——引用≠必然生产关系）

边语义（与 1533 协调声明）：knowledge_relation.rel_type='provenance'，
双端都在 identity 空间——src=mem 身份行 id，dst=vault_doc 身份行 id
（task.md / report.md / 会话日志先登记 vault_doc 身份，顺带补 1524 留白）。

跳过分布全计数（原因可追溯）。分批 500 条/事务（1529 压测：分批安全）。
用法：python3 backfill_provenance.py [--dry-run]
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import psycopg2  # noqa: E402
from nexusos_mem._vault_env import VAULT_ROOT  # noqa: E402
from nexusos_mem.identity import register_many  # noqa: E402
from nexusos_mem.nexus_mem_utils import get_dsn  # noqa: E402

TASK_NUM_RE = re.compile(r"task/(\d{3,4})(?![\d])")   # 路径锚定，池号不误吸
SLUG_NUM_RE = re.compile(r"^(\d{3,4})")
MEM_REF_RE = re.compile(r"MEM#(\d{1,6})")
BATCH = 500


class Stats:
    def __init__(self):
        self.vault_doc_registered = 0
        self.clues = {"task_id": 0, "text_pattern": 0, "report_cite": 0, "session_cite": 0}
        self.edges = {"task_id": 0, "text_pattern": 0, "report_cite": 0, "session_cite": 0}
        self.skipped = {}

    def skip(self, reason):
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def dump(self):
        return {
            "vault_doc_registered": self.vault_doc_registered,
            "clues_found": dict(self.clues),
            "edges_built": dict(self.edges),
            "edges_total": sum(self.edges.values()),
            "skipped": dict(self.skipped),
        }


def task_dirs(vault):
    """task/<NNNN-slug> 目录 → {task_number: [dir_name,...]}。"""
    by_num = {}
    root = vault / "task"
    if not root.is_dir():
        return by_num
    for d in root.iterdir():
        if not d.is_dir():
            continue
        m = SLUG_NUM_RE.match(d.name)
        if m:
            by_num.setdefault(m.group(1), []).append(d.name)
    return by_num


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault-root", default=str(VAULT_ROOT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    vault = Path(args.vault_root)
    st = Stats()

    conn = psycopg2.connect(get_dsn())
    cur = conn.cursor()

    # ── A. 血缘目标登记 vault_doc 身份（task.md / report.md / 会话日志） ──
    targets = set()  # vault 相对路径
    tdirs = task_dirs(vault)

    def rel(p):
        return str(p.relative_to(vault)).replace("\\", "/")

    task_md_of = {}   # dir_name -> task.md 相对路径（存在者）
    report_mds = []
    for _num, dirs in tdirs.items():
        for d in dirs:
            tmd = vault / "task" / d / "input" / "task.md"
            if tmd.is_file():
                task_md_of[d] = rel(tmd)
                targets.add(task_md_of[d])
            rmd = vault / "task" / d / "output" / "report.md"
            if rmd.is_file():
                report_mds.append((d, rel(rmd)))
                targets.add(rel(rmd))
    session_logs = sorted(
        p for p in (vault / "03-Daily").rglob("*.md")
    ) if (vault / "03-Daily").is_dir() else []
    for p in session_logs:
        targets.add(rel(p))

    if not args.dry_run:
        register_many([("vault_doc", None, t, "backfill:1534") for t in sorted(targets)])
    st.vault_doc_registered = len(targets)

    # 身份 id 映射（identity 空间两端解析）
    cur.execute("SELECT id, mem_id FROM knowledge_identity WHERE identity_kind='mem'")
    mem_ident = {r[1]: r[0] for r in cur.fetchall()}
    cur.execute("SELECT id, vault_path FROM knowledge_identity WHERE identity_kind='vault_doc'")
    doc_ident = {r[1]: r[0] for r in cur.fetchall()}

    # citation 边索引（线索②跳过判定：exp↔该 task 的 document chunks 已有 citation）
    cur.execute(
        "SELECT src_id, dst_id FROM knowledge_relation WHERE rel_type='relates' "
        "AND created_by='backfill-citation'")
    cite_pairs = {(r[0], r[1]) for r in cur.fetchall()}
    cur.execute(
        "SELECT id, path FROM knowledge WHERE knowledge_type='document' AND path LIKE 'task/%'")
    doc_chunk_task = {}  # knowledge id -> task number（按 path 前缀 task/<slug>）
    for kid, p in cur.fetchall():
        m = re.match(r"task/(\d{3,4})", p or "")
        if m:
            doc_chunk_task[kid] = m.group(1)
    cite_index = {}  # (exp_id, task_num) 已有 citation
    for s, d in cite_pairs:
        for a, b in ((s, d), (d, s)):
            tn = doc_chunk_task.get(b)
            if tn:
                cite_index[(a, tn)] = True

    edge_rows = []  # [(kid, target_rel, evidence, kind, confidence)]

    def num_of_dir(dir_name):
        m = SLUG_NUM_RE.match(dir_name)
        return m.group(1) if m else None

    # ── B. 线索①a：task_id 列（高置信先到——去重保留权） ──
    cur.execute(
        "SELECT id, task_id FROM knowledge WHERE task_id IS NOT NULL AND NOT deprecated")
    for kid, slug in cur.fetchall():
        st.clues["task_id"] += 1
        tmd = task_md_of.get(slug)
        if not tmd:
            nums = tdirs.get(str(slug))
            if nums and len(nums) == 1:
                tmd = task_md_of.get(nums[0])
            if not tmd:
                st.skip(f"task_id 无对应目录: {slug}")
                continue
        edge_rows.append((kid, tmd, f"clue1a: knowledge.task_id={slug}", "task_id", "high"))

    # ── C. 线索①b：文本 task/NNNN 路径模式 ──
    cur.execute(
        "SELECT id, title, coalesce(path,''), array_to_string(tags,' ') FROM knowledge "
        "WHERE NOT deprecated AND (title ~ 'task/[0-9]{3,4}' OR path ~ 'task/[0-9]{3,4}' "
        "OR array_to_string(tags,' ') ~ 'task/[0-9]{3,4}')")
    seen_1b = set()
    for kid, title, path, tags in cur.fetchall():
        nums = set(TASK_NUM_RE.findall(f"{title} {path} {tags}"))
        for num in nums:
            st.clues["text_pattern"] += 1
            if (kid, num) in seen_1b:
                continue
            seen_1b.add((kid, num))
            dirs = tdirs.get(num)
            if not dirs:
                st.skip(f"文本 task/{num} 无目录")
                continue
            if len(dirs) > 1:
                st.skip(f"文本 task/{num} 目录多义: {dirs}")
                continue
            tmd = task_md_of.get(dirs[0])
            if not tmd:
                st.skip(f"task/{num} 无 task.md")
                continue
            edge_rows.append((kid, tmd, f"clue1b: text task/{num}", "text_pattern", "high"))

    # ── D. 线索②：report.md 的 MEM# 引用 ──
    for d, rrel in report_mds:
        num = num_of_dir(d)
        try:
            text = (vault / rrel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            st.skip(f"report 读取失败: {rrel}")
            continue
        for mid in set(MEM_REF_RE.findall(text)):
            st.clues["report_cite"] += 1
            kid = int(mid)
            if kid not in mem_ident:
                st.skip(f"MEM#{mid} 不存在/已废弃")
                continue
            if cite_index.get((kid, num)):
                st.skip(f"MEM#{mid}↔task{num} 已有 citation 边")
                continue
            edge_rows.append((kid, rrel, f"clue2: {rrel} 引用 MEM#{mid}", "report_cite", "high"))

    # ── E. 线索③：会话日志 MEM# 引用（medium） ──
    for srel in (rel(p) for p in session_logs):
        try:
            text = (vault / srel).read_text(encoding="utf-8", errors="replace")
        except OSError:
            st.skip(f"会话日志读取失败: {srel}")
            continue
        for mid in set(MEM_REF_RE.findall(text)):
            st.clues["session_cite"] += 1
            kid = int(mid)
            if kid not in mem_ident:
                st.skip(f"MEM#{mid} 不存在/已废弃（会话）")
                continue
            edge_rows.append((kid, srel, f"clue3: {srel} 引用 MEM#{mid}", "session_cite", "medium"))

    # ── F. 建边（identity 空间，分批；幂等 ON CONFLICT） ──
    built = 0
    batch = []
    dedup = set()

    def flush():
        nonlocal built
        if not batch or args.dry_run:
            batch.clear()
            return
        cur.executemany(
            "INSERT INTO knowledge_relation (src_id, dst_id, rel_type, evidence, created_by) "
            "VALUES (%s, %s, 'provenance', %s, 'backfill:1534') "
            "ON CONFLICT (src_id, dst_id, rel_type) DO NOTHING",
            batch)
        built += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        conn.commit()
        batch.clear()

    for kid, trel, evidence, kind, conf in edge_rows:
        src = mem_ident.get(kid)
        dst = doc_ident.get(trel)
        if args.dry_run and dst is None and trel in targets:
            dst = -1  # 干跑语义：目标在待登记集合 → 视为可解析（仅计数不落库）
        if not src:
            st.skip(f"经验无 mem 身份行: {kid}")
            continue
        if not dst:
            st.skip(f"目标无 vault_doc 身份行: {trel}")
            continue
        key = (src, dst if dst > 0 else trel)
        if key in dedup:
            st.skip("同对重复线索（多源命中）")
            continue
        dedup.add(key)
        batch.append((src, dst, f"{evidence} conf={conf}"))
        st.edges[kind] += 1
        if len(batch) >= BATCH:
            flush()
    flush()

    # 高置信优先去重说明：1a/1b 同对时按先到（1a 在前）保留，后者计 skip
    print(json.dumps({**st.dump(), "edges_inserted": built,
                      "dry_run": args.dry_run}, ensure_ascii=False, indent=1))
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
