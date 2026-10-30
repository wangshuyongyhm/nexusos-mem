#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""provenance.py — 血缘边（Task1533 M2a：知识生而有链）。

管什么：三条写入链在知识/文档产生时自动挂 provenance 边（事实性血缘，
零错边风险——与 LLM 语义边 M2.5 分层）：

  链① MEM 写入（api_server POST /knowledge）
      新经验 →(produced_in_task)→ task 目录 task.md 代表节点。
      task 线索：body 显式 task_dir > path 参数的 task/NNN 前缀；都无则跳过。
  链② persist 落盘（embed_server fsevents 抓 document 时）
      task/<dir>/output/report.md →(belongs_to_task)→ task/<dir>/input/task.md
      03-Daily/**会话日志 →(belongs_to_index)→ 03-Daily/MOC-日志索引.md
  链③ 快照镜像（knowledge-mirror-gen.py 分片生成后）
      源条目 →(mirrored_in_shard)→ 分片文件代表节点。
      默认关闭（--with-provenance 显式开），防与 1534 存量回填口径撞车；
      全量开启由主控在 1534 落地后定夺。

方向语义（与 relation.py citation 一致）：src=子（新知识/报告/源条目），
dst=父（任务/索引/分片容器）。

目标节点解析：边表端点是 knowledge.id，Vault 文件经 embed_server 抓成
document 行（path 粒度 `<rel>#cN` 分块）。resolve_vault_doc 取该文件
前缀的最小 id 作为**文件代表节点**（c0=文件头，最早插入最稳定）。
文件未被 embed（无 document 行）→ 返回 None → 调用方跳过挂边
（1534 回填 vault_doc 身份后解析率自然上升）。

失败语义（1524 同款）：挂边任何异常只 log 不阻断主写入——血缘是增值
不是主链路。幂等：复用 relation.add_relation 的 ON CONFLICT DO NOTHING。
"""

import re

import psycopg2

from nexusos_mem import relation as _relation

# embed_server 的 document 分块路径规则：`<rel_path>#cN`
_CHUNK_SUFFIX_RE = re.compile(r"#c\d+$")

# 链② 的文件级血缘规则表：(pattern, dst_path_builder, evidence)
# pattern 匹配 rel_path；builder 给出目标文件 rel_path。
_TASK_REPORT_RE = re.compile(r"^task/(?P<dir>[^/]+)/output/.*\.md$")
# 会话日志 = 03-Daily/<YYYY-MM-DD>/*.md（两层；日目录下任意 .md）。
# 03-Daily 直接子级 .md（MOC-日志索引/README）是索引自身，不参与。
_DAILY_SESSION_RE = re.compile(r"^03-Daily/\d{4}-\d{2}-\d{2}/.+\.md$")


def resolve_vault_doc(conn, rel_path: str):
    """Vault 文件 → knowledge.id 文件代表节点。

    匹配 `path = rel_path`（经验类直挂）或 `path LIKE 'rel_path#c%'`
    （document 分块），取最小 id（文件头块最稳定）。无匹配返回 None。
    LIKE 转义：防 rel_path 含 %/_ 通配符误匹配。
    """
    escaped = rel_path.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT min(id) FROM knowledge "
            "WHERE path = %s OR path LIKE %s ESCAPE '\\'",
            (rel_path, escaped + "#c%"),
        )
        row = cur.fetchone()
    return row[0] if row else None


def attach_provenance(conn, src_id: int, dst_rel_path: str,
                      evidence: str, source: str) -> dict:
    """尽力挂一条血缘边（src=知识 id，dst=Vault 文件代表节点）。

    目标文件无 document 行 → {"ok": False, "reason": "no_doc"}（调用方
    记 log 即可，1534 回填后会命中）。其余异常向上抛由调用方兜底
    （统一 try/except log 不阻断）。
    """
    dst_id = resolve_vault_doc(conn, dst_rel_path)
    if dst_id is None:
        return {"ok": False, "reason": "no_doc", "dst": dst_rel_path}
    if dst_id == src_id:
        return {"ok": False, "reason": "self_loop", "dst": dst_rel_path}
    res = _relation.add_relation(conn, src_id, dst_id, "provenance",
                                 source=source, evidence=evidence)
    res["dst_id"] = dst_id
    return res


def resolve_task_dir(task_dir: str = None, path: str = None,
                     title: str = "", content: str = "") -> str:
    """链① 任务目录线索解析（显式 > path > 正文，都不中返回 None=跳过）。

    不从 title/content 猜 task 目录：正文里的 task/ 字样可能是引用他人
    任务（零错边原则——宁可少挂不挂错）。仅当 body 显式传 task_dir 或
    path 参数本身位于 task/<dir>/ 下时挂。
    """
    if task_dir:
        return task_dir.strip("/")
    if path:
        m = re.match(r"^task/([^/]+)/", path.strip("/"))
        if m:
            return f"task/{m.group(1)}"
    return None


def provenance_rules_for_doc(rel_path: str):
    """链② 文件级血缘规则（embed_server 抓 document 时调）。

    返回 (dst_rel_path, evidence) 或 None（无规则命中=不挂边）。
    """
    m = _TASK_REPORT_RE.match(rel_path)
    if m:
        return (f"task/{m.group(1)}/input/task.md", "report belongs_to_task")
    if _DAILY_SESSION_RE.match(rel_path):
        return ("03-Daily/MOC-日志索引.md", "session belongs_to_index")
    return None
