#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""relation.py — 知识关系边表（task 1455：MEM 关系图机器层）。

管什么：knowledge 表只有 superseded_by 一个一等关系信号（旧版指向新版），
多条经验之间的"取代/冲突/相关"关系一直散落在列里，没法做图查询。本模块
在 PG 里建一张轻量边表 knowledge_relation（src→dst，supersedes/conflicts/
relates/provenance 四种边），把 superseded_by 确定性回填成 supersedes 边，并给检索链
提供一跳查询。

在链路哪一环：位于"写入层（supersede 软废弃）"与"检索层（search.py）"之间——
写入层只写 knowledge.superseded_by 列（不动它，本模块只读），回填把列翻译成
边表；检索链命中旧经验时调 one_hop 沿边一跳找到新版，附给用户"这条已有新版"
的指引。

和谁交互：search.py（一跳扩展读边表）、tests/test_relation_graph.py（幂等/
对账/方向语义守卫）、1457（Obsidian 展示层导出，下一单）。

方向语义（全模块最重要的约定）：src supersedes dst = src 是新版、dst 是旧版。
"A 的入边"= 有人取代 A = A 已过时；"A 的出边"= A 取代了谁 = A 的历史版本。
"""

from typing import Optional

import psycopg2

from nexusos_mem.shared import DEFAULT_DSN

# 边类型枚举。supersedes 本单回填；conflicts/relates 只建枚举留空表，
# 抽取是以后的事（任务书 §2：无现成信号，不做臆测抽取）。
# Task1533: +provenance（血缘）——src 产生于/属于/被镜像于 dst。事实性边
# （来源可追溯，零错边），由 provenance.py 三条写入链自动挂（MEM 写入/persist
# 落盘/快照镜像），与 relates（citation 引用关联）分层。方向读法与 citation
# 一致：src=子（新知识/报告/条目），dst=父（任务/索引/分片）。
# Task1537: +semantic（语义相关）——LLM 复审过的高置信相关对（同主题/可互相
# 印证/解决同问题），无方向语义（统一小 id 为 src 规范化存储）。增值层而非
# 保底：riding 默认不含 semantic，search 需显式 include_semantic=True 才扩。
REL_TYPES = ("supersedes", "conflicts", "relates", "provenance", "semantic")

# 建表 SQL。幂等（IF NOT EXISTS）：测试和生产反复跑 ensure_schema 都安全。
# UNIQUE(src_id,dst_id,rel_type) 是回填的"防重入"闸门——同一对关系重复回填
# 会被 ON CONFLICT DO NOTHING 吞掉，天然幂等，不需要 DELETE 重灌（B2）。
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS knowledge_relation (
    id BIGSERIAL PRIMARY KEY,
    src_id BIGINT NOT NULL,
    dst_id BIGINT NOT NULL,
    rel_type VARCHAR(16) NOT NULL CHECK (rel_type IN ('supersedes','conflicts','relates','provenance','semantic')),
    evidence TEXT,
    created_by VARCHAR(32) DEFAULT 'backfill',
    created_at TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT uq_kr_src_dst_type UNIQUE (src_id, dst_id, rel_type)
)
"""

# 双向索引：检索链一跳扩展按 dst_id（旧版）查入边最频繁；1457 导出按
# src_id（新版）查出边。两个方向各建 (id, rel_type) 复合索引。
_INDEX_SQL_DST = (
    "CREATE INDEX IF NOT EXISTS idx_kr_dst_type "
    "ON knowledge_relation (dst_id, rel_type)"
)
_INDEX_SQL_SRC = (
    "CREATE INDEX IF NOT EXISTS idx_kr_src_type "
    "ON knowledge_relation (src_id, rel_type)"
)


def _get_conn(conn=None):
    """复用调用方连接；没传就自建（autocommit——DDL/回填不需要事务包裹，
    且避免连接残留未提交状态）。自建的连接由本函数调用方负责关闭。"""
    if conn is not None:
        return conn, False
    return psycopg2.connect(DEFAULT_DSN), True


def ensure_schema(conn=None) -> bool:
    """幂等建表+索引。跑多少遍结果都一样（幂等是验收标准之一）。

    为什么不用 migration 工具：就一张表两个索引，alembic 之类的引入
    成本大于收益（轻量约束，总规划 P4 定的死规矩）。
    """
    _conn, own = _get_conn(conn)
    try:
        _conn.autocommit = True
        with _conn.cursor() as cur:
            cur.execute(_SCHEMA_SQL)
            cur.execute(_INDEX_SQL_DST)
            cur.execute(_INDEX_SQL_SRC)
            # Task1533: 存量表 CHECK 迁移——CREATE IF NOT EXISTS 对已存在的
            # 旧表无效（旧 CHECK 不含新枚举），检测缺枚举则重建约束。
            # Task1537: 判据改为 semantic（最新枚举）——三枚举（pre-1533）和
            # 四枚举（pre-1537）存量都缺 semantic，一次迁移直接到五枚举终态。
            # 新表走 _SCHEMA_SQL 已含全部枚举，此处探测跳过（幂等）。
            cur.execute(
                "SELECT 1 FROM pg_constraint WHERE conrelid='knowledge_relation'::regclass "
                "AND contype='c' AND pg_get_constraintdef(oid) NOT ILIKE '%semantic%'")
            if cur.fetchone():
                cur.execute(
                    "ALTER TABLE knowledge_relation DROP CONSTRAINT "
                    "knowledge_relation_rel_type_check")
                cur.execute(
                    "ALTER TABLE knowledge_relation ADD CONSTRAINT "
                    "knowledge_relation_rel_type_check "
                    "CHECK (rel_type IN ('supersedes','conflicts','relates','provenance','semantic'))")
        return True
    finally:
        if own:
            _conn.close()


def backfill_supersedes(conn=None) -> dict:
    """把 knowledge.superseded_by 列回填成 supersedes 边（确定性翻译，零抽取）。

    方向：src=superseded_by 的值（新版），dst=该行自身（旧版）。
    例：#A 的 superseded_by=#B（A 被 B 取代）→ 写边 (src=B, dst=A)。

    幂等：ON CONFLICT DO NOTHING——重跑只补新废弃的行，不重复不覆盖。
    对账：返回 dict 带 edge_count（边表 supersedes 总数）和 signal_count
    （有效信号总数）。两者相等=回填完整（验收标准 1）。
    有效信号 = superseded_by 非空 **且不指向自己**。实测库里有 592 条自环
    （superseded_by=id，监控告警类批量操作的系统性脏数据）：自己取代自己
    语义荒谬，入边表会毒化 one_hop（"新版=你自己"），回填直接排除。
    本模块只读 knowledge 列不修数据（治理归写入层），故对账口径=有效信号。
    不相等的合法场景只有一种：边表建好后有人恢复了旧条（superseded_by
    置 NULL），边表残留——那是治理问题不是回填 bug，one_hop 照常工作。
    """
    _conn, own = _get_conn(conn)
    try:
        _conn.autocommit = True
        with _conn.cursor() as cur:
            # 自愈：清掉历史版本回填逻辑灌入的自环边（src=dst"自己取代自己"）。
            # 精确 WHERE 删除（仅本表 supersedes 自环，不碰 knowledge 表）——
            # 这不是 B2 危险操作，是修正本模块早期版本自己的错误产出。
            cur.execute(
                "DELETE FROM knowledge_relation "
                "WHERE rel_type = 'supersedes' AND src_id = dst_id"
            )
            healed = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
            # 回填主体：列→边。evidence 留下溯源 SQL 语义，方便 1457 导出时
            # 区分"确定性回填边"和"未来手工/抽取边"。
            cur.execute(
                """
                INSERT INTO knowledge_relation (src_id, dst_id, rel_type, evidence)
                SELECT k.superseded_by, k.id, 'supersedes',
                       'backfill from knowledge.superseded_by'
                FROM knowledge k
                WHERE k.superseded_by IS NOT NULL
                  AND k.superseded_by != k.id
                ON CONFLICT DO NOTHING
                """
            )
            inserted = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
            cur.execute(
                "SELECT count(*) FROM knowledge_relation WHERE rel_type='supersedes'"
            )
            edge_count = cur.fetchone()[0]
            cur.execute(
                "SELECT count(*) FROM knowledge "
                "WHERE superseded_by IS NOT NULL AND superseded_by != id"
            )
            signal_count = cur.fetchone()[0]
        return {
            "inserted": inserted,
            "healed_selfloops": healed,
            "edge_count": edge_count,
            "signal_count": signal_count,
            "balanced": edge_count == signal_count,
        }
    finally:
        if own:
            _conn.close()


def one_hop(conn, kb_id: int, rel_type: Optional[str] = None, direction: str = "in") -> list:
    """一跳查询：沿边表走一步，返回 [{"src_id","dst_id","rel_type","evidence","title"}, ...]。

    direction 语义（与模块头方向约定一致）：
      "in"  入边——谁取代了 kb_id（kb_id 是旧版，返回 src=新版）。检索链
            一跳扩展用的就是这个：命中旧经验 → 入边 → 新版指引。
      "out" 出边——kb_id 取代了谁（kb_id 是新版，返回 dst=历史旧版）。
            1457 展示层画版本链用。
      "both" 两边都要。

    title 是对端 knowledge 行的标题（方便调用方直接展示，省一次 JOIN 回查）。
    对端已被物理删除时 title 为 None（边表不级联删——保留关系痕迹）。
    """
    if rel_type is not None and rel_type not in REL_TYPES:
        raise ValueError(f"rel_type 必须是 {REL_TYPES} 之一，收到: {rel_type!r}")
    if direction not in ("in", "out", "both"):
        raise ValueError(f"direction 必须是 in/out/both，收到: {direction!r}")

    # 对端标题 JOIN：入边对端是 src，出边对端是 dst。both 时对端可能是任一端，
    # 分别 JOIN 取 COALESCE。
    conds, params = [], []
    if direction in ("in", "both"):
        conds.append("kr.dst_id = %s")
        params.append(kb_id)
    if direction in ("out", "both"):
        conds.append("kr.src_id = %s")
        params.append(kb_id)
    where_dir = "(" + " OR ".join(conds) + ")"
    where = f"{where_dir}"
    if rel_type:
        where += " AND kr.rel_type = %s"
        params.append(rel_type)

    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT kr.src_id, kr.dst_id, kr.rel_type, kr.evidence,
                   CASE WHEN kr.src_id = %s THEN dst.title ELSE src.title END
            FROM knowledge_relation kr
            LEFT JOIN knowledge src ON src.id = kr.src_id
            LEFT JOIN knowledge dst ON dst.id = kr.dst_id
            WHERE {where}
            ORDER BY kr.id
            """,
            [kb_id] + params,
        )
        return [
            {
                "src_id": r[0],
                "dst_id": r[1],
                "rel_type": r[2],
                "evidence": r[3],
                "title": r[4],
            }
            for r in cur.fetchall()
        ]


def one_hop_in_batch(conn, kb_ids: list, rel_type: str = "supersedes") -> dict:
    """批量入边查询：一次 SQL 查一批 id 的"谁取代了我"。

    search.py 一跳扩展用（对 top-k 结果批量附标注，避免每条一次往返）。
    返回 {kb_id: {"id": 新版id, "title": 新版标题}}——一跳扩展只需要新版
    指引这一个信息，多条入边时取 id 最小的（最早的新版；正常情况只有一条，
    因为 superseded_by 列本身单值）。
    """
    if not kb_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT kr.dst_id, kr.src_id, k.title
            FROM knowledge_relation kr
            JOIN knowledge k ON k.id = kr.src_id
            WHERE kr.dst_id = ANY(%s) AND kr.rel_type = %s
            ORDER BY kr.dst_id, kr.src_id
            """,
            (list(kb_ids), rel_type),
        )
        out = {}
        for dst_id, src_id, title in cur.fetchall():
            # ORDER BY src_id 保证同 dst 多边时第一条是最小 id，先到先得不覆盖
            if dst_id not in out:
                out[dst_id] = {"id": src_id, "title": title}
        return out


# ---------------------------------------------------------------------------
# task 1522 A 项：citation 边（正文显式引用 MEM#xxx → relates 边）
# ---------------------------------------------------------------------------

# 行级围栏判定（比正则全文剥码块轻量——回填要扫全库，按行 O(n) 够用且
# 不会误剥跨行内容；围栏内引用多为代码示例/日志原文，不是作者意图性引用）
_CITATION_RE = None  # lazy import re

# 每条知识最多建的 citation 边数（防单条 30 处引用把一跳扩展撑爆）
CITATION_EDGE_CAP = 10


def extract_citations(text: str) -> list:
    """从正文抽显式引用 `MEM#数字` → [id, ...]（去重保序）。

    代码块排除：按行粗判 ``` 围栏（含 ~~~ 不常见不处理），围栏内的
    MEM# 引用不计——那是代码示例/日志粘贴，不是作者在陈述关联。
    行内代码 `MEM#123` 单反引号包裹也不计（同理）。假阳率见 1522 report。
    """
    import re as _re
    global _CITATION_RE
    if _CITATION_RE is None:
        _CITATION_RE = _re.compile(r"MEM#(\d{1,8})")
    ids, seen = [], set()
    in_fence = False
    for line in (text or "").splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        # 行内代码剥除：`...` 段内的引用不算（一行可能多段）
        cleaned = _re.sub(r"`[^`]*`", " ", line)
        for m in _CITATION_RE.finditer(cleaned):
            kid = int(m.group(1))
            if kid not in seen:
                seen.add(kid)
                ids.append(kid)
    return ids


def add_relation(conn, src_id: int, dst_id: int, rel_type: str,
                 source: str = "api", evidence: str = None) -> dict:
    """幂等加一条边。返回 {"ok", "inserted", "edge_id"|None}。

    校验：rel_type 枚举 / src≠dst（自环边无图意义，relation 语义里
    supersedes 自环已是治理债，不让 relates 再添）。ON CONFLICT DO NOTHING
    幂等（重复边不插不报错）。端点存在性由调用方决定是否预检（回填场景
    必须预检——引用了不存在的编号是行文笔误不是关联）。
    """
    if rel_type not in REL_TYPES:
        raise ValueError(f"rel_type 必须是 {REL_TYPES} 之一，收到: {rel_type!r}")
    if src_id == dst_id:
        return {"ok": False, "inserted": False, "reason": "self_loop"}
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO knowledge_relation (src_id, dst_id, rel_type, evidence, created_by)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (src_id, dst_id, rel_type) DO NOTHING
            RETURNING id
            """,
            (src_id, dst_id, rel_type, evidence, source[:32]),
        )
        row = cur.fetchone()
        conn.commit()
        return {"ok": True, "inserted": row is not None,
                "edge_id": row[0] if row else None}


def backfill_citations(conn=None, batch_size: int = 500, dry_run: bool = False) -> dict:
    """全库扫正文显式引用，建 relates 边（1522 A 项一次性回填）。

    语义：k 的正文引用了 MEM#M（M 存在且未废弃且 M≠k）→ 边
    (src=k.id, dst=M, relates, source='backfill-citation')。
    「谁引用了 M」= M 的入边（one_hop in）——与 supersedes 方向读法一致。

    幂等：ON CONFLICT。分页 id 游标防全库 content 载入内存
    （4 万条 × 平均 2KB ≈ 80MB，分批稳）。统计口径全返回供 report。
    dry_run=True 只统计不落库（CLI 预演）。脚本入口
    scripts/backfill_citation_relations.py 是本函数的薄包装。
    """
    _conn, own = _get_conn(conn)
    _conn.autocommit = True
    stats = {"scanned": 0, "edges_inserted": 0, "edges_existing": 0,
             "refs_seen": 0, "refs_missing": 0, "refs_deprecated": 0,
             "refs_self": 0, "cap_truncated": 0}
    try:
        with _conn.cursor() as cur:
            cur.execute("SELECT max(id) FROM knowledge")
            max_id = cur.fetchone()[0] or 0
        last_id = 0
        while True:
            with _conn.cursor() as cur:
                cur.execute(
                    "SELECT id, title, content FROM knowledge "
                    "WHERE id > %s ORDER BY id LIMIT %s",
                    (last_id, batch_size),
                )
                rows = cur.fetchall()
            if not rows:
                break
            all_ids = set()
            for r in rows:
                last_id = r[0]
                stats["scanned"] += 1
                cites = extract_citations((r[1] or "") + "\n" + (r[2] or ""))
                cites = cites[:CITATION_EDGE_CAP]
                if len(cites) >= CITATION_EDGE_CAP:
                    stats["cap_truncated"] += 1
                for cid in cites:
                    stats["refs_seen"] += 1
                    if cid == r[0]:
                        stats["refs_self"] += 1
                        continue
                    all_ids.add((r[0], cid))
            if all_ids:
                # 批量预检端点存在性+废弃态（笔误编号/已废弃目标不建边）
                cids = list({c for _, c in all_ids})
                with _conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, deprecated FROM knowledge WHERE id = ANY(%s)",
                        (cids,),
                    )
                    rows_ = cur.fetchall()
                ok_ids = {r[0] for r in rows_ if not r[1]}
                dep_ids = {r[0] for r in rows_ if r[1]}
                _pending_dry = set()
                for src, dst in all_ids:
                    if dst in dep_ids:
                        stats["refs_deprecated"] += 1
                    elif dst not in ok_ids:
                        stats["refs_missing"] += 1
                    elif dry_run:
                        # dry-run 预演也要区分已存在边（否则 inserted 口径虚高，
                        # 幂等复跑的 report 对不上——1522 reproduce [6] 实证）
                        _pending_dry.add((src, dst))
                    else:
                        res = add_relation(_conn, src, dst, "relates",
                                           source="backfill-citation",
                                           evidence="citation in content")
                        if res.get("inserted"):
                            stats["edges_inserted"] += 1
                        elif res.get("ok"):
                            stats["edges_existing"] += 1
                if _pending_dry:
                    with _conn.cursor() as cur:
                        cur.execute(
                            "SELECT src_id, dst_id FROM knowledge_relation "
                            "WHERE rel_type='relates' AND (src_id, dst_id) IN %s",
                            (tuple(_pending_dry),),
                        )
                        existing = {tuple(x) for x in cur.fetchall()}
                    stats["edges_existing"] += len(existing)
                    stats["edges_inserted"] += len(_pending_dry) - len(existing)
        return stats
    finally:
        if own:
            _conn.close()
