#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""identity.py — 知识身份登记客户端（Task1524 M1单B）。

PG knowledge_identity 表的薄写入端，供快照链产生侧（镜像分片/图谱卡生成器）
与存量回填脚本共用。mem 行的登记不走本模块——由数据库触发器
nexus_mem_identity_register()（sql/001_knowledge_identity.sql）单点把守，
覆盖全部 knowledge 写入口（api_server / embed_server / SharedKnowledge / 直连脚本）。

调用方契约：
- 所有操作幂等（ON CONFLICT DO NOTHING），重跑只补新增。
- 本模块不做异常吞噬：登记失败向上抛，由调用方决定降级方式
  （生成器侧 log 不阻断产出；回填侧如实报数）。
"""
from typing import Iterable, Optional, Tuple

Row = Tuple[str, Optional[int], Optional[str], str]

_INSERT_SQL = (
    "INSERT INTO knowledge_identity (identity_kind, mem_id, vault_path, producer) "
    "VALUES (%s, %s, %s, %s) "
    "ON CONFLICT DO NOTHING"
)


def register_many(rows: Iterable[Row], dsn: Optional[str] = None) -> int:
    """批量登记身份行。rows = [(identity_kind, mem_id, vault_path, producer), ...]

    返回 executemany 影响行数（ON CONFLICT 跳过不计——精确对账用
    identity_stats() / identity_coverage 视图，不依赖本返回值）。
    """
    import psycopg2
    if dsn is None:
        from nexusos_mem.nexus_mem_utils import get_dsn
        dsn = get_dsn()
    rows = list(rows)
    if not rows:
        return 0
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.executemany(_INSERT_SQL, rows)
            n = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        conn.commit()
        return n
    finally:
        conn.close()


def register(kind: str, mem_id: Optional[int] = None,
             vault_path: Optional[str] = None, producer: str = "unknown",
             dsn: Optional[str] = None) -> None:
    """单行登记（生成器低频路径用）。"""
    register_many([(kind, mem_id, vault_path, producer)], dsn=dsn)


def purge_stale(kind: str, under_prefix: str, keep: Iterable[str],
                dsn: Optional[str] = None) -> int:
    """清掉 under_prefix 目录下、不在 keep 集合内的某 kind 身份行（产物文件已删时同步注销）。

    只按前缀圈定范围，绝不越界碰其他目录的行。
    """
    import psycopg2
    if dsn is None:
        from nexusos_mem.nexus_mem_utils import get_dsn
        dsn = get_dsn()
    keep = list(keep)
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM knowledge_identity "
                "WHERE identity_kind = %s AND vault_path LIKE %s "
                "AND NOT (vault_path = ANY(%s))",
                (kind, under_prefix + "%", keep),
            )
            n = cur.rowcount
        conn.commit()
        return n
    finally:
        conn.close()
