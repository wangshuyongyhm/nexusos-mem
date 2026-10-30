#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nexus/embed_corpus.py — task 245: BGE-M3 全量重嵌入脚本

把 knowledge 表行 + Vault .md 文档统一嵌入到 pgvector knowledge.embedding 列。

两种模式:
    1. knowledge 行: UPDATE knowledge SET embedding=... WHERE id=...
       （嵌入 title + content）
    2. 文档语料: 遍历 Vault .md → INSERT knowledge (knowledge_type='document')
       （content_hash 去重，跳过未改动文件）

用法:
    # 嵌入现有 knowledge 行（快，~68 行）
    python3 -m nexus.embed_corpus --knowledge

    # 嵌入 Vault 文档（全量 ~3200，慢）
    python3 -m nexus.embed_corpus --docs --limit 100    # 先 100 验证
    python3 -m nexus.embed_corpus --docs                # 全量
    python3 -m nexus.embed_corpus --docs --batch 64     # 指定 batch

    # 查看进度
    python3 -m nexus.embed_corpus --stats

cron 夜间跑:
    python3 -m nexus.embed_corpus --docs --batch 64 >> /tmp/embed-cron.log 2>&1
"""

import argparse
import hashlib
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

import psycopg2
import psycopg2.extras

# 复用 search.py 的模型单例（同进程内只加载一次）
from nexusos_mem.search import _get_bge_m3, BGE_M3_DIM

DEFAULT_DSN = os.environ.get(
    "NEXUS_PG_DSN", "postgresql://localhost/nexus_mem"
)
from nexusos_mem._vault_env import VAULT_ROOT  # Task702: 单源（env>哨兵推导）
from nexusos_mem.embed_server import _infer_project  # 项目路由单源（yaml 规则驱动）

# 文档内容截断（避免超长 .md 撑爆嵌入输入；BGE-M3 max 8192 tokens）
DOC_MAX_CHARS = 4000
# task 253 P1-4: 分块参数。每篇文档读取上限 + 单块大小 + 每篇最多块数。
DOC_READ_MAX_CHARS = 20000   # 每篇最多读 20K 字（覆盖绝大多数 .md）
CHUNK_SIZE = 1200            # 单块目标字符数
CHUNK_OVERLAP = 150          # 硬切长段时的重叠
CHUNK_MAX_PER_DOC = 12       # 每篇最多 12 块（防超长文档爆行）
# 文档标题/路径黑名单（跳过这些目录）
SKIP_DIRS = {".smart-env", ".obsidian", "node_modules", ".git", "_retired"}
# Task611: 密钥真源文件级黑名单（SKIP_DIRS 管不到顶层文件）
SECRET_MD_FILES = {"SECRETS.md"}


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP,
                max_chunks: int = CHUNK_MAX_PER_DOC) -> list:
    """按段落贪心分块，超长段落硬切。返回块文本列表（≤ max_chunks）。"""
    import re as _re
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    paras = _re.split(r"\n\s*\n", text)
    chunks: list = []
    buf = ""
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if len(chunks) >= max_chunks:
            break
        if len(buf) + len(p) + 1 <= size:
            buf = (buf + "\n" + p).strip() if buf else p
        else:
            if buf:
                chunks.append(buf)
                buf = ""
            if len(p) > size:
                step = max(size - overlap, 1)
                for i in range(0, len(p), step):
                    if len(chunks) >= max_chunks:
                        break
                    chunks.append(p[i:i + size])
            else:
                buf = p
    if buf and len(chunks) < max_chunks:
        chunks.append(buf)
    return chunks[:max_chunks]


def _conn():
    return psycopg2.connect(DEFAULT_DSN)


def _content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def embed_knowledge_rows(batch: int = 64) -> dict:
    """嵌入 knowledge 表现有行（title + content）。"""
    model = _get_bge_m3()
    conn = _conn()
    cur = conn.cursor()
    cur.execute("SELECT id, title, content FROM knowledge WHERE embedding IS NULL")
    rows = cur.fetchall()
    print(f"[knowledge] {len(rows)} 行待嵌入")

    done = skipped = 0
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        texts = [f"{t} {c or ''}"[:DOC_MAX_CHARS] for (_, t, c) in chunk]
        vecs = model.encode(texts, normalize_embeddings=True, batch_size=min(32, len(chunk)))
        for idx, (kid, _, _) in enumerate(chunk):
            cur.execute(
                "UPDATE knowledge SET embedding = %s, content_hash = %s, verified_at = NOW() WHERE id = %s",
                (vecs[idx].tolist(), _content_hash(texts[idx]), kid),
            )
        conn.commit()
        done += len(chunk)
        print(f"  [{done}/{len(rows)}] embedded", flush=True)

    cur.close()
    conn.close()
    return {"knowledge_embedded": done, "skipped": skipped}


def _iter_vault_docs():
    """遍历 Vault .md 文件（跳过 SKIP_DIRS）。缺 vault 显式报错（1957 废静默占位）。"""
    if VAULT_ROOT is None:
        from nexusos_common import ConfigError, VAULT_MISSING_REMEDIATION
        raise ConfigError("vault 不可用，无法遍历 Vault 文档。" + VAULT_MISSING_REMEDIATION)
    for path in VAULT_ROOT.rglob("*.md"):
        # 跳过黑名单目录
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        # Task611: 密钥真源文件级排除
        if path.name in SECRET_MD_FILES:
            continue
        yield path


def embed_docs(limit: Optional[int] = None, batch: int = 64) -> dict:
    """嵌入 Vault .md 文档到 knowledge 表 (knowledge_type='document')。

    task 253 P1-4: 分块嵌入。每篇文档切成多块，每块一行，path=`rel#cN`。
    - content_hash 逐块去重：块未变则跳过。
    - 迁移：删除旧的整文行（path==rel，无 #c）。
    - 剪枝：文档变短后删除多余的尾部块（index >= 新块数）。
    """
    model = _get_bge_m3()
    conn = _conn()
    cur = conn.cursor()

    # 预读已有 document 行的 path → (id, content_hash)（含 chunk-path 和 legacy path）
    cur.execute(
        "SELECT path, id, content_hash FROM knowledge WHERE knowledge_type='document'"
    )
    existing = {r[0]: (r[1], r[2]) for r in cur.fetchall() if r[0]}

    docs = list(_iter_vault_docs())
    if limit:
        docs = docs[:limit]
    print(f"[docs] {len(docs)} 文档待处理（Vault 已有 document 行: {len(existing)}）")

    done = skipped = errors = pruned = 0
    t0 = time.time()
    buf_texts, buf_metas = [], []

    def _flush():
        nonlocal done
        if not buf_texts:
            return
        vecs = model.encode(buf_texts, normalize_embeddings=True,
                            batch_size=min(32, len(buf_texts)))
        for text, meta, vec in zip(buf_texts, buf_metas, vecs):
            chash = meta["hash"]
            cpath = meta["path"]                 # rel#cN
            old = existing.get(cpath)
            vec_list = vec.tolist()
            if old:
                cur.execute(
                    "UPDATE knowledge SET embedding=%s, content_hash=%s, title=%s, "
                    "content=%s, updated=NOW() WHERE id=%s",
                    (vec_list, chash, meta["title"], text[:CHUNK_SIZE + 200], old[0]),
                )
            else:
                cur.execute("""
                    INSERT INTO knowledge
                        (source_agent, knowledge_type, title, content, tags, project,
                         path, embedding, content_hash)
                    VALUES ('nexus-embed', 'document', %s, %s, %s, %s, %s, %s, %s)
                """, (meta["title"], text[:CHUNK_SIZE + 200], ["document"],
                      meta["project"], cpath, vec_list, chash))
            done += 1
        conn.commit()
        buf_texts.clear()
        buf_metas.clear()

    for doc_path in docs:
        try:
            raw = doc_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            errors += 1
            continue
        if not raw.strip():
            skipped += 1
            continue
        try:
            rel = str(doc_path.relative_to(VAULT_ROOT))
        except ValueError:
            rel = str(doc_path)
        project = _infer_project(rel)

        chunks = _chunk_text(raw[:DOC_READ_MAX_CHARS])
        if not chunks:
            skipped += 1
            continue

        # 迁移：删除该文档的 legacy 整文行（path==rel，无 #c 后缀）
        if rel in existing:
            cur.execute("DELETE FROM knowledge WHERE knowledge_type='document' AND path=%s", (rel,))
            existing.pop(rel, None)
            pruned += 1
        # 剪枝：文档变短后删除多余尾部块。用 Python 精确匹配 path 前缀，
        # 避免 SQL LIKE 把文件名里的 _ 当通配符导致误删其它文档的块。
        prefix = rel + "#c"
        stale = []
        for k in list(existing.keys()):
            if k.startswith(prefix):
                m = re.match(r"^#c(\d+)$", k[len(rel):])
                if m and int(m.group(1)) >= len(chunks):
                    stale.append(k)
        for sp in stale:
            cur.execute("DELETE FROM knowledge WHERE knowledge_type='document' AND path=%s", (sp,))
            existing.pop(sp, None)

        for i, chunk in enumerate(chunks):
            cpath = f"{rel}#c{i}"
            chash = _content_hash(chunk)
            old = existing.get(cpath)
            if old and old[1] == chash:
                skipped += 1        # 未变的块在编码前就跳过（断点续跑不重付算力）
                continue
            title = doc_path.stem if i == 0 else f"{doc_path.stem} (part {i+1})"
            buf_texts.append(chunk)
            buf_metas.append({"path": cpath, "title": title, "project": project,
                              "hash": chash})
            if len(buf_texts) >= batch:
                _flush()
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  [{done} chunks] {skipped} skipped, "
                      f"{rate:.1f} chunk/s, elapsed {elapsed:.0f}s", flush=True)
    _flush()

    # 给新写入的 document 行分配 SHADOW 编码（不在 INSERT 时内联，改为批量回填）
    # 复用现有 conn 调用 shadow_match.backfill，避免 subprocess 开销
    try:
        from nexusos_mem.shadow_match import backfill as _shadow_backfill
        shadow_result = _shadow_backfill(conn, dry_run=False, only_null=True)
        if shadow_result["assigned"]:
            print(f"[docs] shadow_code backfill: 扫描={shadow_result['scanned']} "
                  f"分配={shadow_result['assigned']} 写入={shadow_result['updated']}")
    except Exception as e:
        print(f"[docs] shadow_code backfill 失败（非致命）: {e}")

    cur.close()
    conn.close()
    elapsed = time.time() - t0
    print(f"[docs] done: {done} chunks embedded, {skipped} unchanged, "
          f"{pruned} legacy migrated, {errors} errors, {elapsed:.0f}s total")
    return {"chunks_embedded": done, "skipped": skipped, "legacy_migrated": pruned,
            "errors": errors, "elapsed_sec": round(elapsed, 1)}


def show_stats():
    conn = _conn()
    conn.autocommit = True  # Task 379: 只读走 autocommit,防 idle in transaction
    cur = conn.cursor()
    cur.execute("""
        SELECT knowledge_type,
               COUNT(*) AS total,
               COUNT(embedding) AS has_embed
        FROM knowledge GROUP BY knowledge_type ORDER BY 2 DESC
    """)
    print(f"{'type':<16} {'total':>6} {'embedded':>9}")
    for ktype, total, has_emb in cur.fetchall():
        print(f"{(ktype or '-'):<16} {total:>6} {has_emb:>9}")
    cur.execute("SELECT COUNT(*) FROM knowledge WHERE embedding IS NOT NULL")
    print(f"\n总 embedding: {cur.fetchone()[0]}")
    cur.close()
    conn.close()


def main():
    ap = argparse.ArgumentParser(description="task 245 BGE-M3 pgvector 嵌入")
    ap.add_argument("--knowledge", action="store_true", help="嵌入 knowledge 表行")
    ap.add_argument("--docs", action="store_true", help="嵌入 Vault 文档")
    ap.add_argument("--stats", action="store_true", help="显示嵌入统计")
    ap.add_argument("--limit", type=int, default=None, help="文档数上限")
    ap.add_argument("--batch", type=int, default=64, help="batch size")
    args = ap.parse_args()

    if args.stats:
        show_stats()
        return
    if args.knowledge:
        embed_knowledge_rows(batch=args.batch)
    if args.docs:
        embed_docs(limit=args.limit, batch=args.batch)
    if not (args.knowledge or args.docs or args.stats):
        ap.print_help()


if __name__ == "__main__":
    main()
