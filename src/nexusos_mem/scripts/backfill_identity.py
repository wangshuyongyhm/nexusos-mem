#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""backfill_identity.py — Task1524 M1单B：身份注册表存量全量回填（幂等）。

源①：knowledge 全表 → mem 身份行（server-side INSERT..SELECT，含 deprecated——
      身份=存在，活跃口径对账看 identity_coverage 视图）。
源②a：06-Memory/MEM镜像/*.md → mirror_shard 身份行（每片一行；片内条目锚
      仅用于覆盖率报告，身份层零关系列）。
源②b：06-Memory/MEM图谱/MEM-<id>.md → graph_card 身份行（frontmatter 裸
      mem_id 是锚；悬空卡——knowledge 无此 id——跳过并计数，不造 FK 违例）。

重复执行安全：全部 ON CONFLICT DO NOTHING，重跑只补新增。
用法：python3 backfill_identity.py [--vault-root PATH]
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

MIRROR_DIR = "06-Memory/MEM镜像"
GRAPH_DIR = "06-Memory/MEM图谱"
_HEADING_ID_RE = re.compile(r"^## \[#(\d+)\]", re.M)
_FM_MEMIDS_RE = re.compile(r"^mem_ids: \[([\d,\s]*)\]", re.M)
_FM_MEMID_RE = re.compile(r"^mem_id: (\d+)", re.M)


def _kind_count(cur, kind):
    cur.execute("SELECT count(*) FROM knowledge_identity WHERE identity_kind=%s", (kind,))
    return cur.fetchone()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault-root", default=str(VAULT_ROOT))
    args = ap.parse_args()
    vault = Path(args.vault_root)

    conn = psycopg2.connect(get_dsn())
    report = {}
    with conn.cursor() as cur:
        before = {k: _kind_count(cur, k)
                  for k in ("mem", "mirror_shard", "graph_card")}

        # 源①：knowledge 全表 → mem 身份（含 path 残留原样迁入：#cN 锚点保留）
        cur.execute(
            "INSERT INTO knowledge_identity (identity_kind, mem_id, vault_path, producer) "
            "SELECT 'mem', id, path, 'backfill:1524' FROM knowledge "
            "ON CONFLICT DO NOTHING"
        )
        cur.execute("SELECT count(*) FROM knowledge")
        knowledge_rows = cur.fetchone()[0]
        conn.commit()

        # 源②a：镜像分片扫描
        shard_files = sorted((vault / MIRROR_DIR).glob("*.md")) \
            if (vault / MIRROR_DIR).is_dir() else []
        fm_anchor = heading_anchor = 0
        shard_rows = []
        for f in shard_files:
            rel = f"{MIRROR_DIR}/{f.name}"
            shard_rows.append(("mirror_shard", None, rel, "backfill:1524"))
            text = f.read_text(encoding="utf-8", errors="replace")[:20000]
            if _FM_MEMIDS_RE.search(text):
                fm_anchor += 1
            elif _HEADING_ID_RE.search(text):
                heading_anchor += 1
        report["mirror_shard"] = {
            "files": len(shard_files),
            "with_frontmatter_mem_ids": fm_anchor,
            "legacy_heading_anchor_only": heading_anchor,
            "no_anchor": len(shard_files) - fm_anchor - heading_anchor,
        }

        # 源②b：图谱卡扫描（frontmatter mem_id；悬空 FK 跳过计数）
        cur.execute("SELECT id FROM knowledge")
        valid_ids = {r[0] for r in cur.fetchall()}
        card_files = sorted((vault / GRAPH_DIR).glob("MEM-*.md")) \
            if (vault / GRAPH_DIR).is_dir() else []
        card_rows = []
        dangling = no_anchor = 0
        for f in card_files:
            text = f.read_text(encoding="utf-8", errors="replace")[:2000]
            m = _FM_MEMID_RE.search(text)
            if not m:
                no_anchor += 1
                continue
            mid = int(m.group(1))
            if mid not in valid_ids:
                dangling += 1
                continue
            card_rows.append(("graph_card", mid, f"{GRAPH_DIR}/{f.name}", "backfill:1524"))
        report["graph_card"] = {
            "files": len(card_files),
            "with_mem_id": len(card_rows) + dangling,
            "dangling_fk_skipped": dangling,
            "no_mem_id": no_anchor,
        }
        conn.close()

    register_many(shard_rows)
    register_many(card_rows)

    conn = psycopg2.connect(get_dsn())
    with conn.cursor() as cur:
        after = {k: _kind_count(cur, k)
                 for k in ("mem", "mirror_shard", "graph_card")}
        cur.execute("SELECT * FROM identity_coverage")
        cov = cur.fetchone()
        cur.execute("SELECT identity_kind, count(*) FROM knowledge_identity "
                    "GROUP BY identity_kind ORDER BY identity_kind")
        by_kind = {r[0]: r[1] for r in cur.fetchall()}
    conn.close()

    report["knowledge_rows"] = knowledge_rows
    report["inserted"] = {k: after[k] - before[k] for k in after}
    report["identity_by_kind"] = by_kind
    report["coverage"] = {
        "knowledge_total": cov[0], "mem_identity_total": cov[1],
        "knowledge_active": cov[2], "mem_identity_active": cov[3],
        "active_gap": cov[4],
        "graph_card_coverage_pct": round(100.0 * after["graph_card"] / max(len(card_files), 1), 2),
        "mirror_shard_coverage_pct": round(100.0 * after["mirror_shard"] / max(len(shard_files), 1), 2),
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
