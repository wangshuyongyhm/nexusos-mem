#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refresh_readmodel.py — Task1538 M3：统一读模型全量刷新（幂等，只读真源）。

点 = knowledge_identity 全空间（mem/vault_doc/mirror_shard/graph_card）；
   刷新顺带把全部 Vault .md 注册为 vault_doc 身份（1524 遗留收口——路径未
   被其他 kind 占用的 .md 才入，防同文件双身份）。
边 = ① knowledge_relation 全 rel_type（双语义翻译：backfill:1534=identity 空间
     直迁；writechain:*/citation 系=knowledge 空间经 mem 身份行映射，文件代表
     chunk 优先翻成 vault_doc 身份）
   ② Vault wikilink 文件边（复用 nexusos-gate _vaultlink 解析器——Task1301 抗
     代码块围栏版，勿自写）

读模型非真源：TRUNCATE 禁用（B2），代际替换 = DELETE ... WHERE refresh_id <> 本代。
漂移检测：readmodel_meta 对比上次点/边数。
用法：python3 refresh_readmodel.py [--vault-root PATH] [--skip-wikilink]
"""
import argparse
import datetime as _dt
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import psycopg2  # noqa: E402
from nexusos_mem._vault_env import VAULT_ROOT  # noqa: E402
from nexusos_mem.identity import register_many  # noqa: E402
from nexusos_mem.nexus_mem_utils import get_dsn  # noqa: E402

CHUNK_RE = re.compile(r"#c\d+$")

VAULTLINK_DIR = (Path(__file__).resolve().parents[4]
                 / "nexusos-gate" / "src" / "nexusos_gate" / "etc" / "gates" / "_checks")


def _iter_vault_md(vault: Path):
    skip_parts = {".git", ".trash", "05-System" + "/backups"}
    for p in vault.rglob("*.md"):
        rel = p.relative_to(vault)
        if any(part in skip_parts for part in rel.parts):
            continue
        yield p, str(rel).replace("\\", "/")


def _load_vaultlink():
    sys.path.insert(0, str(VAULTLINK_DIR))
    import _vaultlink  # noqa: E402  Task592/1301 共享解析器
    _vaultlink.reset_cache()
    return _vaultlink


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault-root", default=str(VAULT_ROOT))
    ap.add_argument("--skip-wikilink", action="store_true",
                    help="跳过 wikilink 扫描（调试用；点/关系边仍刷新）")
    args = ap.parse_args()
    vault = Path(args.vault_root)
    t0 = time.time()

    conn = psycopg2.connect(get_dsn())
    cur = conn.cursor()

    # ── 1. 节点空间补全：全部 vault .md → vault_doc 身份（幂等） ──
    cur.execute("SELECT vault_path FROM knowledge_identity WHERE identity_kind <> 'mem'")
    taken = {r[0] for r in cur.fetchall()}
    md_rows = []
    md_files = []
    for p, rel in _iter_vault_md(vault):
        md_files.append((p, rel))
        if rel not in taken:
            md_rows.append(("vault_doc", None, rel, "readmodel:1538"))
    if md_rows:
        register_many(md_rows)
    n_new_docs = len(md_rows)

    # ── 2. 代际 refresh_id + 旧代清理（B2：带 WHERE 的代际替换） ──
    cur.execute("SELECT coalesce(max(refresh_id),0) FROM readmodel_meta")
    refresh_id = cur.fetchone()[0] + 1
    cur.execute("DELETE FROM readmodel_edge WHERE refresh_id <> %s", (refresh_id,))
    cur.execute("DELETE FROM readmodel_node WHERE refresh_id <> %s", (refresh_id,))
    conn.commit()

    # ── 3. 点：identity 全空间投影 ──
    cur.execute(
        "INSERT INTO readmodel_node (identity_id, kind, mem_id, vault_path, refresh_id) "
        "SELECT id, identity_kind, mem_id, vault_path, %s FROM knowledge_identity",
        (refresh_id,))

    # 身份解析缓存（🔴 必须 kind='mem' 过滤：graph_card 行也带 mem_id 同值，
    # 不滤则 dict 后写覆盖 → 端点翻错身份——1538 实证 81127 被卡片行覆盖）
    cur.execute("SELECT id, mem_id FROM knowledge_identity "
                "WHERE identity_kind='mem' AND mem_id IS NOT NULL")
    mem_ident = {r[1]: r[0] for r in cur.fetchall()}
    cur.execute("SELECT id, vault_path FROM knowledge_identity WHERE vault_path IS NOT NULL")
    path_ident = {r[1]: r[0] for r in cur.fetchall()}
    # knowledge id -> 文件身份（1533 writechain dst=代表 chunk → 文件 vault_doc 优先）
    cur.execute("SELECT id, path FROM knowledge WHERE path IS NOT NULL")
    kfile = {}
    for kid, p in cur.fetchall():
        base = CHUNK_RE.sub("", p)
        kfile[kid] = path_ident.get(base) or mem_ident.get(kid)

    def to_ident(kid, prefer_file=False):
        if prefer_file and kid in kfile:
            return kfile[kid]
        return mem_ident.get(kid)

    # ── 4. 边①：knowledge_relation 全类型（双语义翻译） ──
    cur.execute("SELECT src_id, dst_id, rel_type, evidence, created_by FROM knowledge_relation")
    edge_rows = []
    for src, dst, rtype, evid, cby in cur.fetchall():
        if cby == "backfill:1534":
            s, d = src, dst  # identity 空间直迁
        else:
            s = to_ident(src)
            d = to_ident(dst, prefer_file=(rtype == "provenance"))
        if s and d and s != d:
            edge_rows.append((s, d, rtype, f"{cby}|{(evid or '')[:80]}"))

    # ── 5. 边②：wikilink 文件边（复用 _vaultlink） ──
    n_wikilink = 0
    if not args.skip_wikilink:
        vl = _load_vaultlink()
        bname_index = {}  # basename(casefold) -> rel_path（首个，Obsidian 同名歧义取先到）
        for _p, rel in md_files:
            bname_index.setdefault(Path(rel).stem.casefold(), rel)
        for p, rel in md_files:
            src_ident = path_ident.get(rel)
            if not src_ident:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for target in vl.extract_wikilinks(text):
                name = target.split("|")[0].split("#")[0].strip().strip('"')
                if not name:
                    continue
                cand = name if name.endswith(".md") else name + ".md"
                trel = None
                if (vault / cand).is_file():
                    trel = cand
                else:
                    trel = bname_index.get(name.casefold())
                dst_ident = path_ident.get(trel) if trel else None
                if dst_ident and dst_ident != src_ident:
                    edge_rows.append((src_ident, dst_ident, "wikilink",
                                      f"[[{target[:60]}]] in {rel[:60]}"))
                    n_wikilink += 1

    # Task1570：读模型边是集合语义（同内容元组每代只应一行）——写入侧先去重。
    # 1547 实证 wikilink 按出现次数插行（铁律文件 17 次引用→17 行同内容边，
    # evidence 不含出现序号=纯冗余）；dict.fromkeys 保序去重，元组可哈希。
    edge_rows = list(dict.fromkeys(edge_rows))

    for i in range(0, len(edge_rows), 2000):
        cur.executemany(
            "INSERT INTO readmodel_edge (src_identity, dst_identity, edge_kind, evidence, refresh_id) "
            "VALUES (%s, %s, %s, %s, %s)", [(s, d, k, e, refresh_id) for s, d, k, e in edge_rows[i:i+2000]])

    # ── 6. 汇总 + 漂移对比 ──
    cur.execute("SELECT count(*) FROM readmodel_node WHERE refresh_id=%s", (refresh_id,))
    n_nodes = cur.fetchone()[0]
    cur.execute("SELECT edge_kind, count(*) FROM readmodel_edge WHERE refresh_id=%s GROUP BY 1", (refresh_id,))
    kinds = {r[0]: r[1] for r in cur.fetchall()}
    n_edges = sum(kinds.values())
    cur.execute("SELECT refreshed_at, nodes, edges, edge_kinds FROM readmodel_meta ORDER BY refreshed_at DESC LIMIT 1")
    prev = cur.fetchone()
    cur.execute(
        "INSERT INTO readmodel_meta (refresh_id, nodes, edges, edge_kinds) VALUES (%s,%s,%s,%s)",
        (refresh_id, n_nodes, n_edges, json.dumps(kinds, ensure_ascii=False)))
    conn.commit()

    report = {
        "refresh_id": refresh_id, "nodes": n_nodes, "edges": n_edges,
        "edge_kinds": kinds, "vault_doc_new": n_new_docs,
        "wikilink_edges": n_wikilink,
        "prev": {"nodes": prev[1], "edges": prev[2], "kinds": prev[3]} if prev else None,
        "drift": (n_edges - prev[2]) if prev else None,
        "elapsed_sec": round(time.time() - t0, 1),
        "refreshed_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
