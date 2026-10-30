#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""backfill_citation_relations.py — 1522 A 项：全库 citation 边一次性回填。

relation.backfill_citations() 的薄 CLI 包装（单一真源在库函数）。
幂等（ON CONFLICT），重跑只补新增。用法:
    python3 backfill_citation_relations.py [--dry-run]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nexusos_mem import relation  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    stats = relation.backfill_citations(dry_run=args.dry_run)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
