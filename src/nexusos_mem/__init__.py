#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nexusos-mem — NEXUS 知识库+检索+打分（Task631 三件套之二）。

组成（自 nexus/ 迁入，模块名不变）:
  - shared: SharedKnowledge 知识库读写核心（PG）
  - search: FTS+pgvector 双模式检索（BGE 三级降级隔离面）
  - hyde: hyDE 查询扩展
  - api_server / embed_server / rerank_server: 三 HTTP 服务（:11537/:11436/:11437）
  - apply_honesty / algorithms / context_capture / shadow_match: 打分体系
  - embed_corpus: 批量回填 embedding
  - parser / env_scrub / nexus_mem_utils: 支撑件
  - config/: 场景配置 yaml（Task1597 收编为包内真源实体——wheel 可打包；旧位 nexus/config 反向 symlink 兼容）

依赖分层（池#72）:
  - 核心: psycopg2 + pyyaml（全 API 服务路径必需）
  - [full] extras: BGE 系（sentence-transformers/torch/FlagEmbedding/transformers）
    ——全部保持函数级懒 import，缺席时 FTS 兜底

旧路径兼容: nexus.shared / nexus.search / ... 经 nexus/ 下 re-export shim
继续可用（Task631 零破坏层）；nexus_mem_utils.py 平铺旧位同 shim 保活。
"""
import sys as _sys
from pathlib import Path as _Path

__version__ = "1.0.0"

# ── 路径自愈（批2 过渡层）────────────────────────────────────────
# 迁移件仍引用 vault 级共享件（nexus.constants 的 VAULT_ROOT/SHADOW_INDEX_PATH、
# 2 处 nexus→nexusos_xdr 懒 import——任务书 §5.5 明令保持现状）。editable 安装时
# 本包位于 vault 内 pypi/nexusos-mem/src/nexusos_mem/，向上 4 层即 自动化脚本/，
# 先入 sys.path 再复用 nexus.paths_setup 单源（_AUTO_DIR+_XDR_DIR+~/.hermes/scripts）。
# 批4 wheel 独立部署时本目录无 nexus/ → 跳过（由 pip 依赖承接）。
_AUTO_DIR = _Path(__file__).resolve().parents[4]
if (_AUTO_DIR / "nexus").is_dir():
    # Task668 ②: 包内自足版（_vault_env.setup_sys_path 内部同款逻辑——vault
    # 布局才副作用，干净环境 no-op；替代对 vault 级 nexus.paths_setup 的顶层依赖）
    from nexusos_mem import _vault_env as _ve_init  # noqa: E402
    _ve_init.setup_sys_path()
