#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nexus.search — 知识库搜索引擎（task 239 #1：从 shared.py 拆出）

单一职责: 实现 PostgreSQL FTS + ILIKE + zvec 三路融合搜索。

接口:
    pgvector_search(conn, query, top_k, ...) → List[dict]
        主路：向量 ∪ FTS 并集融合 + 父文档去重（task 253 P1-3）
    pg_fts_search(conn, query, top_k, ...) → List[dict]
        纯 PostgreSQL 全文+子串匹配
    fts_fallback_search(conn, query, top_k, ...) → List[dict]
        embedding 全表缺失时的降级路径

历史: task 236 P0-4 在 shared.py 中新增了 _zvec_search()。task 245 zvec 退役、
task 253 pgvector_search 成为主路并清理了 zvec 死分支（zvec_semantic_search /
hybrid_search / use_zvec 参数均已移除）。
"""

import os
import re
import math
from datetime import datetime, timezone
from pathlib import Path  # Task 282 P2-4: _load_term_dict 用 Path 读 config YAML
from typing import Any, List, Optional

import psycopg2
import psycopg2.extras


# ---------------------------------------------------------------------------
# task 245: BGE-M3 嵌入模型单例 + pgvector 向量搜索
# ---------------------------------------------------------------------------
# 替代 zvec subprocess 桥接：模型在进程内懒加载（首次 ~5s），后续复用。
# 短命进程（cron/CLI）每次付一次加载成本；长命进程（Hermes）常驻。

_BGE_M3_MODEL = None
BGE_M3_NAME = "BAAI/bge-m3"
BGE_M3_DIM = 1024
# 嵌入输入截断（与 embed_corpus.DOC_MAX_CHARS 一致；BGE-M3 max 8192 tokens）
EMBED_MAX_CHARS = 4000

# task 253 P1-7: 常驻 embed worker（nexus.embed_server）。
# encode_query 优先走 worker(任意 Python 版本零重依赖)，worker 不可用时
# 降级到进程内本地模型（需 3.11 + sentence_transformers）。
# Task 561: 网关类进程（api_server 等）禁用本地降级——worker 失败曾把 BGE-M3
# 模型+torch 全家桶拉进 api_server（footprint 3.8GB）。设
# NEXUS_EMBED_LOCAL_FALLBACK=0 后 worker 失败直接降级 FTS（返回 None），
# 不再加载本地模型。默认 1 保持旧行为（CLI 直跑等场景仍可用本地降级）。
_EMBED_LOCAL_FALLBACK = os.environ.get("NEXUS_EMBED_LOCAL_FALLBACK", "1") == "1"
EMBED_SERVER_URL = os.environ.get(
    "NEXUS_EMBED_URL", "http://127.0.0.1:11436"
)
# 是否禁用 worker（worker 自身进程内必须禁用，否则自我递归调用）
_EMBED_WORKER_DISABLED = os.environ.get("NEXUS_EMBED_NO_WORKER") == "1"
# Task1604: 10→3。LOOP 多线并行时 fsevents 文档嵌入洪峰占用 MPS，查询编码
# 从空闲 0.2s 恶化到 2-3s+；10s 无预算让每次慢 encode 都全额吃到 /search 延迟。
# 3s 是「正常洪峰内可完成、超限快速降级 FTS」的平衡点（环境变量可调）。
_EMBED_WORKER_TIMEOUT = float(os.environ.get("NEXUS_EMBED_TIMEOUT", "3"))
# worker 探测失败后的冷却，避免每次 encode 都吃一次连接超时
_worker_down_until = 0.0
_WORKER_COOLDOWN = 30.0
# task 267 R5: worker 失败时先重试再进冷却（瞬时故障容错）
# Task1604: 2→1。连接拒绝类瞬时故障重试 1 次足够；对「已连接但慢」的请求
# 重试只会把延迟预算翻倍——慢响应不重试，超时即降级。
_EMBED_WORKER_RETRIES = 1  # 首次 + 重试 1 次 = 共 2 次尝试
_EMBED_WORKER_RETRY_DELAYS = [0.5]  # 秒

# Task 458: Reranker cross-encoder 精排（bge-reranker-v2-m3）
# 在 fusion sort 后、父文档去重前插入，对 top-pool 精排。
# worker 不可用时降级（用原融合分排序）。
RERANK_ENABLED = os.environ.get("NEXUS_RERANK_ENABLED", "1") != "0"
RERANK_SERVER_URL = os.environ.get("NEXUS_RERANK_URL", "http://127.0.0.1:11437")
# Task1604: 30→16。rerank 按 chunk=8 分块批推理，16 仍是 2 块（M4 MPS 实测
# 8doc≈1.35s、30doc≈5s）——16 与 8 同块数同延迟，但精排覆盖多一倍候选；
# 30（4 块）在串行批上把每次 /search 基线抬到 5s+，是本次 hang 的主因之一。
rerank_pool = int(os.environ.get("NEXUS_RERANK_POOL", "16"))
_rerank_down_until = 0.0
_RERANK_COOLDOWN = 30.0
# Task1604: 2→0。rerank 是增值层（失败降级用原融合分，语义无损）；进程内锁
# 串行 + 洪峰排队时重试只会把延迟放大 3 倍。单次尝试超时即降级，30s 冷却
# 兜底持续故障（冷却期内直接跳过 rerank，零成本）。
_RERANK_RETRIES = 0
_RERANK_RETRY_DELAYS = [0.5]  # 保留列表结构（RETRIES=0 时不使用）
# Task1604: 30→3。16 文档批空闲 ~2.7s < 3s 正常完成；MPS 争用/排队超 3s
# 说明洪峰中，立即降级融合分保 /search 总延迟。
_RERANK_TIMEOUT = float(os.environ.get("NEXUS_RERANK_TIMEOUT", "3"))
# 防自我递归（rerank_server 内部不调 :11437）
_RERANK_DISABLED = os.environ.get("NEXUS_RERANK_NO_WORKER") == "1"

# Task1612 #244: search 的 PG 查询层预算（秒）。1604 应用层已把 embed 查询 3s/rerank 3s
# 封顶，但 PG 风暴期（舰队批量 DELETE/backfill 拖慢全表 SELECT）search SQL 段无预算——
# 1604 实测 15-22s 窗口。默认 5s（比应用层宽松：正常 recall <1s，风暴期单查询 >5s 即
# 说明撞上长锁等待，降级收益大于硬等）。毫秒值供 SET statement_timeout 用。
_SEARCH_PG_TIMEOUT_SEC = float(os.environ.get("NEXUS_SEARCH_PG_TIMEOUT", "5"))
_SEARCH_PG_TIMEOUT_MS = int(_SEARCH_PG_TIMEOUT_SEC * 1000)


class pg_statement_budget:
    """查询块级 statement_timeout 上下文（Task1612 #244）。

    SET statement_timeout 包住 search 的召回 SQL，finally RESET——
    - 块级而非会话级：同连接的写路径/annotate/后续 enrich 读不受预算约束
      （写查询不该被 5s 读预算误杀）；
    - 无跨线程竞态：api_server 每请求线程一个 SharedKnowledge（thread-local），
      连接线程独享，SET/RESET 只在自身线程可见；
    - 超时抛 psycopg2.errors.QueryCanceled（sqlstate 57014），调用方决定降级路径。
    """

    def __init__(self, conn, timeout_ms=_SEARCH_PG_TIMEOUT_MS):
        self._conn = conn
        self._timeout_ms = timeout_ms

    def __enter__(self):
        cur = self._conn.cursor()
        try:
            cur.execute("SET statement_timeout = %s", (self._timeout_ms,))
        finally:
            cur.close()
        return self

    def __exit__(self, exc_type, exc, tb):
        cur = self._conn.cursor()
        try:
            cur.execute("RESET statement_timeout")
        finally:
            cur.close()
        return False  # 异常照常传播

# Task 459: HyDE 假设文档改写（弱 query 向量召回增强）
# 智能触发：短 query <6字 / 术语词典未命中 / 含口语词。
# Task588: 默认关（hyde.py 单一真源，默认值翻转见彼处注释）；
# 双定义已消除——此处从 hyde 模块导入同一常量（hyde 无 search 依赖，无循环）。
# 失败/超时/provider 不可用 → fail-open 用 expand_query 向量（与无 HyDE 等价）
from nexusos_mem.hyde import HYDE_ENABLED  # noqa: E402  Task588: 单一真源导入


def _get_bge_m3():
    """懒加载 BGE-M3 模型（进程内单例）。

    Task 562: fp16 权重加载（内存 ~5.4G→~3G）。库内旧向量仍是 fp32 生成——
    cos 相似度层面 fp16 查询向量与 fp32 库向量混合检索兼容（质量验证
    task/562-model-fp16：20 词重合率 ≥95% 门槛已验证）。
    """
    global _BGE_M3_MODEL
    if _BGE_M3_MODEL is None:
        # 1636 活②(#20): transformers 4.57+ 的 processor 探测会对 hf.co 发一串
        # HEAD(*_preprocessor_config.json)——网络超时重试数分钟后报
        # Unrecognized processing class(本地缓存本无 processor 文件,BGE-M3 是
        # 纯 encoder)。HF_HUB_OFFLINE=1 让 hub 全离线直走本地缓存(秒级加载,
        # 已实证);setdefault 不覆盖显式要联网的调用方。
        import os as _os
        _os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from sentence_transformers import SentenceTransformer
        _BGE_M3_MODEL = SentenceTransformer(
            BGE_M3_NAME, local_files_only=True,
            model_kwargs={"dtype": torch.float16},
        )
    return _BGE_M3_MODEL


def _rerank_via_worker(query: str, pool: list) -> Optional[list]:
    """Task 458: 调 :11437 rerank worker 精排 pool（list of row dict）。

    失败返回 None（降级用原融合分）。重试 2 次 + 30s 冷却（对齐 _encode_via_worker）。
    """
    global _rerank_down_until
    import time
    if _RERANK_DISABLED or not RERANK_ENABLED or time.time() < _rerank_down_until:
        return None
    if not pool:
        return None
    # 取每行 title + content 前 500 字符（reranker max 512 token）
    documents = [
        (str(row.get("title", "")) + " " + str(row.get("content", ""))[:500]).strip()
        for row in pool
    ]
    last_err = None
    for attempt in range(_RERANK_RETRIES + 1):
        try:
            import json
            import urllib.request
            req = urllib.request.Request(
                f"{RERANK_SERVER_URL}/rerank",
                data=json.dumps(
                    {"query": query, "documents": documents, "top_n": len(pool)},
                    ensure_ascii=False,
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=_RERANK_TIMEOUT) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            results = payload.get("results")
            if results and isinstance(results, list):
                # 按返回的 index 重排 pool。1401-d5a：rerank 归一化分（sigmoid∈[0,1]）
                # 以 _rerank_score 键【原地】附到 row 上（调用方需回写 final_score，
                # 保持「返回顺序=score 降序」契约——原实现只重排不回写，E2E 实测
                # 0.3032 排在 0.3123 前的顺序脱钩）。原地写而非改返回结构：外部
                # 消费者（nexusos_xdr/test_fault_injection_596）断言返回值==row 列表。
                out = []
                for r in results:
                    idx = r.get("index", -1)
                    if not (0 <= idx < len(pool)):
                        continue
                    pool[idx]["_rerank_score"] = max(
                        0.0, min(1.0, float(r.get("score", 0.0) or 0.0)))
                    out.append(pool[idx])
                return out
            return None
        except Exception as e:
            last_err = e
            if attempt < _RERANK_RETRIES:
                time.sleep(_RERANK_RETRY_DELAYS[attempt])
                continue
    _rerank_down_until = time.time() + _RERANK_COOLDOWN
    print(f"[WARN] rerank worker {_RERANK_RETRIES + 1}次重试全失败，"
          f"进{_RERANK_COOLDOWN}s冷却: {last_err}", flush=True)
    return None


def _encode_via_worker(text: str) -> Optional[list]:
    """尝试通过常驻 embed worker 编码。不可用返回 None（触发本地降级）。

    task 267 R5: worker 失败时先重试（瞬时故障容错），全失败才进冷却。
    只在网络/超时类异常重试；worker 响应正常但无 vectors（服务逻辑问题）不重试。
    """
    global _worker_down_until
    import time
    if _EMBED_WORKER_DISABLED or time.time() < _worker_down_until:
        return None
    last_err = None
    for attempt in range(_EMBED_WORKER_RETRIES + 1):
        try:
            import json
            import urllib.request
            req = urllib.request.Request(
                f"{EMBED_SERVER_URL}/embed",
                data=json.dumps({"text": text}, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=_EMBED_WORKER_TIMEOUT) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            vecs = payload.get("vectors")
            if vecs and isinstance(vecs, list):
                return vecs[0]
            # worker 响应正常但无 vectors（服务逻辑问题）→ 不重试
            return None
        except Exception as e:
            # 网络/超时类异常 → 重试
            last_err = e
            if attempt < _EMBED_WORKER_RETRIES:
                time.sleep(_EMBED_WORKER_RETRY_DELAYS[attempt])
                continue
    # 全部重试失败 → 进冷却 + 告警（task 267 R5: 静默失败可见化）
    _worker_down_until = time.time() + _WORKER_COOLDOWN
    print(f"[WARN] embed worker {_EMBED_WORKER_RETRIES + 1}次重试全失败，"
          f"进{_WORKER_COOLDOWN}s冷却: {last_err}", flush=True)
    return None


def encode_query(text: str) -> Optional[list]:
    """把查询文本编码为 BGE-M3 归一化向量。失败返回 None。

    路径: 常驻 worker(HTTP) → 进程内本地模型 → None(降级 FTS)。
    向量口径三条路径一致(normalize_embeddings=True, 1024 维)。
    Task 561: NEXUS_EMBED_LOCAL_FALLBACK=0 时跳过本地模型（网关进程防 torch 载入）。
    """
    vec = _encode_via_worker(text)
    if vec is not None:
        return vec
    if not _EMBED_LOCAL_FALLBACK:
        print("[WARN] embed worker 不可用且本地降级已禁用(NEXUS_EMBED_LOCAL_FALLBACK=0)，降级 FTS",
              flush=True)
        return None
    try:
        model = _get_bge_m3()
        v = model.encode(text, normalize_embeddings=True)
        return v.tolist()
    except Exception as e:
        print(f"[WARN] BGE-M3 encode 失败（worker 与本地均不可用），降级 FTS: {e}",
              flush=True)
        return None
# P3-2: Query Expansion 术语词典（中英混合技术术语同义词）
# Task 282 P2-4: _TERM_DICT 外置到 config/term_dict.yaml（原 16 个术语硬编码）
def _load_term_dict():
    """从 config/term_dict.yaml 加载术语词典，失败降级空字典。"""
    cfg = Path(__file__).parent / "config" / "term_dict.yaml"
    if cfg.exists():
        try:
            import yaml
            return yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        except Exception:
            pass
    return {}

_TERM_DICT = _load_term_dict()


def expand_query(query: str) -> str:
    """P3-2: 查询扩展 — 用术语词典增强中英混合查询。

    拆分查询词，对每个词在术语词典里找同义词/扩展词，
    拼接成增强查询返回。不改变原始查询的语义方向。
    """
    query_lower = query.lower()
    expansions = []
    for term, synonyms in _TERM_DICT.items():
        if term in query_lower:
            # str() 兜底：term_dict.yaml 的纯数字值会被 YAML 解析成 int，
            # 不转直接 join 会撞 TypeError（2026-07-30 503 事故：callback 的 19999）
            expansions.extend(str(s) for s in synonyms[:2])
    if expansions:
        return query + " " + " ".join(expansions[:6])  # 最多扩展6个词
    return query


def has_pgvector_embeddings(conn) -> bool:
    """检查 knowledge 表是否已有 embedding 数据（决定走 pgvector 还是旧路）。

    Task 379: 本函数接收外部 conn（连接所有权归调用方）。为防调用方传非 autocommit
    连接致只读 SELECT 泄漏 idle in transaction,入口做 autocommit 保形:非 autocommit
    时临时切 True,finally 恢复原值。生产调用方 shared.search 已走 _get_read_conn,
    此保形是防御未来误用。
    """
    _ac = getattr(conn, "autocommit", False)
    _needs_restore = (not _ac)
    if _needs_restore:
        try:
            conn.autocommit = True
        except Exception:
            _needs_restore = False
    try:
        try:
            cur = conn.cursor()
            cur.execute("SELECT EXISTS (SELECT 1 FROM knowledge WHERE embedding IS NOT NULL LIMIT 1)")
            return cur.fetchone()[0]
        except Exception:
            return False
        finally:
            try:
                cur.close()
            except Exception:
                pass
    finally:
        if _needs_restore:
            try:
                conn.autocommit = _ac
            except Exception:
                pass


def parent_path(path: Optional[str], row_id=None) -> str:
    """取 chunk 的父文档标识：剥掉尾部 #cN（P1-4 分块）。

    无 path 的结构化知识以 id 为独立父（互不去重）。
    """
    if path:
        return re.sub(r"#c\d+$", "", path)
    return f"id:{row_id}"


def _make_snippet(content: Any, query: str, width: int = 60) -> str:
    """生成匹配片段：定位查询词在 content 中的位置，取前后各 width 字符。

    content 为 None/空时返回空串；查询词未命中时返回 content 前 150 字符。
    （task 253 follow-up: 三小弟审计问题3 — 缺 snippet 预览）
    """
    content_str = str(content or "")
    if not content_str:
        return ""
    low = content_str.lower()
    pos = low.find(query.lower())
    if pos < 0:
        # 整句未命中时退一步：找第一个查询词的子串
        for word in query.split():
            pos = low.find(word.lower())
            if pos >= 0:
                break
    if pos < 0:
        return content_str[:150] + ("..." if len(content_str) > 150 else "")
    start = max(0, pos - width)
    end = min(len(content_str), pos + len(query) + width)
    return (
        ("..." if start > 0 else "")
        + content_str[start:end]
        + ("..." if end < len(content_str) else "")
    )

# 1401-d5a T2：知识类型先验系数（设计定值，详见融合打分段注释）。document 压制
# 幅度保守（0.85 非埋没），经验内核五类上浮 1.25；verification/memory 中性 1.0。
_TYPE_PRIOR = {
    "experience": 1.25, "pitfall": 1.25, "counterexample": 1.25,
    "finding": 1.25, "decision": 1.25,
    "document": 0.85,
    "verification": 1.0, "memory": 1.0, "tacit": 1.0, "uncertainty": 1.0,
}


def pgvector_search(
    conn,
    query: str,
    top_k: int = 5,
    knowledge_type: Optional[str] = None,
    project: Optional[str] = None,
    tag: Optional[str] = None,
    vec_weight: float = 0.7,
    fts_weight: float = 0.3,
    recall_mult: int = 3,
    offset: int = 0,
    core_only: bool = False,
    pool: str = "all",
    decay_override: Optional[float] = None,
    lexical_only: bool = False,
    graph_expand: bool = False,
    graph_mode: str = "annotate",
    readmodel: bool = False,
    include_semantic: bool = False,
) -> List[dict]:
    """task 253 P1-3: 真·混合检索（向量 ∪ FTS 并集融合 + 父文档去重）。

    此前实现是"向量召回后 FTS 重排"：字面精确但语义远的文档进不了向量召回集，
    FTS 只能在向量候选内重排、救不回来（逆向项目 CMD:2/5698 这类精确 token 尤甚）。
    现改为两路独立召回后并集：
        1. 向量路：BGE-M3 编码 → HNSW 召回 top_k*recall_mult
        2. FTS 路：pg_fts_search（FTS + ILIKE）召回 top_k*recall_mult
        3. 按 id 合并，缺失的分数记 0，final = vec_w*vec + fts_w*fts
        4. 父文档去重（P1-4 chunk 折叠到最高分那块），截断 top_k

    Args:
        query: 自然语言查询
        top_k: 最终返回数
        knowledge_type/project/tag: 可选过滤
        vec_weight/fts_weight: 混合权重
        recall_mult: 每路召回倍数
        graph_expand: task 1455 一跳扩展开关。False（默认）= 行为与本任务前
            逐字节一致（生产零变化）。True = 放宽召回层 superseded 过滤
            ——已被取代的旧经验重新可被召回（deprecated 探针照旧过滤，
            因为 superseded 是"版本链还有后继"而 deprecated 是"这条已死"，
            语义不同），命中后沿 knowledge_relation 边表一跳附新版指引。
        graph_mode: graph_expand=True 时的新版处理方式。
            "annotate"（默认保守）= 旧条只附 superseded_by_new 标注，
            结果序列不动。 "promote" = 新版一跳上浮插到旧条位置前
            （用户该看的是新版，旧条顺延一位保演进痕迹）。

    Returns:
        [{"id","title","content",...,"vec_score","fts_score","final_score","source"}, ...]
        source: "vec+fts" | "vec" | "fts"

    Task 379: 本函数接收外部 conn（连接所有权归调用方）。为防调用方传非 autocommit
    连接致只读 SELECT 泄漏,入口做 autocommit 保形:非 autocommit 时临时切 True,
    正常返回路径恢复原值。生产调用方 shared.search 已走 _get_read_conn(True),
    保形不切换;异常路径靠下次 _get_conn 入口强制复位 False 兜底。
    """
    _379_ac = getattr(conn, "autocommit", False)
    _379_restore = (not _379_ac)
    if _379_restore:
        try:
            conn.autocommit = True
        except Exception:
            _379_restore = False
    recall = max(top_k * recall_mult, 20)
    # 1401-d5c 消融调试口（只读实验用，默认参数=生产行为不变，验收第 2 条对照硬卡）：
    #   decay_override: 非 None 时强制 decay_mult 用该值（b 轮传 1.0=去时间衰减），
    #     为什么留调试口——H1/H2 争议（语义距离 vs 时间衰减）需消融法裁定，直接改
    #     生产常量跑实验=改一次测一次不可复现，override 参数让三轮实验同一代码基线。
    #   lexical_only: True 时跳过向量召回（query_vec 置 None → 向量路不跑、FTS 独有
    #     命中不回填向量分）——c 轮纯词面匹配，消融"语义检索"的贡献。
    if lexical_only:
        query_vec = None  # c 轮纯词面：向量路整体跳过（下方 recall 段判 None 不跑）
    else:
        # P3-2: 查询扩展增强向量召回（FTS 仍用原始 query 做 ILIKE 精确匹配）
        expanded = expand_query(query)
        # ---- Task 459: HyDE 假设文档改写（弱 query 向量召回增强）----
        # 仅对短/口语/术语未命中的 query 触发，假设文档替代 expand_query 结果编码
        # 失败/超时/provider 不可用 → fail-open 用 expanded（与无 HyDE 等价）
        # 注意：FTS 路（下方 cur.execute fts sql）仍用原始 query，保精确 token 匹配
        hypoth = None
        _hyde_used = False
        if HYDE_ENABLED:
            try:
                from nexusos_mem import hyde as _hyde_mod
                if _hyde_mod.should_hyde(query, term_dict_hit=(expanded != query)):
                    hypoth = _hyde_mod.generate_hypothetical(query)
                    if hypoth:
                        _hyde_used = True
            except Exception as _e:
                print(f"[WARN] HyDE 模块异常（降级）: {_e}", flush=True)
        query_vec = encode_query(hypoth if hypoth else expanded)

    where_extra = ""
    params_extra = []
    if knowledge_type:
        where_extra += " AND knowledge_type = %s"
        params_extra.append(knowledge_type)
    if project:
        where_extra += " AND project = %s"
        params_extra.append(project)
    if tag:
        where_extra += " AND %s = ANY(tags)"
        params_extra.append(tag)
    if core_only:
        where_extra += " AND knowledge_type IN ('experience','finding','decision','pitfall')"

    # 1401-d5a T2 分池：pool=experience 只搜经验内核（experience/pitfall/
    # counterexample/finding/decision 五类——D5 设计稿块1「活知识」口径）；
    # pool=document 只搜导入切片；all=默认混排（加权后）。与 core_only 的区别：
    # core_only 是旧 core 四类（无 counterexample），保留不动；pool 是新契约参数。
    if pool == "experience":
        where_extra += (" AND knowledge_type IN ('experience','pitfall',"
                        "'counterexample','finding','decision')")
    elif pool == "document":
        where_extra += " AND knowledge_type = 'document'"

    merged: dict = {}

    # task 1455: graph_expand=True 时放宽 superseded 过滤——旧版重新可召回
    # （否则被取代的条目永远进不了候选，"命中旧条附新版指引"无从触发）。
    # deprecated（探针/垃圾）过滤两种模式都保留：死了就是死了，版本链才可回溯。
    _sup_filter = "" if graph_expand else " AND superseded_by IS NULL"

    # ---- 向量路 ----
    if query_vec is not None:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        sql = f"""
            SELECT id, title, content, source_agent, tags, project, task_id,
                   knowledge_type, shadow_code, confidence, created, path,
                   context_coord, outcome, ref_count, updated,
                   1 - (embedding <=> %s::vector) AS vec_score
            FROM knowledge
            WHERE embedding IS NOT NULL{_sup_filter} AND COALESCE(deprecated, FALSE) = FALSE{where_extra}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        cur.execute(sql, [query_vec] + params_extra + [query_vec, recall])
        for r in cur.fetchall():
            row = dict(r)
            row["vec_score"] = max(0.0, float(row.pop("vec_score", 0.0) or 0.0))
            row["fts_score"] = 0.0
            merged[row["id"]] = row
        cur.close()

    # ---- FTS 路（独立召回，把纯关键词命中拉进候选集）----
    fts_rows = pg_fts_search(
        conn, query, top_k=recall,
        knowledge_type=knowledge_type, project=project, tag=tag,
        core_only=core_only, pool=pool,
        graph_expand=graph_expand,
    )
    for r in fts_rows:
        fts_s = max(0.0, min(1.0, float(r.get("rank", 0.0) or 0.0)))
        if r["id"] in merged:
            merged[r["id"]]["fts_score"] = fts_s
        else:
            r["vec_score"] = 0.0
            r["fts_score"] = fts_s
            merged[r["id"]] = r

    # ---- 给 FTS 独有命中回填向量分 ----
    # 不回填的话 FTS-only 行 final=0.3*0.5=0.15，永远排在 20 个泛化向量候选
    # (~0.4+)之后，"并集"名存实亡。回填后精确 token 命中若语义也贴近会自然置顶。
    if query_vec is not None:
        need_vec = [rid for rid, row in merged.items() if row["vec_score"] == 0.0]
        if need_vec:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, 1 - (embedding <=> %s::vector) FROM knowledge "
                "WHERE id = ANY(%s) AND embedding IS NOT NULL",
                (query_vec, need_vec),
            )
            for rid, vs in cur.fetchall():
                merged[rid]["vec_score"] = max(0.0, float(vs or 0.0))
            cur.close()

    # ---- 融合打分 ----
    # 注意：confidence 不参与融合打分（Task 257 证伪 σ 不可解，硬分档 artifact），
    # 仅保留在 SELECT 返回体做向后兼容（L265/L440），LLM 不应据此判断可信度。
    # 见 task/257-catd-prototype/output/report.md。
    # Task616: 候选集内 ref_count 最大值（对数归一分母）——打分前一次聚合
    _ref_count_max = max((int(r.get("ref_count") or 0) for r in merged.values()), default=0)

    results = []
    for row in merged.values():
        vec_s = row.get("vec_score", 0.0)
        fts_s = row.get("fts_score", 0.0)
        # FTS是主要命中源时加权提升（精确token场景，task 253 follow-up: 问题4）
        if fts_s > 0 and fts_s >= vec_s:
            effective_vec_weight = 0.5  # 从默认0.7/0.3调到0.5/0.5
            effective_fts_weight = 0.5
        else:
            effective_vec_weight = vec_weight
            effective_fts_weight = fts_weight
        final = effective_vec_weight * vec_s + effective_fts_weight * fts_s
        # P3-1: 知识质量加权 — 核心经验 > 上下文文档（boost 不改 vec/fts 原始分）
        # Task 355 审计：此 additive boost 已等效 type_boost。实测搜"链路验证"
        # 在 document:experience=22257:294 悬殊比例下 experience 仍包揽 top3（score 0.83/0.78/0.75）。
        # 不再加 multiplicative type_boost（会叠加过度：0.8→+0.15→0.95→×1.15→1.0 饱和）。
        # 本任务唯一调整：uncertainty 0.0→0.10（任务书 §5 HIGH_VALUE_TYPES 含 uncertainty，
        # 原代码漏加，与 finding/pitfall 对齐）。
        _ktype = row.get("knowledge_type", "document")
        _quality_boost = {"experience": 0.15, "finding": 0.10, "decision": 0.12,
                          "pitfall": 0.10, "document": 0.0, "tacit": 0.05,
                          "uncertainty": 0.10}
        # shadow_code 层级加成：A→L0(+0.15 核心约束最高优先) B→L1(+0.10)
        # C→L2(+0.05) D/E/...(L3+)不加成。与 quality_boost 叠加（NULL 行不加成）
        _shadow_code = row.get("shadow_code")
        _shadow_layer_boost = 0.0
        if _shadow_code:
            _shadow_layer_boost = {"A": 0.15, "B": 0.10, "C": 0.05}.get(
                str(_shadow_code)[0].upper(), 0.0)
        # Task387 Part C: outcome 历史战绩加权（认知闭环收紧）
        # success→+0.10(被验证有效,提升), partial→0.0(中性), failed→-0.15(被证伪,降权)
        # 与 quality_boost/shadow_layer_boost 叠加。NULL(未记录)不加成(中性)。
        _outcome_boost = {"success": 0.10, "partial": 0.0, "failed": -0.15}.get(row.get("outcome"), 0.0)
        # Task616 ref_count 维度（v3 w2=0.25 起点调和）：对数归一防大数独大
        # log(1+n)/log(1+max)——n=0→0，n=max→1，n=3（当前实测 max）→1.0。
        # 实测分布 max=3/avg=0.002（38/25699 有引用），乘 w2=0.25 后满配 +0.25
        # 会淹没其他维度（vec/fts 主分 0~1），调和：权重降 0.25→0.10（区分度保留，
        # 与 outcome/quality boost 同量级；v3 原权基于假想 ref_count 上百的分布，
        # 实际冷启动数据下 0.25 过重——T616 论证，样本量上来后可回调）。
        _ref_n = int(row.get("ref_count") or 0)
        _ref_max = max(int(_ref_count_max or 0), 1)  # 候选集内最大引用数（打分前聚合）
        _ref_boost = 0.10 * (math.log1p(_ref_n) / math.log1p(_ref_max)) if _ref_n > 0 else 0.0
        # Task616 decay 维度（v3 w3）：半衰 90 天乘性衰减——老未更新条目整体降权。
        # 乘性（非 additive）：保序语义下压低绝对分，与 boost 系不叠加饱和。
        # updated 空（无时间戳）按 created 兜底，两者皆无 → 不衰减（保守）。
        _updated_raw = row.get("updated") or row.get("created")
        _decay_mult = 1.0
        if _updated_raw is not None:
            try:
                from datetime import timezone as _tz
                _u = _updated_raw
                if hasattr(_u, "tzinfo") and _u.tzinfo is None:
                    _u = _u.replace(tzinfo=_tz.utc)
                _days = max(0.0, (datetime.now(timezone.utc) - _u).total_seconds() / 86400.0)
                _decay_mult = 0.5 ** (_days / 90.0)
            except Exception:
                _decay_mult = 1.0
        # 1401-d5c 消融口：decay_override 非 None 时强制覆盖（b 轮传 1.0=去衰减）。
        # 只影响本次调用的内存计算，不触碰任何存储/生产默认——验收第 2 条对照硬卡。
        if decay_override is not None:
            _decay_mult = float(decay_override)
        final = min(1.0, max(0.0,
            (final + _quality_boost.get(_ktype, 0.0) + _shadow_layer_boost
             + _outcome_boost + _ref_boost) * _decay_mult))
        # 1401-d5a T2 type_prior：知识类型先验——独立【乘法】项，与上方 additive
        # 的 quality_boost 完全分开（MEM#30837 禁二次叠加：boost 已等效 type 加权，
        # 再叠乘会过度饱和）。为什么是 1.25/0.85：42k 混池 document 占 91.5%，无先验
        # 时 document 切片靠词面重叠常挤进 top3（基准集 10 查实测 2 次无关切片高位）；
        # 1.25 保守幅度让真相关经验上浮一档、0.85 只压不埋（R1 反例：真相关 document
        # 仍可命中——基准集含 2 个 document 型查询回归监控）。系数为设计定值
        # （D5 设计稿块1），调整需 need_decision 报主控，禁自行改。
        _type_prior = _TYPE_PRIOR.get(_ktype, 1.0)
        if _type_prior != 1.0:
            final = min(1.0, max(0.0, final * _type_prior))
        row["rank"] = final
        row["vec_score"] = round(vec_s, 4)
        row["fts_score"] = round(fts_s, 4)
        row["final_score"] = round(final, 4)
        if vec_s > 0 and fts_s > 0:
            row["source"] = "vec+fts"
        elif vec_s > 0:
            row["source"] = "vec"
        else:
            row["source"] = "fts"
        # snippet: 取content中匹配查询词前后各60字符（task 253 follow-up: 问题3）
        row["snippet"] = _make_snippet(row.get("content", ""), query)
        # explain: 检索解释性元数据（task 253 P2: 四小弟审计问题2）
        row["explain"] = {
            "vec_score": round(vec_s, 4),
            "fts_score": round(fts_s, 4),
            "final_score": round(final, 4),
            "vec_weight": effective_vec_weight,
            "fts_weight": effective_fts_weight,
            "source": row["source"],
            "shadow_layer_boost": round(_shadow_layer_boost, 4),
            "quality_boost": round(_quality_boost.get(_ktype, 0.0), 4),
            "type_prior": round(_type_prior, 4),      # 1401-d5a：独立乘法先验（与 additive 的 quality_boost 分开显示）
            "outcome_boost": round(_outcome_boost, 4),
            "ref_boost": round(_ref_boost, 4),        # Task616 维度
            "decay_mult": round(_decay_mult, 4),      # Task616 维度
            "rank_method": "weighted_fusion_v3",
        }
        results.append(row)

    results.sort(key=lambda x: x["final_score"], reverse=True)

    # ---- Task 458: Reranker 精排（cross-encoder）----
    # 在 fusion sort 后、父文档去重前对 top-pool 精排。
    # 失败降级（_rerank_via_worker 返 None，跳过用原融合分）。
    if RERANK_ENABLED and len(results) > top_k:
        pool = results[:rerank_pool]
        reranked = _rerank_via_worker(query, pool)
        if reranked:
            # 1401-d5a 契约修正（二轮）：rerank 回写归一化相关性分后【再乘
            # type_prior 并重排】——第一版「精排让位相关性」实测证伪：reranker
            # 按词面语义把检索统计报告切片（#72209）排进 MuMu/superseded 查询
            # top1，经验内核优先目标落空。先验必须贯穿全链（召回融合+精排），
            # 乘后重排保「顺序=score 降序」契约。explain 双记录：rerank_score
            # （纯相关性）+ type_prior（最终生效先验）可分追溯。
            for _row in reranked:
                _rrs = _row.pop("_rerank_score", None)
                if _rrs is None:
                    continue
                _prior = _TYPE_PRIOR.get(_row.get("knowledge_type", "document"), 1.0)
                _fs = min(1.0, max(0.0, _rrs * _prior))
                _row["rank"] = _fs
                _row["final_score"] = round(_fs, 4)
                _exp = _row.get("explain") or {}
                _exp["rerank_score"] = round(_rrs, 4)
                _exp["type_prior"] = round(_prior, 4)
                _exp["rank_method"] = "weighted_fusion_v3+rerank×prior"
                _row["explain"] = _exp
            reranked.sort(key=lambda x: float(x.get("final_score", 0.0) or 0.0), reverse=True)
            results = reranked + results[rerank_pool:]

    # ---- 父文档去重：同一文档的多个 chunk 只保留最高分 ----
    seen_parents = set()
    deduped = []
    for row in results:
        pkey = parent_path(row.get("path"), row.get("id"))
        if pkey in seen_parents:
            continue
        seen_parents.add(pkey)
        deduped.append(row)
        if len(deduped) >= top_k + offset:
            break

    # 应用offset偏移（task 253 P2: 分页检索 — 去重后截断，非SQL层）
    deduped = deduped[offset:offset + top_k]

    # 统一返回字段（task 253 follow-up: 三小弟审计问题2 — 字段不一致）
    for row in deduped:
        row.setdefault("final_score", round(row.get("rank", 0.0), 4))
        row.setdefault("vec_score", 0.0)
        row.setdefault("fts_score", 0.0)
        row.setdefault("source", "unknown")
        row.setdefault("snippet", "")
        row.setdefault("explain", {})

    # ---- task 1455: 关系边表一跳扩展（graph_expand=True 才走）----
    # 命中旧经验（存在 supersedes 入边=已被新版取代）→ 附新版指引。
    # annotate 模式只标注不动序列；promote 模式把新版插到旧条位置前
    # （旧条顺延），分数沿用旧条的——排序契约"顺序=score 降序"不破坏，
    # explain 里标清楚这是图扩展上浮不是检索分。
    if graph_expand and deduped:
        from nexusos_mem import relation as _relation
        _edges = _relation.one_hop_in_batch(
            conn, [r["id"] for r in deduped], rel_type="supersedes")
        if _edges:
            for row in deduped:
                _e = _edges.get(row["id"])
                if _e:
                    row["superseded_by_new"] = _e
                    _exp = row.get("explain") or {}
                    _exp["graph_edge"] = "supersedes_in"
                    row["explain"] = _exp
            if graph_mode == "promote":
                # 新版完整行批量取（字段对齐上面向量路 SELECT，够展示用）
                _new_ids = [e["id"] for e in _edges.values()]
                _cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                _cur.execute(
                    """
                    SELECT id, title, content, source_agent, tags, project,
                           task_id, knowledge_type, shadow_code, confidence,
                           created, path, context_coord, outcome, ref_count,
                           updated
                    FROM knowledge WHERE id = ANY(%s)
                    """,
                    (_new_ids,),
                )
                _new_rows = {r["id"]: dict(r) for r in _cur.fetchall()}
                _cur.close()
                _existing = {r["id"] for r in deduped}
                _out = []
                for row in deduped:
                    _e = row.get("superseded_by_new")
                    if (_e and _e["id"] in _new_rows
                            and _e["id"] not in _existing):
                        _nr = dict(_new_rows[_e["id"]])
                        _nr["vec_score"] = 0.0
                        _nr["fts_score"] = 0.0
                        # 分数沿用被取代旧条的：插进序列不破降序契约，
                        # 消费方按 final_score 排序不会乱
                        _nr["final_score"] = row.get("final_score", 0.0)
                        _nr["rank"] = _nr["final_score"]
                        _nr["source"] = "graph_promote"
                        _nr["snippet"] = _make_snippet(_nr.get("content", ""), query)
                        _nr["promoted_via_supersedes"] = {"from_id": row["id"]}
                        _nr["explain"] = {
                            "rank_method": "graph_promote_v1",
                            "promoted_from": row["id"],
                            "inherited_score": _nr["final_score"],
                        }
                        _out.append(_nr)
                        _existing.add(_e["id"])
                    _out.append(row)
                deduped = _out
        # ---- 1522 C 项: relates 边 riding（citation 关联条目批量一跳）----
        # 只标注不动序列（annotate 语义，无 promote 变体——relates 无主次
        # 方向，上浮没有依据）。防爆: 每命中条目上限 RELATED_CAP 条。
        # Task1533: provenance（血缘）边同池参与一跳扩关联——血缘邻居
        # （同任务产物/同分片）与 citation 邻居对检索者同为高价值上下文。
        # Task1537: semantic（语义）边 opt-in——默认不进扩联池（增值层先试点，
        # 验证数据回流跑稳后再议默认行为），include_semantic=True 才加入；
        # related 条目统一带 rel_type 标记供消费端区分邻居性质。
        _RELATED_CAP = 5
        _hit_ids = [r["id"] for r in deduped]
        _riding_types = ("relates", "provenance", "semantic") if include_semantic \
            else ("relates", "provenance")
        _related_map = {}
        with conn.cursor() as _rc:
            if readmodel:
                # Task1538 M3: 读模型视角（opt-in）——related 换 readmodel_edge 源。
                # 端点统一 identity 空间：命中行经其 mem 身份行找邻居，对端是
                # mem 的折叠回知识条目展示（vault_doc 等非 mem 邻居不进 related
                # ——本接口契约只回知识条目，文件邻居由 knowledge-dig 消费）。
                _rc.execute(
                    """
                    SELECT hi.mem_id AS anchor, k.id, k.title
                    FROM readmodel_edge e
                    JOIN knowledge_identity hi
                      ON hi.identity_kind='mem' AND hi.mem_id = ANY(%s)
                     AND (hi.id = e.src_identity OR hi.id = e.dst_identity)
                    JOIN readmodel_node other
                      ON other.identity_id = CASE WHEN hi.id = e.src_identity
                                                  THEN e.dst_identity ELSE e.src_identity END
                    JOIN knowledge k ON k.id = other.mem_id
                    WHERE e.src_identity <> e.dst_identity
                      AND k.deprecated = FALSE
                    ORDER BY k.id
                    """,
                    (_hit_ids,),
                )
                for _anchor, _oid, _otitle in _rc.fetchall():
                    _lst = _related_map.setdefault(_anchor, [])
                    if len(_lst) < _RELATED_CAP and _oid not in _hit_ids:
                        _lst.append({"id": _oid, "title": _otitle})
            else:
                _rc.execute(
                    """
                    SELECT kr.src_id, kr.dst_id, kr.rel_type, k.id, k.title, k.deprecated
                    FROM knowledge_relation kr
                    JOIN knowledge k
                      ON k.id = CASE WHEN kr.src_id = ANY(%s) THEN kr.dst_id ELSE kr.src_id END
                    WHERE (kr.src_id = ANY(%s) OR kr.dst_id = ANY(%s))
                      AND kr.rel_type = ANY(%s)
                      AND k.deprecated = FALSE
                    ORDER BY kr.id
                    """,
                    (_hit_ids, _hit_ids, _hit_ids, list(_riding_types)),
                )
                for _sid, _did, _rtype, _oid, _otitle, _ in _rc.fetchall():
                    _anchor = _sid if _sid in set(_hit_ids) else _did
                    _lst = _related_map.setdefault(_anchor, [])
                    # 邻居已在结果集里则不重复列（riding 是"结果之外的扩展"语义）
                    if len(_lst) < _RELATED_CAP and _oid not in _hit_ids:
                        _lst.append({"id": _oid, "title": _otitle, "rel_type": _rtype})
        if _related_map:
            for row in deduped:
                _rel = _related_map.get(row["id"])
                if _rel:
                    row["related"] = _rel
                    _exp = row.get("explain") or {}
                    _exp["graph_edge"] = (_exp.get("graph_edge", "")
                                          + "+relates_riding").lstrip("+")
                    if any(_r.get("rel_type") == "semantic" for _r in _rel):
                        _exp["graph_edge"] += "+semantic_riding"
                    row["explain"] = _exp
    if _379_restore:  # Task 379: 恢复传入 conn 的原 autocommit 状态
        try:
            conn.autocommit = _379_ac
        except Exception:
            pass
    return deduped


# ---------------------------------------------------------------------------
# PostgreSQL FTS + ILIKE 搜索（task 245 后为降级路径，保留）
# ---------------------------------------------------------------------------

def pg_fts_search(
    conn,
    query: str,
    top_k: int = 5,
    knowledge_type: Optional[str] = None,
    project: Optional[str] = None,
    tag: Optional[str] = None,
    offset: int = 0,
    core_only: bool = False,
    pool: str = "all",
    graph_expand: bool = False,
) -> List[dict]:
    """PostgreSQL 全文搜索（FTS）+ ILIKE 子串匹配（中文补充）。

    Args:
        conn: psycopg2 连接
        query: 搜索文本
        knowledge_type/project/tag: 可选过滤条件
        graph_expand: task 1455 一跳扩展开关——True 时放宽 superseded 过滤
            （与 pgvector_search 向量路同口径，两路行为必须一致，否则
            "并集"里 FTS 独有命中永远见不到旧条）

    Returns:
        [{"id", "title", ..., "rank", "source"="pg_fts"}, ...]
    """
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    ilike_condition = "(title ILIKE %s OR content ILIKE %s)"
    # task/1452: 中文 FTS jieba 双轨——查询侧与写入侧共用 tokenize_zh 同源分词
    # （两侧词表不一致=词级匹配失效，1449 草稿 §2.3 命门）。flag on：query 分词后
    # 双列 OR（search_vector_zh 词级命中优先，search_vector 保底未回填行+英文），
    # rank 同步双列 GREATEST；flag off：原串+单列条件，SQL 与改造前逐字节一致
    # （双轨灰度回滚生命线）。ILIKE 恒用原始 query——分词插空格破坏连续子串语义。
    from nexusos_mem.tokenize_zh import jieba_enabled as _fts_jieba_on, tokenize_for_query as _tok_query
    if _fts_jieba_on():
        fts_query = _tok_query(query)
        fts_condition = ("(search_vector_zh @@ plainto_tsquery('simple', %s)"
                         " OR search_vector @@ plainto_tsquery('simple', %s))")
        rank_fts = ("ts_rank(search_vector_zh, plainto_tsquery('simple', %s)),"
                    " ts_rank(search_vector, plainto_tsquery('simple', %s))")
        fts_params = [fts_query, fts_query]
    else:
        fts_query = query
        fts_condition = "search_vector @@ plainto_tsquery('simple', %s)"
        rank_fts = "ts_rank(search_vector, plainto_tsquery('simple', %s))"
        fts_params = [fts_query]

    where_extra = ""
    params_extra = []
    if knowledge_type:
        where_extra += " AND knowledge_type = %s"
        params_extra.append(knowledge_type)
    if project:
        where_extra += " AND project = %s"
        params_extra.append(project)
    if tag:
        where_extra += " AND %s = ANY(tags)"
        params_extra.append(tag)
    if core_only:
        where_extra += " AND knowledge_type IN ('experience','finding','decision','pitfall')"
    # 1401-d5a T2：pool 分池（与 pgvector_search 同口径，降级链行为一致）
    if pool == "experience":
        where_extra += (" AND knowledge_type IN ('experience','pitfall',"
                        "'counterexample','finding','decision')")
    elif pool == "document":
        where_extra += " AND knowledge_type = 'document'"

    # task 1455: graph_expand 同口径放宽（与向量路一致，见 pgvector_search）
    _sup_filter = "" if graph_expand else " AND superseded_by IS NULL"

    # task 253 修复: 旧版 DISTINCT ON (id)+ORDER BY id 让 LIMIT 取的是"id 最小"
    # 而非"rank 最高"的行（id 本身是主键，DISTINCT ON 纯属多余）。
    sql = f"""
        SELECT id, title, content, source_agent, tags, project, task_id,
               knowledge_type, shadow_code, confidence, created, path,
               context_coord, outcome, ref_count, updated,
               GREATEST(
                    {rank_fts},
                    CASE WHEN title ILIKE %s OR content ILIKE %s THEN 0.5 ELSE 0 END,
                    CASE WHEN knowledge_type IN ('experience','finding','decision','pitfall')
                         THEN 0.15 ELSE 0 END
                ) AS rank
        FROM knowledge
        WHERE ({fts_condition} OR {ilike_condition})
          {_sup_filter} AND COALESCE(deprecated, FALSE) = FALSE{where_extra}
        ORDER BY rank DESC, id DESC
        LIMIT %s OFFSET %s
    """
    # 参数序：GREATEST（rank_fts 1-2 个 + ILIKE 2 个）→ WHERE（fts_condition 1-2 个 + ILIKE 2 个）
    params = [*fts_params, f"%{query}%", f"%{query}%",  # GREATEST
              *fts_params,  # FTS（WHERE）
              f"%{query}%", f"%{query}%",  # ILIKE（WHERE）
              ] + params_extra + [top_k, offset]

    cur.execute(sql, params)
    results = [dict(row) for row in cur.fetchall()]
    cur.close()

    for r in results:
        r["source"] = "pg_fts"
        r.setdefault("final_score", round(float(r.get("rank", 0.0) or 0.0), 4))
        r.setdefault("vec_score", 0.0)
        r.setdefault("fts_score", round(float(r.get("rank", 0.0) or 0.0), 4))
        r.setdefault("snippet", str(r.get("content", ""))[:150])

    return results


# ---------------------------------------------------------------------------
# 纯 FTS 降级（embedding 未就绪 / 编码不可用时）
# ---------------------------------------------------------------------------

def fts_fallback_search(
    conn,
    query: str,
    top_k: int = 5,
    knowledge_type: Optional[str] = None,
    project: Optional[str] = None,
    tag: Optional[str] = None,
    offset: int = 0,
    core_only: bool = False,
    pool: str = "all",
) -> List[dict]:
    """embedding 全表缺失时的降级路径：纯 PG FTS + ILIKE（task 253 清理）。

    task 245 zvec 已退役、task 253 pgvector_search 已是主路。本函数仅在
    knowledge 表尚无任何 embedding（冷启动/迁移中）时兜底。
    """
    return pg_fts_search(
        conn, query, top_k=top_k,
        knowledge_type=knowledge_type, project=project, tag=tag, offset=offset,
        core_only=core_only, pool=pool,
    )
