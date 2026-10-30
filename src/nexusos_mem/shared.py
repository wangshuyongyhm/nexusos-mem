#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nexus.shared — 多 Agent 共享知识库 API

让所有 AI agent（Hermes/OC/omp/Kimi/DS/ac）通过统一接口访问共享知识库。

用法：
    from nexus.shared import SharedKnowledge

    kb = SharedKnowledge(agent_id="agent-a")

    # 查询
    results = kb.search("frida 注入方法", top_k=5)
    codes = kb.get_shadow_codes(layer="L1")
    params = kb.get_rl_params()

    # 写入
    kb.add_knowledge(
        title="发现XX问题",
        content="详细描述...",
        knowledge_type="finding",
        tags=["nexusos-mem", "bug"],
        project="nexusos-mem",
    )

    # 活动追踪
    kb.log_activity(action="search", target_type="knowledge")
"""

from __future__ import annotations

import json
import os
import psycopg2
import psycopg2.extras
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any


# 默认连接字符串（可通过环境变量覆盖）。
# 1822 hang 根因修复: connect_timeout=5——无此项时半开网络下 psycopg2.connect
# 可无限阻塞（TCP 重传窗），是 /search /knowledge 全 hang 而 /health 存活的路径之一。
DEFAULT_DSN = os.environ.get(
    "NEXUS_PG_DSN",
    "postgresql://localhost/nexus_mem",
)
if "connect_timeout" not in DEFAULT_DSN:
    DEFAULT_DSN += ("&" if "?" in DEFAULT_DSN else "?") + "connect_timeout=5"

# outcome 推断：search 命中≥此值且无反例 → inferred_apply（Task 462）
INFER_MIN_HITS = 3


def apply_session_timeouts(conn):
    """1822 hang 根因修复: 会话级三超时——半开连接上的任意 SQL 最多阻塞 30s；
    写事务等行锁最多 10s；事务挂起 idle 超过 60s 由服务端主动终结
    （斩断"卡死事务持锁→后来者无限排队"的雪崩链）。
    在 autocommit=True 下 SET：GUC 立即生效于会话，不受后续事务回滚影响。
    单一真源：_get_conn 与 api_server 直连端点（/stats /knowledge/{id} 等）共用。
    """
    conn.autocommit = True
    try:
        cur = conn.cursor()
        cur.execute(
            "SET statement_timeout = '30s'; "
            "SET lock_timeout = '10s'; "
            "SET idle_in_transaction_session_timeout = '60s'"
        )
        cur.close()
    except Exception:
        pass  # 老版本 PG 不识别 idle_in_transaction_session_timeout 时保底其余两项
    conn.autocommit = False


# ===========================================================================
# ⚠️  TRUST 字段守卫（Task 337，2026-07-28；Task 655 三维救活后口径更新）
#   - decay_factor (real DEFAULT 0.0): 【Task655 定案：摘除方案】运行时真源是
#     search.py 的 decay_mult（半衰 90 天乘性衰减，Task616 维度）——DB 列不回写，
#     恒 0.0 by design（双真源会造成排序口径漂移）。禁止给本列加 writer。
#   - debt_score (real DEFAULT 0.0): 【Task655 已启用】知识债 = 0.4×failed outcome
#     + 0.3×被活跃 counterexample 引用 + 0.3×superseded_by 非空，clamp 1.0。
#     存量回填 + 每日重算 cron（ai.nexus.mem-debt-daily，03:10）。可安全 SELECT。
#   - confidence_source (varchar DEFAULT 'llm'): Task 257 已证伪（σ 是 artifact），
#     仅 Task 255 一次性 backfill 写过，之后死字段。
# 详见：
#   task/655-scoring-revive/output/report.md (三维救活)
#   task/257-catd-prototype/output/report.md (confidence 证伪)
#   task/247-memory-consolidation/output/report.md (decay/debt 原始设计)
# ===========================================================================


# 1401-d5t6：写入侧 tag 归一映射表（单一真源，检索/批补共用）。
# 出处：T5 消融配对数据（task/1401-d5c）实测漂移对——踩坑 29 条 vs pitfall 8 条
# vs 坑 2 条（中英三分）；c 轮词面旁证=新旧经验检索差距主要来自词汇口径漂移。
# 每条映射一句话理由（陷阱 1：不拍脑袋）：
#   踩坑/坑→pitfall：库内三写法并存（29/8/2），归一到英文枚量（knowledge_type 同源词）
#   经验→experience：同上（1 vs 5），type 枚量对齐
#   裁决/决定→decision：中文变体归一（1 vs 0）
#   发现→finding：中文变体归一（3 vs 0）
#   反例→counterexample：d5a 已将其纳入经验内核五类，tag 同步归一
TAG_ALIAS_MAP = {
    "踩坑": "pitfall", "坑": "pitfall",
    "经验": "experience",
    "裁决": "decision", "决定": "decision",
    "发现": "finding",
    "反例": "counterexample",
}
# 归一后白名单（经验内核五类 + 常用工程词，小写化基准）——非映射词原样保留
# （宁缺勿滥：只归一确定性高的对，不确定词不动避免误伤）。


def _normalize_tags(tags: Optional[List[str]], title: str = "") -> List[str]:
    """1401-d5t6：写入侧 tag 归一（T5 裁定 H1 的落地件）。

    管什么：①中英同义归一（TAG_ALIAS_MAP）②strip+全小写 ③空 tag 时从标题
    保守补 ≤2 个关键词（宁缺勿滥——只取 ≥2 字的中文段/英文词，数字/单字符跳过）。
    为什么在写入侧：T5 消融实证 decay 影响上限 13%，主因是词汇口径漂移
    （c 轮纯词面旁证）——检索侧 fuzzy 是治标且伤性能（任务书已排除）。
    注意：tags 当前不进 search_vector（title+content 生成式）/embedding 文本源，
    归一收益面=?tag=过滤查询与未来索引扩展（如实边界，见 t6 report §架构发现）。
    """
    out: List[str] = []
    _raw = tags
    # 1401-d5t6 验收修复：HTTP 调用方传字符串 tags（如 "踩坑, E2E"）时，
    # 上游 list(tags) 会字符级拆碎（#73184 实证：'踩'/','/'e'/'2'…）。
    # 此处把字符串按中英逗号/分号拆回列表——单一收口兜底，数组输入零影响。
    if isinstance(_raw, str):
        import re as _re_split
        _raw = [t for t in _re_split.split(r"[,，;；]", _raw) if t.strip()]
    for t in (_raw or []):
        t = str(t).strip().lower()
        if not t:
            continue
        t = TAG_ALIAS_MAP.get(t, t)
        if t not in out:
            out.append(t)
    if not out and title:
        # 空 tag 补词：标题分词取前 2 个 ≥2 字段（保守——宁可不补也不错补）。
        # 跳过纯数字与任务号形态（T6/G14/W1 等"字母+数字"短码，是编号不是词）。
        import re as _re
        words = [w for w in _re.split(r"[\s：:，,。/（）()\[\]「」\-—·、|]+", title)
                 if len(w) >= 2 and not w.isdigit()
                 and not _re.fullmatch(r"[A-Za-z]\d+", w)][:2]
        out = words
    return out


_TAG_KTYPE_CN = {"experience": "经验", "pitfall": "踩坑",
                 "decision": "裁决", "counterexample": "反例"}
# 09181120 用户裁决：title 已含场景词面（或 agent 名）即视为自带锚，不补前缀
_TITLE_SCENE_WORDS = ("坑", "经验", "裁决", "反例", "如何", "为什么", "修复", "设计")


def _normalize_title(title: str, knowledge_type: str = "experience",
                     agent_id: str = "") -> str:
    """1401-d5t6 追加段（09181120）：写入侧 title 场景前缀规范（H1 下一手）。

    管什么：title 完全不含场景词面时补前缀 `{agent}·{类型中文}｜{原title}`
    （agent_id 空则只补 `{类型中文}｜`）。为什么：T5/T6 架构确认 tags 不进
    检索索引、H1 载体=title+content 词面——把 agent/类型锚写进 title 是让
    自由文本检索「某 agent 探索 踩坑」类查询能词面命中经验条目的直接手段。
    不补的条件（避免重复前缀/误伤）：title 含任一场景词（坑/经验/裁决/反例/
    如何/为什么/修复/设计）或 agent 名（大小写不敏感），或类型不在经验内核
    映射内（保守——document 等不补）。幂等：已带目标前缀原样返回。
    只影响新写入（add_knowledge 收口），存量 title 不动。
    """
    t = (title or "").strip()
    if not t:
        return title
    # Task1530（池 #200③）：verification 开头的探针 title 免前缀——前缀会让
    # init gate I7 的 title LIKE 'verification%' 永不命中（MEM#81118）。
    # 探针 title 无检索价值（写完即软废弃），免前缀零副作用。
    if t.startswith("verification"):
        return t
    if any(w in t for w in _TITLE_SCENE_WORDS):
        return t
    if agent_id and agent_id.lower() in t.lower():
        return t
    kcn = _TAG_KTYPE_CN.get(knowledge_type or "", "")
    if not kcn:
        return t
    prefix = f"{agent_id}·{kcn}｜" if agent_id else f"{kcn}｜"
    if t.startswith(prefix):
        return t
    return prefix + t


def _read_current_task_id(agent_id: str) -> Optional[str]:
    """Task 322 试运行修复：自动读取小弟当前 task_id（小弟无感关联）。

    dispatch.sh 在发任务时把 task_id 写到 /tmp/nexus_current_task_<session>.txt。
    本函数读该文件，让 search 自动带 task_id（小弟不需要主动传）。

    Args:
        agent_id: Agent 标识（如 'agent-a'/'agent-b'），对应 dispatch 时的 session 名

    Returns:
        task_id 字符串，或 None（文件不存在/读取失败）
    """
    if not agent_id:
        return None
    try:
        ctx_file = Path("/tmp") / f"nexus_current_task_{agent_id}.txt"
        if ctx_file.exists():
            tid = ctx_file.read_text(encoding="utf-8").strip()
            return tid if tid else None
    except Exception:
        pass
    return None


class SharedKnowledge:
    """多 Agent 共享知识库。

    所有 agent 通过此类访问 PostgreSQL 共享知识库。
    每个 agent 用唯一 agent_id 标识自己。

    Args:
        agent_id: Agent 标识（如 "agent-a"/"agent-b"）
        dsn: PostgreSQL 连接字符串（默认从 NEXUS_PG_DSN 环境变量读取）
    """

    def __init__(self, agent_id: str, dsn: str = None):
        self.agent_id = agent_id
        self.dsn = dsn or DEFAULT_DSN
        self._conn = None
        self._register_agent()

    def _get_conn(self):
        """写路径连接：懒加载 + 僵尸重连 + 强制 autocommit=False（显式事务）。

        Task 379: 读写分层后的写路径专用入口。调用方负责 commit/rollback。
        修复(2026-07-29): PG 重启后 _conn.closed 可能仍 False(TCP 半开),
        下一次 execute 才抛 OperationalError。原版无任何重连机制,需手动重启服务。
        修复: 用 ping(SELECT 1) 检测僵尸,失败则重连。
        修复(2026-07-30 Task 379): ① ping 自身曾泄漏(SELECT 1 只 close 不 commit),
           改为 ping 时临时切 autocommit=True 让 SELECT 自动提交,零事务残留;
           ② 入口强制 self._conn.autocommit=False 防御 _get_read_conn 翻转后残留,
           收恢复责任到单一入口(下次 _get_conn 必复位 False),读端无需逐处 finally 恢复。
        """
        if self._conn is not None and not self._conn.closed:
            # 快速 ping 检测僵尸连接(PG 重启/TCP 断开后 closed 仍 False)。
            # ping 用 autocommit=True 临时代理:SELECT 1 自动提交,不开事务,不泄漏。
            try:
                self._conn.autocommit = True
                cur = self._conn.cursor()
                try:
                    cur.execute("SELECT 1")
                    cur.fetchone()
                finally:
                    try: cur.close()
                    except Exception: pass
                # 复位 False:本方法契约是"写路径连接",调用方需显式事务
                self._conn.autocommit = False
                return self._conn
            except (psycopg2.OperationalError, psycopg2.InterfaceError):
                # 僵尸连接,关闭后重连
                try: self._conn.close()
                except Exception: pass
                self._conn = None
            except Exception:
                # autocommit 切换或 ping 失败(非僵尸类异常)也复位 False 兜底
                try: self._conn.autocommit = False
                except Exception: pass
                return self._conn
        # 新建连接
        self._conn = psycopg2.connect(self.dsn)
        apply_session_timeouts(self._conn)
        return self._conn

    def _get_read_conn(self):
        """读路径连接：autocommit=True,SELECT 后自动提交,永不 idle in transaction。

        Task 379: 只读查询专用。所有 get_/search/count/list/fetch_ 改用本方法。
        设计(方案 C 强化变体):
          - 复用 self._get_conn()(共享懒加载 + 僵尸重连逻辑,省连接);
          - _get_conn 入口已强制 autocommit=False,这里翻 True 供 SELECT 自动提交;
          - 不在读端 finally 恢复:下次 _get_conn(写路径或读路径 ping)必复位 False,
            重入污染窗口归零(写方法只调写方法或传 conn 辅助,不调本读路径方法)。
        注意:本连接不可用于 INSERT/UPDATE(autocommit=True 下无原子性);
              写路径继续用 _get_conn()。
        """
        conn = self._get_conn()
        conn.autocommit = True
        return conn

    def _register_agent(self):
        """注册/更新 agent 信息。"""
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO agents (agent_id, last_seen)
                VALUES (%s, NOW())
                ON CONFLICT (agent_id) DO UPDATE SET last_seen = NOW()
            """, (self.agent_id,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            try:
                cur.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 知识查询
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        knowledge_type: Optional[str] = None,
        project: Optional[str] = None,
        tag: Optional[str] = None,
        offset: int = 0,
        core_only: bool = False,
        task_id: Optional[str] = None,  # Task 319: 关联任务（Phase 1 自动检测套用需要）
        annotate: bool = False,  # Task 358: 显式开启套用记录（认知诚实闭环）
        agent_id: Optional[str] = None,  # Task478: 显式归因（修复 _global_kb 单例固化 bug）
        pool: str = "all",  # 1401-d5a T2: experience|document|all（分池检索）
        decay_override: Optional[float] = None,  # 1401-d5c 消融调试口（默认 None=生产行为）
        lexical_only: bool = False,  # 1401-d5c 消融调试口（默认 False=生产行为）
        graph_expand: bool = False,  # task 1455: 关系边表一跳扩展（默认关=生产行为不变）
        graph_mode: str = "annotate",  # task 1455: annotate=只标注 / promote=新版上浮
        readmodel: bool = False,  # Task1538 M3: related 换读模型源（opt-in，默认原路径）
        include_semantic: bool = False,  # Task1537: semantic 边 opt-in 进 riding 扩联池
    ) -> List[dict]:
        """搜索知识库（task 253：真·混合检索）。

        检索路径:
            1. **主路 pgvector_search** — 向量 ∪ FTS 并集融合 + 父文档去重
               （knowledge 表有 embedding 时；BGE-M3 经常驻 worker 或本地模型编码）
            2. **降级 fts_fallback_search** — 全表无 embedding 时纯 FTS + ILIKE

        Args:
            query: 搜索文本
            top_k: 返回数（最终去重后）
            knowledge_type: 过滤类型（experience/finding/decision/pitfall/document）
            project: 过滤项目（项目名如 demo-project-a/nexusos-mem/system）
            tag: 过滤标签
            offset: 分页偏移（task 253 P2）
            core_only: 只搜核心知识（experience/finding/decision/pitfall，task 253 P3-1）

        Returns:
            [{"id", "title", ..., "rank", "source"}, ...]
            source 字段: "vec+fts" | "vec" | "fts" | "pg_fts"
        """
        from nexusos_mem.search import (
            fts_fallback_search, pgvector_search, has_pgvector_embeddings,
            pg_statement_budget, _SEARCH_PG_TIMEOUT_SEC,
        )
        conn = self._get_read_conn()  # Task 379: search 只读走 autocommit(search.py 的 has_pgvector_embeddings/pgvector_search 接收此 conn,根治在此调用方)

        try:
            # Task1612 #244: PG 查询层预算——风暴期（舰队批量 DELETE/backfill 拖慢
            # 全表 SELECT）主路召回 > 预算即超时（QueryCanceled），不无限等锁。
            with pg_statement_budget(conn):
                if has_pgvector_embeddings(conn):
                    results = pgvector_search(
                        conn, query, top_k=top_k,
                        knowledge_type=knowledge_type, project=project, tag=tag,
                        offset=offset, core_only=core_only, pool=pool,
                        decay_override=decay_override, lexical_only=lexical_only,
                        graph_expand=graph_expand, graph_mode=graph_mode,
                        readmodel=readmodel,
                        include_semantic=include_semantic,
                    )
                    engine = "pgvector"
                else:
                    # embedding 全表缺失（冷启动/迁移中）→ 纯 FTS 兜底
                    results = fts_fallback_search(
                        conn, query, top_k=top_k,
                        knowledge_type=knowledge_type, project=project, tag=tag,
                        offset=offset, core_only=core_only, pool=pool,
                    )
                    engine = "fts_fallback"
        except psycopg2.errors.QueryCanceled as _pg_timeout:
            # 风暴期主路超时 → FTS 降级重试（GIN/tsquery 路径通常更轻，同预算内再试一次）；
            # 降级结果带 pg_timeout_degrade 标注，消费端可感知本次检索是降级态。
            # 若 FTS 也超时 → QueryCanceled 冒泡，由 api_server Task487 重试/503 接管。
            import logging
            logging.getLogger(__name__).warning(
                "search 召回 PG 超时(>%ss)→FTS 降级: %s", _SEARCH_PG_TIMEOUT_SEC, _pg_timeout,
            )
            with pg_statement_budget(conn):
                results = fts_fallback_search(
                    conn, query, top_k=top_k,
                    knowledge_type=knowledge_type, project=project, tag=tag,
                    offset=offset, core_only=core_only, pool=pool,
                )
            for row in results:
                row["pg_timeout_degrade"] = True
                _exp = row.get("explain")
                if isinstance(_exp, dict):
                    _exp["rank_method"] = "fts_timeout_degrade(" + str(_exp.get("rank_method", "fts")) + ")"
                else:
                    row["explain"] = {"rank_method": "fts_timeout_degrade"}
            engine = "fts_timeout_degrade"

        # Task 319: detail 加 experience_ids（Phase 1 自动检测依赖）+ task_id 传给 log_activity
        # Task 322 试运行修复: 小弟不会主动传 task_id → 读上下文文件自动关联（小弟无感）
        if task_id is None:
            task_id = _read_current_task_id(self.agent_id)
        experience_ids = [r.get("id") for r in results[:top_k]]
        self.log_activity("search", "knowledge", detail={
            "query": query, "results": len(results), "engine": engine,
            "vec_count": sum(1 for r in results if "vec" in r.get("source", "")),
            "fts_count": sum(1 for r in results if "fts" in r.get("source", "")),
            "experience_ids": experience_ids,  # Task 319: 返回的经验 id 列表（空结果时为 []）
        }, task_id=task_id, agent_id=agent_id)
        self._feedback_shadow_refs(results)
        # 统一返回字段格式（task 253 follow-up: 三小弟审计问题2 — 字段不一致）
        for row in results:
            row.setdefault("final_score", round(row.get("rank", row.get("final_score", 0.0)), 4))
            row.setdefault("vec_score", 0.0)
            row.setdefault("fts_score", 0.0)
            row.setdefault("source", "unknown")
            row.setdefault("snippet", str(row.get("content", ""))[:150])
            row.setdefault("explain", {})
        # task 263: 集成认知诚实体系 Layer 0+2
        # 附 source_coord（SQL 已 SELECT context_coord，无 N+1）+ counterexample_count
        self._enrich_results(results)
        # Task406 Layer2: 批量叠加 calibration_weight（乘法, 与现有 boost 正交）
        try:
            if results:
                _ids = [r["id"] for r in results if r.get("id")]
                if _ids:
                    _conn = self._get_read_conn()
                    _cur = _conn.cursor()
                    _cur.execute(
                        "SELECT id, calibration_weight FROM knowledge WHERE id = ANY(%s)",
                        (_ids,),
                    )
                    _weights = {row[0]: float(row[1] or 1.0) for row in _cur.fetchall()}
                    _cur.close()
                    for row in results:
                        _w = _weights.get(row.get("id"), 1.0)
                        if _w != 1.0 and "final_score" in row:
                            # 1401-d5a：乘后夹回 [0,1]——原乘法在融合夹紧之后无上限，
                            # E2E 实测垃圾查询透出 1.5056 越界分，破坏 0.30 低置信
                            # 阈值的量纲语义。夹紧保序（单调），校准信号不丢排序力。
                            row["final_score"] = round(min(1.0, row["final_score"] * _w), 4)
                        # 对所有 row 暴露 calibration_weight（Task521 confidence_label 依赖）
                        row["calibration_weight"] = _w
                        # Task521: confidence_label（基于 calibration_weight + outcome 推导）
                        _outcome = row.get("outcome")
                        # R5(2026-08-14): 1.5→1.15 弱模式可测阈值。Wave2 配对池3→110后
                        # layer2 能真实校准，但 calibration_weight 单次偏差±0.3，弱模式下
                        # 到 1.5 需多次纯成功。1.15 让 high 在数据积累初期可触发；待真实
                        # outcome 积累充分（calibration_weight 普遍>1.3）可回归 1.5
                        if _w >= 1.15:
                            row["confidence_label"] = "high"           # 多次验证有效（校准被放大）
                        elif _outcome == "failed":
                            row["confidence_label"] = "contradicted"   # 有反例/被证伪（代码值 failed 对应设计文档 contradicted）
                        elif _outcome in ("success", "partial"):
                            # Task536 修复2a: 只认合法回填值（原含 apply/cond_apply 脏值
                            # 标 normal——历史 crucible 枚量已清洗，残留脏值落 unverified）
                            row["confidence_label"] = "normal"         # 有 apply 结果但未大幅校准
                        elif _outcome == "inferred_apply":
                            row["confidence_label"] = "inferred"       # Task516 推断的（非人工确认）
                        elif _outcome == "pending":
                            row["confidence_label"] = "pending"        # 等待结果回填
                        elif _outcome == "uncertain":
                            row["confidence_label"] = "uncertain"      # 推断不确定
                        else:
                            row["confidence_label"] = "unverified"     # outcome NULL，从未被 apply
                    # 1401-d5a：校准乘法后重排序——乘法可能改变相对大小（如 0.8×1.5
                    # 反超 0.9），验收契约「返回顺序=score 降序」要求乘后对齐。
                    # 稳定排序：无校准变动时是 no-op（原序列本已降序）。
                    results.sort(key=lambda x: float(x.get("final_score", 0.0) or 0.0), reverse=True)
        except Exception:
            pass  # calibration_weight 查询失败不影响 search（防御性）
        # 1401-d5a T1 置信门槛：final_score < 阈值的结果显式标 low_confidence=True，
        # 调用方可自筛（roadmap #11 快版）。语义边界（D5 设计稿 R3）：分数量纲不可
        # 跨查询族比较（vec∈[0,1] fts 无量纲混在 final 里），排序为准、阈值筛仅参考
        # ——0.30 是基准集（10 查+3 无结果词）校准出的「几乎必然无关」下界，不是
        # 相关性及格线。可 env 覆盖（NEXUS_LOW_CONFIDENCE_THRESHOLD）。
        try:
            _lc_th = float(os.environ.get("NEXUS_LOW_CONFIDENCE_THRESHOLD", "0.30"))
            for row in results:
                _fs = float(row.get("final_score", 0.0) or 0.0)
                if _fs < _lc_th:
                    row["low_confidence"] = True
        except (TypeError, ValueError):
            pass  # 阈值配置坏值不拖垮 search
        # Task 358: 认知诚实闭环——annotate=True 时记录套用尝试
        # 设计：默认 False（避免每次 search 都刷 apply_attempt 表）；Agent 搜到
        # 经验决定套用时显式 kb.search(..., annotate=True) 触发；只对高价值类型
        # （experience/finding/decision/pitfall）annotate，document 不 annotate
        # （是索引不是经验）；try/except 包裹，annotate 失败不影响 search 返回。
        if annotate and results:
            try:
                from nexusos_mem.apply_honesty import annotate_apply_attempt
                from nexusos_mem.context_capture import capture_current_context
                # 捕获当前场景坐标（task_type=search 触发的套用）
                _extra = {"query": query[:200]}
                if task_id:
                    _extra["task_id"] = task_id
                current_coord = capture_current_context(
                    agent_name=agent_id or self.agent_id,  # Task478: 优先显式归因
                    task_type="search",
                    extra=_extra,
                )
                _ANNOTATE_TYPES = ("experience", "finding", "decision", "pitfall")
                for row in results:
                    if row.get("knowledge_type") not in _ANNOTATE_TYPES:
                        continue
                    exp_id = row.get("id")
                    if exp_id is None:
                        continue
                    # annotate_apply_attempt 内部自动读 source_coord
                    # （通过 kb.get_knowledge(exp_id)），这里只传 current_coord
                    _ann_ret = annotate_apply_attempt(
                        kb=self,
                        exp_id=exp_id,
                        current_coord=current_coord,
                        log_activity=True,
                        task_id=task_id,
                    )
                    # Task531 B2: 回传 apply_attempt_id 给调用方（主控回填要用——
                    # 原返回值被丢弃，Agent 不知道回填哪条）
                    if isinstance(_ann_ret, dict) and _ann_ret.get("activity_id"):
                        row["apply_attempt_id"] = _ann_ret["activity_id"]
            except Exception as _annotate_err:
                # annotate 失败不能影响 search（非致命）
                import logging
                logging.getLogger(__name__).debug(
                    "annotate_apply_attempt failed (non-fatal): %s", _annotate_err,
                )
        return results

    def _enrich_results(self, results: List[dict]) -> None:
        """task 263: search 结果 enrichment（认知诚实体系 Layer 0+2）。

        给每个结果加：
          - source_coord: 经验的场景坐标（从 context_coord 读，供套用算序列用）
          - counterexample_count: 反例数（0=无反例，>0=套用前留意）
          - counterexamples: 前 3 条反例摘要（id/title/attribution/frozen）

        实现策略（避免 N+1）：
          - source_coord：search SQL 已 SELECT context_coord，直接映射，零额外查询
          - counterexample_count：单次批量查询，按 tags 数组聚合，非每行一次
        """
        if not results:
            return

        # 1. source_coord 直接从 SQL 返回的 context_coord 映射（零额外查询）
        exp_ids = []
        for row in results:
            cc = row.get("context_coord")
            # psycopg2 jsonb 返回的是 dict；旧数据可能 None
            row["source_coord"] = cc if isinstance(cc, dict) else None
            if row.get("id") is not None:
                exp_ids.append(row["id"])

        # 2. counterexample_count 批量查（单次 GROUP BY，非 N+1）
        ce_map = self._batch_count_counterexamples(exp_ids) if exp_ids else {}
        for row in results:
            eid = row.get("id")
            count = ce_map.get(eid, 0) if eid else 0
            row["counterexample_count"] = count
            # 反例摘要只在有反例时附（省体积）；完整列表调用方按需查
            row["counterexamples"] = []  # 默认空，避免每行都查
            if count > 0:
                # 只在有反例时附前 3 条摘要
                row["counterexamples"] = self._get_counterexample_summaries(eid, limit=3)

    def _batch_count_counterexamples(self, exp_ids: List[int]) -> Dict[int, int]:
        """批量查多个经验的反例数。

        通过 tags 数组 '原经验:{id}' 关联（走 GIN 索引 idx_knowledge_tags）。
        返回 {exp_id: count}。

        注：理论可用单次 GROUP BY 聚合避免 N 次查询，但 tags 数组 join 的
        SQL 在 psycopg2 参数化下较脆弱。当前逐个查，每个走 GIN 索引，top_k=5
        时 5 次索引查询可接受。数据量大后再优化。
        """
        if not exp_ids:
            return {}
        result = {}
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            for eid in exp_ids:
                cur.execute(
                    "SELECT COUNT(*) FROM knowledge "
                    "WHERE knowledge_type = 'counterexample' "
                    "AND superseded_by IS NULL "
                    "AND %s = ANY(tags)",
                    (f"原经验:{eid}",),
                )
                result[eid] = cur.fetchone()[0]
            cur.close()
            conn.commit()  # 衔接性修复: 只读 SELECT 也需 commit 结束事务,防连接中毒
        except Exception as e:
            try: conn.rollback()
            except Exception: pass
            import sys
            print(f"[WARN] _batch_count_counterexamples 失败（返回空 map）: {e}",
                  file=sys.stderr)
        return result

    def _get_counterexample_summaries(self, exp_id: int, limit: int = 3) -> List[dict]:
        """查某经验的反例摘要列表（供 search 结果标注）。"""
        try:
            conn = self._get_conn()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(
                "SELECT id, title, tags FROM knowledge "
                "WHERE knowledge_type = 'counterexample' "
                "AND superseded_by IS NULL "
                "AND %s = ANY(tags) "
                "ORDER BY created DESC LIMIT %s",
                (f"原经验:{exp_id}", limit),
            )
            rows = cur.fetchall()
            cur.close()
            conn.commit()  # 衔接性修复: 只读 SELECT 需 commit 结束事务
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            summaries = []
            for r in rows:
                tags_list = r.get("tags") or []
                attribution = "unknown"
                frozen_until = None
                for t in tags_list:
                    if t.startswith("attribution:"):
                        attribution = t[len("attribution:"):]
                    elif t.startswith("frozen:"):
                        frozen_until = t[len("frozen:"):]
                is_frozen = False
                if frozen_until:
                    try:
                        from datetime import datetime as _dt
                        freeze_dt = _dt.strptime(frozen_until, "%Y%m%d").replace(
                            tzinfo=timezone.utc)
                        is_frozen = now < freeze_dt
                    except ValueError:
                        pass
                summaries.append({
                    "id": r["id"],
                    "title": r["title"],
                    "attribution": attribution,
                    "frozen": is_frozen,
                })
            return summaries
        except Exception as e:
            try: conn.rollback()
            except Exception: pass
            import sys
            print(f"[WARN] _get_counterexample_summaries({exp_id}) 失败: {e}",
                  file=sys.stderr)
            return []

    def _feedback_shadow_refs(self, results: List[dict]) -> None:
        """task 253 P0-2b: 检索命中回写 SHADOW ref_count（关闭反馈环）。

        此前 search() 从不 increment ref_count → shadow_codes 表 82/84 条恒为 0
        → 熵预算/层级升降没有真实访问信号。现在把命中结果关联的 shadow_code
        计数 +1，让"用得多的浮上来"真实发生。

        注意: 当前 knowledge.shadow_code 覆盖率极低（132 行仅 2 行有码），
        因此本回写短期收效有限——需配套提升写入时的 shadow_code 标注覆盖率
        （见评审文档 follow-up）。失败绝不影响检索主流程。
        """
        try:
            codes = {
                (r.get("shadow_code") or "").upper()
                for r in results
                if r.get("shadow_code")
            }
            for code in codes:
                if code:
                    self.increment_shadow_ref(code, delta=1)
        except Exception as e:
            import sys
            print(f"[WARN] _feedback_shadow_refs 回写失败（不影响检索）: {e}",
                  file=sys.stderr)

    def get_knowledge(self, knowledge_id: int) -> Optional[dict]:
        """按 ID 获取知识条目。"""
        conn = self._get_read_conn()  # Task 379: 只读走 autocommit,防 idle in transaction
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            cur.execute("SELECT * FROM knowledge WHERE id = %s", (knowledge_id,))
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            try: cur.close()
            except Exception: pass

    def get_recent_knowledge(
        self,
        limit: int = 20,
        knowledge_type: Optional[str] = None,
    ) -> List[dict]:
        """获取最近的知识条目。"""
        conn = self._get_read_conn()  # Task 379: 只读走 autocommit
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        sql = "SELECT * FROM knowledge"
        params = []
        if knowledge_type:
            sql += " WHERE knowledge_type = %s"
            params.append(knowledge_type)
        sql += " ORDER BY created DESC LIMIT %s"
        params.append(limit)
        try:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
        finally:
            try: cur.close()
            except Exception: pass

    # ------------------------------------------------------------------
    # 知识写入
    # ------------------------------------------------------------------

    def _check_duplicate_title(self, conn, agent_id, title):
        """Task 355: 写入前相似标题检测（提示，不阻断）。

        查同 agent 最近1天是否写过标题相似（前30字 ILIKE 匹配）的经验。
        返回相似条目列表 [(id, title)] 或空列表。

        注意：只提示不阻断——小弟可能确实要写系列经验（同一任务多维度发现）。
        content_hash 精确去重在 L475-502 已做，这里补模糊标题检测。

        Task 379: 此方法接收外部写路径 conn（被 add_knowledge 调用），不能换
        _get_read_conn（会污染外层写事务的 autocommit）。改用「临时切 autocommit=True
        → finally 恢复 False」模式，只读后立即复位，外层事务语义不变。
        """
        try:
            old_ac = conn.autocommit
            try:
                conn.autocommit = True  # 临时只读自动提交,不开事务
                cur = conn.cursor()
                try:
                    cur.execute(
                        "SELECT id, title FROM knowledge "
                        "WHERE source_agent=%s AND title ILIKE %s "
                        "AND created > NOW() - INTERVAL '1 day' "
                        "AND superseded_by IS NULL LIMIT 3",
                        (agent_id, f"%{title[:30]}%"),
                    )
                    return cur.fetchall()
                finally:
                    try: cur.close()
                    except Exception: pass
            finally:
                # 复位外层写路径 autocommit（add_knowledge 依赖显式事务）
                try: conn.autocommit = old_ac
                except Exception: pass
        except Exception:
            return []

    def add_knowledge(
        self,
        title: str,
        content: str,
        knowledge_type: str = "experience",
        tags: Optional[List[str]] = None,
        project: Optional[str] = None,
        path: Optional[str] = None,
        shadow_code: Optional[str] = None,
        confidence: float = 1.0,
        verification_source: Optional[str] = None,
        embed: bool = True,
        supersedes: Optional[int] = None,
        context_coord: Optional[Dict[str, Any]] = None,
        task_type: Optional[str] = None,
        source_agent: Optional[str] = None,  # Task 284 P0-A: 显式覆盖 self.agent_id
        identity_producer: Optional[str] = None,  # Task1524: 写入链身份登记 producer 标注
    ) -> int:
        """写入知识条目。

        ⚠️ task 241 修正：confidence 的来源决定其可信度
            - verification_source="tool_verification"（实测/grep/curl）→ 高可信
            - verification_source="cross_ref"（多源交叉验证）→ 中可信
            - verification_source="user_asserted"（用户标注）→ 看用户权威性
            - verification_source=None 或 "llm_self_rated"（LLM 自评）→ **不可信**
              LLM 自评的 confidence 只是 token 概率，不反映真实置信度

        Args:
            title: 标题
            content: 内容
            knowledge_type: 类型（experience/finding/decision/pitfall/uncertainty/tacit）
            tags: 标签列表
            project: 项目名
            path: Vault 文件路径
            shadow_code: 关联 SHADOW 编码
            confidence: 置信度 [0, 1]（注意：必须由 verification_source 支撑）
            verification_source: 置信度来源（tool_verification/cross_ref/user_asserted/
                                 llm_self_rated/None）。task 241 新增。
            embed: 是否在写入时同步生成 BGE-M3 向量（task 253 新增，默认 True）。
                   修复"写入不嵌入 → 新知识对向量检索不可见"的 P0 断层。
                   BGE-M3 不可用时（如非 3.11 环境）自动降级为 embedding=NULL 并告警，
                   不阻断写入。
            supersedes: 旧知识条目 ID（task 253 P1-5）。写入成功后把旧条目标记
                        superseded_by=新ID，检索自动排除旧条目。用于"修正/更新
                        既有结论"而不物理删除（保留审计链）。
            context_coord: 场景坐标（task 263 新增）。None 时自动调
                           capture_current_context 捕获当前工作上下文（向下兼容）。
                           传入时用传入的（支持测试/回填/反例回写）。
            task_type: 任务类型（task 263 新增）。context_coord 自动捕获时用，
                       不传则从 knowledge_type 推断默认值。

        Returns:
            新建条目的 ID（若内容与既有条目完全相同则返回既有 ID，不重复写入）
        """
        # Task 284 P0-A: 显式 source_agent 覆盖 self.agent_id（api_server 全局实例复用 bug 修复）
        effective_agent_id = source_agent or self.agent_id

        # #5 修复(2026-07-29): 空 title/content 校验(Python API 路径,HTTP 端已有校验)
        # 空内容会写入空格向量污染检索 + content_hash 碰撞合并成一条垃圾
        if not title or not title.strip():
            raise ValueError("add_knowledge: title 不能为空")
        # 09181120 追加段：title 场景前缀规范（与 tag 归一同一收口点——HTTP/Python
        # 双路径覆盖；存量 title 不动，只影响新写入）
        title = _normalize_title(title, knowledge_type=knowledge_type,
                                 agent_id=effective_agent_id)

        # === Task394 P2 → Task405 收紧: write 凭据 warn+record（无proof→标proof_missing降权）===
        # 经验内容含 proof:xxx:yyy = 有证据指针。无 proof = 标 proof_missing tag + 降权（confidence×0.5）。
        # 硬凭据(file/test/grep): 服务端机械验证。软凭据(reasoning/analogy): 结构校验。
        # **不阻断录入**（warn+record 非reject）：经验仍存，但降权标记。后续检索排序靠后/可过滤。
        _proof_missing = False
        try:
            from nexusos_xdr.credential import verify_proof
            from nexusos_mem._vault_env import VAULT_ROOT as _VR
            _proof_ok, _proof_type, _proof_detail = verify_proof(
                content or "",
                project_root=path or str(_VR),
            )
            if _proof_type == "none":
                _proof_missing = True
                import sys as _sys_p2
                print(f"[Task405] add_knowledge proof_missing: title={title[:40]!r} "
                      f"(warn+record降权confidence×0.5, 不阻断录入)", file=_sys_p2.stderr, flush=True)
        except ImportError:
            pass  # credential 模块不可用时跳过（不标 proof_missing）
        except Exception as _pe:
            import sys as _sys_p2
            print(f"[Task405] proof 校验异常(不阻塞): {_pe}", file=_sys_p2.stderr, flush=True)
        if not content or not content.strip():
            raise ValueError("add_knowledge: content 不能为空")
        # task 241: 把 verification_source 加入 tags 以便过滤查询
        # 1401-d5t6 验收修复：去掉中间 list(tags)——它把字符串 tags 字符级拆碎
        # （#73184/#73186 实证），原始输入直传 _normalize_tags，其 isinstance(str)
        # 分支按分隔符拆分（数组输入行为不变）。verification_source 系统标签
        # 在归一后追加（不参与归一映射）。
        final_tags = _normalize_tags(tags, title=title)
        if verification_source and verification_source not in final_tags:
            final_tags.append(f"verified_by:{verification_source}")
        # Task405: 无 proof 的经验标 proof_missing + 降权 confidence×0.5
        if _proof_missing:
            final_tags.append("proof_missing")
            confidence = confidence * 0.5
            import sys as _sys_pm
            print(f"[Task405] proof_missing 降权: confidence {confidence*2:.1f}→{confidence:.1f}, "
                  f"tag=proof_missing（检索排序靠后/可过滤）", file=_sys_pm.stderr, flush=True)
        # task 241: 警告 LLM 自评的 confidence
        if verification_source == "llm_self_rated" and confidence >= 0.7:
            import sys
            print(f"[WARN] add_knowledge: verification_source='llm_self_rated' "
                  f"但 confidence={confidence} ≥ 0.7 — LLM 自评不可信，建议降低或独立验证",
                  file=sys.stderr)

        # 衔接性修复(2026-07-29): knowledge_type 归一化为小写
        # 写入端不归一化,读取端 search.py 的 IN('experience',...) 硬编码小写,
        # 传入 "Experience"/"FINDING" 会漏检索。统一在入口 lowercase。
        if knowledge_type:
            knowledge_type = knowledge_type.lower().strip()

        # task 263: 场景坐标自动捕获（向下兼容——不传 context_coord 时自动捕获）
        if context_coord is None:
            try:
                from nexusos_mem.context_capture import capture_current_context
                inferred_task_type = task_type or knowledge_type or "general"
                context_coord = capture_current_context(
                    agent_name=effective_agent_id,  # Task 284 P0-A: 用 effective_agent_id
                    task_type=inferred_task_type,
                )
            except Exception as e:
                import sys
                print(f"[WARN] add_knowledge: context_coord 自动捕获失败（写入 NULL）: {e}",
                      file=sys.stderr)
                context_coord = None

        # task 253 P0-1: 写入即嵌入。此前 add_knowledge 只 INSERT 不 embed，
        # 新知识 embedding=NULL → pgvector_search 的 WHERE embedding IS NOT NULL
        # 会静默排除它，直到下次 embed_corpus 批处理才可见（多 Agent 无法实时互通）。
        embedding = None
        content_hash = None
        if embed:
            embedding, content_hash = self._embed_text(f"{title} {content or ''}")
            if embedding is None:
                # task 267 R5: 静默失败可见化（#22991 修复）
                # embedding 失败仍写 NULL（向后兼容不阻断业务），但显式告警
                import sys
                print(f"[WARN] add_knowledge embedding 失败 agent={effective_agent_id} "  # Task 284 P0-A
                      f"title={title[:40]!r}（worker+本地均不可用，降级 embedding=NULL）",
                      file=sys.stderr, flush=True)

        conn = self._get_conn()

        # task 253 P1-5: 精确去重。content_hash 相同 = title+content 前缀完全一致，
        # 直接复用既有条目（刷新 verified_at 表示"再次确认仍有效"），不产生重复行。
        if content_hash:
            cur = conn.cursor()
            cur.execute(
                "SELECT id FROM knowledge "
                "WHERE content_hash = %s AND superseded_by IS NULL "
                "ORDER BY id DESC LIMIT 1",
                (content_hash,),
            )
            dup = cur.fetchone()
            if dup:
                existing_id = dup[0]
                cur.execute(
                    "UPDATE knowledge SET verified_at = NOW(), updated = NOW() "
                    "WHERE id = %s", (existing_id,),
                )
                if supersedes and supersedes != existing_id:
                    cur.execute(
                        "UPDATE knowledge SET superseded_by = %s WHERE id = %s",
                        (existing_id, supersedes),
                    )
                try:
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    try:
                        cur.close()
                    except Exception:
                        pass
                self.log_activity("write", "knowledge", target_id=existing_id,
                                  detail={"dedup": True})
                return existing_id
            cur.close()

        # task 253 follow-up: 未显式指定 shadow_code 时自动确定性分配，
        # 保证全链路关联（编码=join key）+ P0-2b 反馈环有信号流动。
        if not shadow_code:
            try:
                from nexusos_mem.shadow_match import assign_shadow_code
                shadow_code = assign_shadow_code(
                    conn, title=title, content=content, tags=final_tags,
                    project=project, path=path, knowledge_type=knowledge_type,
                )
            except Exception as e:
                import sys
                print(f"[WARN] shadow_code 自动分配失败（写入 NULL）: {e}",
                      file=sys.stderr)

        cur = conn.cursor()
        # task 263: INSERT 加 context_coord 列（一步写入，解决两步非原子）
        context_coord_json = json.dumps(context_coord, ensure_ascii=False) if context_coord else None
        # task/1452: 中文 FTS jieba 双轨——写入侧预分词（flag off 返回 None=列 NULL，
        # 行为与改造前字节级一致；flag on 存空格分词串，查询侧 search_vector_zh 词级匹配）
        from nexusos_mem.tokenize_zh import tokenize_for_write
        tokens_zh_val = tokenize_for_write(title, content)
        # Task 355: INSERT 前相似标题检测（warning 不阻断）。
        # content_hash 精确去重在 L475-502 已做，这里补模糊标题检测——
        # 同 agent 1天内写过相似标题（前30字 ILIKE）就 warning，但仍写入（可能是系列经验）。
        _dup_titles = self._check_duplicate_title(conn, effective_agent_id, title)
        if _dup_titles:
            import sys
            print(f"[WARN][Task355] add_knowledge 疑似重复标题 (agent={effective_agent_id} "
                  f"title={title[:40]!r}): {[(d[0], d[1][:30]) for d in _dup_titles]}. "
                  f"仍写入但建议确认是否重复", file=sys.stderr, flush=True)
        # Task1524: 同事务标注 producer——knowledge 插入触发器读 app.producer
        # 填 knowledge_identity.producer（区分 api_server / 直连 Python 等写入链）。
        # set_config 第三参 true=事务级，随 commit 消失，不污染连接池语义。
        cur.execute(
            "SELECT set_config('app.producer', %s, true)",
            (identity_producer or "nexusos_mem.shared:add_knowledge",),
        )
        cur.execute("""
            INSERT INTO knowledge
                (source_agent, knowledge_type, title, content, tags, project, path,
                 shadow_code, confidence, embedding, content_hash, verified_at,
                 context_coord, entity_type, tokens_zh)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    CASE WHEN %s IS NOT NULL THEN NOW() ELSE NULL END, %s, %s, %s)
            RETURNING id
        """, (
            effective_agent_id, knowledge_type, title, content,  # Task 284 P0-A: self.agent_id → effective_agent_id
            final_tags, project, path, shadow_code, confidence,
            embedding, content_hash, embedding, context_coord_json,
            knowledge_type,  # Task655: entity_type=knowledge_type 写入侧防回归（存量已回填，新行不再 NULL）
            tokens_zh_val,  # task/1452: jieba 预分词（off=NULL 与旧轨一致；on=空格分词串）
        ))
        # Task 337 verified_at 语义注释：上方 CASE WHEN %s(embedding) IS NOT NULL THEN NOW()
        # 的语义是"有 embedding 就 NOW()"，实际含义更接近"embedded_at"。字段名 verified_at
        # 保留做向后兼容（重命名是破坏性，需 schema + 全代码改），勿理解为"已验证可信"。
        # 真正的"已验证"信号看 tags 含 verified_by:* 而非 verified_at 是否非空。
        try:
            kid = cur.fetchone()[0]
        except psycopg2.errors.UniqueViolation:
            # Task 377: 并发写——另一进程已用同 content_hash 入库（部分 UNIQUE 索引
            # idx_knowledge_content_hash_unique 拦截）。UniqueViolation 使事务进入
            # aborted 状态，必须 rollback 才能发新查询；然后回查已存在 id 返回。
            # 这是 L541 check-then-insert 的竞态安全网（SELECT 与 INSERT 之间的窗口）。
            conn.rollback()
            cur.execute(
                "SELECT id FROM knowledge "
                "WHERE content_hash = %s AND superseded_by IS NULL "
                "ORDER BY id DESC LIMIT 1",
                (content_hash,),
            )
            dup = cur.fetchone()
            try:
                cur.close()
            except Exception:
                pass
            if dup:
                existing_id = dup[0]
                self.log_activity("write", "knowledge", target_id=existing_id,
                                  detail={"dedup": True, "concurrent": True})
                return existing_id
            raise  # 理论不可达：UniqueViolation 必有匹配行
        # task 253 P1-5: 新知识修正旧条目 → 旧条目退出检索（保留行做审计）
        if supersedes and supersedes != kid:
            cur.execute(
                "UPDATE knowledge SET superseded_by = %s WHERE id = %s",
                (kid, supersedes),
            )
        try:
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            try:
                cur.close()
            except Exception:
                pass

        self.log_activity("write", "knowledge", target_id=kid,
                          detail={"embedded": embedding is not None,
                                  **({"needs_reembed": True} if embedding is None and embed else {}),
                                  **({"supersedes": supersedes} if supersedes else {})})
        # 问题4修复(2026-07-29): 写入新知识时更新 shadow_code 的 ref_count。
        # 原版只在 search 命中时 increment,写入不更新 → ref_count 严重失真。
        if shadow_code:
            try:
                self.increment_shadow_ref(shadow_code, delta=1)
            except Exception:
                pass  # ref_count 更新失败不阻塞写入
        return kid

    def add_knowledge_batch(self, items: List[dict]) -> List[int]:
        """批量写入知识（减少串行调用样板）。

        Args:
            items: [{title, content, knowledge_type, tags, project, ...}, ...]
        Returns:
            [id1, id2, ...] 写入的ID列表（顺序与 items 一致）
        """
        ids = []
        for item in items:
            kid = self.add_knowledge(
                title=item["title"],
                content=item["content"],
                knowledge_type=item.get("knowledge_type", "finding"),
                tags=item.get("tags", []),
                project=item.get("project"),
            )
            ids.append(kid)
        return ids

    def supersede_knowledge(self, old_id: int, new_id: int) -> bool:
        """把旧知识条目标记为被 new_id 取代（task 253 P1-5）。

        被取代的条目不再出现在任何检索结果中，但物理保留（审计/回溯）。
        不做级联：A→B→C 链上只需保证每个旧条目都有 superseded_by 即可。

        Returns:
            True=标记成功, False=old_id 不存在
        """
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "UPDATE knowledge SET superseded_by = %s, updated = NOW() WHERE id = %s",
            (new_id, old_id),
        )
        ok = cur.rowcount > 0
        try:
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            try:
                cur.close()
            except Exception:
                pass
        if ok:
            self.log_activity("update", "knowledge", target_id=old_id,
                              detail={"superseded_by": new_id})
        return ok

    @staticmethod
    def _embed_text(text: str):
        """生成 BGE-M3 归一化向量 + content_hash（task 253）。

        复用 nexus.search 的进程内模型单例：长命进程（Hermes）零额外成本，
        短命进程首次付一次加载。BGE-M3 不可用（非 3.11 环境 / 模型缺失）时
        返回 (None, None)，调用方降级为 embedding=NULL 写入，不阻断。

        Returns:
            (embedding: list[float] | None, content_hash: str | None)
        """
        import hashlib
        from nexusos_mem.search import encode_query, EMBED_MAX_CHARS

        snippet = (text or "")[:EMBED_MAX_CHARS]
        vec = encode_query(snippet)  # 失败内部已 print 告警并返回 None
        if vec is None:
            return None, None
        chash = hashlib.sha1(snippet.encode("utf-8")).hexdigest()[:16]
        return vec, chash

    # ------------------------------------------------------------------
    # SHADOW 编码
    # ------------------------------------------------------------------

    def get_shadow_codes(self, layer: Optional[str] = None) -> List[dict]:
        """获取 SHADOW 编码列表。

        Args:
            layer: 过滤层级（L0/L1/L2/L3），None = 全部

        Returns:
            [{"code", "layer", "target", "section", "ref_count", "weight", "last_used"}, ...]
        """
        conn = self._get_read_conn()  # Task 379: 只读走 autocommit
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        if layer:
            cur.execute("SELECT * FROM shadow_codes WHERE layer = %s ORDER BY code", (layer,))
        else:
            cur.execute("SELECT * FROM shadow_codes ORDER BY layer, code")
        try:
            return [dict(row) for row in cur.fetchall()]
        finally:
            try: cur.close()
            except Exception: pass

    def increment_shadow_ref(self, code: str, delta: int = 1) -> None:
        """增加 SHADOW 编码引用计数。"""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("""
            UPDATE shadow_codes
            SET ref_count = ref_count + %s, last_used = NOW(), updated = NOW()
            WHERE code = %s
        """, (delta, code.upper()))
        try:
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            try:
                cur.close()
            except Exception:
                pass
        # Task 334: SHADOW 已于 2026-07-23 退役，feedback 日志是孤儿数据（零消费者）。
        # 但 ref_count UPDATE（626-631 行）仍被 injection 注入链路消费，保留。
        # 用 SHADOW_ENABLED 守卫只停 log_activity，不动 UPDATE。
        if os.environ.get("NEXUS_SHADOW_ENABLED", "").lower() in ("1", "true", "yes"):
            self.log_activity("feedback", "shadow_code", detail={"code": code, "delta": delta})

    def get_top_shadow_codes(self, limit: int = 15) -> List[dict]:
        """获取熵预算最高的 SHADOW 编码。

        task 253 P0-2a: 原 SQL 用 `ref_count ^ 1.5 * weight`，而 82/84 条 ref_count=0
        → 0^1.5=0 → 全部并列 0 分 → 退化成按字母序，熵预算失效。
        改为复用 algorithms.compute_entropy_score（task 252 已加 last_access 副项），
        与 injection.py 同一公式，消除公式漂移。
        """
        from datetime import datetime, timezone
        from nexusos_mem.algorithms import compute_entropy_score

        conn = self._get_read_conn()  # Task 379: 只读走 autocommit
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            cur.execute("""
                SELECT * FROM shadow_codes
                WHERE layer IN ('L0', 'L1', 'L2')
            """)
            rows = [dict(row) for row in cur.fetchall()]
        finally:
            try: cur.close()
            except Exception: pass

        now = datetime.now(timezone.utc)
        for r in rows:
            last_used = r.get("last_used")
            if last_used is not None:
                delta_h = max((now - last_used).total_seconds() / 3600.0, 0.0)
            else:
                delta_h = 999.0
            r["score"] = compute_entropy_score(
                R=r.get("ref_count") or 0,
                W=r.get("weight") or 1.0,
                delta_hours=delta_h,
            )

        # 稳定排序：先按 layer 升序（L0>L1>L2 更重要），再按 score 降序覆盖，
        # 使同分时低层级（更重要）优先。
        rows.sort(key=lambda x: (x.get("layer") or "L9"))
        rows.sort(key=lambda x: x["score"], reverse=True)
        return rows[:limit]

    # ------------------------------------------------------------------
    # RL 参数
    # ------------------------------------------------------------------

    def get_rl_params(self) -> Dict[str, float]:
        """获取当前 RL 参数。"""
        conn = self._get_read_conn()  # Task 379: 只读走 autocommit
        cur = conn.cursor()
        try:
            cur.execute("SELECT param_name, value FROM rl_params")
            return {row[0]: row[1] for row in cur.fetchall()}
        finally:
            try: cur.close()
            except Exception: pass

    def update_rl_param(self, param_name: str, value: float) -> None:
        """更新 RL 参数。"""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO rl_params (param_name, value) VALUES (%s, %s)
            ON CONFLICT (param_name) DO UPDATE SET value = EXCLUDED.value, updated = NOW()
        """, (param_name, value))
        try:
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            try:
                cur.close()
            except Exception:
                pass

    def log_activity(
        self,
        action: str,
        target_type: Optional[str] = None,
        target_id: Optional[int] = None,
        detail: Optional[dict] = None,
        task_id: Optional[str] = None,  # Task 319: 写入 agent_activity.task_id 列（原 INSERT 漏写致 99.88% NULL）
        agent_id: Optional[str] = None,  # Task478: 显式归因（修复 _global_kb 单例固化）
    ) -> Optional[int]:
        """记录 agent 活动。返回新建的 activity_id（Task396 Step6 改 RETURNING id）。

        Task 319: 新增 task_id 可选参数，默认 None 向后兼容。
        Task396 Step6: 返回 activity_id 供 annotate_apply_attempt → record_apply_outcome 关联。
        """
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO agent_activity (agent_id, action, target_type, target_id, detail, task_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            agent_id or self.agent_id, action, target_type, target_id,  # Task478: 显式归因优先
            json.dumps(detail, ensure_ascii=False) if detail else None,
            task_id,
        ))
        _activity_id = None
        try:
            _row = cur.fetchone()
            _activity_id = _row[0] if _row else None
        except Exception:
            pass
        try:
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            try:
                cur.close()
            except Exception:
                pass
        return _activity_id  # Task396 Step6: 返回 activity_id

    def resolve_pending_attempt(self, exp_id: int, agent_id: str | None = None) -> dict:
        """Task1401-W3: knowledge_id→apply_attempt_id 解析（G10 三口径对齐）。

        Agent 手头常只有检索所得的 knowledge_id（文档口径），而回填 API 要
        apply_attempt_id（API 口径）——断链致照文档做必报错。本方法按 G10
        判定同源语义（outcome_filled.py）解析该经验最新一条未回填 attempt：

        - action='apply_attempt' AND target_id=exp_id
        - outcome IS NULL OR 'pending'
- 排除已被「真回填」的：result='pending' 是 Task412 通电占位（done 回调钩子
          预写的占位行，不代表主控已裁定），不算已回填——否则占位行会挡住解析，
          恒报「无未回填 attempt」，HTTP 回填链断。旧回填路径的 outcome 行可能
          无 result 键（NULL），按已回填处理（防双重回填），口径同
          apply_honesty.py Task581 COALESCE != 'pending'（G10 outcome_filled 兼容）
        - agent_id 给定时限定该 agent（多 Agent 并发同经验不串号）

        Returns: {"ok": True, "apply_attempt_id": N} 或 {"ok": False, "error": ...}
        """
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            cur.execute("SELECT 1 FROM knowledge WHERE id=%s", (exp_id,))
            if not cur.fetchone():
                return {"ok": False, "error": f"knowledge id={exp_id} 不存在", "status": 404}
            cur.execute(
                """
                SELECT a.id FROM agent_activity a
                WHERE a.action='apply_attempt' AND a.target_id=%s
                  AND (a.outcome IS NULL OR a.outcome='pending')
                  AND NOT EXISTS (
                    SELECT 1 FROM agent_activity o
                    WHERE o.action='apply_outcome' AND o.target_id=a.target_id
                      AND o.detail->>'apply_attempt_id'=a.id::text
                      -- pending=通电占位不算回填；COALESCE 让无 result 键的旧行
                      -- 判「非 pending」=已回填（NULL 语义防双重回填，池#104）
                      AND COALESCE(o.detail->>'result', '') != 'pending')
                  AND (%s IS NULL OR a.agent_id=%s)
                ORDER BY a.id DESC LIMIT 1
                """,
                (exp_id, agent_id, agent_id),
            )
            row = cur.fetchone()
            if not row:
                return {
                    "ok": False,
                    "error": (
                        f"knowledge id={exp_id} 无未回填的 apply_attempt"
                        + (f"（agent={agent_id}）" if agent_id else "")
                        + "——回填前提是 search 带 agent_id 触发 annotate 产生真套用记录"
                    ),
                    "status": 400,
                }
            return {"ok": True, "apply_attempt_id": row[0]}
        finally:
            cur.close()

    def record_apply_outcome(
        self,
        apply_attempt_id: int,
        exp_id: int,
        result: str,
        evidence: str = "",
        agent_id: Optional[str] = None,  # Task1528: 显式归因（HTTP body 传参优先）
    ) -> dict:
        """Task387 Part A: Agent 套用经验后回写真实结果（认知闭环关键）。
        Task396 Step5: 加 pending 校验（通电占位值,主控验收改 success/failed）。

        Args:
            apply_attempt_id: 对应的 apply_attempt 记录 id（agent_activity）
            exp_id: 被套用的经验 knowledge.id
            result: success / partial / failed / not_applicable
            evidence: 结果证据（failed 时作为 counterexample 内容）

        写两处：
        1. agent_activity(action='apply_outcome', detail={result, evidence, ...})
        2. knowledge.outcome = result（覆盖 crucible 的预测值；
           not_applicable 例外——它是对 attempt 的裁定（评审确认非真套用，U8），
           不含经验价值信息，不覆盖 knowledge.outcome、不计 ref_count）

        result='failed' + evidence → 自动创建 counterexample
        （激活反例机制，search 读路径已现成 _batch_count_counterexamples:337）
        """
        result = result.lower().strip()
        if result not in ("success", "partial", "failed", "pending", "not_applicable"):
            return {"ok": False, "error": f"result 必须是 success/partial/failed/pending/not_applicable, 得到: {result}"}

        # 1635(#213 第二行) 幂等守卫：同 attempt 已有非 superseded outcome 行 → 直接复用
        # （并发下 pg_advisory_xact_lock 串行化同 attempt 写入：先提交者写行，后来者
        # 过锁后查见行即复用；配合部分唯一索引兜底防毫秒窗漏网——索引 DDL 见本单 SQL）
        _gcur = self._get_conn().cursor()
        try:
            _gcur.execute("SELECT pg_advisory_xact_lock(%s)", (int(apply_attempt_id),))
            _gcur.execute(
                "SELECT detail->>'result' FROM agent_activity "
                "WHERE action='apply_outcome' "
                "AND (detail->>'superseded') IS DISTINCT FROM 'true' "
                "AND detail->>'apply_attempt_id' = %s "
                # 真值行优先取（1701 叠加态对账发现：909 个 attempt 同时挂着
                # 早期 pending 行+晚期真值行——pre-1635 时代无守卫，两种行共存。
                # 若按 id 取最早会把真值 attempt 误判成 pending 占位，真回填
                # 反杀已有结论。真值存在→取真值走幂等；否则取最早占位走覆盖。）
                "ORDER BY (detail->>'result' IN ('success','partial','failed')) DESC, id LIMIT 1",
                (str(apply_attempt_id),),
            )
            _row = _gcur.fetchone()
            if _row:
                # task/1701（池#277）：旧值是「非终态值」而新值不同 → 让位覆盖——
                # 旧行标 superseded=true，穿透主链写新值行。覆盖两类：
                # ① pending 占位（callback 通电自动写，handler.py Task412）：占位不是
                #   结论，当结论复用=拦死真回填（经验永 pending，G10 误 REJECT，
                #   源单 task/1667 三条 attempt 全靠 DB 手工 UPDATE 消账）；
                # ② not_applicable 裁定（D5B late-override 语义，apply_honesty.py:1028）：
                #   裁定不是终审，迟到真回填仍可覆盖——1635 守卫曾把它一并拦死
                #   （U4/U5 既有测试实证红）。
                # 同值重入保持幂等；pending 是最低权值，永不覆盖任何已有值。
                # 旧值已是真值（success/partial/failed）→ 幂等复用不变（1635 原语义）。
                if _row[0] in ("pending", "not_applicable") and result not in ("pending", _row[0]):
                    _gcur.execute(
                        "UPDATE agent_activity SET detail = detail || '{\"superseded\": \"true\"}'::jsonb "
                        "WHERE action='apply_outcome' "
                        "AND (detail->>'superseded') IS DISTINCT FROM 'true' "
                        "AND detail->>'apply_attempt_id' = %s",
                        (str(apply_attempt_id),),
                    )
                    # 锁继续持有（同连接同事务），随主链 log_activity 写入 + commit 一并释放
                else:
                    self._get_conn().commit()  # 只读未写，提交以释放 advisory 锁
                    return {"ok": True, "result": _row[0], "exp_id": exp_id,
                            "counterexample_created": False, "idempotent_reuse": True}
            # 未命中（或占位已让位）：锁继续持有（同连接同事务），随主链 log_activity 写入 + 步骤2 commit 一并释放
        finally:
            try:
                _gcur.close()
            except Exception:
                pass

        ret = {"ok": True, "result": result, "exp_id": exp_id, "counterexample_created": False}

        # Task532 P-s1: 补查 apply_attempt 行的 task_id（原 outcome 行无 task_id——
        # G10 按任务查回填、D2 类审计都覆盖不到 callback 写的 pending 占位）
        # Task1528: 同查询顺带取 attempt 行 agent_id——body 漏传 agent_id 时
        # 用套用者本人归因（比 "anonymous" 兜底语义更对，池 #204）。
        _ao_task_id = None
        _attempt_agent = None
        try:
            _c_s1 = self._get_conn().cursor()
            _c_s1.execute(
                "SELECT task_id, agent_id FROM agent_activity WHERE id = %s",
                (apply_attempt_id,),
            )
            _r_s1 = _c_s1.fetchone()
            _c_s1.close()
            if _r_s1:
                if _r_s1[0]:
                    _ao_task_id = _r_s1[0]
                if _r_s1[1]:
                    _attempt_agent = _r_s1[1]
        except Exception:
            pass  # 关联查询失败不阻塞 outcome 写入（task_id 置 None 与原行为一致）

        # Task1528 归因优先级：显式传参 > kb 实例真身份（≠anonymous）> attempt
        # 行套用者 > anonymous。已具名的程序化调用方（crucible 裁定 / xdr 回调
        # 占位）行为不变；只有 HTTP 漏传 body agent_id 的洞走 attempt 回填。
        _writer_agent = (
            agent_id
            or (self.agent_id if self.agent_id != "anonymous" else None)
            or _attempt_agent
            or "anonymous"
        )

        # 1. 写 agent_activity
        self.log_activity(
            action="apply_outcome",
            target_type="knowledge",
            target_id=exp_id,
            task_id=_ao_task_id,
            agent_id=_writer_agent,
            detail={
                "result": result,
                "evidence": evidence[:500] if evidence else "",
                "apply_attempt_id": apply_attempt_id,
                "exp_id": exp_id,
            },
        )


        # 1.5 Task1401-D5B: 回写 apply_attempt 行的 outcome 列（治 MEM#49286
        # 双表示断裂——原实现只写 action 行不回写列，列口径统计/G10/裁定筛选
        # 都按「列 OR 行」双查兜底。此处列回写让两表示归一；幂等（同值重写
        # 无害），对既有调用方是纯增强零破坏。
        try:
            cur_lw = self._get_conn().cursor()
            cur_lw.execute(
                "UPDATE agent_activity SET outcome=%s WHERE id=%s AND action='apply_attempt'",
                (result, apply_attempt_id),
            )
            cur_lw.close()
        except Exception:
            pass  # 列回写失败不阻塞主链路（行表示已写入，双查口径仍兼容）

        # 2. 更新 knowledge.outcome（覆盖 crucible 预测值；
        #    not_applicable 跳过——attempt 级裁定不污染经验级 outcome/ref_count）
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            if result != "not_applicable":
                cur.execute(
                    "UPDATE knowledge SET outcome=%s, updated=NOW() WHERE id=%s",
                    (result, exp_id),
                )
            # Task616 ref_count 回写：success/partial 时 +1（真套用=被引用价值；
            # failed 不加——被证伪不算引用，任务书 §5 Step1 口径）。
            # 幂等注意：本函数每次真实 outcome 回调只应调一次；重复回调同 attempt
            # 会在 apply_outcome 行层面留下重复记录（既有语义），ref_count 同步多计
            # 与 outcome 行口径一致，不额外去重。
            if result in ("success", "partial"):
                try:
                    cur.execute(
                        "UPDATE knowledge SET ref_count = ref_count + 1 WHERE id=%s",
                        (exp_id,),
                    )
                except Exception as _rc_e:
                    import logging
                    logging.getLogger(__name__).warning(
                        "[Task616] ref_count 回写失败(不阻断 outcome): %s", _rc_e,
                    )
            # Task548: 同步回写 attempt 行的 outcome 列（单一真源设计——
            # action 行是 API 契约真源，列是一致性快照。原只写 action 行不落列，
            # 致列口径统计（is_real_apply 态1/§13.1 老化）把已回填当 pending 永挂）
            try:
                cur.execute(
                    "UPDATE agent_activity SET outcome=%s WHERE id=%s AND action='apply_attempt'",
                    (result, apply_attempt_id),
                )
            except Exception as _col_e:
                import logging
                logging.getLogger(__name__).warning(
                    "[Task548] attempt outcome 列回写失败(不阻断 action 行语义, id=%s): %s",
                    apply_attempt_id, _col_e,
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            try: cur.close()
            except Exception: pass

        # 3. failed + evidence → 自动创建 counterexample（去重：同 exp 同 evidence 不重复）
        if result == "failed" and evidence:
            try:
                ce_tag = f"原经验:{exp_id}"
                conn = self._get_read_conn()
                cur = conn.cursor()
                cur.execute(
                    "SELECT count(*) FROM knowledge WHERE knowledge_type='counterexample' "
                    "AND superseded_by IS NULL AND %s = ANY(tags) "
                    "AND content = %s",
                    (ce_tag, evidence[:500]),
                )
                exists = cur.fetchone()[0]
                cur.close()
                if exists == 0:
                    self.add_knowledge(
                        title=f"反例: exp:{exp_id} 套用失败",
                        content=evidence[:500],
                        knowledge_type="counterexample",
                        tags=[ce_tag],
                        confidence=0.8,
                        embed=False,
                    )
                    ret["counterexample_created"] = True
            except Exception as e:
                import logging
                logging.getLogger(__name__).debug(
                    "counterexample 创建失败(non-fatal): %s", e,
                )

        return ret

    def expire_stale_pendings(self, days: int = 30) -> dict:
        """Task 462: pending 超 N 天 → 根据 search 命中次数 + 反例数推断 outcome。

        Task1561 (1547-F8): 默认 7→30 天，对齐 loop-auto-advance §13.1 设计口径
        「超阈值时间（建议 30 天）后进入 pending 老化路径」。分层语义：告警层
        （监控 I15）48h 提醒回填保持不变——告警≠执行，两层各司其职。

        推断规则（设计书 v2 line 107/120/165）：
        - 有反例（counterexample 存在）→ failed
        - 命中≥INFER_MIN_HITS 且 反例=0 → inferred_apply
        - 命中<INFER_MIN_HITS 且 反例=0 → uncertain

        推断直接 UPDATE knowledge.outcome，不走 record_apply_outcome
        （避免污染 apply_outcome 统计），每条写 agent_activity 审计。

        Returns:
            {scanned, inferred_apply, failed, uncertain}
        """
        stats = {"scanned": 0, "inferred_apply": 0, "failed": 0, "uncertain": 0}

        conn = self._get_conn()
        cur = conn.cursor()
        try:
            # 1. 查所有超时 pending + 真实缺口
            # Task536 修复3(P1-5): 原只扫 outcome='pending'（全表 1 行）→ 运行时空转。
            # 真实缺口 = 28655 行 NULL 中"有 apply_attempt 但主控从未回填"的行（实测 124）。
            # 只纳入有 apply_attempt 的（全部 NULL 绝大多数是历史死数据，纳入会噪声爆炸）。
            cur.execute(
                "SELECT id FROM knowledge "
                "WHERE updated < NOW() - %s::interval "
                "AND (outcome = 'pending' "
                "     OR (outcome IS NULL AND id IN ("
                "         SELECT target_id FROM agent_activity "
                "         WHERE action = 'apply_attempt' "
                "         AND target_type = 'knowledge')))",
                (f"{days} days",),
            )
            exp_ids = [r[0] for r in cur.fetchall()]
            stats["scanned"] = len(exp_ids)

            if not exp_ids:
                conn.commit()
                return stats

            # 2. 批量算命中次数（jsonb_array_elements_text 展开 search 的 experience_ids）
            hit_counts: Dict[int, int] = {}
            cur.execute(
                "SELECT eid::int AS exp_id, COUNT(*) AS hit_count "
                "FROM agent_activity, "
                "     jsonb_array_elements_text(detail->'experience_ids') AS eid "
                "WHERE action = 'search' "
                "  AND detail IS NOT NULL "
                "  AND jsonb_typeof(detail->'experience_ids') = 'array' "
                "  AND eid::int = ANY(%s) "
                "GROUP BY eid::int",
                (exp_ids,),
            )
            for exp_id, hit_count in cur.fetchall():
                hit_counts[exp_id] = hit_count

            # 3. 批量算反例数（复用 _batch_count_counterexamples，内部独立连接）
            counterexample_counts = self._batch_count_counterexamples(exp_ids)

            # 4. 逐条推断 + UPDATE + 审计记录（同一事务，原子性保证）
            for eid in exp_ids:
                hits = hit_counts.get(eid, 0)
                ce_count = counterexample_counts.get(eid, 0)

                if ce_count > 0:
                    new_outcome = "failed"
                elif hits >= INFER_MIN_HITS:
                    new_outcome = "inferred_apply"
                else:
                    new_outcome = "uncertain"

                cur.execute(
                    "UPDATE knowledge SET outcome = %s, updated = NOW() "
                    "WHERE id = %s",
                    (new_outcome, eid),
                )
                cur.execute(
                    "INSERT INTO agent_activity "
                    "  (agent_id, action, target_type, target_id, detail) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (
                        self.agent_id,
                        "outcome_inferred",
                        "knowledge",
                        eid,
                        json.dumps(
                            {
                                "exp_id": eid,
                                "old": "pending",
                                "new": new_outcome,
                                "hits": hits,
                                "counterexamples": ce_count,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                )
                stats[new_outcome] += 1

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            try:
                cur.close()
            except Exception:
                pass

        return stats

    def get_activity(
        self,
        agent_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[dict]:
        """获取活动日志。

        Args:
            agent_id: 过滤特定 agent（None = 全部）
            limit: 返回数
        """
        conn = self._get_read_conn()  # Task 379: 只读走 autocommit
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        if agent_id:
            cur.execute("""
                SELECT * FROM agent_activity WHERE agent_id = %s
                ORDER BY created DESC LIMIT %s
            """, (agent_id, limit))
        else:
            cur.execute("SELECT * FROM agent_activity ORDER BY created DESC LIMIT %s", (limit,))
        try:
            return [dict(row) for row in cur.fetchall()]
        finally:
            try: cur.close()
            except Exception: pass

    # ------------------------------------------------------------------
    # Agent 管理
    # ------------------------------------------------------------------

    def get_agents(self) -> List[dict]:
        """获取所有注册的 agent。"""
        conn = self._get_read_conn()  # Task 379: 只读走 autocommit
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            cur.execute("SELECT * FROM agents ORDER BY last_seen DESC")
            return [dict(row) for row in cur.fetchall()]
        finally:
            try: cur.close()
            except Exception: pass

    # ------------------------------------------------------------------
    # 写入辅助（task 275 P0-4：供 api_server 复用，避免调 _get_conn 私有方法）
    # ------------------------------------------------------------------

    def find_id_by_title_content(self, title: str, content: str) -> Optional[int]:
        """精确匹配 title+content 的知识条目 ID（供去重判断）。无匹配返回 None。"""
        conn = self._get_read_conn()  # Task 379: 只读走 autocommit
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT id FROM knowledge "
                "WHERE title=%s AND content=%s LIMIT 1",
                (title, content),
            )
            row = cur.fetchone()
            return row[0] if row else None
        finally:
            cur.close()

    def get_shadow_code(self, kid: int) -> Optional[str]:
        """读某条 knowledge 的 shadow_code。无则返回 None。"""
        conn = self._get_read_conn()  # Task 379: 只读走 autocommit
        cur = conn.cursor()
        try:
            cur.execute("SELECT shadow_code FROM knowledge WHERE id=%s", (kid,))
            row = cur.fetchone()
            return row[0] if row else None
        finally:
            cur.close()

    def assign_shadow_code_for_kid(
        self, kid: int, title: str, content: str,
        tags: Optional[List[str]] = None, project: Optional[str] = None,
        path: Optional[str] = None, knowledge_type: str = "experience",
    ) -> Optional[str]:
        """为已写入的知识补 SHADOW 编码并 UPDATE 落库。返回编码或 None。"""
        from nexusos_mem.shadow_match import assign_shadow_code
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            shadow_code = assign_shadow_code(
                conn, title=title, content=content, tags=tags or [],
                project=project, path=path, knowledge_type=knowledge_type,
            )
            if shadow_code:
                cur.execute(
                    "UPDATE knowledge SET shadow_code=%s WHERE id=%s",
                    (shadow_code, kid),
                )
                try:
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
            return shadow_code
        finally:
            try:
                cur.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Embedding 维护（task 275 P0-3：供 mcp_server embed patrol 复用，
    # 避免调 _get_conn 私有方法 + 裸 SQL 绕过封装）
    # ------------------------------------------------------------------

    def count_embedding_null(self) -> int:
        """统计 embedding=NULL 且未废弃的经验条数。"""
        conn = self._get_read_conn()  # Task 379: 只读走 autocommit
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT COUNT(*) FROM knowledge "
                "WHERE embedding IS NULL AND superseded_by IS NULL"
            )
            return cur.fetchone()[0]
        finally:
            cur.close()

    def list_null_embedding_ids(self, limit: int = 50) -> List[tuple]:
        """列出 embedding=NULL 的条目（id, title, content）。"""
        conn = self._get_read_conn()  # Task 379: 只读走 autocommit
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT id, title, content FROM knowledge "
                "WHERE embedding IS NULL AND superseded_by IS NULL "
                "ORDER BY id LIMIT %s",
                (limit,),
            )
            return cur.fetchall()
        finally:
            cur.close()

    def reembed_knowledge(self, kid: int, embedding) -> bool:
        """更新某条 knowledge 的 embedding。返回是否更新成功。

        Args:
            kid: knowledge.id
            embedding: 1024 维向量（list 或 str 形式，会被 ::vector cast）
        """
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            cur.execute(
                "UPDATE knowledge SET embedding = %s::vector WHERE id = %s",
                (str(embedding) if not isinstance(embedding, str) else embedding, kid),
            )
            try:
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            return cur.rowcount > 0
        finally:
            try:
                cur.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """知识库统计。"""
        conn = self._get_read_conn()  # Task 379: 只读走 autocommit(清单遗漏,同根因一并修)
        cur = conn.cursor()
        stats = {}
        try:
            for table in ("shadow_codes", "knowledge", "rl_params", "rl_history",
                           "agents", "agent_activity", "session_health"):
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cur.fetchone()[0]
            return stats
        finally:
            try: cur.close()
            except Exception: pass

    # ------------------------------------------------------------------
    # task 240 #2 + #3 / task 241 修正: 风险校验层支持（不是 LLM 认知）
    # ------------------------------------------------------------------

    def search_with_confidence(
        self,
        query: str,
        top_k: int = 5,
        min_confidence: float = 0.0,
        require_verified: bool = False,
        **kwargs,
    ) -> List[dict]:
        """[DEPRECATED Task 337, 2026-07-28] 零调用方。confidence 不可信（Task 257
        证伪 σ 是 telemetry=1.0 vs llm=0.5 硬分档 artifact，非真实可信度区分）。
        保留做历史参考，不要新增调用。

        原 [SYSTEM_SCAN] 文档（task 241 修正语义，保留供历史参考）：
        带置信度过滤的搜索（task 241 修正语义）。

        ⚠️ task 241 重要修正：
            confidence 字段只在 **来源可信** 时才有意义：
                - tags 含 `verified_by:tool_verification` → 真正高置信
                - tags 含 `verified_by:cross_ref` → 中置信
                - tags 含 `verified_by:user_asserted` → 看用户权威性
                - tags 不含上述任何 + confidence≥0.7 → **可能是 LLM 自评，不可信**

        Args:
            query: 搜索文本
            top_k: 返回数
            min_confidence: 最小置信度阈值（默认 0.0 = 不过滤）
            require_verified: True = 只返回有 verified_by:* 标签的结果（task 241 新增）
            **kwargs: 透传给 search()

        Returns:
            [{"...", "confidence_band": "high|medium|low|unknown",
              "verification": "tool|cross_ref|user|llm_self|none"}, ...]
        """
        results = self.search(query, top_k=top_k * 2, **kwargs)  # 多取一倍给过滤留余量

        # 过滤 + 标注 confidence_band + verification
        filtered = []
        for r in results:
            conf = r.get("confidence")
            tags = r.get("tags", []) or []
            # 判断 verification 来源
            if "verified_by:tool_verification" in tags:
                verification = "tool"
            elif "verified_by:cross_ref" in tags:
                verification = "cross_ref"
            elif "verified_by:user_asserted" in tags:
                verification = "user"
            elif "verified_by:llm_self_rated" in tags:
                verification = "llm_self"
            else:
                verification = "none"

            # require_verified 过滤
            if require_verified and verification in ("llm_self", "none"):
                continue

            # confidence 分档
            if conf is None:
                band = "unknown"
                conf_val = 0.5
            elif conf >= 0.7:
                band = "high"
                conf_val = conf
            elif conf >= 0.3:
                band = "medium"
                conf_val = conf
            else:
                band = "low"
                conf_val = conf

            # task 241: 如果是 LLM 自评的高置信，强制降档
            if verification in ("llm_self", "none") and band == "high":
                band = "medium"  # 不信任 LLM 自评的高置信
                r["_downgraded"] = True

            if conf_val < min_confidence:
                continue
            r["confidence_band"] = band
            r["verification"] = verification
            filtered.append(r)

        return filtered[:top_k]

    def record_uncertainty(
        self,
        question: str,
        context: str = "",
        possible_answers: Optional[List[str]] = None,
        project: Optional[str] = None,
        tags: Optional[List[str]] = None,
        source: str = "system",
    ) -> int:
        """[SYSTEM] 记录"被体系标注为不可靠的 LLM 输出模式"（task 241 修正语义）。

        ⚠️ task 241 重大语义修正：
            - 旧义（task 240 错误）：LLM 自报告"我不确定"——LLM 没有自省能力，
              它填的 question/possible_answers 都是 token 概率最高的输出，不可信
            - 新义（task 241 正确）：体系记录"LLM 输出中被验证为不可靠的模式"
              或"用户标注为不可靠的输出"

        Args 中的 question/context/possible_answers 是 **被记录的事实**，
        不是 LLM 自我评估——它们应该来自：
            - 用户标注："LLM 这里说错了" → 体系调 record_uncertainty 记录
            - 工具验证：grep/curl 发现 LLM 声称的"事实"与实际不符
            - 体系扫描：output_risk_scan 发现 LLM 关键声明未经验证

        Args:
            question: 被标注为不可靠的声明（如 "某项目用 protobuf"——经验证不符）
            context: 触发记录的上下文（如 "frida 实测发现实际是 protobuf-not-json"）
            possible_answers: 候选正确答案（如有）
            project: 关联项目
            tags: 标签
            source: 记录来源（"user_correction" / "tool_verification" / "system_scan"），
                    用于区分这是人类标注还是体系扫描

        Returns:
            新建条目 ID
        """
        content_parts = [f"❓ {question}"]
        if context:
            content_parts.append(f"\n**上下文**: {context}")
        if possible_answers:
            content_parts.append("\n**可能答案**:")
            for i, ans in enumerate(possible_answers, 1):
                content_parts.append(f"{i}. {ans}")
        content_parts.append(f"\n\n_来源: {source}_")
        content_parts.append("_状态: 未验证（待人工/工具确认）_")

        source_tags = list(tags or [])
        if source == "user_correction":
            source_tags.append("user_flagged")
        elif source == "tool_verification":
            source_tags.append("tool_flagged")
        elif source == "system_scan":
            source_tags.append("system_flagged")

        tags = tags or []
        if "uncertainty" not in tags:
            tags.append("uncertainty")
        if "ku" not in [t.lower() for t in tags]:
            tags.append("ku")

        return self.add_knowledge(
            title=f"[KU] {question[:80]}",
            content="\n".join(content_parts),
            knowledge_type="uncertainty",
            tags=source_tags,
            project=project,
            confidence=0.0,
        )

    def get_uncertainties(
        self,
        project: Optional[str] = None,
        resolved: bool = False,
        limit: int = 20,
    ) -> List[dict]:
        """task 240 #3: 获取未解决的"已知的未知"清单。

        Args:
            project: 过滤项目（None = 全部）
            resolved: True = 已解决；False = 未解决（默认）
            limit: 返回数

        Returns:
            [{"id", "title", "content", "tags", "project", "created"}, ...]
        """
        conn = self._get_read_conn()  # Task 379: 只读走 autocommit(清单遗漏,同根因一并修)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # resolved 状态由 tags 包含 'resolved' 判断
        if resolved:
            tag_filter = "AND %s = ANY(tags)"
            tag_param = "resolved"
        else:
            tag_filter = "AND NOT (%s = ANY(tags))"
            tag_param = "resolved"

        project_filter = ""
        params = []
        if project:
            project_filter = " AND project = %s"
            params.append(project)

        try:
            cur.execute(f"""
                SELECT id, title, content, tags, project, created
                FROM knowledge
                WHERE knowledge_type = 'uncertainty'
                {tag_filter}
                {project_filter}
                ORDER BY created DESC
                LIMIT %s
            """, [tag_param] + params + [limit])
            return [dict(row) for row in cur.fetchall()]
        finally:
            try: cur.close()
            except Exception: pass

    def resolve_uncertainty(
        self,
        uncertainty_id: int,
        answer: str,
        source: str = "",
    ) -> int:
        """task 240 #3: 标记一个 KU 为已解决，并把答案存为 KK。

        Args:
            uncertainty_id: record_uncertainty 返回的 ID
            answer: 验证后的答案
            source: 答案来源（如 "实测某项目 client v1.2"）

        Returns:
            新建的 KK 条目 ID
        """
        # 1. 给 uncertainty 条目加 resolved tag
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT tags FROM knowledge WHERE id = %s", (uncertainty_id,))
        row = cur.fetchone()
        if not row:
            cur.close()
            raise ValueError(f"uncertainty id={uncertainty_id} 不存在")
        existing_tags = row[0] or []
        if "resolved" not in existing_tags:
            new_tags = list(existing_tags) + ["resolved"]
            cur.execute("UPDATE knowledge SET tags = %s, updated = NOW() WHERE id = %s",
                        (new_tags, uncertainty_id))
            try:
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                try:
                    cur.close()
                except Exception:
                    pass

        # 2. 把答案存为新的 KK
        return self.add_knowledge(
            title=f"[已验证] {answer[:80]}",
            content=f"**答案**: {answer}\n\n**来源**: {source or '未注明'}\n\n**对应 KU**: #{uncertainty_id}",
            knowledge_type="finding",
            tags=["verified", "kk"],
            confidence=0.9,  # 已验证的高置信
        )

    def record_tacit_knowledge(
        self,
        assumption: str,
        triggered_by: str = "",
        project: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> int:
        """task 240 #4: 捕获"未知的已知"（UK，tacit knowledge）。

        当用户纠正"你又错了因为 XXX"时，XXX 就是 tacit knowledge——
        用户觉得理所当然所以从没写下，但模型不知道。

        Args:
            assumption: 被揭示的隐性假设（如 "某项目服务端用 nodejs 不是 java"）
            triggered_by: 触发场景（如 "用户纠正: 你说 NPK 解密用 Java 但其实是 C"）
            project: 关联项目
            tags: 标签

        Returns:
            新建条目 ID
        """
        tags = tags or []
        for t in ("uk", "tacit"):
            if t not in [x.lower() for x in tags]:
                tags.append(t)

        content = f"**隐性假设**: {assumption}"
        if triggered_by:
            content += f"\n\n**揭示来源**: {triggered_by}"
        content += "\n\n_类型: 用户视为理所当然但未写下的知识_"

        return self.add_knowledge(
            title=f"[UK] {assumption[:80]}",
            content=content,
            knowledge_type="tacit",
            tags=tags,
            project=project,
            confidence=0.8,  # 用户揭示的，可信度高
        )

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
