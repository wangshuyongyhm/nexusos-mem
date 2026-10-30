#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NEXUS-MEM Web API — 公共知识库网关 (127.0.0.1:11537)

让所有 Agent（Hermes/zcode/oc/omp/kimi/ac/ds 等）用 curl 就能读写共享知识库，
无需关心 Python 版本/pyenv/依赖/SHADOW 编码/embedding。

端点:
    GET  /health                              → 服务健康
    GET  /stats                               → 知识库统计
    GET  /search?q=...&top_k=5&offset=0&project=...&type=...
    POST /knowledge   {                       → 写入（自动 SHADOW 补码 + embedding）
        "agent_id": "调用者标识",
        "title": "标题",
        "content": "正文",
        "knowledge_type": "experience|finding|memory|...",
        "tags": ["..."],
        "project": "...",
        "path": "...",          ← 可选，影响 SHADOW 路径推断
        "confidence": 1.0,      ← 可选，默认 1.0
        "verification_source": "user_asserted|tool_verification|..."  ← 可选
    }
    返回 {"id": N, "shadow_code": "XX", "deduped": bool}

鉴权（可选，默认关闭）:
    环境变量 NEXUS_API_TOKEN 设置后，请求需带 Header `X-API-Token: <token>`
    不设置 = 不鉴权（仅 127.0.0.1 本机访问，安全）

常驻:
    launchd plist: ~/Library/LaunchAgents/ai.nexus.api.plist
    参照 ai.nexus.embed.plist 结构

作者: ZCode Agent (agent_id=zcode)
日期: 2026-07-13 (公共化升级)
"""

import faulthandler
import json
import os
import re
import sys
import time
import threading
from http.server import ThreadingHTTPServer as HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# Task1948（池#235）runbook 7.3：凭据模式拦截——POST /knowledge 携带明文凭据特征的写入
# 400 拒收（不落 PG/FTS/向量索引）+ 审计日志留痕。模式清单=任务书 9 类；值长度阈值
# 压误报（文档引用字段名/占位串如 change_me 不拦）。
CREDENTIAL_PATTERNS = [
    ("sk_key",      re.compile(r"sk-[A-Za-z0-9]{16,}")),
    ("password_eq", re.compile(r"(?i)\b(?:PASSWORD|PASSWD)\s*=\s*\"?[A-Za-z0-9/+=_-]{12,}")),
    ("token_eq",    re.compile(r"(?i)\bTOKEN\s*=\s*\"?[A-Za-z0-9/+=_-]{12,}")),
    ("aws_akia",    re.compile(r"AKIA[0-9A-Z]{16}")),
    ("github_ghp",  re.compile(r"ghp_[A-Za-z0-9]{20,}")),
    ("google_aiza", re.compile(r"AIza[0-9A-Za-z_-]{20,}")),
    ("slack_xoxb",  re.compile(r"xoxb-[0-9A-Za-z-]{10,}")),
    ("jwt_eyJ",     re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
]


def _credential_kinds(*texts: str) -> list:
    kinds = []
    for kind, pat in CREDENTIAL_PATTERNS:
        for t in texts:
            if t and pat.search(t):
                kinds.append(kind)
                break
    return kinds


def _audit_credential_block(agent_id: str, kinds: list, title_len: int, content_len: int) -> None:
    """审计只记模式名与长度，不落命中文本本身（防审计文件二次泄露）。"""
    audit_dir = os.environ.get("NEXUS_MEM_AUDIT_DIR") or "/tmp/nexus-mem-audit"
    try:
        os.makedirs(audit_dir, exist_ok=True)
        with open(os.path.join(audit_dir, "credential-blocks.log"), "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "agent_id": agent_id, "patterns": kinds,
                "title_len": title_len, "content_len": content_len,
            }, ensure_ascii=False) + "\n")
    except Exception:
        pass  # silent: 审计尽力而为，不阻断 400 拒收响应

# 脚本自举: python3 api_server.py 直跑时 nexusos_mem 包未安装, 需把 src/ 自身加进
# path 才能解析下面的 from nexusos_mem.shared——pip 安装环境下此 insert 为无害冗余(1636 活①评估保留)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from nexusos_mem.shared import SharedKnowledge, apply_session_timeouts

# 可选鉴权 token（不设置则不校验）
_API_TOKEN = os.environ.get("NEXUS_API_TOKEN", "")

# Task488: ThreadingHTTPServer 多线程——每线程独立 SharedKnowledge 实例（独立 PG 连接）
# 避免 _global_kb 单例的单一 PG 连接被多线程竞争（cursor/事务冲突）。
# _global_kb_lock 废弃但保留（threading.local 天然线程隔离）。
_thread_local = threading.local()

# 1822 hang 修复: 可观测计量（/health 暴露）+ handler 级 socket 超时。
# timeout=65: 半开客户端连接上 readline/recv 最多阻塞 65s 即线程释放
# （线程本地 PG 连接随之 GC）——堵住"死客户端线程永不退出→连接泄漏累积"。
_REQ_STATE = {"active": 0, "total": 0, "errors": 0}
_START_TS = time.time()

# 1822: 重请求并发闸（search/knowledge）——快败 503 替代排队等死。
# 上限 8：压测实证 ThreadingHTTPServer 下搜索均 1-4s、并发 20 即饱和；
# 闸内并发保 PG/嵌入侧不雪崩，闸外快败由客户端退避吸收。
_HEAVY_SEM = threading.BoundedSemaphore(8)
try:
    import psycopg2.errors as psycopg2_errors
except Exception:  # 无 psycopg2 环境（不应发生）时用哑类保 QueryCanceled 分支永不命中
    class psycopg2_errors:  # type: ignore
        class QueryCanceled(Exception):
            pass


class _TrackRequest:
    """请求级计量：active 并发数 / total 累计 / errors 异常计数。"""
    def __enter__(self):
        _REQ_STATE["active"] += 1
        _REQ_STATE["total"] += 1
        return self
    def __exit__(self, exc_type, *_):
        _REQ_STATE["active"] -= 1
        if exc_type is not None:
            _REQ_STATE["errors"] += 1
        return False


def _health_payload():
    return {
        "status": "ok",
        "pid": os.getpid(),
        "uptime_s": round(time.time() - _START_TS, 1),
        "threads": threading.active_count(),
        "active_requests": _REQ_STATE["active"],
        "total_requests": _REQ_STATE["total"],
        "error_requests": _REQ_STATE["errors"],
    }


def _get_kb(agent_id: str) -> SharedKnowledge:
    """返回当前线程独立的 SharedKnowledge 实例。

    ThreadingHTTPServer 每请求一线程，每线程一个 SharedKnowledge → 一个 PG 连接。
    PG max_connections=100，本机 API 并发量远低于此。
    """
    if not hasattr(_thread_local, 'kb'):
        _thread_local.kb = SharedKnowledge(agent_id=agent_id or "api")
    return _thread_local.kb


def _pg_connect():
    """1822: PG 直连统一入口——get_dsn(connect_timeout=5) + 会话级三超时。
    原 /stats /knowledge/{id} 等端点各自 psycopg2.connect(get_dsn())，绕过
    shared._get_conn 的超时保护（悬挂点），收口单一真源。"""
    import psycopg2
    from nexusos_mem.nexus_mem_utils import get_dsn
    conn = psycopg2.connect(get_dsn())
    apply_session_timeouts(conn)
    return conn


def _needs_auth(handler):
    """检查鉴权。token 未配置则放行。"""
    if not _API_TOKEN:
        return True
    return handler.headers.get("X-API-Token", "") == _API_TOKEN


class APIHandler(BaseHTTPRequestHandler):
    timeout = 65  # 1822: 死客户端连接线程 65s 释放

    def handle_one_request(self):
        # 1822: 请求级计量（active/total/errors → /health），零侵入业务 do_* 方法
        with _TrackRequest():
            super().handle_one_request()

    def _json(self, code, data):
        body = json.dumps(data, ensure_ascii=False, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        if parsed.path == "/health":
            self._json(200, _health_payload())
            return
        if parsed.path == "/stats":
            # Task 298 P0-4: 走 get_dsn() helper（显式 user=wang）
            conn = None
            try:
                conn = _pg_connect()
                conn.autocommit = True  # Task 379: 只读 SELECT 走 autocommit,防 idle in transaction
                cur = conn.cursor()
                cur.execute(
                    "SELECT count(*), count(embedding), count(shadow_code) FROM knowledge"
                )
                total, embedded, shadow = cur.fetchone()
                cur.close()
                self._json(
                    200,
                    {
                        "total": total,
                        "embedded": embedded,
                        "shadow_covered": shadow,
                    },
                )
            except Exception as e:
                import sys
                print(f"[api] /stats 失败: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                self._json(503, {"error": f"统计服务暂时不可用: {type(e).__name__}"})
            finally:
                # LOOP-3修复(2026-07-30): 原版 conn.close() 在成功路径末尾,
                # execute/fetchone 异常时 conn 不 close 致泄漏。改 finally 保证释放。
                if conn:
                    try: conn.close()
                    except Exception: pass
            return
        # 1401-w2：GET /knowledge/{id} 单条端点——终结 8 次"知道编号取不到
        # 详情只能 search 碰运气"踩档（MEM#72182）。只读 anonymous，与
        # /search 同口径（无需凭据——读不影响认知诚实闭环的 annotate 面）。
        if parsed.path.startswith("/knowledge/"):
            kid_str = parsed.path[len("/knowledge/"):].strip("/")
            # 1522 A 项：GET /knowledge/{id}/relations — 单条边表查询（双向）
            if kid_str.endswith("/relations"):
                kid_str = kid_str[: -len("/relations")]
                try:
                    kid = int(kid_str)
                except ValueError:
                    self._json(400, {"error": f"invalid knowledge id: {kid_str}"})
                    return
                conn = None
                try:
                    conn = _pg_connect()
                    conn.autocommit = True
                    from nexusos_mem import relation as _relation
                    edges = _relation.one_hop(conn, kid, direction="both")
                    self._json(200, {"id": kid, "count": len(edges), "relations": edges})
                except Exception as e:
                    import sys
                    print(f"[api] /knowledge/{kid}/relations 失败: {type(e).__name__}: {e}",
                          file=sys.stderr, flush=True)
                    self._json(500, {"error": f"读取失败: {type(e).__name__}"})
                finally:
                    if conn:
                        try: conn.close()
                        except Exception: pass
                return
            try:
                kid = int(kid_str)
            except ValueError:
                self._json(400, {"error": f"invalid knowledge id: {kid_str}"})
                return
            conn = None
            try:
                conn = _pg_connect()
                conn.autocommit = True
                cur = conn.cursor()
                cur.execute(
                    "SELECT id, source_agent, knowledge_type, title, content, "
                    "tags, project, deprecated, superseded_by, "
                    "calibration_weight, created, updated "
                    "FROM knowledge WHERE id = %s AND deprecated = FALSE",
                    (kid,),
                )
                row = cur.fetchone()
                cur.close()
                if row is None:
                    self._json(404, {"error": f"knowledge not found: {kid}"})
                    return
                cols = ["id", "source_agent", "knowledge_type", "title", "content",
                        "tags", "project", "deprecated", "superseded_by",
                        "calibration_weight", "created", "updated"]
                self._json(200, dict(zip(cols, row)))
            except Exception as e:
                import sys
                print(f"[api] /knowledge/{kid} 失败: {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                self._json(503, {"error": f"读取失败: {type(e).__name__}"})
            finally:
                if conn:
                    try: conn.close()
                    except Exception: pass  # silent: close 幂等
            return
        if parsed.path == "/search":
            q = params.get("q", [""])[0]
            if not q:
                self._json(400, {"error": "missing 'q' parameter"})
                return
            # Task 280 P1-16: 参数加 try/except，非数字返 400（原抛 ValueError 返 500）
            try:
                top_k = int(params.get("top_k", ["5"])[0])
                offset = int(params.get("offset", ["0"])[0])
            except (ValueError, TypeError):
                self._json(400, {"error": "top_k/offset 必须是整数"})
                return
            # C2 修复(2026-07-29): top_k 无上限致 DoS(top_k=999999→recall=2999997 全表扫描阻塞)
            # clamp 到合理范围: top_k 1-50, offset 0-10000
            top_k = max(1, min(top_k, 50))
            offset = max(0, min(offset, 10000))
            project = params.get("project", [None])[0]
            ktype = params.get("type", [None])[0]
            # 2026-07-30 会话07301929: /search 加 tag 参数透传（底层 search.py 已支持
            # tag 精确过滤 %s = ANY(tags)，HTTP 端点此前未暴露）。版本基线检测等
            # "按元数据查"的场景需此能力，避免语义相关文档挤掉目标条目。
            tag = params.get("tag", [None])[0]
            agent_id = params.get("agent_id", ["anonymous"])[0]
            # Task579: 凭据门禁只对显式传 agent_id 的调用生效（推断值只用于 annotate 归因）。
            # 根因: Task409 从 /tmp 上下文文件推断 agent_id 后，Task405 的 fail-closed 检查
            # 把匿名调用（如 check_agent_versions.sh，无凭据可用）误判为非 anonymous → 403，
            # 违反 B12 文档契约「anonymous 检索豁免 200（人类/兜底通道）」。任何 Agent 在跑
            # 任务时该通道即断——I6「版本不一致」系统性误报的根源。
            _agent_id_explicit = "agent_id" in params
            # Task409: anonymous 时从 nexus_current_task 文件名推断 agent_id
            # 原因: Agent curl search 不带 agent_id → anonymous → annotate=False → 闭环断
            # 修复: 从 nexus_current_task_{agent}.txt 文件名提取 agent（dispatch 时写）
            if agent_id == "anonymous":
                try:
                    from pathlib import Path
                    import time as _time_409
                    _ctx_files_409 = sorted(
                        Path("/tmp").glob("nexus_current_task_*.txt"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    for _cf_409 in _ctx_files_409:
                        if _time_409.time() - _cf_409.stat().st_mtime < 300:
                            agent_id = _cf_409.stem.replace("nexus_current_task_", "")
                            break
                except Exception:
                    pass
            # Task 322 试运行修复：HTTP /search 加 task_id 参数（小弟走 curl 必经此路）
            # 原 Phase 0 只改了 shared.py Python API，漏了 HTTP 端点
            task_id = params.get("task_id", [None])[0]
            # Task 322 fallback: task_id 未显式传时，读上下文文件自动关联（小弟无感）
            if task_id is None:
                from nexusos_mem.shared import _read_current_task_id
                if agent_id != "anonymous":
                    task_id = _read_current_task_id(agent_id)
                else:
                    # Task 326 试运行修复：anonymous 时扫所有上下文文件找最近修改的
                    # 同一时刻通常只有一个 task 在跑，取 mtime 最新的
                    try:
                        from pathlib import Path
                        ctx_files = sorted(
                            Path("/tmp").glob("nexus_current_task_*.txt"),
                            key=lambda p: p.stat().st_mtime,
                            reverse=True,
                        )
                        # 只取 5 分钟内修改的（防旧 task 干扰）
                        # Task1538 修复：此处原 `import time`（函数内二次导入）
                        # 令 time 成为 do_GET 局部名——重试路径 L393 的 time.sleep
                        # 在本分支未执行时 UnboundLocalError（潜伏 bug，readmodel
                        # 路径首次触发）。模块级已 import time，直接用。
                        for cf in ctx_files:
                            if time.time() - cf.stat().st_mtime < 300:
                                task_id = cf.read_text(encoding="utf-8").strip() or None
                                break
                    except Exception:
                        pass
            # 2026-07-30 D3-3: /search 加 annotate 参数（透传给 kb.search 记录套用尝试）
            # 1401-d5a T2: /search 加 pool 参数（experience|document|all 默认混排加权）
            pool = params.get("pool", ["all"])[0]
            if pool not in ("experience", "document", "all"):
                self._json(400, {"error": "pool 必须是 experience|document|all", "got": pool})
                return
            # 1401-d5a T1: explain 参数（默认 true 契约字段；false 时序列化层剔除省 token）
            explain_param = params.get("explain", ["true"])[0].lower() != "false"
            # Task387 Part D: Agent 主动检索(agent_id != anonymous)默认开 annotate,
            # 记录套用尝试供认知闭环。显式 ?annotate=false 可关闭。
            # Task405: search 凭据 fail-closed 收紧（P2 fail-open → P1 fail-closed）
            # 设计：anonymous 豁免放行（设计的一部分非后门）；非anonymous无search_cred→403
            # feature flag: 环境变量 NEXUS_SEARCH_FAIL_CLOSED=1 启用（默认启用），=0 退回 fail-open
            #
            # Task316 G3 凭据 fail 语义成文（双层策略，详见 task-316 报告 §fail）：
            #   ① 授权面（本段，search_cred）：fail-CLOSED——凭据无效即 403，无降级。
            #      理由：泄露代价 >> 可用性损失；签发链自动刷新（register/dispatch），
            #      closed 不会造成持久不可用。仅 verify 内部异常时 fail-open（见下），
            #      防 credential.py 自身故障致全瘫痪——这是「组件故障」不是「凭据无效」。
            #   ② 增值功能面（如 OCRX /analyze 的 VLM key）：fail-OPEN 大声——缺 key
            #      返回明确 error 字段 + 日志，其余端点不受影响；静默吞是第三态，禁止。
            #   ③ 启动面：永远 fail-OPEN——服务必须能起来，缺凭据的功能在使用时暴露
            #      （对齐 Harness：MISSING_CREDENTIAL 在请求级，不在加载级）。
            # Task530 A2: 删除 L199 的 agent_id 重复读取——它在 Task409 推断逻辑
            # （L148-166 从 /tmp 文件名提取）之后执行，把推断结果覆盖回 "anonymous"
            # → annotate 恒 False，闭环断。agent_id 以上方推断后的值为准。
            _search_fail_closed = os.environ.get("NEXUS_SEARCH_FAIL_CLOSED", "1") == "1"
            if _search_fail_closed and _agent_id_explicit and agent_id != "anonymous":
                # 非 anonymous Agent 必须持有 search_cred（通过 X-Credential header 传）
                _search_cred = self.headers.get("X-Credential", "")
                if _search_cred:
                    try:
                        from nexusos_xdr.credential import verify_credential
                        _valid, _detail = verify_credential(_search_cred, "search_cred")
                    except Exception as _e:
                        import sys as _sys_sc
                        print(f"[Task405] search credential verify 异常(fail-open放行): {_e}", file=_sys_sc.stderr, flush=True)
                        _valid = True  # verify 异常时 fail-open（避免 credential.py 故障致全瘫痪）
                else:
                    _valid = False
                if not _valid:
                    self._json(403, {"error": "search凭据不完整", "detail": "非anonymous检索需X-Credential header（search_cred）"})
                    return
                # Task563: 凭据反查兜底 task_id。task_id 解析优先级：
                #   显式 ?task_id 参数 > /tmp 上下文文件 > 凭据绑定的 task_id（本层）
                # 修断场景：主控/小弟任务间隙（/tmp 文件缺失或为旧任务）——search_cred
                # 由 dispatch 随任务签发（Task563 起 task_id 绑定 credentials 列），
                # 凭据即任务契约，反查即得可靠归因。旧凭据（无 task_id 列值）兜底
                # 返回 None，行为与修复前一致。
                if task_id is None and _search_cred:
                    try:
                        _c563 = _pg_connect()
                        _cur563 = _c563.cursor()
                        _cur563.execute(
                            "SELECT task_id FROM credentials WHERE credential_id = %s AND task_id IS NOT NULL",
                            (_search_cred,),
                        )
                        _row563 = _cur563.fetchone()
                        _cur563.close()
                        _c563.close()
                        if _row563 and _row563[0]:
                            task_id = _row563[0]
                    except Exception as _e563:
                        print(f"[Task563] 凭据反查 task_id 失败（非致命，跳过兜底）: {_e563}", file=sys.stderr, flush=True)
            elif agent_id == "anonymous":
                import sys as _sys_p2
                print(f"[Task405] search anonymous 豁免放行（设计的一部分）", file=_sys_p2.stderr, flush=True)
            elif not _search_fail_closed:
                import sys as _sys_p2
                print(f"[Task405] search fail-open 模式（NEXUS_SEARCH_FAIL_CLOSED=0）放行", file=_sys_p2.stderr, flush=True)
            annotate_param = params.get("annotate", ["auto"])[0]
            # task 1455: /search 加 graph_expand/graph_mode 透传（关系边表一跳扩展）。
            # 1522 C 项: 默认 false→true（检索 riding——#78554 断点③接通：
            # 日常检索自动带 supersedes 新版指引 + relates 关联条目）。
            # 向后兼容: 显式 expand=false/0/no 关闭（1455 时代生产行为）。
            # 注: true 同时放宽 superseded 过滤（search.py L459）——旧版重新
            # 可召回但带 superseded_by_new 指引，这正是 riding 的设计意图。
            graph_expand = params.get("graph_expand", ["true"])[0].lower() in ("true", "1", "yes")
            graph_mode = params.get("graph_mode", ["annotate"])[0]
            readmodel = params.get("readmodel", ["false"])[0].lower() in ("true", "1", "yes")  # Task1538 M3 opt-in
            # Task1537: semantic 边 riding opt-in（默认关=生产行为不变；
            # 增值层试点期显式 include_semantic=1 才扩，见 search.py riding 段）
            include_semantic = params.get("include_semantic", ["false"])[0].lower() in ("true", "1", "yes")
            if annotate_param.lower() in ("false", "0", "no"):
                annotate = False
            elif annotate_param.lower() in ("true", "1", "yes"):
                annotate = True
            else:
                # auto(默认): Agent 检索时开, anonymous 检索时关
                annotate = agent_id != "anonymous"
            kb = _get_kb(agent_id)
            # [Task 376 P0-2] kb.search() 走 _get_conn()→psycopg2.connect，
            # PG 断连/索引损坏时抛 OperationalError，冒泡到 BaseHTTPRequestHandler
            # 不回响应，curl 收到 "Empty reply from server" 调用方无感知系统挂了。
            # 现显式捕获，返 503 + 明确 error。
            # Task487: 并发重试——embed 单 worker 并发排队时指数退避重试（0.5s/1s/2s）
            # 1822: ①重请求并发闸——超限 503 busy 快败（排队等死=客户端感知 hang；
            #   快败+退避才是诚实语义）；②QueryCanceled(语句超时)不再重试——超时
            #   本身就是过载信号，重试是负载放大器（压测实锤 3×30s 死亡螺旋）。
            if not _HEAVY_SEM.acquire(blocking=False):
                self._json(503, {"error": "busy: 重请求并发已满,请退避重试", "hint": "retry-after 1s"})
                return
            results = None
            try:
                for _attempt in range(3):
                    try:
                        results = kb.search(
                            q, top_k=top_k, offset=offset, project=project, knowledge_type=ktype,
                            tag=tag,
                            task_id=task_id,
                            annotate=annotate,
                            agent_id=agent_id,
                            pool=pool,
                            graph_expand=graph_expand,
                            graph_mode=graph_mode,
                            readmodel=readmodel,  # Task1538: 读模型视角（默认 false 走原路径）
                            include_semantic=include_semantic,
                        )
                        break
                    except psycopg2_errors.QueryCanceled:
                        import sys
                        print(f"[1822] /search 语句超时(过载),不重试快败: {q[:40]}", file=sys.stderr, flush=True)
                        self._json(503, {"error": "busy: 检索语句超时(过载保护),请退避重试"})
                        return
                    except Exception as e:
                        if _attempt < 2:
                            # 1822/C2: handler 线程禁 sleep 退避（占死线程=worker 耗尽帮凶），立即重试；退避责任移交客户端 Retry-After
                            import sys
                            print(f"[Task487] /search 第{_attempt+1}次失败，立即重试(无线程退避): {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                        else:
                            import sys
                            print(f"[Task487] /search 3次重试全失败: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                            self._json(503, {"error": f"检索服务繁忙(3次重试全失败): {type(e).__name__}", "query": q})
                            return
            finally:
                _HEAVY_SEM.release()
            if results is None:
                self._json(503, {"error": "检索服务返回空(未知原因)", "query": q})
                return
            # Task 333 Layer 2: freshness 标注
            # 注意：search.py SELECT 未取 content_hash 列，r.get("content_hash") 恒为空，
            # 故按 id 批量查 PG 拿真实 hash 对比（原任务书代码此点有 bug，会永远报 stale）
            _doc_rs = [r for r in results if r.get("path") and "#c" in r["path"]]
            if _doc_rs:
                from pathlib import Path as _P
                from nexusos_mem.embed_corpus import _chunk_text, _content_hash
                from nexusos_mem._vault_env import VAULT_ROOT as _VR
                _vault = str(_VR)
                _hash_map = {}
                _conn = None
                try:
                    _conn = _pg_connect()
                    _conn.autocommit = True  # Task 379: 只读走 autocommit,防 idle in transaction
                    _cur = _conn.cursor()
                    _ids = [r["id"] for r in _doc_rs if r.get("id")]
                    _cur.execute("SELECT id, content_hash FROM knowledge WHERE id = ANY(%s)", (_ids,))
                    _hash_map = {row[0]: row[1] for row in _cur.fetchall()}
                    _cur.close()
                except Exception:
                    pass  # PG 不可用时降级：freshness 全部 unknown
                finally:
                    # LOOP-3修复(2026-07-30): 原版 conn.close() 在 try 成功路径末尾,
                    # execute/fetchall 异常时 conn 不 close 致泄漏。改 finally 保证释放。
                    if _conn:
                        try: _conn.close()
                        except Exception: pass
                for r in _doc_rs:
                    _path = r["path"]
                    _file_path = _path.split("#c")[0]
                    _abs = _P(_vault) / _file_path
                    if not _abs.exists():
                        r["freshness"] = "orphan"
                        continue
                    try:
                        _content = _abs.read_text(encoding="utf-8", errors="replace")[:20000]
                        _chunks = _chunk_text(_content)
                        _cidx = int(_path.split("#c")[1])
                        if _cidx < len(_chunks):
                            _live = _content_hash(_chunks[_cidx])
                            _pg = _hash_map.get(r.get("id"), "")
                            r["freshness"] = "fresh" if _live == _pg else "stale"
                        else:
                            r["freshness"] = "stale"
                    except Exception:
                        r["freshness"] = "unknown"
            for r in results:
                if "freshness" not in r:
                    r["freshness"] = "n/a"
            # 1401-d5a T1 契约化：顶层 score = final_score（原序列化层从未输出，
            # 调用方只见黑箱排序——mem-stats §5 实测 score 恒 0/缺失）。
            # explain 量纲说明：vec∈[0,1]、fts 无量纲（ts_rank/ILIKE 混合）、
            # type_prior/decay 乘法、quality/outcome/ref additive——final 仅保证
            # 同查询内排序可比，不保证跨查询族绝对含义（R3：阈值筛参考用）。
            for r in results:
                r["score"] = r.get("final_score", r.get("rank", 0.0))
                if not explain_param:
                    r.pop("explain", None)
            # Task1543 crucible 埋点（1537 方案落地）：related 含 semantic 边邻居 →
            # 记 sem_impression（action 列 varchar(16) 上限——1537 原案名
            # semantic_riding_impression 27 字符超长 INSERT 静默失败，实测改短名；
            # target_id=被曝光 exp_id + detail.via_exp_id=锚点条目）。
            # 频率护栏：同 (agent, exp) 每日 1 条（防压测/批量检索刷量）。
            # fail-open：埋点失败不影响检索返回（条级失败记 stderr 不纯吞）。
            try:
                _sem_pairs = []  # (anchor_id, neighbor_id)
                for r in results:
                    for rel in (r.get("related") or []):
                        if rel.get("rel_type") == "semantic" and rel.get("id"):
                            _sem_pairs.append((r.get("id"), rel["id"]))
                if _sem_pairs and agent_id != "anonymous":
                    _conn_imp = kb._get_conn()
                    _cur_imp = _conn_imp.cursor()
                    _inserted_imp = 0
                    for _anchor, _nb in _sem_pairs[:20]:  # 单次上限 20 邻居（防超长爆量）
                        try:
                            _cur_imp.execute(
                                "SELECT 1 FROM agent_activity "
                                "WHERE agent_id=%s AND action='sem_impression' "
                                "AND target_id=%s AND created::date=CURRENT_DATE LIMIT 1",
                                (agent_id, _nb),
                            )
                            if _cur_imp.fetchone():
                                continue  # 护栏：当日已记，跳过
                            _cur_imp.execute(
                                "INSERT INTO agent_activity (agent_id, action, target_type, target_id, detail) "
                                "VALUES (%s, 'sem_impression', 'knowledge', %s, %s)",
                                (agent_id, _nb, json.dumps({
                                    "via_exp_id": _anchor,
                                    "query": q[:200],
                                    "source": "api:/search semantic_riding",
                                })),
                            )
                            _inserted_imp += 1
                        except Exception as _e_row:
                            # 单条失败不阻断其余（fail-open 到条级），但记 stderr——
                            # 纯吞会让列约束类错误静默丢数据（本单 varchar(16) 实锤）
                            print(f"[Task1543] impression 单条失败: {type(_e_row).__name__}: {_e_row}", file=sys.stderr, flush=True)
                    if _inserted_imp:
                        _conn_imp.commit()
                        print(f"[Task1543] semantic_riding_impression: {_inserted_imp} 条（agent={agent_id}, query={q[:40]}）", file=sys.stderr, flush=True)
                    _cur_imp.close()
            except Exception as _e_imp:
                print(f"[Task1543] impression 埋点失败（fail-open 不影响检索）: {_e_imp}", file=sys.stderr, flush=True)
            self._json(200, {
                "query": q, "count": len(results), "results": results,
                "explain_scale_note": (
                    "vec∈[0,1]; fts unbounded-then-clamped; final for ranking "
                    "within same query only; low_confidence flag < "
                    + os.environ.get("NEXUS_LOW_CONFIDENCE_THRESHOLD", "0.30")
                ) if explain_param else None,
            })
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if not _needs_auth(self):
            self._json(401, {"error": "invalid or missing X-API-Token"})
            return
        parsed = urlparse(self.path)
        if parsed.path == "/apply_outcome":
            # Task387 Part A: 认知闭环 — Agent 套用经验后回写真实结果
            # Task1401-W3: 双口径兼容——apply_attempt_id（原生）或 knowledge_id
            # （文档口径，自动解析最新 pending attempt；outcome 别名 result）。
            # 三口径断裂实证：旧文档教 knowledge_id+outcome，API 只认
            # apply_attempt_id+exp_id+result → 照文档做必报错（1401 征评 5 连发）。
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                self._json(400, {"error": f"invalid request: {e}"})
                return
            # 字段别名（文档→API）：knowledge_id≈exp_id，outcome≈result
            exp_id = body.get("exp_id", body.get("knowledge_id"))
            result = body.get("result", body.get("outcome"))
            agent_id = body.get("agent_id", "anonymous")  # kb 实例键；Task1528: 漏传时归因走 attempt 行回填（池 #204）
            if exp_id is None or result is None:
                self._json(400, {"error": "missing required field: exp_id(或 knowledge_id)/result(或 outcome)"})
                return
            result = str(result).lower().strip()
            if result not in ("success", "partial", "failed", "not_applicable"):
                self._json(400, {"error": "result must be success/partial/failed/not_applicable"})
                return
            kb = _get_kb(agent_id)
            try:
                apply_attempt_id = body.get("apply_attempt_id")
                if apply_attempt_id is None:
                    # knowledge_id 口径：解析该经验最新未回填 attempt
                    r = kb.resolve_pending_attempt(int(exp_id), agent_id if agent_id != "anonymous" else None)
                    if not r.get("ok"):
                        self._json(r.get("status", 400), {"error": r.get("error", "resolve failed")})
                        return
                    apply_attempt_id = r["apply_attempt_id"]
                ret = kb.record_apply_outcome(
                    apply_attempt_id=int(apply_attempt_id),
                    exp_id=int(exp_id),
                    result=result,
                    evidence=body.get("evidence", ""),
                    agent_id=body.get("agent_id"),  # Task1528: 显式传参优先；None 时 record 内回填 attempt 套用者
                )
                self._json(200, ret)
            except Exception as e:
                import sys
                print(f"[api] /apply_outcome 失败: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                self._json(500, {"error": f"内部错误: {type(e).__name__}"})
            return
        # 1522 A 项：POST /knowledge/relations — 显式写边（幂等）
        # body: {from_id, to_id, relation: relates|conflicts, source?, evidence?}
        # 方向语义同 relation.py 模块头：src=from_id。citation 边的常规来源
        # 是写入时自动抽取（source=citation），本端点服务 Agent/主控显式声明。
        if parsed.path == "/knowledge/relations":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                self._json(400, {"error": f"invalid request: {e}"})
                return
            from_id = body.get("from_id")
            to_id = body.get("to_id")
            relation = body.get("relation", "relates")
            source = body.get("source", "api")
            evidence = body.get("evidence")
            if from_id is None or to_id is None:
                self._json(400, {"error": "missing required field: from_id/to_id"})
                return
            if relation not in ("relates", "conflicts"):
                # supersedes 边由 supersede 联动产生（单一真源），不接受手写
                self._json(400, {"error": "relation 必须是 relates|conflicts（superseded_by 列是 supersedes 边的唯一真源）"})
                return
            conn = None
            try:
                conn = _pg_connect()
                from nexusos_mem import relation as _relation
                with conn.cursor() as cur:
                    cur.execute("SELECT id, deprecated FROM knowledge WHERE id IN (%s,%s)",
                                (from_id, to_id))
                    rows_ = cur.fetchall()
                ok_ids = {r[0] for r in rows_ if not r[1]}
                if from_id not in ok_ids or to_id not in ok_ids:
                    self._json(400, {"error": f"端点无效或已废弃: from={from_id in ok_ids} to={to_id in ok_ids}"})
                    return
                ret = _relation.add_relation(conn, int(from_id), int(to_id), relation,
                                             source=source, evidence=evidence)
                self._json(200, ret)
            except ValueError as e:
                self._json(400, {"error": str(e)})
            except Exception as e:
                import sys
                print(f"[api] /knowledge/relations 失败: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                self._json(500, {"error": f"内部错误: {type(e).__name__}"})
            finally:
                if conn:
                    try: conn.close()
                    except Exception: pass
            return
        # 08040834: /knowledge/supersede-by-tag — nexus-monitor 告警恢复用
        # 按 tag 查 knowledge + 标 superseded_by（软废弃，不删数据）
        # 修复：原 nexus-monitor 调此端点但端点不存在，导致 57 条监控告警永不恢复
        if parsed.path == "/knowledge/supersede-by-tag":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                self._json(400, {"error": f"invalid request: {e}"})
                return
            tag = body.get("tag")
            agent_id = body.get("agent_id", "anonymous")
            new_id = body.get("new_id")  # 可选：指向新条目（不传则只标废弃不指向）
            if not tag:
                self._json(400, {"error": "missing required field: tag"})
                return
            kb = _get_kb(agent_id)
            try:
                conn = kb._get_conn()
                cur = conn.cursor()
                # 1522 B 项：RETURNING 拿受影响 id，同事务建 supersedes 边
                # （#78554 项 7 预警：首个新 supersede 不进边表即 1388 巧平衡漂移）
                if new_id:
                    cur.execute(
                        "UPDATE knowledge SET superseded_by = %s, updated = NOW() "
                        "WHERE %s = ANY(tags) AND superseded_by IS NULL AND id <> %s "
                        "RETURNING id",
                        (new_id, tag, new_id)
                    )
                    superseded_ids = [r[0] for r in cur.fetchall()]
                    if superseded_ids:
                        from nexusos_mem import relation as _relation
                        for old_id in superseded_ids:
                            # src=新版 dst=旧版（relation.py 方向约定）；
                            # 幂等由 UNIQUE 约束兜底，这里不逐条判重
                            cur.execute(
                                "INSERT INTO knowledge_relation "
                                "(src_id, dst_id, rel_type, evidence, created_by) "
                                "VALUES (%s, %s, 'supersedes', "
                                "'sync from supersede-by-tag', 'supersede-sync') "
                                "ON CONFLICT (src_id, dst_id, rel_type) DO NOTHING",
                                (new_id, old_id)
                            )
                else:
                    # 无 new_id：标为 self-superseded（废弃，不指向新条目）
                    # self-loop 无图意义，不建边（与 backfill_supersedes 口径一致）
                    cur.execute(
                        "UPDATE knowledge SET superseded_by = id, updated = NOW() "
                        "WHERE %s = ANY(tags) AND superseded_by IS NULL",
                        (tag,)
                    )
                    superseded_ids = []
                affected = len(superseded_ids) if new_id else cur.rowcount
                conn.commit()
                conn.close()
                self._json(200, {"ok": True, "superseded": affected, "tag": tag})
            except Exception as e:
                import sys
                print(f"[api] /knowledge/supersede-by-tag 失败: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                self._json(500, {"error": f"内部错误: {type(e).__name__}"})
            return
        if parsed.path != "/knowledge":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            self._json(400, {"error": f"invalid request: {e}"})
            return

        # 必填校验
        for field in ("agent_id", "title", "content"):
            if not body.get(field):
                self._json(
                    400, {"error": f"missing required field: {field}"}
                )
                return

        # H3 修复(2026-07-29): 输入长度上限防膨胀攻击(原无上限,1MB title+5MB content 实测可写)
        agent_id = body["agent_id"]
        title = body["title"]
        content = body["content"]
        # Task1948 runbook 7.3：凭据模式拦截——在任何 DB/索引写入前拒收
        hit_kinds = _credential_kinds(title, content)
        if hit_kinds:
            _audit_credential_block(agent_id, hit_kinds, len(title), len(content))
            self._json(400, {"error": "credential pattern detected", "patterns": hit_kinds})
            return
        if len(title) > 500:
            self._json(400, {"error": "title 过长(上限500字符)"})
            return
        if len(content) > 100000:
            self._json(400, {"error": "content 过长(上限100000字符)"})
            return
        tags = body.get("tags", [])
        if isinstance(tags, list) and len(tags) > 30:
            self._json(400, {"error": "tags 过多(上限30个)"})
            return

        ktype = body.get("knowledge_type", "experience")
        tags = body.get("tags", [])
        project = body.get("project")
        path = body.get("path")
        # Task 337: confidence 入参仅做记录，不影响检索排名（融合打分公式不读它），
        # Task 257 已证伪 σ 不可解，confidence 是 telemetry=1.0 vs llm=0.5 硬分档 artifact。
        # 保留入参是为向后兼容已有写入方。
        # Task 280 P1-16: confidence 加 try/except，非数字返 400（原抛 ValueError 返 500）
        try:
            confidence = float(body.get("confidence", 1.0))
        except (ValueError, TypeError):
            self._json(400, {"error": "confidence 必须是数字"})
            return
        verification_source = body.get("verification_source")
        # Task1533 链①: 显式任务目录线索（血缘边目标）；缺省时从 path 推。
        task_dir = body.get("task_dir")

        # 1822: 写路径同受重请求并发闸保护（add_knowledge 含 embedding+PG 写事务）
        if not _HEAVY_SEM.acquire(blocking=False):
            self._json(503, {"error": "busy: 重请求并发已满,请退避重试", "hint": "retry-after 1s"})
            return
        try:
            kb = _get_kb(agent_id)

            # 查写入前是否存在（用于判断是否去重命中）
            pre = kb.find_id_by_title_content(title, content)

            # 写入（add_knowledge 内部自动 embed + content_hash 去重）
            kid = kb.add_knowledge(
                title=title,
                content=content,
                knowledge_type=ktype,
                tags=tags,
                project=project,
                path=path,
                confidence=confidence,
                verification_source=verification_source,
                embed=True,
                source_agent=agent_id,  # Task 284 P0-A: 显式传 POST body 的 agent_id（_get_kb 全局实例 self.agent_id 永远是 "ac"）
                identity_producer="api_server:POST /knowledge",  # Task1524: 身份登记 producer
            )
        finally:
            _HEAVY_SEM.release()

        # SHADOW 自动补码（assign_shadow_code 保底 BZ）
        shadow_code = None
        try:
            existing = kb.get_shadow_code(kid)
            if existing:
                shadow_code = existing
            else:
                shadow_code = kb.assign_shadow_code_for_kid(
                    kid, title=title, content=content, tags=tags,
                    project=project, path=path, knowledge_type=ktype,
                )
        except Exception as e:
            # SHADOW 补码失败不阻断写入
            # Task 280 P1-17: 不把异常信息返给客户端（信息泄漏），只记日志 + shadow_code=None
            import sys
            print(f"[ERROR] api_server SHADOW 补码失败 kid={kid} path={path}: {e}", file=sys.stderr, flush=True)
            shadow_code = None

        deduped = pre is not None
        # 1522 A 项：citation 抽取——正文显式引用 MEM#xxx 自动建 relates 边。
        # 失败不阻断写入（边是增值不是主链路）；dedupe 命中（同内容已存在）
        # 不重复建边（写入时建过一次）。响应加 relations_added（向后兼容新增）。
        relations_added = 0
        if not deduped:
            try:
                from nexusos_mem import relation as _relation
                _cites = _relation.extract_citations(
                    (title or "") + "\n" + (content or ""))
                _cites = _cites[:_relation.CITATION_EDGE_CAP]
                if _cites:
                    _rel_conn = kb._get_conn()
                    try:
                        with _rel_conn.cursor() as _rc:
                            _rc.execute(
                                "SELECT id, deprecated FROM knowledge WHERE id = ANY(%s)",
                                (_cites,),
                            )
                            _ok = {r[0] for r in _rc.fetchall() if not r[1]}
                        for _cid in _cites:
                            if _cid in _ok and _cid != kid:
                                _r = _relation.add_relation(
                                    _rel_conn, kid, _cid, "relates",
                                    source="citation",
                                    evidence="citation in content at write")
                                if _r.get("inserted"):
                                    relations_added += 1
                    finally:
                        try: _rel_conn.close()
                        except Exception: pass
            except Exception as _e:
                import sys
                print(f"[1522] citation 抽取失败（不阻断写入）kid={kid}: {_e}",
                      file=sys.stderr, flush=True)
        # Task1533 链①: 血缘边——新经验 →(produced_in_task)→ task.md 代表节点。
        # 失败/无目标只 log 不阻断（血缘是增值非主链）；dedupe 命中不重复挂
        # （ON CONFLICT 幂等兜底，行为一致）。响应加 provenance_attached。
        provenance_attached = 0
        if not deduped:
            try:
                from nexusos_mem import provenance as _prov
                _tdir = _prov.resolve_task_dir(task_dir=task_dir, path=path)
                if _tdir:
                    _pr = _prov.attach_provenance(
                        kb._get_conn(), kid, f"{_tdir}/input/task.md",
                        evidence="produced_in_task",
                        source="writechain:mem")
                    if _pr.get("ok"):
                        provenance_attached = 1
                    else:
                        print(f"[1533] 血缘跳过 kid={kid} reason={_pr.get('reason')} "
                              f"dst={_pr.get('dst')}", file=sys.stderr, flush=True)
            except Exception as _pe:
                import sys
                print(f"[1533] 血缘挂边失败（不阻断写入）kid={kid}: {_pe}",
                      file=sys.stderr, flush=True)
        # [Task 376 P0-3] 降级检测：add_knowledge 内部 embedding 失败会写 NULL
        # （向量检索永远搜不到），shadow_code 分配失败写 None，但原版仍返 201，
        # 调用方以为完全成功。现查 PG 确认 embedding 状态，响应加 degraded 标记。
        # 不删旧字段，仅新增 degraded/degraded_reason（向后兼容）。
        degraded_reasons = []
        try:
            _check_conn = kb._get_read_conn()  # Task 379: 只读走 autocommit(原 _get_conn 致 idle in transaction)
            _check_cur = _check_conn.cursor()
            try:
                _check_cur.execute(
                    "SELECT embedding FROM knowledge WHERE id = %s", (kid,)
                )
                _emb_row = _check_cur.fetchone()
            finally:
                try: _check_cur.close()
                except Exception: pass
            if _emb_row and _emb_row[0] is None:
                degraded_reasons.append(
                    "embedding_missing(向量检索不可见)"
                )
        except Exception:
            # 查询失败不阻塞写入响应（降级标记是锦上添花）
            pass
        if shadow_code is None:
            degraded_reasons.append("shadow_code_missing(认知体系降级)")

        response = {
            "id": kid,
            "shadow_code": shadow_code,
            "deduped": deduped,
            "agent_id": agent_id,
            "relations_added": relations_added,
            "provenance_attached": provenance_attached,  # Task1533
        }
        if degraded_reasons:
            response["degraded"] = True
            response["degraded_reason"] = "; ".join(degraded_reasons)
        self._json(201, response)

    def log_message(self, *args):
        pass  # quiet


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 11537
    # 1828: hang 自诊断——任一线程卡 >45s 周期性 dump 全线程序列栈（含业务函数名）至持久文件
    try:
        _fh_dir = os.environ.get("NEXUS_MEM_FH_DIR") or "/tmp/nexus-mem-faulthandler"
        os.makedirs(_fh_dir, exist_ok=True)
        _fh = open(os.path.join(_fh_dir, "traceback.log"), "a", encoding="utf-8")
        faulthandler.dump_traceback_later(45, repeat=True, file=_fh)
    except Exception as _e:
        print(f"[1828] faulthandler 注入失败(不影响服务): {_e}", flush=True)
    print(f"NEXUS-MEM API Server on :{port}")
    print(f"  auth: {'enabled' if _API_TOKEN else 'disabled (localhost only)'}")
    HTTPServer(("127.0.0.1", port), APIHandler).serve_forever()
