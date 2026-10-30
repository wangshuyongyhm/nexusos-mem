#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nexus.embed_server — 常驻 BGE-M3 嵌入 worker + FSEvents 响应式增量 embed

一次性解决四件事:
    1. 版本分裂: BGE-M3 只需在本 worker(Python 3.11.6)加载;其余 6 个 Agent /
       cron 用任意 Python(含 3.14)通过 HTTP 拿向量,不再 import sentence_transformers,
       也就不会在 3.14 上挂起。
    2. 重复加载: 模型常驻内存,短命进程(cron/CLI)不再每次付 ~5s 冷启动。
    3. 接入过重: 客户端零重依赖,只发文本收向量。
    4. 响应式增量: fswatch 监听 Vault 文件变更（底层 macOS FSEvents 内核 API）,
       自动 embed 变化的文件。零轮询、事件驱动、5秒延迟（2026-07-14 新增）。

设计（稳定性优先）:
    - 只绑 127.0.0.1（不暴露公网，安全）。
    - 启动即 warmup（先 encode 一次），首个真实请求不卡。
    - encode 加进程内锁串行化，避免 MPS 并发争用导致的偶发崩溃。
    - 输入截断到 EMBED_MAX_CHARS，防超长撑爆。
    - 与 search.py / embed_corpus.py 共用同一模型单例常量，向量口径一致
      (normalize_embeddings=True, 1024 维)——保证与已入库向量可比。
    - 由 launchd KeepAlive 托管（ai.nexus.embed.plist）。
    - FSEvents 监听线程：文件变更→去抖2秒→content_hash比对→只embed变化的.md文件。
      无变更时零CPU，不依赖任何Agent/cron/fswatch。

端点:
    GET  /health           → {"status":"ok","ready":bool,"model":...,"dim":1024,"encoded":N}
    POST /embed  {"texts":[...]} | {"text":"..."}
                           → {"vectors":[[...],...],"dim":1024,"model":...}

用法:
    python3 -m nexus.embed_server                 # 默认 127.0.0.1:11436
    NEXUS_EMBED_PORT=11500 python3 -m nexus.embed_server
"""

import json
import os
import sys
import threading
import time
from pathlib import Path

from nexusos_mem.env_scrub import scrubbed_env  # Task600 池#50残留: 子进程 env 净化
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 离线，避免每次 encode 触发 HF Hub 网络探测（国内环境更稳）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# 防御：worker 进程内强制走本地模型，杜绝 encode_query 意外回连自己成环
os.environ["NEXUS_EMBED_NO_WORKER"] = "1"

DEFAULT_HOST = os.environ.get("NEXUS_EMBED_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("NEXUS_EMBED_PORT", "11436"))

_ENCODE_LOCK = threading.Lock()
_STATE = {"ready": False, "encoded": 0, "started": time.time()}


def _model():
    """复用 search.py 的模型单例，保证向量口径与入库时完全一致。"""
    from nexusos_mem.search import _get_bge_m3
    return _get_bge_m3()


def _encode(texts):
    """并发 encode，返回 list[list[float]]。

    Task488: 移除 _ENCODE_LOCK——M4 MPS 支持并发 encode，原锁是 M1/M2 时代的防崩溃保护。
    ThreadingHTTPServer + api_server ThreadingHTTPServer 让多请求并发到达，
    锁导致串行排队。移除后每个请求独立 encode（batch_size 按自身输入大小定）。

    Task582: 超长 texts 分片推理（防 OOM）——单次 HTTP 传入超大列表时，
    逐片（_BATCH_SLICE 条）喂模型，语义等价于一次全量 encode。
    """
    from nexusos_mem.search import EMBED_MAX_CHARS
    clipped = [(t or "")[:EMBED_MAX_CHARS] for t in texts]
    # Task582: 分片上限（对齐 embed_corpus 批量惯例 64×4；单请求超万条不再整载内存）
    slice_max = 256
    out = []
    for i in range(0, len(clipped), slice_max):
        chunk = clipped[i:i + slice_max]
        vecs = _model().encode(
            chunk, normalize_embeddings=True,
            batch_size=min(32, max(1, len(chunk))),
        )
        out.extend(v.tolist() for v in vecs)
    _STATE["encoded"] += len(clipped)
    return out


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # 静默默认访问日志（避免刷屏），只在错误时手动打印
        pass

    def do_GET(self):
        if self.path.startswith("/health"):
            from nexusos_mem.search import BGE_M3_NAME, BGE_M3_DIM
            self._send(200, {
                "status": "ok",
                "ready": _STATE["ready"],
                "model": BGE_M3_NAME,
                "dim": BGE_M3_DIM,
                "encoded": _STATE["encoded"],
                "uptime_sec": round(time.time() - _STATE["started"], 1),
            })
        elif self.path.startswith("/reconcile"):
            # Task 333 Layer 1d: 手动触发对账（扫近N小时.md，对比hash，re-embed stale）
            from urllib.parse import urlparse, parse_qs as _pqs
            _parsed = urlparse(self.path)
            # 修复(2026-07-29): int() 未 try,?hours=abc 会 ValueError 致连接挂死
            try:
                hours = int(_pqs(_parsed.query).get("hours", ["2"])[0])
            except (ValueError, TypeError):
                self._send(400, {"error": "hours 参数必须是整数"})
                return
            self._send(200, _run_reconcile(hours))
            return
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.startswith("/reconcile-file"):
            # Task 333 Layer 3: reconcile cron 走 HTTP 触发 re-embed
            # （BGE-M3 模型只在 embed_server 进程加载，不能直接调 _embed_single_file）
            try:
                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(length) if length else b"{}")
                self._send(200, _embed_single_file(data.get("path", "")))
            except Exception as e:
                self._send(400, {"error": f"bad request: {e}"})
            return
        if not self.path.startswith("/embed"):
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw.decode("utf-8"))
        except Exception as e:
            self._send(400, {"error": f"bad request: {e}"})
            return

        texts = data.get("texts")
        if texts is None and "text" in data:
            texts = [data["text"]]
        if not isinstance(texts, list) or not texts:
            self._send(400, {"error": "需要 texts:[...] 或 text:'...'"})
            return

        try:
            from nexusos_mem.search import BGE_M3_NAME, BGE_M3_DIM
            vectors = _encode(texts)
            self._send(200, {"vectors": vectors, "dim": BGE_M3_DIM,
                             "model": BGE_M3_NAME, "count": len(vectors)})
        except Exception as e:
            print(f"[embed_server] encode 失败: {e}", file=sys.stderr, flush=True)
            # BUG-12 修复(2026-07-29): 原版把原始异常返给客户端(信息泄漏,可能含路径/DSN)
            # 改为通用错误信息,详情只记 stderr。对齐 api_server.py Task280 P1-17 的修复方式。
            self._send(500, {"error": "encode failed (见 stderr 日志)"})


# ---------------------------------------------------------------------------
# FSEvents 响应式增量 embed（2026-07-14 新增）
# ---------------------------------------------------------------------------

from nexusos_mem._vault_env import VAULT_ROOT as _VR_SRC  # Task702: 单源（env>哨兵推导）
_VAULT_ROOT = str(_VR_SRC)
_DEBOUNCE_SEC = 0.8  # Task 333: 去抖 2.0→0.8 降延迟（content_hash 去重防高频写爆 encode）
_FSEVENTS_SKIP_DIRS = {
    ".obsidian", ".git", ".trash", ".zcode", "__pycache__",
    ".smart-env", "node_modules", "copilot", ".DS_Store",
    "99-Archive",  # 归档文件不 embed
    "hermes-scripts-backup",  # 脚本镜像不 embed
    "health-reports",  # 健康报告不 embed
    "reports",  # 运行报告不 embed
    ".hermes",  # Hermes 配置不 embed
    # Task 333: 移除 "sessions" — 01-Projects/sessions/ 是有效归档不应跳过
    # （原用 s in parts 匹配任意层级 sessions 目录，误杀 01-Projects/sessions/）
}
# Task611: 密钥真源文件永不 embed（目录级 SKIP 管不到顶层文件——文件级黑名单）
_SECRET_MD_FILES = {"SECRETS.md"}

# 待 embed 的文件集合（线程安全）
_pending_files: set = set()
_pending_lock = threading.Lock()
_fsevents_started = False
_native_fsevents_running = False  # 原生 FSEvents 是否成功启动并阻塞在 CFRunLoopRun


# Task 282 P2-7: project 推断从 config/project_rules.yaml 加载
def _load_project_rules():
    """从 config/project_rules.yaml 加载项目路由规则，失败降级默认规则（兼容旧行为）。"""
    cfg = Path(__file__).parent / "config" / "project_rules.yaml"
    if cfg.exists():
        try:
            import yaml
            return yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        except Exception:
            pass
    # 降级默认（yaml 缺失时）：通用示例规则，不含私项目名（live 环境 yaml 必在，降级仅 OSS 兜底）
    return {
        "rules": [
            {"keyword": "demo-project", "project": "demo-project"},
            {"keyword": "NEXUS", "project": "nexusos-mem"},
        ],
        "default": None,
    }

_PROJECT_RULES_CFG = _load_project_rules()


# ============================================================
# Task 564: 任务关联 + 镜像准入
# 设计原则（用户原话）：一个任务关联一个事，任务内的内容关联在一起。
#   ① 任务目录识别：task/<NNN-slug>/...（主 task/）或 01-Projects/sessions/task-<NNN-slug>/...（归档）
#   ② 准入拒收（检索价值≈0，文件系统已有原文）：input/（任务书+dispatched 副本）、
#      process/（full-log 管道记录）、coord-traces/（apply 溯源 trace——crucible 走
#      文件系统读，不依赖 knowledge 镜像）
#   ③ 命中任务目录的合规文件：task_id 列 + context_coord 任务维度 + title 带任务名
# ============================================================
import re as _re564

_TASK_DIR_RE = _re564.compile(r'(?:^|/)(task-)?(\d+-[\w.-]+?)/')
_COORD_TRACES_MARK = 'coord-traces/'

# Task 566: 通用词文档名集合——命中时分块 title 自动带路径前缀（防无名分块退化）
_GENERIC_TITLES_566 = frozenset((
    'report', 'README', 'readme', 'index', 'full-log', 'notes', 'note',
    'draft', 'untitled', '会话', '输出',
))


def _path_prefix_566(rel_path: str) -> str:
    """取 path 的 1-2 级目录前缀（目录段取末两级——信息密度最高）。

    00-Inbox/README.md                      → 00-Inbox（仅一级目录）
    02-Skills/productivity/task-manager.md  → productivity/task-manager 前的 productivity…即 02-Skills/productivity
    05-System/Agent/agent-a/README.md       → Agent/agent-a（跳过顶层编号目录，末两级定位最准）
    """
    parts = [p for p in rel_path.split('/') if p]
    dirs = parts[:-1]
    if not dirs:
        return 'vault'
    return '/'.join(dirs[-2:]) if len(dirs) >= 2 else dirs[0]


def _parse_task_slug(rel_path: str):
    """从镜像路径解析任务 slug（如 '561-api-torch-leak'）。非任务路径返回 None。

    Task 564 识别根域两种形态；Task 565 泛化到 vault 域全部任务目录模式：
      task/561-xxx/output/report.md                     → 561-xxx
      task/455/output/report.md（纯数字）                → 455
      01-Projects/sessions/task-307-agent-a-xxx/input/x.md → 307-agent-a-xxx
      05-System/task/389-standard-lib/output/a.md       → 389-standard-lib
      01-Projects/demo-project-a/task/152-xxx/output/a.md → 152-xxx
      01-Projects/demo-project-a/07-会话与归档/sessions/…（编号）  → 编号 slug
      01-Projects/demo-project-b/task/12-xxx/output/a.md → 12-xxx
      task/LOOP-全力以赴-08130832/output/a.md            → LOOP-全力以赴-08130832（LOOP 命名空间原样）
    防误挂铁律：候选段必须是**目录**（其后还有文件段）——项目 sessions 目录下的
    日期命名会话日志（2026-06-18-174216-xxx.md，数字开头）是文件不是目录，
    永不挂 task_id。
    """
    parts = rel_path.split('/')
    n = len(parts)
    for i, seg in enumerate(parts):
        if i >= n - 1:
            break  # 候选必须是目录：其后至少还有文件段
        cand = None
        if seg in ('task', 'sessions'):
            nxt = parts[i + 1]
            # 候选目录还必须是中间层（nxt 不能是最后的文件段）
            if i + 1 < n - 1 and _re564.match(r'^\d+(-|$)', nxt):
                cand = nxt
        elif seg == '07-会话与归档' and parts[i + 1] == 'sessions' and i + 2 < n - 1:
            nxt = parts[i + 2]
            if _re564.match(r'^\d+(-|$)', nxt):
                cand = nxt
        elif seg.startswith('task-') and _re564.match(r'^task-\d+-', seg):
            cand = seg[len('task-'):]
        elif seg.startswith('LOOP-'):
            cand = seg  # LOOP 命名空间：目录名原样作 task_id
        if cand:
            return cand
    return None


def _mirror_admission(rel_path: str):
    """Task 564 镜像准入判定。返回 (admit: bool, reason: str, task_slug)。

    拒收三类（增量从源头减量，拒收留日志不静默丢）：
      - coord-traces/ 下的 trace-* 文件（crucible 走文件系统，镜像无消费方）
      - 任务目录 input/ 子树（任务书、dispatched 副本——dispatch 流程已有原文）
      - 任务目录 process/ 子树（full-log 管道自动记录）
    其余保持旧行为（vault 文档镜像面不变）。

    Task1533 例外：input/task.md 本体放行——它是血缘边（provenance）的目标
    节点（新经验→produced_in_task→task.md 代表节点），且任务书原文有检索
    价值；input/ 其余（dispatched 副本/references）仍拒收，564 减量初衷不变。
    """
    if _COORD_TRACES_MARK in rel_path:
        return False, "coord-traces(apply溯源,crucible走文件系统)", None
    slug = _parse_task_slug(rel_path)
    if slug:
        after = rel_path.split(slug + '/', 1)[-1] if slug + '/' in rel_path else ''
        first_seg = after.split('/', 1)[0] if after else ''
        if first_seg == 'input':
            _rest = after.split('/', 1)[1] if '/' in after else ''
            if _rest == 'task.md':
                return True, "", slug
            return False, f"task {slug}/input(任务书+dispatched副本)", slug
        if first_seg == 'process':
            return False, f"task {slug}/process(full-log管道记录)", slug
    return True, "", slug


def _infer_project(rel_path: str):
    """按 YAML 规则推断 project（命中第一个返回，否则 default）。

    Task 282 P2-7: 替代原私项目名硬编码推断，改 yaml 规则驱动。
    """
    for rule in _PROJECT_RULES_CFG.get("rules", []):
        keyword = rule.get("keyword")
        if keyword and keyword in rel_path:
            return rule.get("project")
    return _PROJECT_RULES_CFG.get("default")


def _should_embed(rel_path: str) -> bool:
    """判断文件是否需要 embed（.md 且不在排除目录）。"""
    if not rel_path.endswith(".md"):
        return False
    parts = rel_path.replace("\\", "/").split("/")
    # Task611: 密钥真源文件级排除（防 SECRETS.md 内容进向量库）
    if parts[-1] in _SECRET_MD_FILES:
        return False
    for skip in _FSEVENTS_SKIP_DIRS:
        if skip in parts:
            return False
    # Task 564: 镜像准入（拒收任务过程文件/trace——检索价值≈0，从源头减量）
    admit, reason, _slug = _mirror_admission(rel_path)
    if not admit:
        print(f"[fsevents] 🚫 拒收 {rel_path}: {reason}", flush=True)
        return False
    return True


def _escape_like(s: str) -> str:
    """Escape PostgreSQL LIKE wildcards (_ and %), for use with ESCAPE '\\'."""
    return s.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')

def _embed_single_file(abs_path: str) -> dict:
    """embed 单个 .md 文件到 PG。

    复用 embed_corpus 的 embed 逻辑，但只处理一个文件。
    返回 {"ok": bool, "chunks": N, "error": str?}

    [task254] 修复 3 个 path 匹配 bug：
    B1: 去重 SELECT 改前缀 LIKE 匹配（原 path=rel_path 永不命中 #c 后缀行）
    B2: 删旧 DELETE 改前缀 LIKE 匹配（同上）
    B3: content_hash 统一用 embed_corpus._content_hash 分块 hash（原用全文 hash）
    """
    from pathlib import Path
    try:
        from nexusos_mem.embed_corpus import (
            _chunk_text, CHUNK_MAX_PER_DOC, DOC_READ_MAX_CHARS, _content_hash,
        )
    except ImportError:
        return {"ok": False, "error": "import failed"}

    try:
        p = Path(abs_path)
        if not p.exists() or not p.is_file():
            # 文件不存在（Removed/Renamed 事件的旧路径）：删 PG 旧 document 条目
            # 2026-07-23 补：原版只返回 error 不清理 PG，导致改名/删除后旧快照残留。
            # 现按 rel_path 前缀匹配删旧 chunks（document 类型）。
            try:
                rel_path = str(p.relative_to(_VAULT_ROOT)).replace("\\", "/")
                import psycopg2
                dsn = os.environ.get("NEXUS_PG_DSN", "postgresql://localhost/nexus_mem")
                conn = psycopg2.connect(dsn)
                cur = conn.cursor()
                try:
                    # [Task 376 P0-1] 改名/删除清理：原版只删 knowledge_type='document'
                    # 导致 experience/finding/decision/pitfall（path=纯 rel_path 无 #c 后缀）
                    # 成为孤儿永不清理。现用两条 DELETE 精确覆盖：
                    # ① path LIKE rel_path#c% → document chunks（path 格式 rel_path#c0#c1...）
                    # ② path = rel_path      → 非 document 经验类（path 格式纯 rel_path）
                    # PG 实测：document 100% 含 #c，非 document 0% 含 #c，两条互不重叠。
                    prefix = rel_path + "#c"
                    cur.execute(
                        "DELETE FROM knowledge "
                        "WHERE path LIKE %s ESCAPE '\\'",
                        (_escape_like(prefix) + "%",),
                    )
                    deleted_chunks = cur.rowcount
                    cur.execute(
                        "DELETE FROM knowledge WHERE path = %s",
                        (rel_path,),
                    )
                    deleted_other = cur.rowcount
                    deleted = deleted_chunks + deleted_other
                    conn.commit()
                    if deleted > 0:
                        print(
                            f"[fsevents] 🗑️ {rel_path}: 文件已删，清理 {deleted} 条 PG 旧快照"
                            f"（document chunks={deleted_chunks}, 经验类={deleted_other}）",
                            flush=True,
                        )
                    return {"ok": True, "chunks": 0, "deleted_stale": deleted}
                finally:
                    try: cur.close()
                    except Exception: pass
                    try: conn.close()
                    except Exception: pass
            except Exception as e:
                # 清理失败不阻塞（记录即可，下次 embed 会自然覆盖）
                print(f"[fsevents] ⚠️ {abs_path} 清理旧条目失败: {e}", file=sys.stderr, flush=True)
                return {"ok": False, "error": "file gone"}

        content = p.read_text(encoding="utf-8", errors="replace")[:DOC_READ_MAX_CHARS]

        # chunk（在连 PG 前完成，以便逐块 hash 去重）
        chunks = _chunk_text(content)
        if not chunks:
            return {"ok": True, "chunks": 0, "skipped": True}
        chunks = chunks[:CHUNK_MAX_PER_DOC]

        # 连 PG
        import psycopg2
        dsn = os.environ.get("NEXUS_PG_DSN", "postgresql://localhost/nexus_mem")
        conn = psycopg2.connect(dsn)
        cur = conn.cursor()
        # Task 280 P1-8: 异常路径 conn/cur 泄漏修复——
        # 原版 L295 except 只 return error 不 close，每次 embed 失败泄漏一个 PG 连接。
        # 用 try/finally 包裹整个 DB 操作区段，确保异常时也 close。
        try:
            rel_path = str(p.relative_to(_VAULT_ROOT)).replace("\\", "/")

            # [B1修复] 读取已存 chunk 行（前缀精确匹配，ESCAPE 避免 _ 和 % 通配符误匹配）
            prefix = rel_path + "#c"
            cur.execute(
                "SELECT path, id, content_hash FROM knowledge "
                "WHERE knowledge_type='document' AND path LIKE %s ESCAPE '\\'",
                (_escape_like(prefix) + "%",),
            )
            existing = {r[0]: (r[1], r[2]) for r in cur.fetchall()}

            # [B1+B3修复] 逐块 hash 去重：所有块 hash 未变且无多余尾部块 → 跳过
            new_hashes = {f"{rel_path}#c{i}": _content_hash(chunk)
                          for i, chunk in enumerate(chunks)}
            all_unchanged = all(
                existing.get(cpath) and existing[cpath][1] == chash
                for cpath, chash in new_hashes.items()
            )
            stale_tail = [k for k in existing if k not in new_hashes]
            if all_unchanged and not stale_tail:
                return {"ok": True, "chunks": 0, "skipped": True}

            # embed（通过本进程的 _encode，复用 BGE-M3 单例）
            vectors = _encode(chunks)

            # [T671 upsert] 原 B2「DELETE 全部旧 chunks + 全量重插」= 快照失忆：重嵌
            # 冲掉行上人工字段（tags/knowledge_type——652 打标成果被管道吃回，665
            # 实证镜像 tag 26→20），且多路径副本（pypi/dist-repos/build 同内容文件）
            # 每次落盘都造新重复组（R5 清零后回积 16+ 组，现已 22 组）。改原地更新：
            #   ① path 同位旧行、hash 未变 → 不动（人工字段零触碰）
            #   ② path 同位旧行、hash 变 → UPDATE content/content_hash/embedding/title
            #      （tags/knowledge_type/project/context_coord 等其余字段保留）
            #   ③ 无旧行 → 幂等护栏 INSERT：content_hash 已有活跃 document 行则跳过
            #      （同内容副本文件不造重复组）
            #   ④ chunk 收缩的尾部旧行 → 按 id 精确清单 DELETE（WHERE 带本文件 path，
            #      不误删他文件——陷阱2）
            import psycopg2.extras

            title_base = p.stem
            # Task 282 P2-7: project 推断从 config/project_rules.yaml 加载（原私项目名硬编码）
            project = _infer_project(rel_path)

            # Task 285 P0-B: 预先计算 context_coord（同一文件所有 chunks 共享）
            # Task 564: 任务目录文件补任务维度（task_id + phase）——修 phase=null 空壳
            task_slug = _parse_task_slug(rel_path)
            file_coord_json = None
            try:
                from nexusos_mem.context_capture import capture_current_context
                import json as _json
                _extra564 = {"mirror_path": rel_path}
                if task_slug:
                    _extra564["task_id"] = task_slug
                file_coord = capture_current_context(
                    agent_name="nexus-fsevents",
                    task_type="document",
                    phase="mirror" if task_slug else None,
                    extra=_extra564,
                )
                file_coord_json = _json.dumps(file_coord, ensure_ascii=False)
            except Exception as e:
                print(f"[embed_server] context_coord 捕获失败（写 NULL）: {e}", file=sys.stderr, flush=True)

            # Task1524: producer 标注（事务级）——document 行插入触发器读它登记
            # knowledge_identity（fsevents 增量嵌入链的精确身份来源）。
            cur.execute(
                "SELECT set_config('app.producer', 'embed_server:fsevents', true)")

            _n_same = _n_updated = _n_inserted = _n_dup_skip = 0
            for i, (chunk_text_data, vec) in enumerate(zip(chunks, vectors)):
                # [B3修复] content_hash 用逐块 hash（16字符），与 embed_corpus 一致
                chunk_hash = new_hashes[f"{rel_path}#c{i}"]
                path_with_chunk = f"{rel_path}#c{i}"
                # Task 564: 分块 title 带任务名（如 task561-api-torch-leak/report p5）
                # ——修"report (part 5)"丢任务上下文；非任务文件保持原命名
                # Task 566: 非任务文件的 title 若是通用词（README/report/index…），
                # 自动带 1-2 级目录前缀（如 02-Skills/README p2）——杜绝无名分块。
                if task_slug:
                    title = (f"task{task_slug}/{title_base}" if i == 0
                             else f"task{task_slug}/{title_base} p{i+1}")
                elif title_base in _GENERIC_TITLES_566:
                    _pref = _path_prefix_566(rel_path)
                    title = (f"{_pref}/{title_base}" if i == 0
                             else f"{_pref}/{title_base} p{i+1}")
                else:
                    title = title_base if i == 0 else f"{title_base} (part {i+1})"

                if path_with_chunk in existing:
                    if existing[path_with_chunk][1] == chunk_hash:
                        # ① 内容未变 → 行原样保留（人工字段零触碰）
                        _n_same += 1
                        continue
                    # ② 内容变了 → 原地更新内容四元组，其余字段（含人工 tags）保留
                    cur.execute(
                        "UPDATE knowledge SET content=%s, content_hash=%s, "
                        "embedding=%s, title=%s, updated=NOW() "
                        "WHERE knowledge_type='document' AND path=%s",
                        (chunk_text_data, chunk_hash,
                         psycopg2.extras.Json(vec), title, path_with_chunk),
                    )
                    _n_updated += cur.rowcount
                    continue

                # ③ 幂等护栏：同 hash 活跃 document 行已存在（他路径副本）→ 跳过
                cur.execute(
                    "SELECT 1 FROM knowledge "
                    "WHERE knowledge_type='document' AND content_hash=%s "
                    "AND NOT deprecated AND superseded_by IS NULL LIMIT 1",
                    (chunk_hash,),
                )
                if cur.fetchone():
                    _n_dup_skip += 1
                    continue
                cur.execute(
                    "INSERT INTO knowledge "
                    "(path, content, content_hash, embedding, knowledge_type, "
                    " source_agent, tags, title, project, context_coord, task_id) "
                    "VALUES (%s,%s,%s,%s,'document','nexus-fsevents',%s,%s,%s,%s,%s)",
                    (path_with_chunk, chunk_text_data, chunk_hash,
                     psycopg2.extras.Json(vec), ["document"], title, project,
                     file_coord_json,  # Task 285 P0-B: context_coord
                     task_slug),       # Task 564: task_id（非任务路径 NULL）
                )
                _n_inserted += cur.rowcount

            # ④ 尾部旧行（chunk 收缩）按 id 精确清单清理——只动本文件自己的行。
            # FK 护栏（T671）：被其他行 superseded_by 引用的行跳过——它们是 R5 合并
            # 的 supersede 锚点（checkpoint-latest.md 实证：DELETE 撞
            # knowledge_superseded_by_fkey 整事务回滚）。锚点行留档符合 B1 软删语义。
            _stale_ids = [existing[k][0] for k in stale_tail]
            if _stale_ids:
                cur.execute(
                    "DELETE FROM knowledge WHERE id = ANY(%s) "
                    "AND NOT EXISTS (SELECT 1 FROM knowledge k2 "
                    "WHERE k2.superseded_by = knowledge.id)",
                    (_stale_ids,),
                )

            conn.commit()

            # Task1533 链②: persist 落盘血缘——document upsert 成功后按文件级
            # 规则挂 provenance 边（report→task.md / 会话日志→日志索引）。
            # 内容未变 early-return（skipped）不会到这里=不重复挂。目标文件
            # 未被 embed（no_doc）或规则不命中→静默跳过；异常只 log 不阻断。
            try:
                from nexusos_mem import provenance as _prov1533
                _rule1533 = _prov1533.provenance_rules_for_doc(rel_path)
                if _rule1533:
                    _dst_path1533, _ev1533 = _rule1533
                    _doc_id1533 = _prov1533.resolve_vault_doc(conn, rel_path)
                    if _doc_id1533 is not None:
                        _prov1533.attach_provenance(
                            conn, _doc_id1533, _dst_path1533,
                            evidence=_ev1533, source="writechain:persist")
            except Exception as _e1533:
                print(f"[1533] persist 血缘挂边失败（不阻断 embed）"
                      f"rel_path={rel_path}: {_e1533}", file=sys.stderr, flush=True)

            # 分配 shadow_code（best-effort，失败不阻塞）
            try:
                from nexusos_mem.shadow_match import backfill as _shadow_backfill
                _shadow_backfill(conn, dry_run=False, only_null=True)
            except Exception:
                pass  # shadow_code 由定时 backfill 补

            return {"ok": True, "chunks": len(chunks), "upsert": {
                "same": _n_same, "updated": _n_updated,
                "inserted": _n_inserted, "dup_skipped": _n_dup_skip,
                "stale_deleted": len(_stale_ids)}}
        finally:
            # Task 280 P1-8: 无论成功/异常/early return（skipped），都 close conn/cur
            try:
                cur.close()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass

    except Exception as e:
        return {"ok": False, "error": str(e)}


def _fsevents_worker():
    """文件监听：使用 fswatch（macOS FSEvents 的 CLI 封装）。

    fswatch 比 pyobjc FSEvents C-API 更稳定（回调注册简单），
    且底层同样走内核 FSEvents（零轮询、事件驱动）。
    如果 fswatch 不可用，降级为 10 秒 content_hash 轮询。
    """
    import shutil
    import subprocess
    from pathlib import Path

    vault = Path(_VAULT_ROOT)
    if not vault.exists():
        print(f"[fsevents] Vault 不存在: {vault}，监听线程退出", file=sys.stderr, flush=True)
        return

    fswatch_bin = shutil.which("fswatch")
    if fswatch_bin:
        print(f"[fsevents] 使用 fswatch 原生事件监听: {vault}", flush=True)
        _fswatch_mode(vault, fswatch_bin)
    else:
        print("[fsevents] fswatch 不可用，降级为 10 秒轮询模式", file=sys.stderr, flush=True)
        _polling_fallback(vault)


def _fswatch_mode(vault, fswatch_bin):
    """用 fswatch subprocess 监听文件变更（零轮询，内核 FSEvents 驱动）。

    fswatch 输出变更的文件路径（每行一个），我们逐行读 → 去抖 → embed。
    """
    import subprocess

    vault_str = str(vault)
    # fswatch 参数：
    # --event=Created/Updated/Removed/Renamed  监听四种事件
    #   Renamed（2026-07-23 补）：原只监听 Created/Updated/Removed，mv 改名时
    #   fswatch 把 rename 解读为 Removed，新路径不触发 Created → PG 旧条目残留 +
    #   新文件不 embed。加 Renamed 后改名能正确触发重建（配合 _embed_single_file
    #   的"文件不存在则删旧条目"逻辑）
    # --latency=1.0  FSEvents 聚合窗口（Task 333: 3.0→1.0 降延迟，和原生 API 一样）
    # --recursive    递归监听子目录
    # 最后是监听路径
    cmd = [
        fswatch_bin,
        "--event=Created",
        "--event=Updated",
        "--event=Removed",
        "--event=Renamed",
        "--latency=1.0",
        "--recursive",
        vault_str,
    ]

    print(f"[fsevents] fswatch 启动: {' '.join(cmd[:6])} ... {vault_str}", flush=True)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # 行缓冲
            env=scrubbed_env(),
        )

        global _native_fsevents_running
        _native_fsevents_running = True
        print(f"[fsevents] ✅ fswatch 监听已启动（latency=1s + debounce={_DEBOUNCE_SEC}s）", flush=True)

        _debounce_timers: list = []

        for line in proc.stdout:
            path = line.strip()
            if not path:
                continue
            if not path.endswith(".md"):
                continue
            # 转相对路径判断
            rel = path.replace(vault_str + "/", "").replace(vault_str, "")
            if not _should_embed(rel):
                continue

            with _pending_lock:
                _pending_files.add(path)

            # 去抖：取消之前的 timer，重新计时
            for t in _debounce_timers:
                t.cancel()
            _debounce_timers.clear()
            timer = threading.Timer(_DEBOUNCE_SEC, _flush_pending)
            timer.daemon = True
            _debounce_timers.append(timer)
            timer.start()

    except Exception as e:
        print(f"[fsevents] fswatch 失败: {e}，降级轮询", file=sys.stderr, flush=True)
        _native_fsevents_running = False
        _polling_fallback(vault)


def _fsevents_native(vault):
    """原生 FSEvents 模式（零轮询，事件驱动）。

    使用 pyobjc-framework-FSEvents 的 C 风格 API。
    FSEvents 自带 latency 聚合（Task 333: 设为 1.0 秒），加上去抖 0.8 秒，
    总延迟约 2 秒（远好于 4 小时轮询）。
    """
    import ctypes
    from pathlib import Path

    try:
        from FSEvents import (
            FSEventStreamCreate,
            FSEventStreamScheduleWithRunLoop,
            FSEventStreamStart,
            CFRunLoopGetCurrent,
            CFRunLoopRun,
            kFSEventStreamEventIdSinceNow,
            CFArrayCreate,
            CFStringCreateWithCString,
            kCFStringEncodingUTF8,
            kFSEventStreamCreateFlagUseCFTypes,
        )
    except ImportError as e:
        print(f"[fsevents] FSEvents API 不可用: {e}，降级轮询", file=sys.stderr, flush=True)
        _polling_fallback(vault)
        return

    vault_str = str(vault)

    # 去抖 timer 池（必须在回调引用前定义，否则 NameError）
    _debounce_timers: list = []

    # FSEvents 回调（C 函数指针形式）
    # 回调签名: (stream, ctx_ref, num_events, event_paths_ref, event_flags_ref, event_ids_ref)
    import FSEvents as _fsevents_mod

    # 定义 Python 回调
    def _fsevents_callback(stream, info, num_events, event_paths, event_flags, event_ids):
        """FSEvents 事件回调。event_paths 是带 \0 分隔的 C 字符串。"""
        try:
            # event_paths 是一个 ctypes 指针指向 char** 数组
            # pyobjc 会把它转成 tuple of bytes
            if isinstance(event_paths, (tuple, list)):
                paths = [p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in event_paths]
            else:
                # 尝试从原始指针解析
                paths = []
                for i in range(num_events):
                    try:
                        p = event_paths[i]
                        if isinstance(p, bytes):
                            paths.append(p.decode("utf-8"))
                        else:
                            paths.append(str(p))
                    except Exception:
                        continue

            for path in paths:
                if not path.endswith(".md"):
                    continue
                # 转相对路径判断
                rel = path.replace(vault_str + "/", "").replace(vault_str, "")
                if not _should_embed(rel):
                    continue
                with _pending_lock:
                    _pending_files.add(path)

            # 去抖
            if _pending_files:
                # 取消之前的 timer（如果有），重新计时
                for t in _debounce_timers:
                    t.cancel()
                _debounce_timers.clear()
                timer = threading.Timer(_DEBOUNCE_SEC, _flush_pending)
                timer.daemon = True
                _debounce_timers.append(timer)
                timer.start()

        except Exception as e:
            print(f"[fsevents] 回调异常: {e}", file=sys.stderr, flush=True)

    try:
        # 创建 CFString 和 CFArray 包装的路径
        # pyobjc 需要 bytes 不是 str
        cf_path = CFStringCreateWithCString(None, vault_str.encode("utf-8"), kCFStringEncodingUTF8)
        cf_paths = CFArrayCreate(None, [cf_path], 1, None)

        # 创建事件流
        # 参数: allocator, callback, ctx, paths, sinceWhen, latency, flags
        # 使用 kFSEventStreamCreateFlagUseCFTypes 让回调直接收到 Python 类型
        stream = FSEventStreamCreate(
            None,                   # allocator
            _fsevents_callback,     # callback
            None,                   # ctx
            cf_paths,               # paths to watch
            kFSEventStreamEventIdSinceNow,  # since (现在起的所有事件)
            1.0,                    # latency（Task 333: 3.0→1.0 降延迟）
            kFSEventStreamCreateFlagUseCFTypes,  # flags: 回调收 CFType（Python 兼容）
        )

        if not stream:
            print("[fsevents] FSEventStreamCreate 返回 None，降级轮询", file=sys.stderr, flush=True)
            _polling_fallback(vault)
            return

        # 注册到当前 RunLoop
        run_loop = CFRunLoopGetCurrent()
        FSEventStreamScheduleWithRunLoop(stream, run_loop, "kCFRunLoopDefaultMode")

        # 启动流
        if not FSEventStreamStart(stream):
            print("[fsevents] FSEventStreamStart 失败，降级轮询", file=sys.stderr, flush=True)
            _polling_fallback(vault)
            return

        print(f"[fsevents] ✅ FSEvents 监听已启动（latency=1s + debounce={_DEBOUNCE_SEC}s）", flush=True)

        # 阻塞在 RunLoop（主线程，进程退出时自动结束）
        global _native_fsevents_running
        _native_fsevents_running = True
        CFRunLoopRun()

    except Exception as e:
        print(f"[fsevents] 原生模式失败: {e}，降级轮询", file=sys.stderr, flush=True)
        _polling_fallback(vault)


def _polling_fallback(vault):
    """降级轮询模式（30秒扫一次 content_hash）。"""
    import time
    import hashlib
    from pathlib import Path

    SCAN_INTERVAL = 10  # 秒（降级模式，比原生 FSEvents 慢但远好于 4 小时 cron）
    hash_cache: dict = {}  # {rel_path: sha1}

    while True:
        try:
            changed = []
            for md_file in vault.rglob("*.md"):
                rel = str(md_file.relative_to(vault)).replace("\\", "/")
                if not _should_embed(rel):
                    continue
                try:
                    content = md_file.read_text(encoding="utf-8", errors="replace")[:20000]
                    h = hashlib.sha1(content.encode("utf-8")).hexdigest()
                    if hash_cache.get(rel) != h:
                        hash_cache[rel] = h
                        changed.append(str(md_file))
                except Exception:
                    continue

            if changed:
                print(f"[fsevents-poll] 检测到 {len(changed)} 个文件变更，开始 embed", flush=True)
                for f in changed:
                    result = _embed_single_file(f)
                    if result.get("ok") and not result.get("skipped"):
                        print(f"[fsevents-poll] ✅ {f}: {result.get('chunks', 0)} chunks", flush=True)
                    elif not result.get("ok"):
                        print(f"[fsevents-poll] ❌ {f}: {result.get('error')}", file=sys.stderr, flush=True)

        except Exception as e:
            print(f"[fsevents-poll] 扫描失败: {e}", file=sys.stderr, flush=True)

        time.sleep(SCAN_INTERVAL)


def _flush_pending():
    """处理去抖后积攒的待 embed 文件。"""
    with _pending_lock:
        files = list(_pending_files)
        _pending_files.clear()

    if not files:
        return

    print(f"[fsevents] 去抖完成，embed {len(files)} 个文件", flush=True)
    for f in files:
        result = _embed_single_file(f)
        if result.get("ok") and not result.get("skipped"):
            print(f"[fsevents] ✅ {f}: {result.get('chunks', 0)} chunks", flush=True)
        elif not result.get("ok"):
            print(f"[fsevents] ❌ {f}: {result.get('error')}", file=sys.stderr, flush=True)



def _run_reconcile(hours: int = 2) -> dict:
    """Task 333 Layer 1d: 扫近 hours 小时修改的 .md，对比 PG content_hash，re-embed stale。

    兜底对账：fsevents 丢事件时（实测 119 文件 11 个 stale），手动/定时触发补齐。
    Returns: {"scanned": N, "fresh": N, "stale": N, "reembedded": N, "errors": N}
    """
    import subprocess
    import psycopg2
    from nexusos_mem.embed_corpus import _chunk_text, _content_hash

    vault = Path(_VAULT_ROOT)
    find_cmd = [
        "find", str(vault), "-name", "*.md", "-mmin", f"-{hours * 60}",
        "-not", "-path", "*/.obsidian/*", "-not", "-path", "*/.git/*",
        "-not", "-name", "SECRETS.md",  # Task611: 密钥真源不进扫描
    ]
    result = subprocess.run(find_cmd, capture_output=True, text=True, timeout=30,
                            env=scrubbed_env())
    files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]

    stats = {"scanned": 0, "fresh": 0, "stale": 0, "reembedded": 0, "errors": 0}

    for abs_path in files:
        rel = abs_path.replace(str(vault) + "/", "").replace(str(vault), "")
        if not _should_embed(rel):
            continue
        stats["scanned"] += 1

        p = Path(abs_path)
        if not p.exists():
            continue

        content = p.read_text(encoding="utf-8", errors="replace")[:20000]
        chunks = _chunk_text(content)
        if not chunks:
            continue

        rel_path = str(p.relative_to(vault)).replace("\\", "/")
        new_hashes = {f"{rel_path}#c{i}": _content_hash(c) for i, c in enumerate(chunks)}

        conn = psycopg2.connect(os.environ.get("NEXUS_PG_DSN", "postgresql://localhost/nexus_mem"))
        conn.autocommit = True  # Task 379: 只读(SELECT content_hash 对比)走 autocommit,防 idle in transaction
        cur = conn.cursor()
        prefix = rel_path + "#c"
        cur.execute(
            "SELECT path, content_hash FROM knowledge "
            "WHERE knowledge_type='document' AND path LIKE %s ESCAPE '\\'",
            (_escape_like(prefix) + "%",),
        )
        existing = {r[0]: r[1] for r in cur.fetchall()}
        cur.close()
        conn.close()

        all_match = all(existing.get(cp) == ch for cp, ch in new_hashes.items())
        if all_match and len(existing) == len(new_hashes):
            stats["fresh"] += 1
        else:
            stats["stale"] += 1
            r = _embed_single_file(abs_path)
            if r.get("ok"):
                stats["reembedded"] += 1
            else:
                stats["errors"] += 1

    print(f"[reconcile] {stats}", flush=True)
    return stats

def serve(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    # Task 333 Layer 1c: atexit hook，防进程退出时丢失最后的 pending 写入
    import atexit

    def _final_flush():
        """进程退出前 flush 所有 pending 文件。"""
        if _pending_files:
            print(f"[fsevents] 进程退出前 flush {len(_pending_files)} 个 pending 文件", flush=True)
            _flush_pending()

    atexit.register(_final_flush)
    print(f"[embed_server] 加载 BGE-M3 …", flush=True)
    t0 = time.time()
    _encode(["warmup"])            # 触发模型加载 + 预热
    # Task 562: 释放 warmup 峰值页——fp16 权重加载/warmup 期间 MPS allocator
    # 缓存的 IOAccelerator 页（实测峰值 ~2.2G）不主动 empty_cache 会常驻 footprint
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    _STATE["ready"] = True
    print(f"[embed_server] 就绪，加载耗时 {time.time() - t0:.1f}s，"
          f"监听 http://{host}:{port}", flush=True)

    # HTTPServer 放子线程（daemon）
    httpd = ThreadingHTTPServer((host, port), _Handler)
    httpd.daemon_threads = True
    http_thread = threading.Thread(target=httpd.serve_forever, daemon=True, name="http-server")
    http_thread.start()
    print(f"[embed_server] HTTP 服务已启动（子线程）", flush=True)

    # FSEvents 监听在主线程跑（fswatch subprocess 的 stdout 读取阻塞主线程）
    # 如果 fswatch 不可用，_fsevents_worker 内部降级为轮询（也阻塞主线程）
    global _fsevents_started
    if not _fsevents_started:
        _fsevents_started = True
        _fsevents_worker()  # 阻塞主线程（fswatch 或轮询）
    else:
        try:
            http_thread.join()
        except KeyboardInterrupt:
            pass

    print("[embed_server] 收到中断，退出", flush=True)
    httpd.server_close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="NEXUS-MEM BGE-M3 嵌入 worker")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()
    serve(args.host, args.port)
