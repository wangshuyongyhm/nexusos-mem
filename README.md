# nexusos-mem

**NEXUS-MEM — 知识库 + 检索 + 打分体系（SharedKnowledge / pgvector / BGE 可选）**

多 Agent 体系的共享真相源：所有 Agent 把经验、发现、教训写进同一个 PostgreSQL 知识库，检索时 FTS 全文 + 向量双模式召回。解决问题先搜 MEM，完成任务写回 MEM——知识在 Agent 之间即时流动。

## 核心特性

- **双模式检索**：PostgreSQL FTS 全文检索打底，`[full]` extras 提供 BGE-M3 向量嵌入 + bge-reranker-v2-m3 精排；向量组件缺席时自动降级纯 FTS，功能不断
- **读写一体 HTTP 网关**：`api_server`（默认 `127.0.0.1:11537`）提供 `/search`、`/knowledge` 等 REST 端点，任何语言的 Agent 都能接入
- **Python API**：`SharedKnowledge` 类一行接入（`kb.search("关键词", top_k=5)` / `kb.add_knowledge(...)`）
- **认知诚实层**：检索带 `agent_id` 自动记录套用尝试（annotate），配合 outcome 回填形成经验有效性闭环
- **常驻嵌入 worker**：`embed_server`（默认 :11436）FSEvents 增量嵌入；`rerank_server`（默认 :11437）cross-encoder 精排
- **SHADOW 编码**：knowledge → 确定性编码匹配（`shadow_match` / `parser`）

## 安装

最小安装（纯 FTS 检索，不装向量组件）：

```bash
# 发布至 PyPI 前：本地路径安装（仓库根目录执行；三包目录并列时同命令安装三件套）
pip install ./nexusos-mem
pip install ./nexusos-gate ./nexusos-mem ./nexusos-xdr   # 三件套一条命令（1597 干净 venv 实证口径）
```

完整安装（含 BGE-M3 向量 + 精排，体积大，含 torch）：

```bash
pip install "./nexusos-mem[full]"
```

要求 Python >= 3.10。运行时需要可访问的 PostgreSQL（知识库存 `nexus_mem` 库；向量模式另需 pgvector 扩展）。

PyPI 发布后可用 `pip install nexusos-mem` / `pip install "nexusos-mem[full]"`（当前包未发布，请勿直接照抄）。

`[full]` extras 内容：`sentence-transformers>=2.2`、`torch>=2.0`、`transformers>=4.30`、`FlagEmbedding>=1.2`。BGE 系全部函数级懒 import——装了才启用，不装不报错。

## 快速上手

安装后导入首验（1597 实证命令）：

```bash
python -c "import nexusos_mem; print('ok')"
```

```python
from nexusos_mem.shared import SharedKnowledge

kb = SharedKnowledge(agent_id="my-agent")

# 检索（FTS + 向量双模式，支持中英文）
results = kb.search("dispatch 超时", top_k=5)

# 写入（其他 Agent 立即可见）
kb.add_knowledge(
    title="XDR /dispatch 客户端超时须 ≥280s",
    content="现象/根因/修法……",
    knowledge_type="experience",
    tags=["xdr", "dispatch"],
)
```

HTTP 方式（非 Python Agent 适用）：

```bash
curl -s "http://127.0.0.1:11537/search?q=关键词&top_k=5&agent_id=my-agent"

**score 语义（1401-d5a 起变更）**：顶层 `score` = 排序融合分 `final_score`（原缺失/恒 0）。量纲：vec∈[0,1]、fts 无量纲、type_prior/decay 乘法、quality/outcome/ref 加法——**仅保证同查询内排序可比，跨查询族不可比**（阈值筛参考用）。`explain` 默认返回五分量（`?explain=false` 剔除省 token）；低于阈值（默认 0.30，env `NEXUS_LOW_CONFIDENCE_THRESHOLD`）的结果带 `"low_confidence": true`。新参数 `pool=experience|document|all`（默认 all=加权混排）。旧客户端不带新参数行为=混排+真实 score，兼容。
curl -s -X POST http://127.0.0.1:11537/knowledge \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"my-agent","title":"标题","content":"内容","knowledge_type":"experience","tags":["关键词"]}'
```

## 知识身份层（Task1524，2026-09-24）

同一知识单元在体系里有多套地址（PG mem_id / Vault 路径 / 镜像分片 / 图谱卡），
`knowledge_identity` 表（`sql/001_knowledge_identity.sql`）是唯一身份入口：

- **表结构**：`identity_kind`（mem/vault_doc/mirror_shard/graph_card/index）+
  `mem_id`（FK→knowledge，ON DELETE CASCADE：知识行删除身份同步注销）+
  `vault_path` + `producer`（哪条写入链产生）。零关系列——血缘/语义边是 M2 的边表。
- **登记机制（单一真源=数据库触发器）**：`AFTER INSERT ON knowledge` 自动登记
  mem 行，覆盖全部写入口（含直连脚本）。写入链用 `set_config('app.producer',...)`
  标注精确 producer（如 `api_server:POST /knowledge`、`embed_server:fsevents`）；
  未标注落 `pg:knowledge-insert`。登记失败（锁/权限）只 WARNING 不阻断知识写入
  （lock_timeout 250ms 有界等待 + 异常吞噬）。
- **快照链产生侧**：`knowledge-mirror-gen.py`（分片 frontmatter 带 `mem_ids`
  列表=分片→条目机器锚）与 `relation_graph_export.py`（图谱卡 frontmatter 裸
  `mem_id`）各自登记产物身份；分片删除时同步注销陈旧行。
- **存量回填**：`scripts/backfill_identity.py`（幂等，可重跑）。
- **消费**：`identity_stats()` 按 kind/顶层目录计数；`identity_coverage` 视图对账
  活跃知识 vs mem 身份覆盖（M1A 域监控 / M3 统一读模型的接入点）。

## 环境变量（常用）

| 变量 | 默认值 | 作用 |
|------|--------|------|
| `NEXUS_PG_DSN` | `postgresql://localhost/nexus_mem` | PostgreSQL 连接串（知识库/向量/元数据同库） |
| `NEXUS_VAULT_ROOT` | env > 包位置哨兵推导 > 占位 | Vault 根目录（fsevents 监听/文档切片用，干净环境可缺省） |
| `NEXUS_EMBED_TIMEOUT` | `3` | 查询编码客户端超时（秒），超时降级纯 FTS |
| `NEXUS_RERANK_TIMEOUT` | `3` | 精排客户端超时（秒），超时降级融合分 |
| `NEXUS_RERANK_POOL` | `16` | 精排候选池大小（进入 rerank 的 top-N） |
| `NEXUS_RERANK_ENABLED` | `1` | `0` 关闭精排（向量/全文融合后直接返回） |
| `NEXUS_LOW_CONFIDENCE_THRESHOLD` | `0.30` | 低置信标记阈值（低于此分带 `low_confidence: true`） |

（完整环境变量面见 `search.py` 常量段与 `_vault_env.py`——上表为公开文档常用集）

## 依赖分层

| 层 | 内容 |
|----|------|
| 硬依赖 | `psycopg2-binary>=2.9`、`pyyaml>=5.4` |
| 可选依赖 | `[full]`：BGE-M3 向量嵌入 + reranker 精排（sentence-transformers / torch / transformers / FlagEmbedding） |
| 数据层 | PostgreSQL `nexus_mem` 库（pgvector 扩展仅向量模式需要） |
| 服务层 | api :11537（必需）/ embed :11436（可选增强）/ rerank :11437（可选增强） |

## 与另两件的关系

nexusos-mem 是三件套的**共享真相源**：nexusos-xdr 的 dispatch 链路会调 MEM 检索注入相关经验、done 回调会触发经验套用回填；nexusos-gate 的检查插件可选地查 MEM 留痕。本包可独立部署使用——不装 xdr/gate 也能跑完整的知识库服务。

## License

MIT License。详见仓库根目录 [LICENSE](./LICENSE)。

## 血缘边 provenance（Task1533 M2a，2026-09-24）

`rel_type='provenance'`——事实性血缘边（零错边风险，与 LLM 语义边 M2.5 分层），
方向语义与 citation 一致：**src=子（新知识/报告/源条目），dst=父（任务/索引/分片容器）**。
模块 `nexusos_mem/provenance.py`；检索 riding 一跳默认参与（`search.py` 白名单
`relates+provenance`，related 字段混合展示）。

三条写入链自动挂边（失败/无目标只 log 跳过，幂等 ON CONFLICT）：

| 链 | 挂点 | 边 |
|----|------|----|
| ① MEM 写入 | `POST /knowledge` body 可选 `task_dir`（缺省从 `path` 的 `task/NNN` 前缀推；都无则跳过） | 新经验 →(produced_in_task)→ `task/<dir>/input/task.md` 代表节点 |
| ② persist 落盘 | `embed_server` document upsert 后按文件级规则 | `output/*.md` →(belongs_to_task)→ task.md；`03-Daily/<date>/*.md` →(belongs_to_index)→ MOC-日志索引.md |
| ③ 快照镜像 | `knowledge-mirror-gen.py --provenance-tag=<tag>`（默认关，防与 1534 存量口径撞车；全量开启待主控定夺） | 源条目 →(mirrored_in_shard)→ 分片代表节点 |

**目标节点解析**：边表端点是 `knowledge.id`；Vault 文件经 embed_server 抓成
document 行（`path` 粒度 `<rel>#cN` 分块），`resolve_vault_doc()` 取该文件前缀
最小 id 作为文件代表节点。为使 `input/task.md` 可解析，`_mirror_admission`
放行 task.md 本体（dispatched 副本/references 仍拒收，564 减量初衷不变）；
存量 task.md 的 document 行与 vault_doc 身份回填归 1534。

**DB 迁移**：`knowledge_relation` 的 CHECK 约束扩为四枚举
（supersedes/conflicts/relates/**provenance**）——`relation.ensure_schema()`
幂等迁移（检测旧约束自动重建）；手工路径见 Task1533 reproduce.sh。

## 语义边 semantic（Task1537 M2.5，2026-09-24）

`rel_type='semantic'`——LLM 复审过的高置信相关对（同主题/可互相印证/解决同问题），
**增值层而非保底**：双端 knowledge id 空间，无方向语义（统一小 id 为 src），
evidence 带 `conf=high; sim=..; 理由`，created_by=`semantic:<task>`。

**riding opt-in**（默认不进扩联池，试点期验证数据回流后再议默认）：

- HTTP：`GET /search?...&include_semantic=1`
- Python：`kb.search(..., include_semantic=True)`
- related 条目统一带 `rel_type` 标记（relates/provenance/semantic 可区分），
  explain 附 `+semantic_riding`；与 1538 的 `readmodel` opt-in 正交可叠加。

试点管线 `scripts/semantic_pilot.py`：`candidates`（BGE-M3 HNSW 预筛，核心类型
白名单，排除同 task_id/已有边/相似度窗口外）→ LLM 复审决议 → `apply`（仅
confidence=high 落库，幂等 ON CONFLICT）。DB 迁移 `sql/003_semantic_rel_type.sql`
（CHECK 五枚举，幂等；ensure_schema 判据同步）。
