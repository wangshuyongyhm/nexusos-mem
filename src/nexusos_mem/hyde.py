#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nexus.hyde — HyDE: Hypothetical Document Embeddings (Task 459 + Task 460 双模型分工).

对弱 query 用线上 LLM（OpenAI 兼容）生成假设文档，提升向量召回精度。

原理：
    BGE-M3 是 bi-encoder，query 和 doc 各自独立编码。
    短 query（如"经验怎么管"5字）信息量少，编码出的向量在语义空间定位模糊。
    假设文档（如 LLM 生成的"经验治理手册描述经验回流机制..."200字）信息丰富，
    向量定位精准。用假设文档的向量替代原 query 向量召回 → 召回质量提升。

Task 460 双模型分工（修复 Task 459 语言错配失败）：
    - 中文 query → qwen2.5:7b 生成中文假设文档（与 MEM 90%+ 中文语料对齐）
    - 英文 query → llama3.2:3b 生成英文假设文档（保英文专有名词 + 英文 document 召回）
    - 语言检测 heuristic：中文字符占比 >30% 判中文
    - prompt 强化：分中英文模板，明确"必须用 X 语言"+"专有名词可保留英文"

智能触发（should_hyde）：
    - query <6 字 → 触发（信息量少）
    - 术语词典未命中（expand_query 没扩展）→ 触发
    - 含口语词（"怎么"/"为啥"/"如何"等）→ 触发
    - 默认走原 query（延迟可控）

叠加不互斥：
    - expand_query（术语同义词扩展）+ HyDE（语义改写）叠加使用
    - HyDE 改的是 query_vec（向量召回），不动 FTS（仍用原 query 保精确 token）
    - HyDE + reranker 叠加 = 召回用假设文档向量 + 精排用原 query 与 doc 的 cross-encoder

降级（fail-open）：
    provider env 缺失 / 生成失败 / 超时 → 返回 None → 用原 expand_query 向量（与无 HyDE 等价）

依赖：
    Task590: 线上 LLM（OpenAI 兼容 chat/completions，env 配置）。
    本地 Ollama 已退役（模型外迁+服务停用，0819 用户终局）——无本地 GPU 依赖。
"""
import json
import os
import urllib.error
import urllib.request

# Task588: 默认值翻转 fail-open→fail-closed（0819 幽灵 ollama 调用根因）。
# 原默认 "1"（env 未设=开）使 dispatchd/pytest/直连库场景每次 search 都触发
# ollama generate（18 日 559 次，pytest 单跑 Δlog 486 行）。用户终局：不养本地
# 生成式 LLM——默认关，显式 NEXUS_HYDE_ENABLED=1 才开（复测时设）。
# 🔴 单一真源：本常量全库唯一定义处，search.py 从此处 import。
HYDE_ENABLED = os.environ.get("NEXUS_HYDE_ENABLED", "0") == "1"

# === Task590: OpenAI 兼容线上 provider（单一真源，模块头集中读 env）===
# kimi coding 端点（已实测 models+chat 200）。key 只从 env 读（密钥纪律：绝不进代码/日志/Vault）。
# provider 配置缺失 → generate_hypothetical fail-fast 返回 None（fail-open 到「无假设文档」），
# 默认状态（HYDE_ENABLED=0）下整条路径不激活——零网络调用（MEM#57461 幽灵调用教训）。
LLM_PROVIDER = os.environ.get("NEXUS_LLM_PROVIDER", "")
LLM_BASE_URL = os.environ.get("NEXUS_LLM_BASE_URL", "")
LLM_MODEL = os.environ.get("NEXUS_LLM_MODEL", "")
LLM_API_KEY = os.environ.get("NEXUS_LLM_API_KEY", "")
HYDE_TIMEOUT = float(os.environ.get("NEXUS_HYDE_TIMEOUT", "30"))  # 秒（线上推理 ~5-10s + 余量；思考型模型（k3）建议 env 拉到 90）
HYDE_MAX_TOKENS = int(os.environ.get("NEXUS_HYDE_MAX_TOKENS", "200"))  # 假设文档 ~200 字内（0=省略参数，思考型模型 reasoning 吃 token 时用）
# Task591: k3-256k 等思考型模型只允许 temperature=1（实测 400 "invalid temperature"）。
# 默认不发 temperature（服务端默认）；显式 NEXUS_HYDE_TEMPERATURE 才附带。
HYDE_TEMPERATURE = os.environ.get("NEXUS_HYDE_TEMPERATURE", "")


def _llm_configured() -> bool:
    """Task590: provider 配置完整性（base_url/model/api_key 三者齐备）。"""
    return bool(LLM_BASE_URL and LLM_MODEL and LLM_API_KEY)

# Task 460: 记录最后一次假设文档的语言（'zh'/'en'/None），供 explain 字段追溯
_LAST_LANG = None


def get_last_lang() -> str | None:
    """返回最后一次 generate_hypothetical 调用使用的语言（'zh'/'en'/None）。

    供 explain 字段追溯（hyde_used + hyde_lang）。search.py 可在 explain dict 构造时
    调用 `hyde.get_last_lang()` 获取。
    """
    return _LAST_LANG


# 口语化触发词（命中即触发 HyDE）
_COLLOQUIAL = {
    "怎么", "为啥", "为什么", "咋", "如何", "哪", "啥", "怎样", "怎么办",
    "怎样做", "为什么", "如何实现", "介绍一下",
}


def should_hyde(query: str, term_dict_hit: bool | None = None) -> bool:
    """判断是否触发 HyDE。

    Args:
        query: 用户原始查询
        term_dict_hit: 术语词典是否命中（expand_query 后 expanded != query 即命中）。
                       None=未指定（不基于此判断，仅看长度+口语词，单测时不传）。
                       False=明确未命中（触发 HyDE）。
                       True=明确命中（不基于此触发）。
    Returns:
        True=触发 HyDE 生成假设文档；False=走原 expand_query 向量

    触发条件：
        1. query <6 字（信息量少）→ 触发
        2. term_dict_hit=False（术语词典明确未命中）→ 触发
        3. 含口语化词（"怎么"/"为啥"/"如何"等）→ 触发
        其他（含 term_dict_hit=None/True 且长且无口语）→ 不触发
    """
    if not HYDE_ENABLED:
        return False
    if not query:
        return False
    # 1. 短 query
    if len(query) < 6:
        return True
    # 2. 术语词典明确未命中（False 才触发，None 不基于此）
    if term_dict_hit is False:
        return True
    # 3. 口语化词
    for w in _COLLOQUIAL:
        if w in query:
            return True
    return False


# ============================================================================
# Task 460: 语言检测 + 模型路由
# ============================================================================

def _detect_language(text: str) -> str:
    """简单 heuristic 语言检测。返回 'zh' 或 'en'。

    Task582 修复（池 08-07 #22）：原版分母=len(去空格文本)，把数字/标点/符号
    计入分母，「trace 膨胀」=2中文/(7有效字符)=28.6%≤30% 误判 en。
    新口径：
    - 分母只计「文字类字符」：CJK 中文 + ASCII 字母（空格/数字/标点不计）
    - 阈值 30% → 25%：中英混合技术查询（如「trace 膨胀」28.6%）判 zh，
      对 MEM 语料 90%+ 中文更安全（宁 zh 勿 en）
    空字符串默认中文（保持原语义）。
    纯数字/符号串：文字字符=0，分母 max(1,0)=1，占比 0 → en（不崩）。
    """
    if not text:
        return "zh"
    chinese_chars = 0
    ascii_letters = 0
    for c in text:
        if "\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf":
            chinese_chars += 1
        elif "a" <= c <= "z" or "A" <= c <= "Z":
            ascii_letters += 1
    letter_total = chinese_chars + ascii_letters
    if letter_total == 0:
        return "en"  # 纯数字/标点：无文字信号，按 en（不崩、可确定性输出）
    return "zh" if chinese_chars / letter_total > 0.25 else "en"


# Task590: _select_model 已删（双模型路由随 ollama 退役——线上单一模型由
# NEXUS_LLM_MODEL 指定，语言差异体现在 prompt 选择而非模型选择）


# ============================================================================
# Task 461: Few-shot 样例（MEM 真实经验，引导 LLM 生成领域准确的假设文档）
# ============================================================================

_FEW_SHOT_EXAMPLES = [
    {
        "query": "磁盘空间清理",
        "doc": "本地磁盘满时：1.纯缓存(npm/yarn/pnpm)symlink外迁到外接盘，每次插盘自动生效 2.App缓存(Chrome/Telegram/微信)mv到Trash删除自动重建 3.不外迁huggingface模型(NEXUS-EMBED在用)和Hermes state.db(高频读写)。收益18.7GB。关键认知：du -sh虚高要用find -type f算真实大小；symlink方式稳定(npm验证3个月)。",
    },
    {
        "query": "session 残留文件",
        "doc": "01-Projects/sessions/ 下偶发的只有 started header 无内容的 session 日志(source=unknown)，不是异常中断，而是 tool_notify_server.py 接收小弟 POST /report(status=started)时 archive_obsidian 写下的正常产物。当小弟发了 started 回调但任务未完成/被中断/没发 done 回调时，就留下这种残留。处理：直接 mv 到 ~/.Trash 即可。",
    },
    {
        "query": "经验治理手册",
        "doc": "经验治理手册描述经验回流与防膨胀机制。经验不直接变规则，而是提炼为检查点。三层架构：COLD(经验库)→WARM(检查点提炼)→HOT(规则固化)。防膨胀：定期 densify + superseded 标记 + 向量去重。经验回流路径：小弟 done 回调带 experiences → write_knowledge 入 MEM → crucible 熔炉验证 → 提炼为检查点 → 固化为 AGENTS.md 规则。",
    },
]


# ============================================================================
# Task 460/461: 分中英文 prompt 模板（修复语言错配 + Few-shot 引导领域知识）
# ============================================================================

# 中文 prompt：明确"必须用中文" + Few-shot 样例引导 MEM 领域风格
_HYDE_PROMPT_ZH = """请根据查询，用中文生成一段技术经验文档（约200字）。要求：
1. 必须使用中文书写（专有名词如 API/Gateway/Python 可保留英文）
2. 内容像真实的技术经验记录，包含背景、做法、关键概念、踩坑
3. 直接输出文档内容，不要解释、不要复述查询

以下是真实经验文档样例（参考其风格和详细度）：

样例1（查询：磁盘空间清理）：
{fewshot_1_doc}

样例2（查询：经验治理手册）：
{fewshot_2_doc}

现在请根据以下查询生成：
查询：{query}

中文技术文档："""

# 英文 prompt（Task 461 陷阱 6：英文 prompt 不加 Few-shot，英文 query 少且主要矛盾是中文）
_HYDE_PROMPT_EN = """Based on the following query, generate a technical experience document (~200 words) in English. Requirements:
1. Write in English
2. Content should read like a real technical experience note with background, approach, key concepts
3. Output document content directly, no explanation, no restating the query

Query: {query}

Technical document:"""


def generate_hypothetical(query: str) -> str | None:
    """调线上 LLM（OpenAI 兼容 chat/completions）生成假设文档。

    Task590: ollama api/generate → OpenAI 兼容 chat/completions（本地 LLM 已退役）。
    语言路由保留（zh/en 对应 prompt），但模型由 env 统一指定（NEXUS_LLM_MODEL，
    线上单一模型同时懂中英，不再双模型）。

    Args:
        query: 用户原始查询
    Returns:
        假设文档字符串（约 200 字），失败返 None（降级用原 query）

    设计：
        - temperature 0.7（平衡多样性和相关性）
        - max_tokens 200（假设文档长度上限，省配额）
        - provider 配置缺失 → fail-fast 返 None（不抛异常，不打断 search）
        - 记录 _LAST_LANG 供 explain 追溯
    """
    global _LAST_LANG
    if not query:
        _LAST_LANG = None
        return None

    # Task590: provider 未配置 → fail-fast（fail-open 到无假设文档，非崩溃）
    if not _llm_configured():
        print("[WARN] HyDE provider env 未配置（NEXUS_LLM_BASE_URL/MODEL/API_KEY），"
              "跳过假设文档生成", flush=True)
        _LAST_LANG = None
        return None

    lang = _detect_language(query)
    # Task 461: 中文 prompt 注入 Few-shot 样例，英文 prompt 不加（陷阱 6 简化）
    if lang == "zh":
        prompt = _HYDE_PROMPT_ZH.format(
            query=query,
            fewshot_1_doc=_FEW_SHOT_EXAMPLES[0]["doc"],
            fewshot_2_doc=_FEW_SHOT_EXAMPLES[2]["doc"],  # 用经验治理手册样例引导领域词
        )
    else:
        prompt = _HYDE_PROMPT_EN.format(query=query)

    try:
        # Task591: 参数按 env 组装——思考型模型（k3）只允许 temperature=1，
        # 默认省略 temperature/max_tokens 由服务端定（max_tokens 0=省略）。
        body = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
        }
        if HYDE_MAX_TOKENS > 0:
            body["max_tokens"] = HYDE_MAX_TOKENS
        if HYDE_TEMPERATURE != "":
            body["temperature"] = float(HYDE_TEMPERATURE)
        req = urllib.request.Request(
            f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LLM_API_KEY}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=HYDE_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        # OpenAI 兼容响应: choices[0].message.content
        text = str(payload["choices"][0]["message"]["content"]).strip()
        if text:
            _LAST_LANG = lang  # 记录成功的语言供 explain 追溯
            return text
        _LAST_LANG = None
        return None
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError,
            KeyError, IndexError) as e:
        # 端点不可达 / 超时 / 响应损坏 → fail-open
        print(f"[WARN] HyDE 生成失败（lang={lang}, model={LLM_MODEL}, 降级用原 query）: "
              f"{type(e).__name__}: {e}", flush=True)
        _LAST_LANG = None
        return None
    except Exception as e:
        # 未知异常也降级（绝不阻塞 search）
        print(f"[WARN] HyDE 未知异常（lang={lang}, 降级）: {type(e).__name__}: {e}", flush=True)
        _LAST_LANG = None
        return None


def warmup() -> bool:
    """Task590: warmup 线上 provider（zh + en 两种 prompt 各跑一次极小生成）。

    用于 eval 跑前预热（验证链路+度量首 token 延迟）。
    Returns: True=两种都成功 False=至少一个失败（仍可降级运行）
    """
    if not _llm_configured():
        print("[hyde] provider 未配置，warmup 跳过", flush=True)
        return False
    print(f"[hyde] warmup 线上 provider ({LLM_MODEL})...", flush=True)
    ok = True
    zh_result = generate_hypothetical("测试查询")
    if zh_result:
        print(f"[hyde] ✅ zh prompt warmup 完成（{len(zh_result)} chars）", flush=True)
    else:
        print("[hyde] ❌ zh prompt warmup 失败", flush=True)
        ok = False
    en_result = generate_hypothetical("test query")
    if en_result:
        print(f"[hyde] ✅ en prompt warmup 完成（{len(en_result)} chars）", flush=True)
    else:
        print("[hyde] ❌ en prompt warmup 失败", flush=True)
        ok = False
    return ok


if __name__ == "__main__":
    # 自测：语言检测 + 路由 + 触发判断 + 生成
    import sys

    test_query = sys.argv[1] if len(sys.argv) > 1 else "经验怎么管"
    print(f"=== _detect_language ===")
    print(f"  query: {test_query!r}")
    print(f"  lang: {_detect_language(test_query)}")
    print()
    print(f"=== should_hyde('{test_query}') ===")
    print(f"  short<6: {len(test_query) < 6}")
    print(f"  口语词: {[w for w in _COLLOQUIAL if w in test_query]}")
    print(f"  → 触发: {should_hyde(test_query)}")
    print()
    print(f"=== generate_hypothetical('{test_query}') ===")
    if should_hyde(test_query):
        result = generate_hypothetical(test_query)
        if result:
            print(f"  生成 ({len(result)} chars, lang={get_last_lang()}):")
            for line in result.splitlines()[:5]:
                print(f"    {line}")
        else:
            print("  ❌ 生成失败（provider 未配置/不可达？）")
    else:
        print("  跳过（不触发 HyDE）")
