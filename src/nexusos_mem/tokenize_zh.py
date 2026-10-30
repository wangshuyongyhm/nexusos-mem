#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tokenize_zh.py — 中文 FTS jieba 分词共用模块（task/1452 路线 B 双轨）。

写入侧（shared.add_knowledge 的 tokens_zh 列）与查询侧（search.pg_fts_search 的
tsquery 构造）共用同一分词实现——两侧分词不一致=词表错位，词级匹配即失效
（1449 施工草稿 §2.3 命门，故抽共用模块绝不两处实现）。

双轨灰度：NEXUS_FTS_JIEBA 环境变量（默认 off）。off=完全走旧轨（写入列存
NULL、查询用原 query，字节级等价改造前行为，回滚生命线）；on=新轨
（jieba.lcut 精确模式分词，空格 join 后仍用 'simple' config——simple 按空格
切词，预分词文本即获得中文词级匹配能力，无需 PG 扩展，1449 路线 A 已排除）。

fail-open 铁律：jieba 缺失/异常时写入侧返回 None（列 NULL=未回填语义）、
查询侧返回原串——分词失败绝不阻断写入或检索。
"""
import os
import sys

_TRUTHY = ("1", "true", "yes", "on")
_jieba = None  # 进程内惰性单例（首次 lcut 自动加载词典 ~1s，之后复用）


def jieba_enabled() -> bool:
    """双轨 flag：env 现读现判（评测 on/off 两组切换无需改代码/重启）。"""
    return os.environ.get("NEXUS_FTS_JIEBA", "").strip().lower() in _TRUTHY


def _lcut_join(text: str) -> str:
    global _jieba
    if _jieba is None:
        import jieba  # 惰性导入：flag off 的环境不装 jieba 也零依赖
        _jieba = jieba
    return " ".join(_jieba.lcut(text))


def tokenize_zh(text: str):
    """统一入口：flag off / 空串 / 分词失败 → None（调用方决定列 NULL 或回退原串）。"""
    if not jieba_enabled() or not text or not text.strip():
        return None
    try:
        return _lcut_join(text)
    except Exception as e:  # fail-open：分词失败降级，不阻断调用方
        print(f"[WARN][tokenize_zh] jieba 分词失败（降级 None）: {e}", file=sys.stderr)
        return None


def tokenize_for_write(title: str, content: str):
    """写入侧：title+' '+content 整体分词（文本源与旧列 search_vector 生成式一致）。"""
    return tokenize_zh(f"{title or ''} {content or ''}".strip())


def tokenize_for_query(query: str) -> str:
    """查询侧：分词失败/关轨回退原串（SQL 参数恒为 str，防 None 入参炸查询）。"""
    return tokenize_zh(query) or query
