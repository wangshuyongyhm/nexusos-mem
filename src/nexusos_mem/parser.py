#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nexus.parser — SHADOW 索引解析器（统一 v2）

v2 变更（2026-07-14）：
  - 层级定义从固定范围（A-E/F-Z/AA-BZ/ZA-ZF）改为首字母通配（A*/B*/C*/D*...）
  - 正则用 \(X\*\) 匹配"首字母X的所有码"
  - 支持任意层级扩展（E*-Z* 自动识别）
  - 同层级内码越长越优先（A < AA < AAA）— 由调用方按 len(code) 排序

接口：
    parse_shadow_index(path) -> {code: {target, layer, section?}}
    parse_shadow_by_layer(path) -> {layer: {code: target}}
    parse_shadow_flat(path) -> {code: target}
"""

import re
from pathlib import Path
from typing import Dict, Optional


# SHADOW 索引各层的标题正则（v2: 首字母通配）
# 格式: ## L0常驻(A*) payload  或  ## L2扩展(CA*-CZ*) payload
# 匹配: ## + 层级名 + ( + 字母序列 + 可选*-范围 + * + ) + 空格 + payload
_LAYER_PATTERNS = [
    ("L0", r"## L0.*?\(A\*?\).*?\s+(.*)$"),
    ("L1", r"## L1.*?\(B\*?\).*?\s+(.*)$"),
    ("L2", r"## L2.*?\(C.*?\*?\).*?\s+(.*)$"),
    ("L3", r"## L3.*?\(D\*?\).*?\s+(.*)$"),
    ("L4", r"## L4.*?\(E\*?\).*?\s+(.*)$"),
    ("L5", r"## L5.*?\(F\*?\).*?\s+(.*)$"),
    ("L6", r"## L6.*?\(G\*?\).*?\s+(.*)$"),
    ("L7", r"## L7.*?\(H\*?\).*?\s+(.*)$"),
    ("L8", r"## L8.*?\(I\*?\).*?\s+(.*)$"),
    ("L9", r"## L9.*?\(J\*?\).*?\s+(.*)$"),
]

# 通用兜底正则：匹配 ## L{数字}...({字母开头}*) payload
# 支持 (C*), (CA*-CZ*), (CA*) 等所有变体
_LAYER_FALLBACK_RE = re.compile(r"^## L(\d+).*?\(([A-Z][A-Z]*\*[^)]*)\)\s+(.*)$")


def parse_shadow_index(path: Path) -> Dict[str, dict]:
    """解析 SHADOW-索引.md，返回 {code: {target, layer, section?}}。

    v2: 支持首字母通配层级（A*=L0, B*=L1...）和任意层级扩展。
    通用兜底正则捕获 _LAYER_PATTERNS 未显式列出的高层级。

    Args:
        path: SHADOW-索引.md 路径

    Returns:
        dict: {code: {"target": str, "layer": "L0"|"L1"|..., "section": str|None}}

    Example:
        >>> entries = parse_shadow_index(Path("SHADOW-索引.md"))
        >>> entries["A"]
        {"target": "NEXUS-CONSTRAINT/index.md", "layer": "L0", "section": None}
        >>> entries["AA"]
        {"target": "NEXUS-CONSTRAINT/init.md", "layer": "L0", "section": None}
    """
    path = Path(path)
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8")
    entries: Dict[str, dict] = {}

    # 第一遍：用显式 _LAYER_PATTERNS 匹配
    matched_layers = set()
    for layer, pattern in _LAYER_PATTERNS:
        for line in text.splitlines():
            m = re.search(pattern, line.strip())
            if not m:
                continue
            payload = m.group(1).strip()
            _parse_payload_into(payload, layer, entries)
            matched_layers.add(layer)

    # 第二遍：通用兜底，捕获未显式列出的层级（L10+ 或遗漏的）
    for line in text.splitlines():
        m = _LAYER_FALLBACK_RE.search(line.strip())
        if not m:
            continue
        layer_num = m.group(1)
        layer = f"L{layer_num}"
        if layer in matched_layers:
            continue  # 已被显式 pattern 处理
        payload = m.group(3).strip()
        _parse_payload_into(payload, layer, entries)
        matched_layers.add(layer)

    return entries


def parse_shadow_by_layer(path: Path) -> Dict[str, Dict[str, str]]:
    """解析并按层分组，返回 {layer: {code: target}}。

    v2: 动态发现所有层级，不再硬编码 L0-L3。
    适用于 entropy-budget.py / cross-project-migration.py 等需要分层处理的脚本。
    """
    entries = parse_shadow_index(path)
    by_layer: Dict[str, Dict[str, str]] = {}
    for code, meta in entries.items():
        layer = meta.get("layer", "L1")
        if layer not in by_layer:
            by_layer[layer] = {}
        by_layer[layer][code] = meta["target"]
    return by_layer


def parse_shadow_flat(path: Path) -> Dict[str, str]:
    """最简形式：{code: target}（不含 layer/section）。

    适用于 rl-feedback.py / knowledge-evolution.py 等只需要 code→target 映射的场景。
    """
    entries = parse_shadow_index(path)
    return {code: meta["target"] for code, meta in entries.items()}


# ---------------------------------------------------------------------------
# 内部实现
# ---------------------------------------------------------------------------

def _parse_payload_into(payload: str, layer: str, entries: Dict[str, dict]) -> None:
    """解析一行的 payload（"A→铁律 B→..."），写入 entries dict。

    支持指向路径中包含空格（如 "环境经验/macOS"），按 "编码→" 分割。
    支持 section 标记（§）。
    """
    tokens = payload.split()
    current_code: Optional[str] = None
    current_target_parts: list = []

    def flush():
        nonlocal current_code, current_target_parts
        if current_code and current_target_parts:
            target = " ".join(current_target_parts).strip()
            section = None
            if "§" in target:
                base, _, section = target.partition("§")
                target = base.strip()
                section = section.strip() or None
            entries[current_code] = {
                "target": target,
                "layer": layer,
                "section": section,
            }
        current_code = None
        current_target_parts = []

    for token in tokens:
        if "→" in token:
            flush()
            parts = token.split("→", 1)
            current_code = parts[0].strip()
            current_target_parts = [parts[1].strip()] if len(parts) > 1 and parts[1] else []
        elif current_code is not None:
            current_target_parts.append(token)
    flush()
