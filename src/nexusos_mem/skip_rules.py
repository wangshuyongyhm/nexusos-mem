# -*- coding: utf-8 -*-
"""embed 目录排除单一真源（task 2611111530）。

全量 corpus（embed_corpus._iter_vault_docs）与 FSEvents 增量
（embed_server._should_embed）共用 should_skip 一个判定函数，
废除各自为政、已经互相漂移的枚举清单——同一业务规则只定义一处。

规则（用户定向：.XXXX 不进知识库）:
    1. 隐藏目录：任一目录组件以 "." 开头 → 排除。
       .wt/.zcode/.obsidian/.git/.pytest_cache/.codewhale/.opencode/
       .atomcode/.smart-env/.hermes/.trash/.DS_Store 等全覆盖，
       未来新工具的隐藏目录零维护。
    2. 具名排除：非隐藏但明确不 embed 的目录（NAMED_SKIP_DIRS），
       由原两通道枚举清单并集去重而来。
"""

HIDDEN_PREFIX = "."

# 非隐藏排除清单（原 corpus+FSEvents 两清单并集去重；
# 隐藏项由规则 1 通配覆盖，不再逐个枚举）
NAMED_SKIP_DIRS = {
    "node_modules",
    "__pycache__",           # 点规则兜不住的非隐藏缓存目录
    "copilot",
    "99-Archive",             # 归档文件不 embed
    "hermes-scripts-backup",  # 脚本镜像不 embed
    "health-reports",         # 健康报告不 embed
    "reports",                # 运行报告不 embed
    "_retired",               # corpus 通道遗留排除项
}


def should_skip(dir_parts) -> bool:
    """目录排除判定（单一真源，两 embed 通道共用）。

    dir_parts: 路径的目录组件序列（不含文件名），
    如 PurePath.parts[:-1] 或 rel_path.split("/")[:-1]。
    文件名不参与目录规则——SECRETS.md 文件级黑名单由调用方处理（Task611）。

    语义保留（Task 333）：sessions 不在排除清单——
    01-Projects/sessions/ 是有效归档，必须可 embed。
    """
    for part in dir_parts:
        if part.startswith(HIDDEN_PREFIX) or part in NAMED_SKIP_DIRS:
            return True
    return False
