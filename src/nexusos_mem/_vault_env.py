"""nexusos_mem._vault_env — vault 级依赖的包内自足替代（Task668 批4b ②；1957 锚点收敛）。

锚点解析单源=nexusos_common（env > 哨兵推导 > 缺 vault 三态，见其 anchors.py）。
本模块保留 mem 侧常量派生与 sys.path 自愈副作用（哨兵复用 common，无第二套推导）。

缺 vault 行为（1955 design §3.2 统一，废 /nonexistent 静默占位）：
mem 的 vault 依赖功能（SHADOW 索引 / apply_honesty trace / embed_corpus）属
「必须 vault」——常量落 None + WARN；功能入口经显式报错（含 remediation）。

替代（历史契约不变）：
  - apply_honesty.py / shadow_match.py 的 `from nexus.constants import …`
  - __init__.py 自愈块的 `import nexus.paths_setup`
vault 上下文行为零变化；干净环境（wheel 独立部署）全 no-op 可 import。
"""
import logging
import sys

from nexusos_common import VAULT_MISSING_REMEDIATION, auto_scripts_dir, vault_root

logger = logging.getLogger(__name__)

VAULT_ROOT = vault_root(__file__)  # Path | None（缺 vault 不再静默占位）
if VAULT_ROOT is None:
    logger.warning(
        "NEXUS vault 不可用——SHADOW 索引/apply_honesty trace/embed_corpus 等 "
        "vault 依赖功能将显式报错。" + VAULT_MISSING_REMEDIATION
    )
    MEMORY_DIR = None
    SHADOW_INDEX_PATH = None
else:
    MEMORY_DIR = VAULT_ROOT / "06-Memory"
    SHADOW_INDEX_PATH = MEMORY_DIR / "SHADOW-索引.md"


def setup_sys_path() -> None:
    """vault 布局的 sys.path 初始化（nexus.paths_setup 的包内等价；哨兵=common 单源）。"""
    auto_dir = auto_scripts_dir(__file__)
    if auto_dir is None:
        return
    if str(auto_dir) not in sys.path:
        sys.path.insert(0, str(auto_dir))
    try:
        import nexus.paths_setup  # noqa: F401  副作用单源：尽力而为（原 Task668 语义保持）
    except Exception:  # noqa: BLE001, S110
        pass
