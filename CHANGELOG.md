# Changelog

All notable changes to the nexusos packages are documented in this file.

格式：Keep a Changelog 1.1；四包 lockstep 单文件共享（同一版本号 = 四包同版本），
每个版本下按 `### nexusos-common / gate / mem / xdr` 四节分账，节内按
Added / Changed / Deprecated / Removed / Fixed / Security 分组。
版本策略与 layout_version 轴见各包 README 与 design §7（task/1955）。

## [Unreleased]

### nexusos-common

**Added**
- anchors 锚点解析包：`nexus_home` / `vault_root` / `task_root` / `subpath` / `expand_env`，显式报错 + WARN 降级规则（task 1957）
- 私路径 lint 扫描器 `nexusos_common.lint`（15 pattern 数据文件化），基线 204 hits/13 类与 audit §1 对账（task 1957）
- `LAYOUT_VERSION` / `SUPPORTED_LAYOUTS` 布局版本轴常量——独立于包 semver 的目录布局版本，与 `.nexus/config.toml [meta] layout_version` 对账（task 1963）

### nexusos-gate

**Changed**
- `_vault_env` / `gate_engine._ensure_vault_root` / `_checks/_vault_root.py` 锚点解析改调 nexusos-common，parents[N] 内联推导删除（不留双源）（task 1957）
- 五私人 check OSS 化处置：config_key_refs/codegraph_wrapper 插件化、persist_pool_ledger/doc_consistency 示例化、e2e_suite_evidence 配置迁 `$NEXUS_HOME/gate/e2e-paths.yaml`（无配置 = WARN skip 非 REJECT）（task 1959）
- 包更名 `nexus-gate` → `nexusos-gate`（目录/模块/pyproject/import 一次切净，无 shim）（task 1962）
- 版本 0.1.0 → 1.0.0，加入四包 lockstep 版本线（task 1963）

**Added**
- `nexusos.gate.checks` entry_points 插件协议：engine sorted(name) 稳定序加载 + `checks.d` 声明式启用（task 1959）

### nexusos-mem

**Changed**
- `_vault_env` 锚点解析改调 nexusos-common，parents[N] 内联推导删除（task 1957）
- 包更名 `nexus-mem` → `nexusos-mem`（目录/模块/pyproject/import 一次切净，无 shim）（task 1962）
- 版本 0.1.0 → 1.0.0，加入四包 lockstep 版本线（task 1963）

### nexusos-xdr

**Fixed**
- events 四事故修复：迁移函数 samefile+realpath 双防线防自引用环、guard 转 deque tail 读（治 readlines 整读假死）、消费者全盘点对账（monitor 三脚本 + mcp_server gate-rollup + checks.yaml 同修）（task 1958）

**Changed**
- `~/.hermes` 家族 15 条映射迁移 `.nexus`（env 单源 + 幂等 `nexus migrate`，三次重跑 0 MOVED）（task 1960）
- 包更名 `nexus-xdr` → `nexusos-xdr`（目录/模块/pyproject/import 一次切净，无 shim）（task 1962）
- 版本线与四包 lockstep 对齐 1.0.0（task 1963）

**Added**
- 运维 CLI `python3 -m nexusos_xdr.cli`（update/doctor）：PyPI 版本比对、layout_version 迁移提示、one-liner 升级；doctor 六服务/DB/守卫/版本一致性对账（`nexus` 命令名归 nexusos-common 向导 CLI——1963/1965 并行撞名收敛，xdr 不再声明 scripts 入口）（task 1963）
- tag 驱动发布流：GitHub Actions push tag `v*` → build → TestPyPI（正式 PyPI 为用户授意节点，不自动）（task 1963）

### 迁移说明

- 0.x 版本线无兼容义务（design §7：v1.0 全新起跳）。包名更名后旧名 `nexus-gate/mem/xdr` 不再发布。
