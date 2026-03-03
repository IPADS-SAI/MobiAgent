# UI Semantic Boxer 中文说明

`ui_semantic_boxer.py` 用于从真实设备页面生成单步可点击元素语义数据。  
核心思路是：**hierarchy 提供可点击框**，**VLM 提供语义理解与歧义修正**。

输出数据包括：
- `ui_semantics.json`
- 每个目标目录下的 `actions.json` / `react.json`
- VLM 过程调试文件

## 1. 简介

该工具用于自动构建 UI 点击样本：
- 基于 hierarchy 抽取可点击元素及其 bbox
- 结合 VLM 生成/复核 `text` 与 `ui_kind`
- 自动生成可执行的单步点击任务描述

## 2. 快速开始

在仓库根目录执行：

```bash
bash ./run-ui-semantic-boxer.sh
```

主要配置脚本：
- [`run-ui-semantic-boxer.sh`](/home/zhaoxi/ipads/llm-agent/test/MobiAgent/run-ui-semantic-boxer.sh)

核心程序：
- [`collect/auto/ui_semantic_boxer.py`](/home/zhaoxi/ipads/llm-agent/test/MobiAgent/collect/auto/ui_semantic_boxer.py)

## 3. 处理流程

1. 采集页面截图与层级树（Android/Harmony）。
2. 从 hierarchy 提取可点击框并做去重/筛选。
3. 生成元素文本 `text`：
- 默认：优先 hierarchy 文本，缺失时用 VLM。
- `VLM_TEXT_ONLY=on`：所有可点击元素都由 VLM 生成文本。
4. 生成/复核元素类型 `ui_kind`（规则优先 + 可选整页 VLM 复核）。
5. 执行歧义处理（可选）：
- VLM 文本精炼（减少“状态标识/空白区域”等泛化标签）
- 重名目标短语去歧义
- 最终整页任务描述歧义复核
6. 导出 `ui_semantics.json` 与每个目标的 `actions.json` / `react.json`。

## 4. 关键参数

以下参数来自 `run-ui-semantic-boxer.sh`：

- `USE_VLM`：是否启用 VLM。
- `VLM_TEXT_ONLY`：开启后不直接用 hierarchy 文本，所有元素文本均由 VLM 生成。
- `ENABLE_KIND_VLM`：是否启用整页 `ui_kind` 复核。
- `ENABLE_VLM_TEXT_REFINE`：是否对 VLM 文本做二次精炼，减少泛化和歧义。
- `ENABLE_DUPLICATE_DESC_VLM`：是否对同名目标进行短语级去歧义。
- `ENABLE_TASK_DESC_VLM_REVIEW`：是否在最终阶段做整页 `task_description` 歧义检查。

常见预算参数：
- `MAX_VLM_CALLS`
- `MAX_DUPLICATE_DESC_VLM_CALLS`
- `MAX_VLM_TEXT_REFINE_CALLS`

## 5. 输出目录结构

输出目录示例：

`collect/auto/ui-semantic-output/<timestamp>/`

典型文件：
- `screenshot.jpg`
- `annotated.jpg`
- `hierarchy.xml` 或 `hierarchy.json`
- `ui_semantics.json`
- `<id>/1.jpg`
- `<id>/1_annotated.jpg`
- `<id>/actions.json`
- `<id>/react.json`

## 6. 调试文件说明

用于排查 VLM 行为与合并决策：

- `ui_kind_vlm_review_debug.json`  
  整页 `ui_kind` 复核请求、解析与规则合并决策。
- `vlm_text_refine_debug.json`  
  `source=vlm` 的文本是否被精炼、精炼前后差异与预算跳过原因。
- `task_description_ambiguity_review_debug.json`  
  最终任务描述歧义检查结果、冲突目标与重写记录。

## 7. 注意事项

- 当 `USE_VLM=on` 时必须设置 `OPENROUTER_API_KEY`。
- `VLM_TEXT_ONLY=on` 依赖 `USE_VLM=on`。
- `MAX_ITEMS` 越大，时延与 token 成本越高。
- 如需降低成本，可关闭部分阶段（如 `ENABLE_TASK_DESC_VLM_REVIEW` 或 `ENABLE_VLM_TEXT_REFINE`）。
