# Auto Search 使用说明

`runner/mobiagent/auto-search.py` 用于 App 自动探索（DFS + 回溯）：

1. Explorer 模型生成当前页面候选单步任务（Top-H）。
2. Decider 模型把单步任务转为可执行动作（click/click_input/input/swipe/wait）。
3. 执行动作并递归探索到指定深度。
4. 在叶子路径保存完整 path 结果，并追加 `done` 作为路径终止标记。

## 1. 前置要求

- 设备已连接（Android/Harmony）
- Decider 服务已启动（OpenAI 兼容接口）
- OpenRouter API Key 可用
- 已安装依赖（如 `openai`、`uiautomator2`）

## 2. 参数说明

- `--app_name`：目标 App 名称（必须在设备映射表中）
- `--depth`：探索深度，必须 `> 0`
- `--breadth`：每层候选数，必须 `> 0`
- `--device`：`Android` 或 `Harmony`
- `--service_ip`：Decider 服务 IP
- `--decider_port`：Decider 端口
- `--openrouter_base_url`：默认 `https://openrouter.ai/api/v1`
- `--openrouter_api_key`：OpenRouter Key（必填）
- `--explorer_model`：Explorer 模型名，默认 `google/gemini-3-flash-preview`
- `--use_qwen3`：`on` / `off`，是否使用 qwen3 坐标换算
- `--allow_hierarchy_text_decider`：`on` / `off`，是否允许用 hierarchy 文本直接定位点击 bbox
- `--data_dir`：输出目录（可选，不传则自动按时间戳创建）

## 3. 运行方式

推荐以模块方式运行：

```bash
python -m runner.mobiagent.auto-search \
  --app_name "美团" \
  --depth 3 \
  --breadth 4 \
  --device Android \
  --service_ip localhost \
  --decider_port 8000 \
  --openrouter_api_key "$OPENROUTER_API_KEY" \
  --explorer_model "google/gemini-3-flash-preview" \
  --use_qwen3 on \
  --allow_hierarchy_text_decider on
```

也可使用仓库根目录模板脚本：`auto-search-run-template.sh`。

## 4. Decider 消息与 done 规则

- auto-search 使用 `build_auto_decider_messages` 构造 Decider 消息。
- system prompt 基于 `prompts/e2e_qwen3_system.md`，并在加载时移除 `done` 动作空间与响应格式里的 `done` 候选。
- 运行中不接受 Decider 输出 `done`（会重试/报错），即探索步骤本身不执行 `done`。
- `done` 只在 **path 落盘** 时自动追加，用于形成完整轨迹。

## 5. 输出目录结构

默认输出目录：

- `runner/mobiagent/data-auto-search/<app_name>/<timestamp>/`

子目录：

- `steps/step_0001`, `steps/step_0002`, ...
- `paths/path_0001`, `paths/path_0002`, ...

### 5.1 steps（单步原子结果）

每个 `step_xxxx/` 保存一次候选执行结果，包含：

- `<step_idx>.jpg`（执行该步前的截图）
- `<step_idx>.xml`/`<step_idx>.json`（层级）
- `<step_idx>_highlighted.jpg`、`<step_idx>_bounds.jpg`、`<step_idx>_click_point.jpg` 或 `<step_idx>_swipe.jpg`（可视化）
- `actions.json`、`react.json`（单步）

### 5.2 paths（完整 DFS 路径）

每个 `path_xxxx/` 保存一条完整路径：

- 从 `steps` 复制并重编号后的截图/层级/标注文件（`1.*`, `2.*`, ...）
- `actions.json`、`react.json`（整条路径）

关键约束：

- `actions.json` 与 `react.json` 的条数一致。
- path 末尾会追加 `done`（`status=success`）及对应 reasoning。
- 为保证多模态样本对齐，`done` 也会有对应观测：额外保存 `n.jpg` 与 `n.xml/json`（`n` 为追加 done 后的最终步数）。

## 6. 推荐实践

- 先小规模验证：`depth=2`, `breadth=2`
- 再逐步增大参数
- 若结果异常，优先检查 `OPENROUTER_API_KEY` 是否有效
- 若结果异常，优先检查 `app_name` 是否存在于设备映射表
- 若结果异常，优先检查 Decider 服务连通性与端口是否正确
