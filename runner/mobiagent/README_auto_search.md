# Auto Search 使用说明

本目录新增了自动探索脚本：`auto-search.py`。

该脚本用于：

1. 指定目标 App 名称。
2. 指定探索深度 `D` 与探索广度 `H`。
3. 调用远端通用大模型（如 Gemini）生成当前页面 Top-H 单步任务。
4. 调用 MobiAgent e2e decider 执行候选任务并进入下一层。
5. 到达最大深度后自动回退并继续探索未执行候选（DFS + 回溯）。
6. 保存截图、层级、动作与推理 JSON。

---

## 1. 前置要求

- 设备已连接（Android/Harmony）
- Decider 服务已启动（OpenAI 兼容接口）
- OpenRouter API Key 可用
- 已安装项目依赖（含 `openai`、`uiautomator2` 等）

---

## 2. 关键参数

- `--app_name`: 目标 App 名称（需在设备映射表中存在）
- `--depth`: 探索深度 D，必须 > 0
- `--breadth`: 每层候选数量 H，必须 > 0
- `--device`: `Android` 或 `Harmony`
- `--service_ip`: decider 服务 IP（模型由服务端决定）
- `--decider_port`: decider 端口
- `--openrouter_api_key`: OpenRouter Key（必填）
- `--openrouter_base_url`: 默认为 `https://openrouter.ai/api/v1`
- `--explorer_model`: 默认为 `google/gemini-2.5-flash`
- `--use_qwen3`: `on` / `off`
- `--data_dir`: 输出目录（可选）

---

## 3. 快速运行示例

```bash
python runner/mobiagent/auto-search.py \
  --app_name "美团" \
  --depth 3 \
  --breadth 4 \
  --device Android \
  --service_ip localhost \
  --decider_port 8000 \
  --openrouter_api_key "$OPENROUTER_API_KEY" \
  --explorer_model "google/gemini-2.5-flash" \
  --use_qwen3 on
```

---

## 4. 输出数据格式

默认输出到：

- `runner/mobiagent/data-auto-search/<app_name>/<timestamp>/`

目录内包括：

- `1.jpg`, `2.jpg`, ...
- `1.xml` / `1.json`（UI hierarchy，Android/Harmony）
- `actions.json`
- `react.json`

其中：

- `actions.json`：记录执行动作轨迹（点击/输入/滑动/等待等）
- `react.json`：记录 decider 每步推理与动作参数

---

## 5. 推荐实践

- 先用小参数验证链路：`D=2`, `H=2`
- 再逐步增大：如 `D=4`, `H=5`
- 若探索结果为空，优先检查：
  - OpenRouter Key 是否有效
  - `--decider_model` 是否正确
  - App 名是否存在于设备映射表
