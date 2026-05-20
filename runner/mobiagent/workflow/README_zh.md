# Workflow 使用说明

这个目录提供一套独立于现有 `python -m runner.mobiagent.mobiagent` 单任务入口的 workflow 运行能力。它通过**新增代码**复用 MobiAgent 已有的 planner、GUI 执行、设备控制和模型客户端，但不会修改原有任务执行路径。

## 1. 设计目标

workflow 适合描述一串按顺序执行的自动化步骤，每个步骤都属于下面六类之一：

1. `gui_task`：执行一次由模型推理驱动的 GUI 任务，例如“点击 xxx 按钮”“搜索 xxx 博主”。
2. `gui_action`：执行一次显式 GUI 操作，例如固定坐标点击、固定方向滑动、输入文本、截图、等待。
3. `command`：执行一次 shell / adb / hdc / python / bash 等命令。
4. `tool`：调用一个内置工具。首版内置工具是 `vlm_qa`，用于根据图片回答问题或做总结，并保留了后续扩展接口。
5. `loop`：按给定次数重复执行一组子步骤，适合“截图一次、下滑一次，再重复”的场景。
6. `if`：按条件决定执行哪一组子步骤，适合“如果不是最后一次循环就继续滑动”的场景。

## 2. 启动方式

使用独立入口：

```bash
python -m runner.mobiagent.workflow_runner \
  --workflow_file runner/mobiagent/workflow/examples/00_command_only_minimal.json \
  --service_ip xxx \
  --decider_port 7003 \
  --grounder_port 7003 \
  --planner_port 7002 \
  --device Harmony

python -m runner.mobiagent.workflow_runner --workflow_file ./runner/mobiagent/workflow/examples/01_basic_gui_task.json --service_ip xxxx --decider_port 7003 --grounder_port 7003 --planner_port 7002 --device Harmony --output_dir runner/mobiagent/workflow/test-runs
```

常用参数：

- `--workflow_file`：workflow JSON 文件路径。
- `--service_ip` / `--decider_port` / `--grounder_port` / `--planner_port`：复用现有模型服务。
- `--device`：workflow 默认设备类型，支持 `Android` 或 `Harmony`。
- `--user_profile on|off`：是否启用 user profile 记忆能力，供 `gui_task` 节点复用。
- `--use_graphrag on|off`：是否为 user profile 启用 GraphRAG。
- `--use_qwen3 on|off`：默认是否在 `gui_task` 节点使用 Qwen3。
- `--e2e`：默认是否在 `gui_task` 节点使用 e2e 模式。
- `--output_dir`：workflow 运行产物目录。
- `--accept_planner_changes on|off`：默认是否自动接受 `gui_task` 中 planner 改写的任务描述。

## 3. 输入格式

workflow 输入格式使用 JSON，与现有 `runner/mobiagent/task.json` 风格保持一致。

顶层字段：

- `metadata`：workflow 名称、描述等元信息。
- `defaults`：默认执行参数，例如默认设备、默认模型开关。
- `context`：全局上下文变量，可被后续步骤引用。
- `steps`：按顺序执行的步骤列表。

每个步骤的公共字段：

- `id`：步骤唯一标识，推荐使用从 `1` 开始递增的数字；程序内部会自动转成字符串处理。
- `type`：步骤类型，支持 `gui_task`、`gui_action`、`command`、`tool`、`loop`、`if`。
- `name`：步骤说明，可选；不提供也可以正常运行。
- `on_error`：失败后的处理方式，支持 `stop` 或 `continue`，默认 `stop`。

## 4. 节点类型说明

### 4.1 `gui_task`

`gui_task` 在 workflow 中是一个特殊节点：

1. 只有 `id=1` 的 `gui_task` 会调用 planner，并且只调用 planner。它的作用是根据任务描述选择 app / package，并启动对应 app。
2. 后续 `gui_task` 步骤不会再调用 planner，而是直接复用现有 MobiAgent 的 decider + grounder + device 执行链。
3. 每个后续 `gui_task` 会持续调用 decider，直到 decider 返回 `done`，此时该步骤结束并进入下一个步骤。
4. 下一个 `gui_task` 会拿自己的 `task_description` 和当前页面截图继续调用 decider，因此它是“在当前 app / 当前页面上下文中继续做下一段 GUI 任务”。

主要字段：

- `task_description`：自然语言任务描述。
- `device`：可选，覆盖 workflow 默认设备。
- `use_experience`：可选，是否为该步启用 experience。
- `use_qwen3`：可选，是否为该步启用 Qwen3。
- `use_e2e`：可选，是否为该步启用 e2e。
- `use_graphrag`：可选，是否为该步启用 GraphRAG。
- `accept_planner_changes`：可选，是否自动接受 planner 改写。

示例：

```json
{
  "id": 1,
  "type": "gui_task",
  "task_description": "打开微博"
}
```

### 4.2 `gui_action`

`gui_action` 直接调用设备抽象，适合做确定性的页面操作。

它可以覆盖你提到的这些场景：

1. 点击一个具体坐标：使用 `action="click"`，配合 `x` 和 `y`。
2. 滑动一定距离的屏幕：使用 `action="swipe_with_coords"`，明确给出起点和终点坐标；如果只是做固定方向翻页，也可以用 `action="swipe"`。
3. 在输入框已经激活的情况下输入文字：先用一次 `click` 激活输入框，再用 `action="input"` 输入文本。

需要注意的是，`gui_action` 和 decider 的动作空间不是完全一一对应的。当前它们是**部分重合**：

- 已支持：`click`、`input`、`swipe`、`keyevent`、`screenshot`、等待、启停 app
- 通过组合动作可覆盖：先点击输入框，再输入文字
- 还没有直接暴露成同名 action 的：例如 `click_input`、`press_back`、`press_home`、`wait` 这种 decider 语义级动作

如果你只是想手工精确控制页面，这一版 `gui_action` 已经足够实现“点坐标、滑动距离、激活输入框后输入文字”这类需求。

当前支持的 `action`：

- `click`：固定坐标点击，需要 `x`、`y`
- `input`：输入文本，需要 `text`
- `swipe`：按方向滑动，需要 `direction`，取值如 `up/down/left/right`
- `swipe_with_coords`：按绝对坐标滑动，需要 `start_x`、`start_y`、`end_x`、`end_y`
- `keyevent`：按键事件，需要 `key`
- `sleep`：等待，需要 `seconds`
- `screenshot`：截图，可选 `file_name`
- `app_start`：启动包名，需要 `package_name`
- `app_stop`：停止包名，需要 `package_name`

示例：

```json
{
  "id": 1,
  "type": "gui_action",
  "action": "screenshot",
  "file_name": "page.jpg"
}
```

更完整的手工操作示例见：

- `runner/mobiagent/workflow/examples/04_gui_action_touch_swipe_input.json`

这个示例演示了：

1. 第 1 步先用 `gui_task` 打开微博
2. 第 2 步点击一个固定坐标
3. 第 3 步按固定起止点上滑一段距离
4. 第 4 步点击搜索框坐标
5. 第 5 步在输入框已激活的情况下输入“雷军”
6. 第 6 步发送一个按键事件
7. 第 7 步截图保存结果

其中最关键的三种写法分别是：

点击具体坐标：

```json
{
  "id": 2,
  "type": "gui_action",
  "action": "click",
  "x": 540,
  "y": 2080
}
```

按固定距离滑动屏幕：

```json
{
  "id": 3,
  "type": "gui_action",
  "action": "swipe_with_coords",
  "start_x": 540,
  "start_y": 1700,
  "end_x": 540,
  "end_y": 900
}
```

激活输入框后输入文字：

```json
{
  "id": 4,
  "type": "gui_action",
  "action": "click",
  "x": 540,
  "y": 240
}
```

```json
{
  "id": 5,
  "type": "gui_action",
  "action": "input",
  "text": "雷军"
}
```

### 4.3 `command`

`command` 用于执行脚本或命令行操作。

支持两种形式：

1. 直接写 `command` 字符串，按 shell 执行。
2. 写 `program` + `args`，按非 shell 方式执行。

常用字段：

- `command`：完整命令字符串。
- `program` / `args`：程序名和参数列表。
- `cwd`：命令工作目录。相对路径会相对于 workflow 文件所在目录解析。
- `env`：附加环境变量。
- `timeout_sec`：超时时间。
- `capture_output`：是否捕获 stdout/stderr，默认 `true`。
- `allow_failure`：是否允许失败继续执行，默认 `false`。

示例：

```json
{
  "id": 1,
  "type": "command",
  "command": "adb shell wm size"
}
```

### 4.4 `tool`

`tool` 节点通过内置 registry 调用可扩展工具。

首版内置工具：

- `vlm_qa`：复用 planner client 执行图片问答或图片总结。

`vlm_qa` 支持字段：

- `question`：问题文本。
- `image`：图片路径。
- `mode`：`answer` 或 `summary`。
- `json_schema`：可选，要求模型按结构化 JSON 返回结果。适合返回 `summary` 之外的判断字段，例如 `contains_today_chat`、`is_today_order` 等。

当 `vlm_qa` 以 `mode="summary"` 运行时，workflow 会自动把“总结文本本身”追加保存到输出目录下的 `daily-log/` 中，不会把其他返回字段写进去。

日志文件名规则如下：

- 基础目录：`<output_dir>/daily-log/YYYY-MM-DD/`，其中 `YYYY-MM-DD` 表示本次 workflow 执行日期。
- 文件名：`<app.package_name>__<workflow_json文件名去掉.json后的名字>__<context中的值按顺序拼接>.md`
- 如果没有 `context`，则文件名中不会追加 `context` 部分。
- 每次追加 summary 时，都会自动在前面加递增序号，例如 `1.`、`2.`、`3.`，用于区分同一天内的多次总结。
- 每个 md 顶部都会带一个 metadata 头，里面包含 workflow JSON 顶层的 `metadata` 信息，以及当前最新条目的 `latest_entry_index`，后续追加 summary 时会基于这个值继续递增。

例如，当前包名是 `com.tencent.mm`，workflow 文件是 `01_basic_gui_task_weixin_v2.json`，`context` 中有 `contact_name: 小赵`，则文件名会类似于：

```text
daily-log/2026-05-20/com.tencent.mm__01_basic_gui_task_weixin_v2__小赵.md
```

md 文件头部会类似于：

```text
<!-- DAILY_LOG_METADATA
{
  "workflow_metadata": {
    "name": "basic-gui-task",
    "description": "收集指定微信用户的今天聊天记录"
  },
  "latest_entry_index": 6
}
-->
```

如果 `vlm_qa` 使用了 `json_schema`，并且其中包含 `summary` 字段，则写入 md 文件的内容只会取 `structured_output.summary`；否则写入原始 summary 文本响应。

示例：

```json
{
  "id": 2,
  "type": "tool",
  "tool_name": "vlm_qa",
  "inputs": {
    "mode": "summary",
    "image": "${steps.1.output.image_path}",
    "question": "请总结这张截图的主要内容。"
  }
}
```

如果你希望 `vlm_qa` 除了总结之外，还返回可供后续逻辑判断的字段，可以这样写：

```json
{
  "id": 2,
  "type": "tool",
  "tool_name": "vlm_qa",
  "inputs": {
    "mode": "summary",
    "image": "${steps.1.output.image_path}",
    "question": "请判断当前聊天记录是否仍然属于今天，并总结内容。",
    "json_schema": {
      "summary": {
        "type": "string",
        "description": "当前截图内容总结"
      },
      "contains_today_chat": {
        "type": "boolean",
        "description": "当前截图中的聊天记录是否仍然属于今天"
      }
    }
  }
}
```

这时工具输出中会多一个：

- `${steps.2.output.structured_output.summary}`
- `${steps.2.output.structured_output.contains_today_chat}`

### 4.5 `loop`

`loop` 用于重复执行一组子步骤，当前支持按固定次数循环。

主要字段：

- `times`：循环次数。
- `max_times`：循环最多执行多少次。适合和 `break_if` 一起使用。
- `break_if`：可选，循环体每轮执行完后进行判断；如果条件成立，则提前停止循环。
- `steps`：循环体中的子步骤列表。

循环体中可用变量：

- `${loop.index}`：当前轮次，从 `1` 开始。
- `${loop.index0}`：当前轮次，从 `0` 开始。
- `${loop.count}`：总轮次。
- `${loop.first}`：是否第一轮，布尔值。
- `${loop.last}`：是否最后一轮，布尔值。

示例：

```json
{
  "id": 3,
  "type": "loop",
  "times": 3,
  "steps": [
    {
      "id": 1,
      "type": "gui_action",
      "action": "screenshot",
      "file_name": "chat_${loop.index}.jpg"
    }
  ]
}
```

### 4.6 `if`

`if` 用于条件分支执行。

主要字段：

- `condition`：条件表达式对象。
- `then_steps`：条件成立时执行的子步骤列表。
- `else_steps`：条件不成立时执行的子步骤列表，可为空。

`condition` 当前支持的字段：

- `left`
- `operator`
- `right`

当前支持的 `operator`：

- `==`
- `!=`
- `contains`
- `not_contains`
- `>`
- `>=`
- `<`
- `<=`
- `in`
- `not_in`

示例：

```json
{
  "id": 3,
  "type": "if",
  "condition": {
    "left": "${loop.last}",
    "operator": "==",
    "right": false
  },
  "then_steps": [
    {
      "id": 1,
      "type": "gui_action",
      "action": "swipe_with_coords",
      "start_x": 540,
      "start_y": 1700,
      "end_x": 540,
      "end_y": 900
    }
  ],
  "else_steps": []
}
```

## 5. 变量引用

workflow 支持在字符串中引用运行时变量。当前支持：

- `${context.xxx}`：引用顶层 `context` 中的变量。
- `${loop.xxx}`：引用当前循环轮次变量，例如 `${loop.index}`、`${loop.last}`。
- `${run.dir}`：引用当前 workflow 运行目录。
- `${run.workflow_file}`：引用 workflow 文件路径。
- `${run.workflow_dir}`：引用 workflow 文件所在目录。
- `${steps.step_id.output.xxx}`：引用前序步骤输出，其中 `step_id` 使用当前作用域内步骤的 `id`，例如 `${steps.1.output.image_path}`。在 `loop` / `if` 内部，这个引用会优先指向当前循环体或当前分支中的同级步骤。
- `${steps.step_id.status}`：引用前序步骤状态。

示例：

```json
{
  "command": "echo '${steps.1.output.image_path}'"
}
```

## 6. 样例文件

本目录附带以下样例：

- `examples/00_command_only_minimal.json`：最小可运行样例，只执行一条命令。
- `examples/01_basic_gui_task.json`：执行一个自然语言 GUI 任务。
- `examples/01_basic_gui_task_weixin.json`：打开微信聊天界面，然后通过 `loop` + `if` 连续截图三次并逐次下滑。
- `examples/01_basic_gui_task_weixin_v2.json`：打开微信聊天界面，通过 `vlm_qa + json_schema` 判断当前聊天记录是否仍然属于今天；如果已经翻到更早聊天则停止，同时最多只翻 10 次。
- `examples/02_gui_action_and_command.json`：组合 GUI action 与 command。
- `examples/03_vlm_summary.json`：截图后调用 `vlm_qa` 做总结。
- `examples/04_gui_action_touch_swipe_input.json`：点击坐标、滑动屏幕、激活输入框后输入文字。
- `examples/05_loop_and_if_minimal.json`：不依赖设备的最小 loop/if 示例，适合先验证控制流逻辑。

## 7. 运行产物

每次 workflow 执行会在输出目录下创建一个独立运行目录，包含：

- `steps/<step_id>/`：每个步骤的单独产物目录，例如 `steps/1/`、`steps/2/`。
- `run_summary.json`：整次 workflow 的运行摘要、每步状态和输出。

命令节点默认会额外保存：

- `stdout.txt`
- `stderr.txt`

## 8. 扩展方式

如果你后续要加新的工具能力，可以在 `runner/mobiagent/workflow/tools.py` 中新增一个实现 `WorkflowTool` 的类，并注册到 `ToolRegistry`。

建议后续扩展方向：

1. OCR 工具
2. HTTP API 调用工具
3. 业务函数工具
4. 条件判断与分支执行

## 9. 当前限制

当前版本是首版实现，有以下限制：

1. 只支持串行执行步骤，不支持 DAG 并发调度。
2. 变量引用是轻量字符串替换，不是完整模板引擎。
3. `tool` 首版只内置 `vlm_qa`。
4. `gui_task` 直接复用现有 MobiAgent 单任务链路，因此它的行为边界与现有单任务模式一致。
