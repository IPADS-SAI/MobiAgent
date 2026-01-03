# 统一 GUI Agent Runner 框架使用指南

这是一个统一的 GUI Agent 任务执行框架，支持接入多种模型（MobiAgent, UI-TARS, Qwen, AutoGLM 等）并执行移动端自动化任务。该框架提供了统一的接口、参数管理、动作规范化以及执行过程可视化功能。

##  快速开始

### 1. 执行单个任务

可以通过 `run.py` 直接执行单条任务指令。建议参考 `examples/` 目录下的脚本。

```bash
# 使用 MobiAgent (HarmonyOS)
python run.py --provider mobiagent --task "在淘宝上搜索电动牙刷" --device-type Harmony

# 使用 UI-TARS (带可视化绘制)
python run.py --provider uitars --task "在微信发送消息给张三，说我的AI助手很厉害" --draw --api-base http://localhost:8000/v1

# 使用 Qwen VLM
python run.py --provider qwen --task "在微博查看热搜" --api-base http://localhost:8080/v1
```

### 2. 批量执行任务

支持从 JSON 文件中读取任务列表进行批量测试。

```bash
# 执行通用任务列表
python run.py --provider mobiagent --task-file task.json

# 执行 MobiFlow 格式任务列表
python run.py --provider uitars --task-file task_mobiflow.json
```

---

## 📂 项目结构

```text
runner/
├── run.py                 # 统一执行入口，处理核心命令行逻辑
├── base_task.py           # 基础任务抽象类，定义统一接口和动作映射
├── task_manager.py        # 任务管理器，负责 Provider 实例化
├── device.py              # 设备抽象层 (支持 Android 和 Harmony)
├── providers/             # 模型适配器目录
│   ├── mobiagent/         # MobiAgent 相关适配逻辑
│   ├── uitars/            # UI-TARS 适配器
│   ├── qwen/              # Qwen-VL 适配器
│   └── autoglm/           # AutoGLM 适配器
├── examples/              # 各模型的启动脚本示例
└── results/               # 默认执行结果输出路径
```

---

## 🛠️ 参数说明

### 1. 基础参数
| 参数 | 说明 | 默认值 | 可选值 |
| :--- | :--- | :--- | :--- |
| `--provider` | 模型提供者 | `mobiagent` | `uitars`, `qwen(或其他任意VLM模型)`, `autoglm` |
| `--device-type`| 设备系统类型 | `Android` | `Android`, `Harmony` |
| `--device-id` | 设备 ID 或 IP | None | - |
| `--max-steps` | 单个任务最大执行步数 | 40 | - |
| `--log-level` | 日志详细程度 | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `--draw` | 是否在截图上绘制操作可视化 | `False` | - |

### 2. 任务参数
| 参数 | 说明 |
| :--- | :--- |
| `--task` | 单个任务描述字符串 |
| `--task-file` | 任务列表 JSON 文件路径 |
| `--output-dir` | 结果输出根目录 (默认 `results`) |

### 3. 通用模型参数
框架对常见的模型调用参数进行了统一，各 Provider 会映射到各自的底层实现。
| 参数 | 说明 |
| :--- | :--- |
| `--api-base` | 模型服务基础 URL |
| `--api-key` | API 密钥 |
| `--model` | 模型名称 |
| `--temperature` | 生成温度 |

### 4. 特定 Provider 参数
- **MobiAgent**:
  - `--service-ip`: 服务主机 IP
  - `--decider-port`: Decider 端口 (默认 8000)
  - `--grounder-port`: Grounder 端口 (默认 8001)
  - `--planner-port`: Planner 端口 (默认 8002)
  - `--enable-planning`: 启用任务规划模式
  - `--use-e2e`: 启用端到端执行模式
- **UI-TARS**:
  - `--step-delay`: 每步操作间的延迟 (秒)

---

## 📊 输出结果

每个任务执行后会在 `output-dir` 下生成以时间戳命名的目录，包含：

- `1.jpg, 2.jpg...`: 每一步的屏幕截图。
- `1_draw.jpg...`: 标注了动作（如点击红点、滑动箭头）的可视化截图（需开启 `--draw`）。
- `1.xml` 或 `1.json`: 每一步的 UI 结构树。
- `actions.json`: 完整的动作序列记录，包含坐标和参数。
- `react.json`: 模型每一步推导的思维过程 (Reasoning)。

---

## 🧩 接入新模型 (Provider)

接入新模型只需遵循以下步骤：

1. **创建适配器**:
   在 `providers/` 下创建子目录，实现一个继承自 `base_task.py:BaseTask` 的类。
   - 实现 `execute_step(self, step_index)`：接收当前步数，返回动作序列列表。
   - 或者实现 `_execute_task(self)`：如果你希望自己接管整个任务循环。

2. **动作规范化**:
   在 `BaseTask` 中已定义了统一的动作映射表 `ACTION_TYPE_ALIASES`。如果新模型的动作名称不同，只需在 `base_task.py` 中添加映射即可，例如将 `tap` 映射为 `click`。

3. **注册 Provider**:
   在 `task_manager.py` 的 `_get_task_map` 方法中添加你的模型名称与类的映射。

4. **添加配置**:
   在 `run.py` 的 `PROVIDER_DEFAULTS` 中添加该模型的默认 API 地址和模型名。

---

## 🙏 感谢

感谢 **[23japhone](https://github.com/23japhone)** 的测试框架参考。
