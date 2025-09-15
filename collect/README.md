# 数据收集标注工具

[English](README_en.md) | [中文](README.md)

## 数据收集

### 数据格式

通过人工/自动收集工具，收集每个action前的手机截图，并记录每个action的信息，并汇总到一个actions.json文件中。action格式如下：
```
{{
    "app_name": str
    "task_description": ["The description of the task list."],
    "action_count": "The count of the actions.",
    "actions": [
        {{
            "type": "The type of the action",
            "parameters": "etc.",  
        }},
        {{
            "type": "click",
            "position_x": "x-coordinate of click",
            "position_y": "y-coordinate of click action",
            "bounds": "the bound of the clicked element",
        }},
        {{
            "type": "swipe",
            "press_position_x": "x-coordinate of press",
            "press_position_y": "y-coordinate of press",
            "release_position_x": "x-coordinate of release",
            "release_position_y": "y-coordinate of release",
            "direction": "The direction of the user's swipe gesture. UP: swipe finger upward to scroll content up and reveal content below. DOWN: swipe finger downward to scroll content down and reveal content above. LEFT: swipe finger leftward to scroll content left. RIGHT: swipe finger rightward to scroll content right."
        }},
        {{
            "type": "input",
            "text": "The text to input",
        }},
        {{
            "type": "done"
        }},
        {{
            "type": "wait"
        }},
    ]
}}
```

### 手动数据收集

**启动服务器**
```bash
python -m collect.manual.server
```
启动成功后，访问 http://localhost:9000 进入Web操作界面。

**操作步骤**

1. **开始收集**：在Web界面点击 **开始收集** 按钮

2. **配置应用信息**：在弹出的 **应用信息配置** 窗口中填写：
- **应用名称**：如 "饿了么"、"微信"、"淘宝" 等
- **任务类型**：如 “tpye1”、 “tpye2” 等，具体参考收集任务文档

3. **输入任务描述**
   - 在 **任务描述** 窗口中详细描述当前要执行的具体任务
   - 确保描述清晰明确，便于后续数据分析和模型训练

4. **执行操作**
   - 在Web界面的手机截图上进行以下操作：
     - **点击操作**：直接点击截图上的目标位置
     - **滑动操作**：按住鼠标左键拖拽到目标位置后松开（注意保持在屏幕范围内）
     - **文本输入**：点击 **文本输入** 按钮，在弹出框中输入文本内容

5. **保存数据**
   - 完成一个任务序列后，根据需要选择：
     - **下一条数据**：继续收集同类型任务的更多数据样本
     - **结束收集**：完成当前收集会话并保存所有数据
     - **删除任务**：丢弃当前数据（用于处理错误操作或无效数据）

**数据存储格式**

收集的数据自动保存到 `collect/manual/data/` 目录，按以下层级结构组织：

```
data/
├── <应用名称>/
│   ├── <任务类型>/
│   │   ├── 1/
│   │   │   ├── 1.jpg          # 第1个操作前的截图
│   │   │   ├── 2.jpg          # 第2个操作前的截图
│   │   │   ├── ...
│   │   │   └── actions.json   # 操作记录和任务信息
│   │   ├── 2/
│   │   │   └── ...            # 第2条数据
│   │   └── ...
│   └── <其他任务类型>/
└── <其他应用名称>/
```

每个数据样本包含：
- **截图序列**：记录每个操作步骤前的界面状态
- **actions.json**：包含完整的操作序列、任务描述和应用信息

### 自动数据收集
先在 `collect/auto/task.json` 写入需要完成的任务列表，格式为字符串数组：
```json
[
    "在淘宝搜索iPhone手机",
    "在微信给张三发消息说你好",
    "在b站关注up主李四"
]
```

运行自动数据收集程序：
```bash
python -m collect.auto.server --model <模型名称> --api_key <API密钥> --base_url <API基础URL> [--max_steps <最大步数>]
```

**必需参数：**
- `--model`：LLM模型名称
- `--api_key`：API密钥
- `--base_url`：API基础URL

**可选参数：**
- `--max_steps`：每个任务的最大执行步数，默认为 15

**工作流程：**
1. 程序读取 `task.json` 中的任务列表
2. 对每个任务：
   - AI智能体根据任务描述自动选择并启动相应的应用
   - 自动执行操作序列（点击、滑动、输入等）
   - 每步操作前自动截图并记录操作信息
   - 达到最大步数或任务完成时停止
3. 自动保存数据到指定目录

**存储数据格式：**
- 原始日志数据存储在 `collect/auto/data_log/` 
- 转换后的标准格式数据存储在 `collect/auto/data/`
- 数据结构与手动收集保持一致，包含截图序列和 `actions.json` 文件

## 数据标注

数据标注模块将原始的操作数据转换为带有视觉标注的数据，为通用AI模型提供更丰富的上下文信息，使得其能够提供更加准确的reasoning。

### 视觉标注格式

**操作标注**
- 用户每个时间步的操作以 **红色字体** 标注在对应截图的顶部
- 辅助信息同时在截图中进行可视化标注：
  - **点击操作**：在操作位置标注 **红色圆圈**
  - **滑动操作**：用 **红色箭头** 标示从起始位置到结束位置的方向

**数据生成**
系统将标注后的截图序列和任务描述发送给大模型，生成 `react.json` 文件，包含推理过程和操作决策：

```json
[
    {
        "reasoning": "选择此操作类型的推理过程和原因",
        "function": {
            "name": "click",
            "parameters": {
                "target_element": "点击目标的高级语义描述"
            }
        }
    },
    {
        "reasoning": "滑动操作的推理过程",
        "function": {
            "name": "swipe",
            "parameters": {
                "direction": "UP, DOWN, LEFT, RIGHT"
            }
        }
    },
    {
        "reasoning": "文本输入的推理过程",
        "function": {
            "name": "input",
            "parameters": {
                "text": "要输入的文本内容"
            }
        }
    },
    {
        "reasoning": "任务完成的判断依据",
        "function": {
            "name": "done",
            "parameters": {}
        }
    },
    {
        "reasoning": "等待操作的原因说明",
        "function": {
            "name": "wait",
            "parameters": {}
        }
    }
]
```

### 自动标注执行

**启动命令**
```bash
python -m collect.annotate --data_path <数据路径> --model <模型名称> --api_key <API密钥> --base_url <API基础URL>
```

**参数说明**
- `--data_path`：原始轨迹数据存储路径（可选，默认为当前目录下的 `data` 目录）
- `--model`：大语言模型名称（必需）
- `--api_key`：模型服务API密钥（必需）
- `--base_url`：模型服务基础URL（必需）

**处理流程**
1. 读取原始数据目录中的截图序列和 `actions.json` 文件
2. 根据操作信息在截图上添加视觉标注
3. 将标注后的数据发送给大模型进行推理分析
4. 生成包含推理过程的 `react.json` 文件
5. 保存完整的标注数据集，用于后续模型训练

**数据存储格式**

收集的数据自动保存到对应目录，最小的子目录有如下结构：
```
dir/
├── 1.jpg          # 第1个操作前的截图
├── 2.jpg          # 第2个操作前的截图
├── ...
└── actions.json   # 操作记录和任务信息
└── react.json     # 标注数据
```

## 数据构建

数据构建模块将标注后的数据转换为适合模型训练的格式，支持监督微调（SFT）数据集的生成。

### 启动命令

```bash
python -m collect.construct_sft --data_path <原始数据路径> --ss_data_path <单步数据路径> --unexpected_img_path <意外图片路径> --out_path <输出路径> [--factor <缩放因子>] [--train_ratio <训练比例>]
```

### 参数说明

- `--data_path`：原始轨迹数据存储路径（默认：`data`）
- `--out_path`：训练数据集输出路径（默认：`output`）
- `--ss_data_path`：单步数据存储路径（默认：`ss_data`）
- `--unexpected_img_path`：意外图片数据路径（默认：`unexpected_img`）
- `--factor`：图片缩放因子，用于减小图片尺寸（默认：`0.5`）
- `--train_ratio`：训练集与验证集的划分比例（默认：`0.9`）

其中，`data_path`存放完整的、VLM标注后的操作轨迹，示例目录结构为：

```
data/
|-- some-subpath1
|   |-- 1.jpg
|   |-- 2.jpg
|   |-- ...
|   |-- actions.json
|   `-- react.json
`-- some-subpath2
    |-- 1.jpg
    |-- 2.jpg
    |-- ...
    |-- actions.json
    `-- react.json
```

`some-subpath`代表任意深度的路径，可根据数据组织的需要任意指定，一个最深的子目录包含 `n` 个屏幕截图，以及长度为 `n` 的动作列表 `react.json` + `actions.json`，动作和截图按照下标一一对应，且这 `n` 个截图-动作对构成一个任务的完整操作轨迹。

`ss_data_path`存放手动收集的单步动作数据，示例目录结构为：

```
ss_data/
|-- decider
|   `-- some-subpath
|       |-- 1.jpg
|       |-- 2.jpg
|       |-- ...
|       |-- react.json
|       `-- tasks.json
`-- grounder
    `-- some-subpath
        |-- 1.jpg
        |-- 2.jpg
        |-- ...
        `-- react.json
```

`ss_data_path`内必须仅包含 `decider` 和 `grounder` 作为一级目录，分别代表用于训练 `decider` 和 `grounder` 模型的单步动作数据。`some-subpath` 的深度和命名均任意，一个最深的子目录包含 `n` 个屏幕截图，长度为 `n` 的动作列表 `react.json`，动作和截图按照下标一一对应，且这 `n` 个截图-动作对均为单步操作，彼此之间没有联系。子目录不包含 `actions.json`。

* `ss_data/decider` 下的子目录还包含一个长度任意的任务列表 `tasks.json` ，构建训练数据集时会为每个截图-动作对，从列表中随机采样一个任务，用于填充训练时模型输入提示词中的任务描述部分
* `ss_data/grounder` 下的子目录下的 `react.json` 中，每一个动作应为`click`，且包含一个额外的 `bbox` 字段，代表此次点击的元素的边界框（绝对坐标），例如：

```json
[
    {
        "reasoning": "...",
        "function": {
            "name": "click",
            "parameters": {
                "target_element": "..."
            }
        },
        "bbox": [100, 200, 300, 400]
    }
]
```