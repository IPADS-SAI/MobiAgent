# UI-TARS 批量任务执行器使用指南

## 快速开始

### 1. 环境准备

确保已安装必要的依赖：
```bash
pip install -r requirements.txt
```

确保Android设备已连接并启用USB调试：
```bash
adb devices
```

可选：安装 `ADBKeyboard.apk`（用于发送文本输入）
```bash
adb install -r ADBKeyboard.apk
```

### 2. 脚本说明

| 脚本文件 | 功能说明 | 适用场景 |
|---------|----------|----------|
| `batch_task_executor.py` | 执行auto_task-3.json中的所有任务 | 生产环境批量执行 |
| `test_batch_executor.py` | 执行少量测试任务 | 测试验证 |
| `test_single_task.py` | 执行单个自定义任务 | 调试和测试 |
| `quick_start.py` | 原有的快速启动示例 | 单任务快速测试 |

### 3. 使用步骤

#### 方案一：测试单个任务（推荐新手）
```bash
python3 test_single_task.py
```
1. 输入模型服务地址
2. 选择应用（bilibili、淘宝等）
3. 输入任务描述
4. 确认执行

#### 方案二：测试少量任务（推荐调试）
```bash
python3 test_batch_executor.py
```
- 自动执行bilibili的前2个type1任务
- 验证批量执行流程是否正常

#### 方案三：执行所有任务（生产环境）
```bash
python3 batch_task_executor.py
```
- 执行auto_task.json中的所有210个任务
- 需要较长时间完成

## 配置说明

### 模型服务配置
- 默认地址：http://192.168.12.152:8000/v1
- 模型名称：UI-TARS-7B-SFT
- 可在运行时修改地址

### 执行参数
- 最大步数：30步
- 步骤延迟：2秒
- 语言：中文
- 任务间隔：5秒

### 数据保存
执行结果保存在以下目录结构：
```
data_example/          # 生产数据
test_data_example/     # 测试数据
├── bilibili/
│   ├── type1/
│   │   ├── 1/
│   │   │   ├── task_data.json      # 任务执行数据（原格式）
│   │   │   ├── actions.json        # 操作记录（参考淘宝格式）
│   │   │   ├── 1/screenshot_1.jpg  # 第1步截图
│   │   │   ├── 1/hierarchy_1.xml   # 第1步XML
│   │   │   ├── 2/screenshot_2.jpg  # 第2步截图
│   │   │   ├── 2/hierarchy_2.xml   # 第2步XML
│   │   │   └── ...
│   │   ├── 2/
│   │   └── 3/
│   └── type2/
└── ...
```

### 数据格式说明

#### task_data.json格式（保持原有格式）
```json
{
    "task_description": "在B站搜一下\"元神4.8版本更新\"",
    "app_name": "bilibili",
    "task_type": "type1",
    "task_index": 1,
    "package_name": "tv.danmaku.bili",
    "execution_time": "2025-08-27T22:05:03.639290",
    "action_count": 3,
    "actions": [
        {
            "reasoning": "需要点击搜索框以便输入搜索内容",
            "function": {
                "name": "click",
                "parameters": {"x": 546, "y": 201}
            }
        }
    ],
    "success": true
}
```

#### actions.json格式（参考淘宝格式）
```json
{
    "app_name": "bilibili",
    "task_type": "type1",
    "task_description": "在B站搜一下\"元神4.8版本更新\"",
    "action_count": 3,
    "actions": [
        {
            "type": "click",
            "position_x": 546,
            "position_y": 201,
            "bounds": [204, 153, 773, 264],
            "action_index": 1
        },
        {
            "type": "input",
            "text": "元神4.8版本更新",
            "action_index": 2
        },
        {
            "type": "done",
            "action_index": 3
        }
    ]
}
```

## 任务结构

### auto_task-3.json结构
```json
[
  {
    "app": "bilibili",
    "type": "type1", 
    "tasks": [
      "在B站搜一下"元神4.8版本更新"",
      "在B站搜一下"决战库班之王"",
      "在B站搜一下"黑神话悟空最新实机""
    ]
  }
]
```

### 支持的应用
- bilibili (B站)
- 淘宝
- 携程
- 网易云音乐
- 小红书
- 高德地图
- 饿了么

## 监控和日志

### 日志文件
- `batch_execution.log` - 批量执行日志
- `test_batch_execution.log` - 测试执行日志
- `automation.log` - 框架执行日志

### 实时监控
执行过程中会在终端显示：
- 当前执行的任务
- 执行步骤详情
- 成功/失败状态
- 执行统计

## 故障排除

### 常见问题

1. **设备连接失败**
```bash
adb devices
adb kill-server
adb start-server
```

2. **应用启动失败**
- 检查应用是否已安装
- 检查包名是否正确
- 尝试手动启动应用

3. **模型调用失败**
- 检查网络连接
- 验证模型服务地址
- 检查服务是否运行

4. **任务执行超时**
- 检查任务复杂度
- 增加最大步数限制
- 检查设备响应速度

### 调试技巧

1. **先测试单个任务**
```bash
python3 test_single_task.py
```

2. **检查保存的截图**
查看步骤目录中的screenshot文件

3. **查看详细日志**
```bash
tail -f batch_execution.log
```

4. **验证应用状态**
确保应用处于预期界面

## 性能优化

### 提高成功率
1. 确保设备性能良好
2. 关闭不必要的后台应用
3. 保持网络连接稳定
4. 适当增加步骤延迟

### 提高执行效率
1. 批量执行相同应用的任务
2. 合理设置任务间隔
3. 优化任务描述的准确性

## 数据分析

### task_data.json 字段说明
- `task_description`: 任务描述
- `app_name`: 应用名称
- `task_type`: 任务类型
- `task_index`: 任务索引
- `package_name`: 应用包名
- `execution_time`: 执行时间
- `action_count`: 操作步数
- `actions`: 详细操作记录
- `success`: 是否成功

### 统计分析
可以通过分析task_data.json文件来：
- 计算各应用的成功率
- 分析平均执行步数
- 识别常见失败原因
- 优化任务描述

## 扩展开发

### 添加新应用
在app_packages字典中添加新的包名映射：
```python
app_packages = {
    "新应用名": "com.example.package"
}
```

### 修改执行参数
调整ExecutionConfig中的配置：
```python
config = ExecutionConfig(
    max_steps=50,      # 增加最大步数
    step_delay=3.0,    # 增加延迟
    temperature=0.1    # 调整模型温度
)
```

### 自定义数据保存
重写save_task_data方法以自定义数据格式。

## 注意事项

1. **资源占用**：长时间执行会产生大量截图数据
2. **设备稳定性**：建议定期重启设备和清理缓存
3. **网络稳定**：确保模型服务连接稳定
4. **权限设置**：确保应用有必要的权限
5. **存储空间**：预留足够的存储空间保存数据

## 版本历史

- v1.0: 基础批量执行功能
- v1.1: 添加测试脚本和数据保存优化
- v1.2: 完善错误处理和日志记录

## 快速使用指南（补充）
此文档补充了 `quick_start.py` 与 `batch_task_executor.py` 的常用运行示例、配置说明和故障排查要点。

### 环境准备
1. 确保已安装依赖：
```bash
pip install -r requirements.txt
```
2. 准备 Android 设备并启用 adb：
```bash
adb devices
```
3. 可选：安装 `ADBKeyboard.apk`（用于发送文本输入）
```bash
adb install -r ADBKeyboard.apk
```

### quick_start.py（手动/交互式运行）
- 目的：快速启动单任务执行，用于调试和试验。
- 运行：
```bash
python quick_start.py --task-file <path-to-task-json>
```
- 常见参数：
  - `--task-file`：指定单个任务的 JSON 配置文件（如果不传则使用内置示例）。
  - `--no-screenshot`：禁用截图保存以加速执行（若需要节省空间）。

### batch_task_executor.py（批量运行）
- 目的：从 JSON 列表中批量执行任务，适合收集多次样本或回放历史动作集。
- 运行：
```bash
python batch_task_executor.py --config auto_task.json
```
- 建议：先运行小批量（例如 2~5 个任务）进行 smoke 测试，再扩大规模。

### 日志与数据输出
- 执行数据（截图、dump XML、动作日志）由 `DataManager` 保存，默认会在 `data_base_dir` 下按时间戳和任务名建立子目录。
- 如果需要自定义保存策略，请查看并调整 `ui_tars_automation/data_manager.py` 中的 `DataManager` 类。

### 常见问题与排查
- adb 无设备：确认设备已连接，且运行 `adb devices` 可以看到设备 ID。
- 权限问题：某些设备需要开启开发者选项与 USB 调试权限。
- 执行卡住：查看日志文件（同目录下）以定位是哪一步失败，常见为坐标匹配或超时。

如果你希望我把 `USAGE_GUIDE.md` 扩展为带参数文档的 CLI 参考（包含 `--help` 输出模拟），告诉我优先级，我会继续完善。
