# UI-TARS 自动化执行框架

基于UI-TARS-7B-SFT模型的移动应用智能自动化框架，支持视觉理解和自然语言指令执行,能够收集到支持当前项目MobiAgent格式的trace数据。

## 核心文件

- `automation_framework_simple.py` - 核心自动化框架
- `automation_examples.py` - 任务示例和交互界面
- `requirements.txt` - Python依赖包

## 快速开始

### 1. 安装依赖

安装项目根目录下的相关依赖即可

```bash
pip install -r requirements.txt
```

### 2. 启动模型服务
```bash
# 使用vLLM启动UI-TARS模型服务
python -m vllm.entrypoints.openai.api_server \
    --model UI-TARS-7B-SFT \
    --served-model-name UI-TARS-7B-SFT \
    --host 0.0.0.0 \
    --port 8000
```

### 3. 连接Android设备
- 开启USB调试
- 安装ADB键盘 (用于文本输入)(在手机上安装项目根目录中的 ADBKeyboard.apk 文件)
- 确保设备可通过ADB连接

### 4. 运行示例

下面是最小步骤，帮助你在本地运行 UI-TARS完成手机任务的核心脚本。

1. 安装依赖：
```bash
pip install -r requirements.txt
```

1. 运行单任务示例：
```bash
python quick_start.py
```
quick_start 会使用仓库内默认配置并引导你通过交互或指定参数运行单个测试任务。

1. 批量执行任务（示例）：
```bash
python batch_task_executor.py --config auto_task.json
```
默认配置文件 `auto_task.json` 放在仓库根目录，用于演示批量任务的 JSON 格式。

## 目录与主要文件说明（补充）
- `quick_start.py`：单任务执行的快速入口，适合手工调试和快速验证。
- `batch_task_executor.py`：批量任务执行器，支持从 JSON 列表加载多个任务并顺序执行。
- `ui_tars_automation/`：核心库，包含设备/坐标处理、数据管理、日志封装和执行框架。
  - `data_manager.py`：负责保存截图、XML、动作记录等执行数据。
  - `config.py`：包含执行相关的可调整配置项（例如保存路径、最大步数等）。
  - `framework.py`：任务执行流程入口与调度逻辑。

## 常见问题与提示（补充）
- 日志：脚本会输出到标准输出并保存到当前目录下的日志文件，若需要修改日志级别或格式，可编辑 `ui_tars_automation/logger.py`。
- ADB 输入：如需通过键盘输入模拟，请先将 `ADBKeyboard.apk` 安装到目标设备。
- 数据输出路径：可在 `ui_tars_automation/config.py` 中调整 `data_base_dir`。

## 预定义任务

框架包含以下预定义任务：

1. **淘宝购物**: 打开淘宝应用，搜索'手机壳'，查看搜索结果
2. **微信聊天**: 打开微信，找到好友列表，查看最近的聊天记录  
3. **系统设置**: 打开系统设置，找到WiFi设置，查看当前连接的WiFi信息
4. **网页浏览**: 打开浏览器，搜索'UI-TARS'相关信息
5. **短视频**: 打开抖音或B站，浏览视频内容
6. **地图导航**: 打开地图应用，搜索附近的餐厅
7. **音乐播放**: 打开音乐应用，搜索并播放一首歌曲

## 自定义任务

支持三种方式执行任务：

1. **预定义任务**: 从任务列表选择执行
2. **交互模式**: 输入自然语言描述执行自定义任务
3. **动态添加**: 将自定义任务添加到任务列表

## 支持的操作

- `click(point)` - 点击指定坐标
- `long_press(point)` - 长按指定坐标
- `type(content)` - 输入文本
- `scroll(point, direction)` - 滚动(上/下/左/右)
- `drag(start_point, end_point)` - 拖拽操作
- `press_home()` - 按Home键
- `press_back()` - 按返回键
- `finished(content)` - 任务完成

## 配置参数

```python
config = ExecutionConfig(
    model_base_url="http://192.168.12.152:8000/v1",  # 模型服务地址
    model_name="UI-TARS-7B-SFT",                     # 模型名称
    max_steps=30,                                    # 最大执行步数
    step_delay=2.0,                                  # 步骤间延迟(秒)
    language="Chinese"                               # 思考语言
)
```

## 使用示例

### 编程方式使用
```python
from automation_framework_simple import UITarsSimpleFramework, ExecutionConfig

# 创建配置
config = ExecutionConfig(
    model_base_url="http://192.168.12.152:8000/v1",
    max_steps=20,
    step_delay=2.0
)

# 创建框架实例
framework = UITarsSimpleFramework(config)

# 执行任务
success = framework.execute_task("打开微信，查看朋友圈")

# 获取执行摘要
summary = framework.get_execution_summary()
print(f"任务{'成功' if success else '失败'}, 共{summary['total_steps']}步")
```

### 命令行使用
```bash
python automation_examples.py
```

## 输出文件

- `automation.log` - 详细执行日志
- `execution_log_*.json` - 任务执行摘要
- `automation_screenshots/` - 每步执行的截图

## 注意事项

1. 确保模型服务正常运行
2. 设备网络连接稳定
3. 合理设置最大步数和延迟时间
4. 任务描述要清晰具体

## 许可证

Apache 2.0 License
