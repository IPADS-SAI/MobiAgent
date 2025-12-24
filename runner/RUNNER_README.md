# 统一GUI Agent Runner框架使用指南

这是一个统一的GUI Agent任务执行框架，支持将多种模型（MobiAgent、UI-TARS等）接入并执行移动端自动化任务。

## 快速开始

### 1. 执行单个任务

```bash
# 使用MobiAgent
python run.py --provider mobiagent --task "在淘宝上搜索电动牙刷"

# 使用UI-TARS  
python run.py --provider uitars --task "在淘宝上搜索电动牙刷"
```

### 2. 批量执行任务

```bash
# 从task.json执行
python run.py --provider mobiagent --task-file mobiagent/task.json

# 从task_mobiflow.json执行
python run.py --provider mobiagent --task-file mobiagent/task_mobiflow.json
```

## 项目结构

```
runner/
├── run.py                    # 统一入口脚本
├── base_task.py             # 基础任务类
├── task_manager.py          # 任务管理器
├── config.json              # 配置文件
├── providers/               # 模型适配器
│   ├── mobiagent_task.py   # MobiAgent适配器
│   └── uitars_task.py      # UI-TARS适配器
├── mobiagent/              # MobiAgent源码
└── UI-TARS-agent/          # UI-TARS源码
```

## 输出格式

每个任务会创建独立目录，包含：

- `1.jpg, 2.jpg, ...`: 每步截图
- `1.xml/1.json, 2.xml/2.json, ...`: UI层级结构
- `1_bounds.jpg, 1_click_point.jpg, ...`: 标注图（如有）
- `actions.json`: 动作记录
- `react.json`: 推理记录

## 主要参数

### 基础参数
- `--provider`: 模型（mobiagent, uitars）
- `--task`: 单个任务描述
- `--task-file`: 批量任务文件
- `--max-steps`: 最大步数（默认40）

### MobiAgent参数
- `--service-ip`: 服务IP
- `--decider-port`: Decider端口（默认8000）
- `--grounder-port`: Grounder端口（默认8001）
- `--planner-port`: Planner端口（默认8002）

### UI-TARS参数
- `--model-url`: 模型地址
- `--model-name`: 模型名称

更多详情请查看完整文档。

## 添加新模型

1. 在`providers/`创建适配器继承`BaseTask`
2. 在`providers/__init__.py`注册
3. 在`task_manager.py`添加映射
4. 在`run.py`添加参数

## 许可证

Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
