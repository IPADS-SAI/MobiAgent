# 通用任务测试执行器

## 概述

通用任务测试执行器是一个灵活的测试框架，可以通过 JSON 配置文件轻松配置和执行不同任务的自动化测试。

## 核心特性

- **灵活配置**: 通过 JSON 配置文件定义任务类型、规则文件、数据目录等
- **多种测试模式**: 支持测试所有类型、指定类型、指定 trace 等
- **详细日志**: 自动记录测试过程中的所有输出到指定文件
- **结果汇总**: 生成详细的测试报告，包括成功率、匹配节点、失败原因等
- **易于扩展**: 新增任务只需添加对应的配置文件

## 使用方法

### 1. 基本用法

```bash
# 测试所有类型
python universal_test_runner.py task_configs/taobao.json

# 测试指定类型
python universal_test_runner.py task_configs/taobao.json type3

# 测试指定类型的指定trace
python universal_test_runner.py task_configs/taobao.json type3:150

# 测试指定的trace编号
python universal_test_runner.py task_configs/taobao.json 150,151,152
```

### 2. 配置文件示例

#### 淘宝任务配置 (task_configs/taobao.json)

```json
{
  "task_name": "taobao",
  "description": "淘宝任务测试配置",
  
  "rules_base_dir": "task_rules/taobao",
  "data_base_dir": "data",
  
  "task_types": {
    "3": {
      "name": "加购物车任务",
      "rule_file": "type3-taobao_add_cart-new.yaml",
      "data_traces": [150, 151, 152, 153, 154, 155],
      "description": "添加商品到购物车"
    },
    "4": {
      "name": "排序/筛选任务", 
      "rule_file": "type4-taobao_add_cart.yaml",
      "data_traces": [120, 121, 122, 123, 124, 125],
      "description": "商品排序和筛选功能"
    }
  },

  "test_options": {
    "enable_ocr": true,
    "enable_llm": true,
    "force_llm": false,
    "ocr_frame_exclusive": true,
    "llm_frame_exclusive": true,
    "prevent_frame_backtrack": true
  },

  "logging": {
    "level": "DEBUG",
    "use_colors": true,
    "show_time": true,
    "show_module": true,
    "output_file": "test-{task_name}-{timestamp}.log"
  },

  "output": {
    "summary_file": "test-{task_name}-summary-{timestamp}.txt",
    "detailed_results_file": "test-{task_name}-detailed-{timestamp}.json"
  }
}
```

#### 小红书任务配置 (task_configs/xiaohongshu.json)

```json
{
  "task_name": "xiaohongshu",
  "description": "小红书任务测试配置",
  
  "rules_base_dir": "task_rules/xiaohongshu",
  "data_base_dir": "data/xiaohongshu",
  
  "task_types": {
    "2": {
      "name": "type2任务",
      "rule_file": "xiaohongshu-type2.yaml",
      "data_traces": "type2",
      "description": "小红书type2功能测试"
    },
    "3": {
      "name": "type3任务",
      "rule_file": "xiaohongshu-type3.yaml", 
      "data_traces": "type3",
      "description": "小红书type3功能测试"
    }
  },

  "test_options": {
    "enable_ocr": true,
    "enable_llm": true,
    "force_llm": false,
    "ocr_frame_exclusive": true,
    "llm_frame_exclusive": true,
    "prevent_frame_backtrack": true
  },

  "logging": {
    "level": "DEBUG",
    "use_colors": true,
    "show_time": true,
    "show_module": true,
    "output_file": "test-{task_name}-{timestamp}.log"
  },

  "output": {
    "summary_file": "test-{task_name}-summary-{timestamp}.txt",
    "detailed_results_file": "test-{task_name}-detailed-{timestamp}.json"
  }
}
```

## 配置文件说明

### 基本配置

- `task_name`: 任务名称，用于生成日志和结果文件名
- `description`: 任务描述
- `rules_base_dir`: 规则文件基础目录
- `data_base_dir`: 数据文件基础目录

### 任务类型配置

每个任务类型包含：
- `name`: 类型显示名称
- `rule_file`: 对应的规则文件名
- `data_traces`: 测试数据配置，支持多种格式：
  - **数字列表**: `[150, 151, 152]` - 明确指定trace编号
  - **字符串**: `"type2"` - 指定type目录名
  - **空列表**: `[]` - 自动发现可用traces
  - **不配置**: 完全不包含`data_traces`字段 - 自动发现可用traces
- `description`: 类型描述

#### data_traces 配置详解

1. **明确指定trace编号列表** (最高优先级)
   ```json
   "data_traces": [150, 151, 152, 153, 154]
   ```
   系统会测试编号为 150、151、152、153、154 的trace。

2. **指定type目录**
   ```json
   "data_traces": "type2"
   ```
   系统会在 `data_base_dir/type2` 目录下查找测试数据。

3. **自动发现模式**
   ```json
   "data_traces": []
   ```
   或者完全不配置 `data_traces` 字段，系统会自动扫描 `data_base_dir` 目录：
   - 优先查找 `type{task_type}` 格式的目录
   - 然后查找数字编号的目录
   - 最后包含其他匹配的目录

4. **优先级规则**
   - 如果配置了 `data_traces` 且不为空，优先使用配置的值
   - 如果配置为空或未配置，则启用自动发现模式
   - 如果配置的路径不存在，会回退到自动发现模式

### 测试选项

- `enable_ocr`: 启用 OCR 功能
- `enable_llm`: 启用 LLM 功能
- `force_llm`: 强制使用 LLM
- `ocr_frame_exclusive`: OCR 帧独占模式
- `llm_frame_exclusive`: LLM 帧独占模式
- `prevent_frame_backtrack`: 防止帧回退

### 日志配置

- `level`: 日志级别 (DEBUG, INFO, WARNING, ERROR)
- `use_colors`: 使用彩色输出
- `show_time`: 显示时间戳
- `show_module`: 显示模块名
- `output_file`: 日志文件名模板

### 输出配置

- `summary_file`: 汇总文件名模板
- `detailed_results_file`: 详细结果文件名模板

## 输出文件说明

### 日志文件

记录测试过程中的所有详细信息，包括：
- 系统初始化信息
- 每个测试用例的执行过程
- 验证结果和匹配详情
- 错误信息和警告

### 汇总文件 (.txt)

人类可读的测试结果汇总，包括：
- 测试基本信息
- 总体成功率
- 分类型结果统计
- 每个测试用例的详细结果

### 详细结果文件 (.json)

机器可读的详细结果数据，包括：
- 完整的测试配置
- 每个测试用例的详细结果
- 执行时间统计
- 错误信息

## 新增任务步骤

1. **创建配置文件**: 在 `task_configs/` 目录下创建新的 JSON 配置文件
2. **配置规则目录**: 在 `task_rules/` 下创建对应的规则文件目录
3. **准备测试数据**: 在 `data/` 下准备对应的测试数据
4. **运行测试**: 使用新配置文件运行测试

## 示例输出

### 控制台输出

```
=== 通用任务测试执行器 ===
任务名称: taobao
任务描述: 淘宝任务测试配置
日志文件: test-taobao-20240822_143025.log
汇总文件: test-taobao-summary-20240822_143025.txt
详细结果: test-taobao-detailed-20240822_143025.json

--- 测试 150 [加购物车任务] ---
规则文件: type3-taobao_add_cart-new.yaml
数据路径: /path/to/data/150
验证结果: ✓ 成功
匹配节点: ['search', 'add_to_cart']
任务得分: 100.0分
执行时间: 2.34秒

--- 类型 3 汇总 ---
trace 150: ✓ | score: 100.0 | nodes: ['search', 'add_to_cart'] | reason: 
trace 151: ✗ | score: 60.0 | nodes: ['search'] | reason: 未找到加购物车操作
成功率: 1/2 (50.0%)
```

### 汇总文件示例

```
任务测试汇总报告
============================================================
任务名称: taobao
测试时间: 2024-08-22 14:30:25
配置文件: task_configs/taobao.json
总测试数: 10
总成功数: 8
总成功率: 80.0%
总执行时间: 25.67秒

分类型结果:
----------------------------------------
类型 3 (加购物车任务):
  测试数: 5
  成功数: 4
  成功率: 80.0%

类型 4 (排序/筛选任务):
  测试数: 5
  成功数: 4
  成功率: 80.0%
```

## 扩展和定制

### 自定义验证选项

可以在配置文件的 `test_options` 中添加更多验证选项，然后在 `UniversalTestRunner._create_verifier_options()` 方法中处理。

### 自定义输出格式

可以修改 `save_results()` 方法来支持更多输出格式，如 CSV、XML 等。

### 添加新的测试模式

可以在 `main()` 函数中添加更多参数解析逻辑来支持新的测试模式。

## 常见问题

### Q: 如何添加新任务？
A: 创建新的配置文件，配置规则目录和数据目录，然后运行测试。

### Q: 如何修改日志级别？
A: 在配置文件的 `logging.level` 中设置，支持 DEBUG、INFO、WARNING、ERROR。

### Q: 如何禁用 LLM 或 OCR？
A: 在配置文件的 `test_options` 中设置 `enable_llm` 或 `enable_ocr` 为 false。

### Q: 测试结果保存在哪里？
A: 根据配置文件中的 `output` 部分设置，默认保存在当前目录下，文件名包含时间戳。
