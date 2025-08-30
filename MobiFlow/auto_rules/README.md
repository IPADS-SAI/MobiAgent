# Auto Rules 自动化任务配置生成

基于 LLM 的任务验证配置自动生成系统。从用户操作轨迹中提取任务描述，结合模版通过大语言模型生成验证配置文件。

## 功能特性

- **自动提取**: 从目录结构中提取actions.json中的任务描述
- **模版结合**: 结合现有的示例模版（orders/modes）
- **统一提示词**: 集中管理LLM提示词，避免重复
- **LLM生成**: 使用大语言模型生成验证配置
- **简化流程**: 一键从任务描述到完整配置文件

## 安装依赖

```bash
pip install PyYAML openai
```

## 快速开始

### 基础使用

```bash
# 使用orders模版生成配置
python main.py /path/to/taobao_test YOUR_API_KEY

# 指定输出文件
python main.py /path/to/taobao_test YOUR_API_KEY --output-file config.yaml

# 使用modes模版
python main.py /path/to/taobao_test YOUR_API_KEY --template-type modes
```

ordes和modes只是最基础的模版，可以自定义后作为LLM的参考使用。

### 试运行模式

```bash
# 仅提取任务描述，不调用LLM
python main.py /path/to/taobao_test dummy_key --dry-run

# 实际使用
python main.py ../data/taobao/type2 --output-file ./test_output.yaml
```

## 命令行参数

### 必需参数

- `target_dir`: 包含actions.json文件的目标目录路径
- `api_key`: OpenAI API密钥

### 可选参数

- `--output-file, -o`: 输出配置文件路径
- `--template-type, -t`: 模版类型，可选 `orders` 或 `modes`（默认：orders）
- `--app-name`: 应用名称（默认：从任务描述中提取）
- `--dry-run`: 仅提取任务描述，不调用LLM

## 输入数据格式

系统会自动搜索目标目录及其子目录中的 `actions.json` 文件，提取其中的 `task_description` 字段。

### actions.json 示例

```json
{
    "app_name": "淘宝",
    "task_description": "在淘宝搜一下苹果数据线，然后挑一款合适的",
    "action_count": 5,
    "actions": [...]
}
```

## 输出配置示例

生成的配置文件将基于选择的模版类型，包含完整的任务验证节点和条件检查。

```yaml
task_id: taobao_search_and_select
app_id: com.taobao.app
description: 在淘宝搜索商品并选择的任务验证配置

nodes:
  - id: launch_app
    name: 启动淘宝应用
    condition:
      type: escalate
      params:
        ocr:
          any: ["淘宝", "启动"]
        llm:
          prompt: "该步是否成功启动了淘宝应用？"
          expected_true: true
    next: [search_entry]
  # ... 更多节点

success:
  any_of: [complete_action]
```

## 注意事项

1. **API费用**: LLM调用会产生费用，建议先用--dry-run测试
2. **网络连接**: 需要稳定的网络连接访问LLM API
3. **输出验证**: 建议人工审核生成的配置文件
