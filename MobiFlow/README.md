# MobiFlow: 基于dag的移动代理基准测试框架

一个离线验证框架：在收集到某次任务执行的完整轨迹（关键帧/事件）后，读取任务的 DAG 配置，检查是否存在满足依赖与顺序约束的"满足路径"，从而判断该次执行是否达到任务目标。

## 功能特性

### 核心验证能力
- **多层级条件检查**：支持文本匹配、正则表达式、UI状态、XML解析、图标检测、OCR识别、LLM推理等多种判定方法
- **图标检测识别**：基于OpenCV模板匹配的图标检测，支持多尺度匹配和相似度阈值控制，快速识别UI界面元素
- **逐级升级策略**：当前一种判定方法不可用时，自动升级到更高级的判定方式
- **多路径DAG验证**：基于有向无环图验证任务节点间的依赖关系，支持AND和OR两种语义的路径分支
- **动态条件匹配**：支持根据任务描述动态提取条件并进行相应验证

### 高级验证功能
- **双语义依赖系统**：
  - `deps` 字段：AND语义，所有前置节点必须完成
  - `next` 字段：OR语义，支持多分支路径选择
- **路径感知验证**：智能帧分配机制，避免跨路径的帧冲突
- **约束检查**：自动检测deps/next配置冲突并发出警告
- **路径分析日志**：执行前显示所有可能的成功路径，便于调试和配置验证
- **界面文字识别**：识别当前界面文字内容，通过文章元素匹配关键节点
- **智能图标识别**：`icons_match` 检查器支持基于模板匹配的图标检测，提供快速的UI状态验证能力
- **多模态LLM验证**：支持结合截图和上下文信息进行LLM推理验证
- **灵活配置系统**：通过YAML配置文件灵活定义各种验证条件和模式

## 目录结构

- `avdag/` 核心库
  - `conditions.py` 条件检查器与注册表（含动态匹配检查器、图标检测检查器）
  - `dag.py` DAG 定义、拓扑排序、路径分析和约束检查
  - `verifier.py` 核心验证逻辑（路径感知的帧分配和多路径验证）
  - `loader.py` 从 YAML/JSON 读取任务配置（支持deps和next字段）
  - `trace_loader.py` 从目录结构读取轨迹数据
  - `types.py` 基本类型（Frame/Result/VerifierOptions 等，NodeSpec支持next字段）
- `tools/` 辅助工具
  - `Icon_detection/` 图标检测工具包
    - `icon_detector.py` 基于OpenCV的多尺度模板匹配核心检测器
    - `icon_detection.py` 高级图标检测服务接口
    - `config.py` 图标检测配置管理
    - `test_*.py` 图标检测功能测试脚本
- `task_configs/icons/` 图标资源库
  - `weixin/` 微信应用图标模板
  - `bilibili/` B站应用图标模板
  - `xiecheng/` 携程应用图标模板
- `task_configs/` 任务配置与图标资源（本目录中的可用示例配置位于此目录）
  - `task_configs/*.json` JSON 格式的任务配置示例
  - `task_configs/icons/` 图标资源模板
- `docs/` 使用说明与设计文档（包含多路径、检查器模式与OCR/LLM 使用说明）
- `tests/` 单元测试
  - `test_dependency_validation.py` 依赖约束检查测试
  - `test_next_paths.py` next节点OR语义测试
  - `test_path_analysis.py` 路径分析日志测试
- 测试脚本
  - `test_dynamic_filter.py` 动态筛选条件验证测试
  - `test_filter_verification.py` 筛选功能验证测试
  - `test_image_verification.py` 图像验证测试

## 安装依赖

安装项目根目录的相关库即可

```bash
pip install -r requirements.txt
```

可选额外安装一个OCR辅助工具，配合Paddle进行检测：

```bash
# 安装Tesseract OCR
sudo apt-get install tesseract-ocr

# 安装中文语言包
sudo apt-get install tesseract-ocr-chi-sim

# 检查是否正确安装
tesseract --version
```

### 图标检测功能额外依赖

```bash
pip install opencv-python numpy
```

## 快速开始

1) 查看本目录内的示例配置：`task_configs/` 中包含若干任务配置及图标资源。

2) 使用最小演示脚本运行（示例使用 `task_configs` 中的配置）：

```bash
python -m avdag.verifier task_configs/taobao.json trace_folder/
```

输出包括：
- 路径分析日志（所有可能的成功路径）
- 是否成功（存在一条满足约束的满足路径）
- 被满足的节点与对应匹配到的帧索引
- 一个按时间排序的满足序列（可视为线性化的 trace）

示例输出：
```
[INFO] === DAG 路径分析 ===
[INFO] 发现 2 条可能的成功路径:
  路径 1: activate_search -> input_keyword -> results_page -> open_profile -> follow_author
  路径 2: activate_search -> input_keyword -> results_page -> follow_author
[INFO] === 路径分析结束 ===
```

## 配置格式（YAML/JSON）

可以使用`MobiFlow/auto_rules`中自动工具由LLM分析生成任务配置，也可手动按照如下规则编写。

### 基础配置示例

```yaml
task_id: shop_search
nodes:
  - id: open_app
    name: 打开购物 App
    condition:
      type: text_match
      params:
        any: ["打开了淘宝", "商城首页"]
  - id: search_page
    deps: [open_app]  # AND语义：必须等待open_app完成
    condition:
      type: ui_flag
      params:
        key: screen
        equals: search
  - id: search_keyword
    deps: [search_page]
    condition:
      type: regex_match
      params:
        pattern: ".*iPhone 15.*"
  - id: result_list
    deps: [search_keyword]
    condition:
      type: text_match
      params:
        any: ["结果", "共", "商品"]
# 成功条件（可选）。若省略，则默认任一"汇点"节点被满足即成功。
success:
  any_of: [result_list]
```

### 多路径配置示例

```yaml
task_id: bilibili_search_follow
nodes:
  - id: activate_search
    name: 激活搜索功能
    condition:
      type: text_match
      params:
        any: ["搜索", "search"]
    next: [input_keyword]  # OR语义：可以进入input_keyword
  
  - id: input_keyword
    name: 输入搜索关键词
    condition:
      type: text_match
      params:
        any: ["关键词", "搜索词"]
    next: [results_page]
  
  - id: results_page
    name: 搜索结果页面
    condition:
      type: text_match
      params:
        any: ["搜索结果", "结果页"]
    next: [follow_author, open_profile]  # OR语义：两条可选路径
  
  - id: open_profile
    name: 打开用户主页
    condition:
      type: text_match
      params:
        any: ["用户主页", "个人页面"]
    next: [follow_author]
  
  - id: follow_author
    name: 关注作者
    condition:
      type: text_match
      params:
        any: ["关注", "已关注"]

success:
  any_of: [follow_author]
```


### 图标检测配置示例

```yaml
task_id: wechat_send_message
app_id: com.tencent.mm
description: 在微信中给指定联系人或群聊发送消息
nodes:
  - id: find_contact_entry
    name: 查找联系人或群聊
    condition:
      type: escalate
      params:
        icons:
          all: ["icon_001_通讯录", "icon_002_微信", "icon_000_我"]  # 必须检测到所有图标
        ocr:
          all: ["微信", "通讯录", "发现", "我"]
        llm:
          prompt: 当前页面是否为微信主界面、通讯录或搜索页面？
          expected_true: true
    next: [send_message_success]

  - id: send_message_success
    name: 成功发送消息
    condition:
      type: juxtaposition  # 要求图标和OCR都成功
      params:
        icons:
          any: ["icon_001_回车", "icon_002_发送"]  # 匹配任意发送相关图标
          threshold: 0.85  # 自定义相似度阈值
        ocr:
          all: ["发送"]

success:
  any_of: [send_message_success]
```

### 配置说明

- `nodes[].deps`：该节点的前置依赖（AND 关系）- 所有依赖节点必须完成
- `nodes[].next`：该节点的后继节点（OR 关系）- 任一后继节点可以执行
- `condition`：由 `type` 指定检查器，`params` 为该检查器参数
- `success`：
  - `any_of: [node_id...]` 任一节点满足即判成功
  - `all_of: [node_id...]` 列表中全部节点满足才判成功
  - 若不配置，默认检查"汇点"节点（无出边）中是否存在满足的节点

#### 依赖系统语义

- **deps（AND语义）**：严格的前置依赖，所有listed节点必须先完成
- **next（OR语义）**：灵活的后继选择，可以进入任一listed节点
- **约束检查**：当节点同时定义deps和作为其他节点的next目标时，系统会发出警告，deps优先

#### 路径分析

验证执行前会自动分析并输出所有可能的成功路径：
```
[INFO] 发现 2 条可能的成功路径:
  路径 1: activate_search -> input_keyword -> results_page -> open_profile -> follow_author
  路径 2: activate_search -> input_keyword -> results_page -> follow_author
```

### 支持的检查器类型

#### 基础检查器
- `text_match`: 文本匹配
- `regex_match`: 正则表达式匹配  
- `ui_flag`: UI状态检查
- `xml_text_match`: XML内容匹配
- `action_match`: 动作类型匹配

#### 图像识别检查器
- `icons_match`: 图标检测匹配，基于OpenCV模板匹配技术快速识别UI界面图标

#### 高级检查器
- `escalate`: 按策略升级顺序尝试多种检查方法
- `juxtaposition`: 并列检查器，要求所有配置的检查器都必须通过
- `dynamic_match`: 动态条件匹配，根据任务描述提取条件并验证相应操作

## 轨迹数据（Frames）

框架支持两种轨迹数据格式：

### 1. JSON格式（简单）

把执行过程中的关键帧/事件整理为按时间排序的数组：

```json
{
  "timestamp": 1723456789.123,
  "text": "打开了淘宝，进入搜索页",
  "ui": {"screen": "search"},
  "payload": {"extra": "自由扩展"}
}
```

### 2. 目录格式（移动端自动化）

支持包含截图、XML、动作记录的目录结构：

```
trace_folder/
├── 1.jpg          # 截图
├── 1.xml          # UI布局信息
├── 2.jpg
├── 2.xml
├── ...
├── actions.json   # 动作序列
└── react.json     # 推理记录
```

每帧包含的字段：
- `image`: 截图文件路径
- `screenshot`: 截图数据（numpy数组或文件路径，用于图标检测）
- `xml_text`: UI布局的XML文本
- `reasoning`: 推理过程描述
- `action`: 执行的动作信息
- `task_description`: 任务描述（用于动态匹配）
- `text`: 综合文本信息（用于简单匹配）
- `app_id`: 应用包名（用于图标路径解析）

内置检查器会在上述字段中查找信息，你也可以注册自定义检查器（见下）。

## 自定义检查器

```python
from avdag.conditions import register_condition, ConditionChecker

@register_condition("my_checker")
class MyChecker(ConditionChecker):
    def check(self, frame: dict, params: dict, options=None) -> bool:
        # 读取 frame / params 做任意判断
        return frame.get("payload", {}).get("flag") == params.get("flag")
```

注册后即可在配置中使用：

```yaml
condition:
  type: my_checker
  params:
    flag: true
```

## 运行测试

请使用下面的通用方式运行测试或直接运行工具自带的测试脚本：

### 自定义验证选项

```python
from avdag.verifier import make_llm_options, verify_task_folder

# 配置LLM验证
opts = make_llm_options(
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",
    model="gpt-4o",
    force_llm=True  # 强制使用LLM验证
)

# 运行验证
result = verify_task_folder("task.yaml", "trace_folder", opts)
```

## 设计说明

### 核心算法
- **多路径DAG验证**：支持AND（deps）和OR（next）两种语义的路径分支
- **路径感知验证**：智能帧分配，避免跨路径的帧冲突，确保验证准确性
- **约束检查**：检查顺序严格：某节点匹配到的帧索引必须晚于其所有依赖节点
- **验证流程**：
  - 路径分析：输出所有可能的成功路径
  - 候选收集：为每个节点收集"候选帧索引集合"（基于可达性）
  - 拓扑验证：计算每个节点的"最小可行索引"
  - 成功判定：若 `success` 中的任一/全部目标节点存在可行索引，则判成功
  - 结果输出：给出按匹配索引排序的线性化满足序列

### 支持功能特性
- **多语义依赖系统**：
  - deps：AND语义，严格的前置依赖关系
  - next：OR语义，灵活的分支路径选择
- **路径感知帧分配**：基于动态可达性分析的智能帧分配机制
- **约束冲突检测**：自动检测deps/next配置冲突并发出警告
- **路径分析日志**：执行前输出所有可能路径，便于调试和配置验证
- **智能图标检测**：
  - 基于OpenCV模板匹配的多尺度图标检测
  - 支持any/all匹配模式和自定义相似度阈值
  - 智能路径解析，根据应用ID自动查找图标资源
  - 非极大值抑制去重，提高检测准确性
- **动态条件匹配**：`dynamic_match` 检查器支持根据任务描述动态提取条件并验证相应操作
- **多模态验证**：支持结合截图、XML、推理文本、图标检测进行LLM验证
- **升级策略**：`escalate` 检查器支持从简单到复杂的逐级验证策略（text → regex → ui → action → dynamic_match → icons → ocr → llm）
- **灵活配置**：通过YAML配置文件可以灵活定义各种复杂的验证条件

### 适用场景
- **移动端多路径任务**：特别适合B站、淘宝等存在多种操作路径的应用验证
- **UI界面状态检测**：通过图标检测快速识别应用界面状态，如微信主界面、聊天窗口等
- **复杂分支逻辑**：支持用户可以选择不同操作路径的任务验证
- **条件筛选验证**：支持根据任务要求动态判断是否执行了正确的筛选操作
- **多模态验证链**：结合图标检测、OCR识别、LLM推理的逐级验证策略
- **配置调试**：通过路径分析日志快速定位配置问题
- **人工复核标记**：当自动验证不确定时，自动标记需要人工复核

## 图标检测功能详细说明

### 技术原理
图标检测功能基于OpenCV模板匹配技术，采用多尺度检测和相似度阈值控制，能够在移动应用截图中快速准确地识别UI图标元素。

### 核心特性
- **多尺度模板匹配**：支持0.5x到2.0x的缩放范围，适应不同分辨率的设备
- **智能相似度控制**：可配置相似度阈值，平衡检测精度和召回率
- **非极大值抑制**：自动去除重复检测结果，提高检测准确性
- **路径智能解析**：根据应用包名自动查找对应图标资源
- **批量检测优化**：支持同时检测多个图标，提高验证效率

### 配置参数说明

#### icons检查器参数
```yaml
icons:
  any: ["icon_001_通讯录", "icon_002_微信"]  # 匹配任意一个图标
  all: ["icon_001_回车", "icon_002_发送"]   # 必须匹配所有图标
  threshold: 0.85                          # 可选：自定义相似度阈值
```

#### 图标资源组织
```
task_configs/icons/
├── weixin/              # 微信应用图标
│   ├── icon_001_通讯录.jpg
│   ├── icon_002_微信.jpg
│   └── icon_000_我.jpg
├── bilibili/            # B站应用图标
└── taobao/             # 淘宝应用图标
```

### 升级策略中的位置
在escalate检查器中，图标检测位于OCR之前，LLM之后：
```
text → regex →  action  → icons → ocr → llm
```
这样的设计确保：
1. 优先使用快速的文本和UI检查
2. 图标检测提供视觉验证能力
3. OCR处理复杂文本识别
4. LLM作为最终的智能判断

### 性能优化
- **图标缓存机制**：已加载的图标模板会被缓存，避免重复读取
- **早期终止**：escalate模式下图标检测成功即返回，无需后续检查
- **尺寸预检查**：避免处理过大的缩放模板，提高检测速度

### 使用建议
1. **图标质量**：使用清晰、特征明显的图标模板
2. **阈值调优**：根据实际效果调整相似度阈值，通常0.8-0.9为佳
3. **命名规范**：采用统一的图标命名规则，便于管理和配置
4. **组合使用**：结合其他检查器使用，提高验证的可靠性

---

若你需要适配真实移动端各类设备采集（OCR、UI dump、强 LLM 审核回调等），可将其加工为上述 `frames` 数组再进行离线验证；也可通过自定义检查器接入更复杂的判断逻辑。
