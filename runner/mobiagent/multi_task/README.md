# multi_task 多任务运行模块

## 项目简介
multi_task模块通过自动化的方式完成用户在移动设备上的复杂任务。能够分析任务的复杂度，选择合适的执行策略，并支持单阶段任务（单个应用内完成）和多阶段任务（涉及多个应用协同完成）。

## 系统架构
MobiAgent multi_task 的架构采用模块化设计，主要由以下核心组件组成：

### 1. Planner 模块
- **功能**：
  - 分析任务复杂度（单阶段或多阶段）
  - 生成多阶段任务的执行计划
  - 提取子任务的结构化数据（Artifact）
  - 根据已完成任务的结果优化后续任务描述
- **主要方法**：
  - `analyze_task`：分析任务类型（单阶段/多阶段）
  - `generate_plan`：生成任务计划
  - `extract_artifact`：从子任务结果中提取数据
  - `refine_next_subtask_description`：优化子任务描述

### 2. Executor 模块
- **功能**：
  - 执行单个子任务
  - 集成 Decider 和 Grounder 的决策与执行逻辑
  - 处理多种 UI 动作（点击、输入、滑动、等待、完成）
  - 截图获取与可视化生成
- **主要方法**：
  - `task_in_app`：执行单个子任务的核心逻辑
  - `get_screenshot`：获取设备截图
  - `create_swipe_visualization`：生成滑动动作的可视化图像

### 3. Decider 组件
- **功能**：
  - 基于任务描述、操作历史和截图，决定下一步动作
- **输入**：任务描述、操作历史、当前截图
- **输出**：结构化的动作计划（JSON 格式）

### 4. Grounder 组件
- **功能**：
  - 定位 UI 元素的坐标或边界框
- **输入**：用户意图、目标元素描述、截图
- **输出**：坐标或边界框信息

### 5. 设备抽象层
- **功能**：
  - 封装 Android 设备操作
- **实现**：
  - `AndroidDevice` 类，支持 APP 启动、截图、点击、输入、滑动等操作

### 6. 状态管理
- **功能**：
  - 维护任务执行状态
- **数据结构**：
  - `State` 对象，包含任务描述、计划、当前进度、Artifacts 等

### 7. 数据模型
- **定义**：
  - `ActionPlan`：动作决策结果
  - `GroundResponse`：Grounder 响应
  - `Subtask`：单个子任务定义
  - `Plan`：完整任务规划
  - `Artifact`：子任务执行结果
  - `State`：任务执行状态

---

## 执行流程

### 总体流程
1. **任务输入**：用户提供任务描述。
2. **任务分析**：Planner 分析任务复杂度。
3. **策略选择**：
   - 单阶段任务：直接执行。
   - 多阶段任务：生成执行计划。
4. **任务执行**：按计划依次执行子任务。
5. **结果输出**：返回执行结果和统计信息。

### 单阶段任务执行流程
```
用户任务描述 → Planner 分析 → 确定 APP → Executor 执行 → 返回结果
```
详细步骤：
1. 调用 `get_app_package_name` 获取 APP 信息和完善任务描述。
2. 调用 `task_in_app` 执行任务。
3. Executor循环执行：截图 → Decider 决策 → Grounder 定位 → 执行动作。
4. 直到任务完成（done 动作）。

### 多阶段任务执行流程
```
用户任务描述 → Planner 分析 → 生成 Plan → 依次执行子任务 → 提取 Artifact → 更新状态
```
详细步骤：
1. **任务分析** (`analyze_task`):
   - 使用 `PLANNER_TASK_ANALYSIS_PROMPT` 分析任务类型。
   - 判断是否需要多 APP 协同。
2. **计划生成** (`generate_plan`):
   - 使用 `PLANNER_PLAN_GENERATION_PROMPT` 生成子任务列表。
   - 每个子任务包含：APP 信息、描述、依赖关系、需要提取的数据格式。
3. **状态初始化**:
   - 创建 `State` 对象，包含 Plan 和初始状态。
4. **子任务执行循环**:
   - 选择下一个待执行子任务。
   - 根据依赖关系检查前置条件。
   - 使用 `refine_next_subtask_description` 重写子任务描述（融入前置 Artifact）。
   - 调用 `task_in_app` 执行子任务。
   - 提取 Artifact (`extract_artifact`)。
   - 更新状态，标记完成。
5. **流程控制**:
   - 检查是否所有子任务完成。
   - 处理失败情况和最大步骤限制。

### 子任务内部执行流程
每个子任务的执行遵循以下循环：
```
while True:
    获取截图
    Decider 决策 → 解析动作和参数
    根据动作类型处理：
        - click: Grounder 定位 → 执行点击 → 可视化
        - input: 执行输入
        - swipe: 执行滑动 → 可视化
        - wait: 等待
        - done: 结束任务
    保存执行数据
    检查终止条件
```

---

## 数据流

### 输入数据
- 用户任务描述（字符串）
- 设备连接（Android 设备对象）
- 模型服务配置（IP、端口）

### 中间数据
- Plan：结构化的子任务列表
- State：执行状态跟踪
- Artifacts：各子任务的结构化输出
- 截图序列：执行过程可视化

### 输出数据
- 执行结果：成功/失败状态
- 统计信息：子任务完成情况
- 可视化文件：截图、动作标注图

---

## 关键技术细节

### 1. OCR 集成
- **双引擎支持**：PaddleOCR（主）+ Tesseract（备）
- **临时文件管理**：自动创建和清理OCR处理的临时文件

### 2. 经验检索
- 集成本地经验检索系统 (`PromptTemplateSearch`)。
- 使用历史任务模板优化生成质量。
- **增强检索**：在计划生成和任务重写时自动检索相关经验,使描述更加完整。

### 4. 可视化支持
- 生成点击和滑动动作的可视化图像。
- 保存执行截图和 UI 层次结构。

### 5. 轨迹数据持久化
- 保存任务分析结果、执行计划、状态信息。
- 导出执行历史和结果统计。

---

## 配置选项

### ExtractArtifactConfig 配置类
```python
class ExtractArtifactConfig:
    use_ocr: bool = True                    # 是否使用OCR提取页面文本
    num_screenshots: int = 1                # 使用最后几张截图进行提取
    use_text_with_image: bool = True        # 是否将文本和截图一起发送
    enable_hybrid_ocr: bool = True          # 是否启用混合OCR识别
```

### 命令行参数
```bash
python mobiagent_refactored.py \
    --service_ip localhost \
    --decider_port 8000 \
    --grounder_port 8001 \
    --planner_port 8002 \
    --use_ocr \                             # 启用OCR
    --num_screenshots 3 \                   # 使用最后3张截图
    --use_text_with_image \                 # 文本+图像模式
    --enable_hybrid_ocr \                   # 启用混合OCR
    --use_qwen3 \
    --use_experience \
    --task "我在小红书找一下推荐的2025年性价比最高单反相机，然后在淘宝搜这一款相机，把淘宝中的相机品牌、名称和价格用微信发给小赵"
```
或在根目录下执行
```
python -m runner.mobiagent.multi_task.mobiagent_refactored \
  --service_ip localhost \
  --decider_port 8000 \
  --grounder_port 8001 \
  --planner_port 80002 \
  --use_qwen3 \
  --use_experience \
  --task "帮我在小红书找一下推荐的最畅销的男士牛仔裤，然后在淘宝搜这一款裤子，把淘宝中裤子品牌、名称和价格用微信发给小赵"
```

### 配置示例
```python
# 基础配置
basic_config = ExtractArtifactConfig(
    use_ocr=True,
    num_screenshots=1,
    use_text_with_image=True,
    enable_hybrid_ocr=True
)

# 高精度配置（多截图+混合OCR）
advanced_config = ExtractArtifactConfig(
    use_ocr=True,
    num_screenshots=3,              # 使用最后3张截图
    use_text_with_image=True,
    enable_hybrid_ocr=True
)

# 纯图像模式
image_only_config = ExtractArtifactConfig(
    use_ocr=False,
    num_screenshots=2,
    use_text_with_image=False,      # 只发送图像，不发送文本
    enable_hybrid_ocr=False
)
```

---
##  新增功能
### 1. OCR 增强支持
- **OCR 文本提取**：在 artifact 提取过程中，自动使用 OCR 工具提取页面重要文本信息
- **混合OCR引擎**：支持 PaddleOCR + Tesseract 双引擎，提高识别准确率
- **智能文本处理**：自动清理、标准化OCR文本，提高匹配准确性

### 2. 多截图配置
- **可配置截图数量**：支持灵活使用最后 N 张截图进行 artifact 提取，默认使用最后一张
- **灵活输入模式**：
  - **文本+图像模式**：将OCR提取的文本和截图一起发送给模型
  - **纯图像模式**：只发送截图给模型进行分析

### 3. 经验检索增强
- **计划生成优化**：在生成多阶段任务计划时，自动检索相关历史经验
- **任务重写优化**：在重写下一阶段任务描述时，结合历史经验和前置任务结果
- **智能提示增强**：基于经验库提供更准确的任务执行指导

---