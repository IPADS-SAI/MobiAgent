"""
提示词模板
包含所有Planner、Decider、Grounder使用的提示词
"""

# ==================== Decider Prompts ====================

DECIDER_PROMPT_TEMPLATE = """
You are a phone-use AI agent. Now your task is "{task}".
Your action history is:
{history}
Please provide the next action based on the screenshot and your action history. You should do careful reasoning before providing the action.
Your action space includes:
- Name: click, Parameters: target_element (a high-level description of the UI element to click).
- Name: swipe, Parameters: direction (one of UP, DOWN, LEFT, RIGHT).
- Name: input, Parameters: text (the text to input).
- Name: wait, Parameters: (no parameters, will wait for 1 second).
- Name: done, Parameters: (no parameters).
Your output should be a JSON object with the following format:
{{"reasoning": "Your reasoning here", "action": "The next action (one of click, input, swipe, done)", "parameters": {{"param1": "value1", ...}}}}

Remember your task is "{task}".
"""

DECIDER_PROMPT_TEMPLATE_ZH = """
你是一个手机使用AI代理。现在你的任务是"{task}"。
你的操作历史如下：
{history}
请根据截图和你的操作历史提供下一步操作。在提供操作之前，你需要进行仔细的推理。
你的操作范围包括：
- 名称：点击（click），参数：目标元素（target_element，对要点击的UI元素的高级描述）。
- 名称：滑动（swipe），参数：方向（direction，UP、DOWN、LEFT、RIGHT中的一个）。
- 名称：输入（input），参数：文本（text，要输入的文本）。
- 名称：等待（wait），参数：（无参数，将等待1秒）。
- 名称：完成（done），参数：（无参数）。
你的输出应该是一个如下格式的JSON对象：
{{"reasoning": "你的推理分析过程在此", "action": "下一步操作（click、input、swipe、done中的一个）", "parameters": {{"param1": "value1", ...}}}}"""


# ==================== Grounder Prompts ====================

GROUNDER_PROMPT_NO_BBOX = """
Based on the screenshot, user's intent and the description of the target UI element, provide the coordinates of the element using **absolute coordinates**.
User's intent: {reasoning}
Target element's description: {description}
Your output should be a JSON object with the following format:
{{"coordinates": [x, y]}}"""

GROUNDER_PROMPT_BBOX = """
Based on the screenshot, user's intent and the description of the target UI element, provide the bounding box of the element using **absolute coordinates**.
User's intent: {reasoning}
Target element's description: {description}
Your output should be a JSON object with the following format:
{{"bbox": [x1, y1, x2, y2]}}"""

GROUNDER_QWEN3_BBOX_PROMPT = """
Based on user's intent and the description of the target UI element, locate the element in the screenshot.
User's intent: {reasoning}
Target element's description: {description}
Report the bbox coordinates in JSON format."""

GROUNDER_PROMPT_NO_BBOX_ZH = """
根据截图、用户意图和目标UI元素的描述，使用**绝对坐标**提供该元素的坐标。
用户意图：{reasoning}
目标元素描述：{description}
你的输出应该是一个如下格式的JSON对象：
{{"coordinates": [x, y]}}"""

GROUNDER_PROMPT_BBOX_ZH = """
根据截图、用户意图和目标UI元素的描述，使用**绝对坐标**提供该元素的边界框。
用户意图：{reasoning}
目标元素描述：{description}
你的输出应该是一个如下格式的JSON对象：
{{"bbox": [x1, y1, x2, y2]}}"""


# ==================== Planner Prompts ====================

PLANNER_TASK_ANALYSIS_PROMPT = """
你是一个任务分析AI助手。用户给出了一个任务描述，你需要分析这个任务的复杂度并决定执行策略。

用户任务描述：{task_description}


请分析这个任务并输出JSON格式的结果：

{{
    "task_type": "single" 或 "multi",  // single=单阶段任务（单个APP可完成）, multi=多阶段任务（需要多个APP协同或复杂流程）
    "reasoning": "你的分析理由",
    "app_name": "主要使用的应用名称（如果是single类型必填）",
    "package_name": "应用包名（如果是single类型必填）",
    "refined_description": "完善后的任务描述（如果是single类型）"
}}

判断标准：
1. 如果任务只涉及单个APP内的操作，且不需要提取和传递结构化数据 → single
2. 如果任务涉及多个APP的协同，或需要在不同APP间传递信息 → multi
3. 如果任务需要比较、聚合多个来源的数据 → multi
4. 如果任务有明确的"先...后..."、"然后..."等多步骤描述 → multi

示例：
- "在携程搜索明天从上海到北京的机票" → single（单APP）
- "在携程订票后把行程发给张三" → multi（携程→微信）
- "比较淘宝和京东的iPhone价格" → multi（需要聚合数据）
- "查看携程订单号然后在微信发给客服" → multi（需要传递数据）
"""

PLANNER_PLAN_GENERATION_PROMPT = """
你是一个任务规划AI助手。用户给出了一个可能涉及多个应用的复杂任务，你需要：
1. 将任务分解为多个子任务（Subtasks）
2. 为每个子任务确定需要使用的应用（app_name 和 package_name）
3. 定义每个子任务需要提取的结构化数据（artifact_schema）
4. 确定子任务之间的依赖关系

APP列表及其包名参考：
{app_list}

用户任务描述：{task_description}

相关经验参考：
{experience_content}

请输出JSON格式的规划，包含以下字段：
{{
    "task_description": "原始任务描述",
    "subtasks": [
        {{
            "subtask_id": 1,
            "app_name": "应用名称",
            "package_name": "应用包名",
            "description": "子任务详细描述",
            "artifact_schema": {{
                "字段名1": "字段描述1",
                "字段名2": "字段描述2"
            }},
            "depends_on": []  // 依赖的子任务ID列表
        }}
    ]
}}

注意：
- 任务要求：
    - 每个子任务应该是原子性的，在单个APP内可以完成
    - 浏览任务需要适当滑动以找到目标信息
    - app_name和package_name必须从提供的APP列表中选择
- artifact_schema定义需要从该子任务中提取的关键信息
- depends_on指定该子任务依赖哪些前置子任务的输出
"""

PLANNER_EXTRACT_ARTIFACT_PROMPT = """
你是一个信息提取AI助手。一个子任务刚刚执行完毕，你需要结合截图和提取的文崩从执行结果中提取结构化信息。

子任务描述：{subtask_description}

需要提取的字段（artifact_schema）：
{artifact_schema}

执行历史：
{execution_history}

页面文本内容（OCR提取）：
{ocr_text_section}

请分析最后几步的截图和操作记录，提取出所需的结构化信息。

输出JSON格式：
{{
    "subtask_id": {subtask_id},
    "success": true/false,  // 子任务是否成功完成
    "summary": "自然语言总结子任务执行情况",
    "data": {{
        "字段名1": "提取的值1",
        "字段名2": "提取的值2"
    }}
}}
"""

PLANNER_NEXT_STEP_PROMPT = """
你是一个任务规划AI助手。当前正在执行一个多阶段任务。

当前已完成的子任务及其结果：
{artifacts_desc}

下一个待执行的子任务：
{next_subtask_description}

请基于已完成子任务的结果（artifacts），生成一个完整、详细的、口语化任务描述，用于执行下一个子任务。
这个任务描述应该：
1. 包含前置子任务提取的关键信息（如商品名称、价格、订单号等），替换占位符为具体内容
2. 清晰描述在下一个APP中需要完成什么操作
3. 确保任务可以在1个APP独立执行

输出JSON格式：
{{
    "refined_description": "完善后的子任务描述，需要包含前置任务提取的具体数据值，不含占位符和变量符号"
}}

示例：
如果前置任务提取了：shirt_name="【免烫款】海澜之家白衬衫", shirt_price="131"
下一个任务是：将商品信息发送给小赵
则refined_description应该是："打开微信，找到联系人'小赵'，发送消息：'【免烫款】海澜之家白衬衫 价格：131元'"
"""
