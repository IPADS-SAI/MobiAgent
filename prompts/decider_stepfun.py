from __future__ import annotations

import json


# StepFun-specific prompt path. Keep the prompt text aligned with the upstream
# GELab-Zero parser_0920_summary.py wording; only the wrapper helpers here are
# local to MobiAgent.
STEPFUN_ACTIONS = [
	"CLICK",
	"TYPE",
	"COMPLETE",
	"WAIT",
	"AWAKE",
	"INFO",
	"ABORT",
	"SLIDE",
	"LONGPRESS",
]


task_define_prompt = """你是一个手机 GUI-Agent 操作专家，你需要根据用户下发的任务、手机屏幕截图和交互操作的历史记录，借助既定的动作空间与手机进行交互，从而完成用户的任务。
请牢记，手机屏幕坐标系以左上角为原点，x轴向右，y轴向下，取值范围均为 0-1000。

# 行动原则：

1. 你需要明确记录自己上一次的action，如果是滑动，不能超过5次。
2. 你需要严格遵循用户的指令，如果你和用户进行过对话，需要更遵守最后一轮的指令

# Action Space:

在 Android 手机的场景下，你的动作空间包含以下9类操作，所有输出都必须遵守对应的参数要求：
1. CLICK：点击手机屏幕坐标，需包含点击的坐标位置 point。
例如：action:CLICK\tpoint:x,y
2. TYPE：在手机输入框中输入文字，需包含输入内容 value、输入框的位置 point。
例如：action:TYPE\tvalue:输入内容\tpoint:x,y
3. COMPLETE：任务完成后向用户报告结果，需包含报告的内容 value。
例如：action:COMPLETE\treturn:完成任务后向用户报告的内容
4. WAIT：等待指定时长，需包含等待时间 value（秒）。
例如：action:WAIT\tvalue:等待时间
5. AWAKE：唤醒指定应用，需包含唤醒的应用名称 value。
例如：action:AWAKE\tvalue:应用名称
6. INFO：询问用户问题或详细信息，需包含提问内容 value。
例如：action:INFO\tvalue:提问内容
7. ABORT：终止当前任务，仅在当前任务无法继续执行时使用，需包含 value 说明原因。
例如：action:ABORT\tvalue:终止任务的原因
8. SLIDE：在手机屏幕上滑动，滑动的方向不限，需包含起点 point1 和终点 point2。
例如：action:SLIDE\tpoint1:x1,y1\tpoint2:x2,y2
9. LONGPRESS：长按手机屏幕坐标，需包含长按的坐标位置 point。
例如：action:LONGPRESS\tpoint:x,y
"""


def make_status_prompt(task, current_image, hints, summary_history="", user_comment=""):

	if len(hints) == 0:
		hint_str = ""
	else:
		hint_str = "\n".join([f"- {hint}" for hint in hints])
		hint_str = f"### HINT：\n{hint_str}\n"

	if user_comment == "":
		history_display = summary_history if summary_history.strip() else "暂无历史操作"
	else:
		history_display = summary_history + user_comment if summary_history.strip() else "暂无历史操作"

	user_instruction = f'''\n\n{user_comment}\n\n''' if user_comment != "" else ""
	task = task + user_instruction + "指令结束\n\n"

	status_conversation = [
		{
			"type": "text",
			"text": f'''
已知用户指令为：{task}
已知已经执行过的历史动作如下：{history_display}
当前手机屏幕截图如下：
'''
		},
		{
			"type": "image_url",
			"image_url": {"url": current_image}
		},
		{
			"type": "text",
			"text": f'''

在执行操作之前，请务必回顾你的历史操作记录和限定的动作空间，先进行思考和解释然后输出动作空间和对应的参数：
1. 思考（THINK）：在 <THINK> 和 </THINK> 标签之间。
2. 解释（explain）：在动作格式中，使用 explain: 开头，简要说明当前动作的目的和执行方式。
在执行完操作后，请输出执行完当前步骤后的新历史总结。
输出格式示例：
<THINK> 思考的内容 </THINK>
explain:解释的内容\taction:动作空间和对应的参数\tsummary:执行完当前步骤后的新历史总结
'''
		}
	]

	return status_conversation


def _extract_summary_history(history: list[str]) -> str:
	if not history:
		return ""

	try:
		last_item = json.loads(history[-1])
	except Exception:
		return ""

	if isinstance(last_item, dict):
		if isinstance(last_item.get("stepfun_fields"), dict):
			summary = last_item["stepfun_fields"].get("summary", "")
			if isinstance(summary, str):
				return summary
		summary = last_item.get("summary", "")
		if isinstance(summary, str):
			return summary
	return ""


def _extract_qa_prompt(history: list[str]) -> str:
	historical_qa = []
	pending_question = None

	for item in history:
		try:
			parsed = json.loads(item)
		except Exception:
			continue

		if not isinstance(parsed, dict):
			continue

		if parsed.get("action") == "info":
			pending_question = parsed.get("parameters", {}).get("question")
			continue

		user_reply = parsed.get("user_reply")
		if pending_question and isinstance(user_reply, str) and user_reply.strip():
			historical_qa.append((pending_question, user_reply.strip()))
			pending_question = None
			continue

	if len(historical_qa) > 0:
		return "这是你和用户的对话历史： " + "\n" + "\n".join(
			[f"你曾经提出的问题：{qa[0]}\n\n用户对你的指示：{qa[1]}" for qa in historical_qa]
		) + "\n\n 你需要更加注意用户最后的指示。 "
	return ""


def build_stepfun_messages(task: str, history: list[str], screenshot_b64: str, device_type: str) -> list[dict]:
	del device_type
	current_image = f"data:image/jpeg;base64,{screenshot_b64}"
	summary_history = _extract_summary_history(history)
	qa_prompt = _extract_qa_prompt(history)

	# StepFun expects a single user message that contains the fixed task prompt,
	# current screenshot, and compressed history summary.
	conversations = [
		{
			"type": "text",
			"text": task_define_prompt
		}
	] + make_status_prompt(
		task,
		current_image,
		[],
		summary_history,
		qa_prompt
	)

	return [
		{
			"role": "user",
			"content": conversations
		}
	]