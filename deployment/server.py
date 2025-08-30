from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, List, Any
import json
import traceback
from openai import OpenAI
import copy

app = FastAPI()

decider_client = None
grounder_client = None
planner_client = None

terminate_checklist = [
    "当前页面未按预期加载",
    "进入了错误的页面",
    "打开了不合预期的页面",
    "当前打开了错误页面",
    "当前页面不合预期",
    "需要用户介入",
    "需要用户接管",
]

supported_apps = {
    "微信": "com.tencent.mm",
    "QQ": "com.tencent.mobileqq",
    "微博": "com.sina.weibo",
    "饿了么": "me.ele",
    "美团": "com.sankuai.meituan",
    "bilibili": "tv.danmaku.bili",
    "B站": "tv.danmaku.bili",
    "爱奇艺": "com.qiyi.video",
    "腾讯视频": "com.tencent.qqlive",
    "淘宝": "com.taobao.taobao",
    "京东": "com.jingdong.app.mall",
    "携程": "ctrip.android.view",
    "去哪儿": "com.Qunar",
    "知乎": "com.zhihu.android",
    "小红书": "com.xingin.xhs",
    "QQ音乐": "com.tencent.qqmusic",
    "网易云": "com.netease.cloudmusic",
    "高德": "com.autonavi.minimap"
}

def should_terminate(reasoning: str):
    for phrase in terminate_checklist:
        if phrase in reasoning:
            return True
    return False

def try_find_app(task_description: str):
    longest_match = ""
    for app in supported_apps:
        if app.lower() in task_description.lower() and len(app) > len(longest_match):
            longest_match = app
    if longest_match != "":
        return longest_match, supported_apps[longest_match]
    else:
        return None, None

PLANNER_PROMPT = '''
## 角色定义
你是一个任务描述优化专家和智能手机应用选择助手。你需要根据用户的任务描述，选择一个最合适的应用启动。

## 任务描述
用户想要完成的任务是："{task_description}"

## 可用应用列表
以下是可用的应用及其包名：
- 微信: com.tencent.mm
- QQ: com.tencent.mobileqq
- 新浪微博: com.sina.weibo
- 饿了么: me.ele
- 美团: com.sankuai.meituan
- bilibili: tv.danmaku.bili
- 爱奇艺: com.qiyi.video
- 腾讯视频: com.tencent.qqlive
- 淘宝: com.taobao.taobao
- 京东: com.jingdong.app.mall
- 携程: ctrip.android.view
- 去哪儿: com.Qunar
- 知乎: com.zhihu.android
- 小红书: com.xingin.xhs
- QQ音乐: com.tencent.qqmusic
- 网易云音乐: com.netease.cloudmusic
- 高德地图：com.autonavi.minimap

## 默认应用列表
以下是各个应用类别的默认应用：

通讯应用：
- 微信: com.tencent.mm

外卖应用：
- 饿了么: me.ele

视频应用：
- bilibili: tv.danmaku.bili

酒店/旅行应用：
- 携程: ctrip.android.view

社区应用：
- 小红书: com.xingin.xhs

音乐应用：
- 网易云音乐: com.netease.cloudmusic

地图/打车应用：
- 高德地图：com.autonavi.minimap


## 输出格式
请严格按照以下JSON格式输出：
```json
{{
    "reasoning": "分析任务内容，说明为什么选择这个应用最合适",
    "app_name": "选择的应用名称",
    "package_name": "选择的应用包名",
}}
```

## 重要规则
1. 只能从上述可用应用列表中选择
2. 如果应用列表中不存在能够完成用户任务的应用，或者用户显式指定了不在列表中的应用，"app_name"和"package_name"请返回空字符串，也就是""
3. 必须选择最符合任务需求的应用
4. 包名必须完全匹配列表中的包名，不能修改
5. 若用户没有显式指定应用名称，你只能根据任务类型，从**默认应用列表**中挑选，**不能挑选非默认应用**
'''.strip()

DECIDER_PROMPT = '''
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
{{"reasoning": "Your reasoning here", "action": "The next action (one of click, input, swipe, wait, done)", "parameters": {{"param1": "value1", ...}}}}'''

GROUNDER_PROMPT = '''
Based on the screenshot, user's intent and the description of the target UI element, provide the bounding box of the element using **absolute coordinates**.
User's intent: {reasoning}
Target element's description: {description}
Your output should be a JSON object with the following format:
{{"bbox": [x1, y1, x2, y2]}}'''

# Define the response body model using Pydantic
class ResponseBody(BaseModel):
    reasoning: str
    action: str
    parameters: Dict[str, Any]

# Define the request body model using Pydantic
class RequestBody(BaseModel):
    task: str
    image: str
    history: List[str]

def get_model_output(model_client, prompt, image_b64=None):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if image_b64 is not None:
        messages[0]["content"].insert(0, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})

    response = model_client.chat.completions.create(
        model="",
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content

def validate_history(history: List[str]):
    filtered = []
    allowed_keys = {
        "click": {"target_element"},
        "input": {"text"},
        "swipe": {"direction"},
        "done": {}
    }
    for h in history:
        old = json.loads(h)
        new = copy.deepcopy(old)
        action = old["action"]
        if action not in allowed_keys:
            continue
        for k in old["parameters"]:
            if k not in allowed_keys[action]:
                new["parameters"].pop(k)
        filtered.append(new)
    
    return [json.dumps(act, ensure_ascii=False) for act in filtered]

@app.post("/v1", response_model=ResponseBody)
async def v1(request_body: RequestBody):
    try:
        if request_body.task.strip() == "":
            return ResponseBody(
                reasoning="任务不能为空，任务终止",
                action="terminate",
                # action="done",
                parameters={}
            )
        history = request_body.history
        if len(history) == 0:
            app_name, package_name = try_find_app(request_body.task)
            if app_name is None:
                planner_prompt = PLANNER_PROMPT.format(task_description=request_body.task)
                planner_output = get_model_output(planner_client, planner_prompt)
                print(planner_output)
                planner_output = planner_output.replace("```json", "").replace("```", "")
                planner_output_json = json.loads(planner_output)
                app_name = planner_output_json["app_name"]
                package_name = planner_output_json["package_name"]
                if package_name not in supported_apps.values():
                    app_name, package_name = None, ""
            if app_name is None or app_name == "" or package_name == "":
                reasoning = f"无法识别用户任务\"{request_body.task}\"需要打开的应用，任务终止"
                return ResponseBody(
                    reasoning=reasoning,
                    action="terminate",
                    # action="done",
                    parameters={}
                )
            else:
                reasoning = f"为了完成用户任务\"{request_body.task}\", 我需要打开应用\"{app_name}\""
                return ResponseBody(
                    reasoning=reasoning,
                    action="open_app",
                    parameters={
                        "package_name": package_name,
                    }
                )
        
        # print("raw history: ", history)
        history = validate_history(history)
        # print("cleaned history: ", history)
        if len(history) == 0:
            history_str = "(No history)"
        else:
            history_str = "\n".join(f"{idx}. {act}" for idx, act in enumerate(history, start=1))

        img_b64 = request_body.image
        decider_prompt = DECIDER_PROMPT.format(task=request_body.task, history=history_str)
        decider_output = get_model_output(decider_client, decider_prompt, img_b64)
        print(decider_output)
        decider_output_json = json.loads(decider_output)
        reasoning = decider_output_json["reasoning"]
        if should_terminate(reasoning):
            return ResponseBody(
                reasoning=reasoning,
                action="terminate",
                parameters={}
            )
        action = decider_output_json["action"]
        parameters = decider_output_json["parameters"]
        if action == "click":
            grounder_prompt = GROUNDER_PROMPT.format(reasoning=reasoning, description=parameters["target_element"])
            grounder_output = get_model_output(grounder_client, grounder_prompt, img_b64)
            print(grounder_output)
            grounder_output_json = json.loads(grounder_output)
            bbox = grounder_output_json["bbox"]
            parameters["x"] = (bbox[0] + bbox[2]) // 2
            parameters["y"] = (bbox[1] + bbox[3]) // 2
        response = ResponseBody(
            reasoning=reasoning,
            action=action,
            parameters=parameters
        )
        return response

    except Exception as e:
        traceback.print_exc()
        # Handle potential errors
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )

# Optional: Add a root endpoint for health checks
@app.get("/")
async def root():
    return {}

if __name__ == "__main__":
    import uvicorn, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--service_ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=22334)
    parser.add_argument("--planner_port", type=int, default=18003)
    parser.add_argument("--decider_port", type=int, default=18001)
    parser.add_argument("--grounder_port", type=int, default=18002)
    args = parser.parse_args()
    decider_client = OpenAI(api_key="0", base_url=f"http://{args.service_ip}:{args.decider_port}/v1")
    grounder_client = OpenAI(api_key="0", base_url=f"http://{args.service_ip}:{args.grounder_port}/v1")
    planner_client = OpenAI(api_key="0", base_url=f"http://{args.service_ip}:{args.planner_port}/v1")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
