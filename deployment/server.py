from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any
import json
import traceback
from openai import OpenAI
import copy
import io, base64
from PIL import Image
import time

app = FastAPI()

decider_client = None
grounder_client = None
planner_client = None

use_qwen3 = False

PLANNER_PROMPT = '''
## 角色定义
你是一个用户意图识别和智能手机应用分类助手。你需要根据用户的任务描述，输出能完成用户指定任务的应用类别。

## 任务描述
用户想要完成的任务是："{task_description}"

## 应用分类列表

- 通讯：示例应用：微信；示例任务：帮我给xxx发一条微信

- 社交：示例应用：微博，知乎，小红书；示例任务：帮我关注博主xxx

- 外卖：示例应用：饿了么，美团；示例任务：帮我点一份麦当劳的汉堡外卖

- 网购：示例应用：淘宝；示例任务：帮我下单购买一条白色连衣裙

- 视频：示例应用：爱奇艺，bilibili；示例任务：帮我播放xxx的第一条视频

- 酒店：示例应用：携程，飞猪；示例任务：帮我订一间汉庭酒店的大床房

- 旅行：示例应用：12306；示例任务：帮我查询上海到北京，9月1日出发的火车票

- 音乐：示例应用：网易云音乐；示例任务：帮我放一首周杰伦的歌曲

- 地图：示例应用：高德地图；示例任务：帮我导航到上海人民广场

- 打车：示例应用：滴滴出行；示例任务：帮我打车到上海交通大学


## 输出格式
请严格按照以下JSON格式输出：
```json
{{
    "reasoning": "分析任务内容，说明你选择这个应用分类的原因",
    "class": "任务分类，只能从上述应用分类列表中选择一个"
}}
```

## 重要规则
1. 应用分类只能从上述列表中选择
2. 如果应用分类列表中，没有一类应用能够贴合用户需求，"class"字段中请返回空字符串，也就是""
3. 类别必须完全匹配列表中的类别名，不能进行任何修改
4. 请你综合考虑每类应用的应用场景、示例任务，以及用户的实际输入，进行选择
'''.strip()

rewrite_cache = {}
task_refcnt = {}

terminate_checklist = [
    "当前页面未按预期加载",
    "进入了错误的页面",
    "打开了不合预期的页面",
    "当前打开了错误页面",
    "当前页面不合预期",
    "需要用户介入",
    "需要用户接管",
]

class_default_app = {
    "通讯": "微信",
    "社交": "小红书",
    "外卖": "饿了么",
    "网购": "淘宝",
    "视频": "bilibili",
    "酒店": "携程",
    "旅行": "携程",
    "音乐": "网易云",
    "地图": "高德",
    "打车": "高德",
}

supported_apps = {
    "支付宝": "com.eg.android.AlipayGphone",
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
    "高德": "com.autonavi.minimap",
    "12306": "com.MobileTicket",
    "钉钉": "com.alibaba.android.rimet"
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
{{"reasoning": "Your reasoning here", "action": "The next action (one of click, input, swipe, wait, done)", "parameters": {{"param1": "value1", ...}}}}
Remember your task is "{task_repeat}".'''

GROUNDER_PROMPT = '''
Based on the screenshot, user's intent and the description of the target UI element, provide the bounding box of the element using **absolute coordinates**.
User's intent: {reasoning}
Target element's description: {description}
Your output should be a JSON object with the following format:
{{"bbox": [x1, y1, x2, y2]}}'''

GROUNDER_PROMPT_QWEN3 = '''
Based on user's intent and the description of the target UI element, locate the element in the screenshot.
User's intent: {reasoning}
Target element's description: {description}
Report the bbox coordinates in JSON format.'''

class ResponseBody(BaseModel):
    reasoning: str
    action: str
    parameters: Dict[str, Any]

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

    start = time.perf_counter()
    response = model_client.chat.completions.create(
        model="",
        messages=messages,
        temperature=0,
    )
    print(f"Model response time: {time.perf_counter() - start:.2f} seconds")
    return response.choices[0].message.content

def rewrite_task(original_task: str, app_name):
    ret = original_task
    # add your custom rewriting rules here
    return ret

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

# async 
def cleanup_task(task):
    # async with cache_lock:
    if task_refcnt.get(task, 0) > 0:
        task_refcnt[task] -= 1
        if task_refcnt[task] == 0:
            rewrite_cache.pop(task, None)
            task_refcnt.pop(task, None)

# Define the POST endpoint
@app.post("/v1", response_model=ResponseBody)
async def v1(request_body: RequestBody):
    try:
        if request_body.task.strip() == "":
            return ResponseBody(
                reasoning="任务不能为空，任务终止",
                action="terminate",
                parameters={}
            )
        history = request_body.history
        task = request_body.task
        if len(history) == 0:
            try:
                print(f"Task: {task}")
                app_name, package_name = try_find_app(task)
                if app_name is None:
                    planner_prompt = PLANNER_PROMPT.format(task_description=task)
                    planner_output = get_model_output(planner_client, planner_prompt)
                    print(planner_output)
                    planner_output = planner_output.replace("```json", "").replace("```", "")
                    planner_output_json = json.loads(planner_output)
                    classification = planner_output_json["class"]
                    if classification not in class_default_app:
                        app_name, package_name = None, ""
                    else:
                        app_name = class_default_app[classification]
                        package_name = supported_apps[app_name]
            except Exception as e:
                traceback.print_exc()
                app_name, package_name = None, ""
            if app_name is None or app_name == "" or package_name == "":
                reasoning = f"暂不支持用户任务\"{task}\"需要打开的应用，任务终止"
                return ResponseBody(
                    reasoning=reasoning,
                    action="terminate",
                    parameters={}
                )
            else:
                reasoning = f"为了完成用户任务\"{task}\", 我需要打开应用\"{app_name}\""
                # async with cache_lock:
                task_refcnt[task] = task_refcnt.get(task, 0) + 1
                if task_refcnt[task] == 1:
                    rewrite_cache[task] = rewrite_task(task, app_name)
                return ResponseBody(
                    reasoning=reasoning,
                    action="open_app",
                    parameters={
                        "package_name": package_name,
                    }
                )
                    
        # async with cache_lock:
        rewritten_task = rewrite_cache.get(task, task)
        
        # print("raw history: ", history)
        history = validate_history(history)
        # print("cleaned history: ", history)
        if len(history) == 0:
            history_str = "(No history)"
        else:
            history_str = "\n".join(f"{idx}. {act}" for idx, act in enumerate(history, start=1))

        img_b64 = request_body.image
        
        pil_img = Image.open(io.BytesIO(base64.b64decode(img_b64)))
        width, height = pil_img.size
        print(f"Received image of size: {width}x{height}")

        decider_prompt = DECIDER_PROMPT.format(task=rewritten_task, history=history_str, task_repeat=rewritten_task)
        decider_output = get_model_output(decider_client, decider_prompt, img_b64)
        print(decider_output)
        decider_output_json = json.loads(decider_output)
        reasoning = decider_output_json["reasoning"]
        if should_terminate(reasoning):
            # await 
            cleanup_task(task)
            return ResponseBody(
                reasoning=reasoning,
                action="terminate",
                parameters={}
            )
        action = decider_output_json["action"]
        parameters = decider_output_json["parameters"]
        if action == "click":
            grounder_prompt_fmt = GROUNDER_PROMPT_QWEN3 if use_qwen3 else GROUNDER_PROMPT
            grounder_prompt = grounder_prompt_fmt.format(reasoning=reasoning, description=parameters["target_element"])
            grounder_output = get_model_output(grounder_client, grounder_prompt, img_b64)
            print(grounder_output)
            if grounder_output.startswith("```json"):
                grounder_output = grounder_output.replace("```json", "").replace("```", "")
            grounder_output_json = json.loads(grounder_output)
            if isinstance(grounder_output_json, list):
                grounder_output_json = grounder_output_json[0]
            bbox = grounder_output_json.get("bbox", grounder_output_json.get("bbox_2d", None))
            if use_qwen3:
                bbox[0] = bbox[0] / 1000 * width
                bbox[2] = bbox[2] / 1000 * width
                bbox[1] = bbox[1] / 1000 * height
                bbox[3] = bbox[3] / 1000 * height
            parameters["x"] = (bbox[0] + bbox[2]) // 2
            parameters["y"] = (bbox[1] + bbox[3]) // 2
        elif action == "done":
            cleanup_task(task)
        response = ResponseBody(
            reasoning=reasoning,
            action=action,
            parameters=parameters
        )
        return response

    except Exception as e:
        traceback.print_exc()
        # await 
        cleanup_task(request_body.task)
        # Handle potential errors
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )

# Optional: Add a root endpoint for health checks
@app.get("/")
async def root():
    return {"message": "Welcome to the Simple FastAPI Server! Use /docs for API documentation."}

if __name__ == "__main__":
    import uvicorn, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=22334)
    parser.add_argument("--planner_url", type=str, help="Base URL for planner model service")
    parser.add_argument("--decider_url", type=str, help="Base URL for decider model service")
    parser.add_argument("--grounder_url", type=str, help="Base URL for grounder model service")
    parser.add_argument("--use_qwen3", action='store_true', help="Use Qwen3-VL model format")
    args = parser.parse_args()
    use_qwen3 = args.use_qwen3
    decider_client = OpenAI(api_key="0", base_url=args.decider_url)
    grounder_client = OpenAI(api_key="0", base_url=args.grounder_url)
    planner_client = OpenAI(api_key="0", base_url=args.planner_url)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
