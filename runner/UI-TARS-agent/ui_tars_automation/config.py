#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置和数据结构定义
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class ActionResult:
    """操作结果"""
    success: bool
    message: str
    screenshot_path: Optional[str] = None
    xml_path: Optional[str] = None
    error: Optional[str] = None

@dataclass
class ExecutionConfig:
    """执行配置"""
    model_base_url: str = "http://192.168.12.152:8000/v1"
    model_name: str = "UI-TARS-7B-SFT"
    device_ip: Optional[str] = None  # None表示USB连接
    max_steps: int = 50
    step_delay: float = 1.5
    language: str = "Chinese"
    temperature: float = 0.0
    max_tokens: int = 400
    
    # 数据保存配置
    save_data: bool = True
    data_base_dir: str = "automation_data"
    save_screenshots: bool = True
    save_xml: bool = True

# 应用包名映射
APP_PACKAGES = {
    "微信": "com.tencent.mm",
    "QQ": "com.tencent.mobileqq", 
    "微博": "com.sina.weibo",
    
    "饿了么": "me.ele",
    "美团": "com.sankuai.meituan",

    "bilibili": "tv.danmaku.bili",
    "爱奇艺": "com.qiyi.video",
    "腾讯视频": "com.tencent.qqlive",
    "优酷": "com.youku.phone",

    "淘宝": "com.taobao.taobao",
    "京东": "com.jingdong.app.mall",

    "携程": "ctrip.android.view",
    "同城": "com.tongcheng.android",
    "飞猪": "com.taobao.trip",
    "去哪儿": "com.Qunar",
    "华住会": "com.htinns",

    "知乎": "com.zhihu.android",
    "小红书": "com.xingin.xhs",

    "QQ音乐": "com.tencent.qqmusic",
    "网易云音乐": "com.netease.cloudmusic",
    "酷狗音乐": "com.kugou.android"
}

# 内置提示词模板
MOBILE_PROMPT_TEMPLATE = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(point='<point>x1 y1</point>')
long_press(point='<point>x1 y1</point>')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
press_home()
press_back()
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.

## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.
- To open an app, use click() to tap on the app icon you can see in the screenshot, don't use open_app().
- Always look for app icons, buttons, or UI elements in the current screenshot and click on them.

## User Instruction
{instruction}"""
