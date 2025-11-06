from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import time
import json
import base64
import shutil
import uvicorn
import sys
import logging

# 添加当前目录到Python路径，以便导入device模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.insert(0, project_root)

from utils.parse_xml import find_clicked_element
from device import create_device, Device

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 数据模型
class DeviceConfig(BaseModel):
    """Device configuration model"""
    device_type: str = "Android"  # "Android" or "Harmony"
    adb_endpoint: Optional[str] = None  # Optional ADB endpoint for Android

class ClickAction(BaseModel):
    x: int
    y: int

class SwipeAction(BaseModel):
    startX: int
    startY: int
    endX: int
    endY: int
    direction: str  # 'up', 'down', 'left', 'right'

class InputAction(BaseModel):
    text: str

class TaskDescription(BaseModel):
    description: str
    app_name: str
    task_type: str

screenshot_path = "screenshot-collect.jpg"

currentDataIndex = 0
action_history = []
current_task_description = ""  # 当前任务描述
current_app_name = ""  # 当前应用名称
current_task_type = ""  # 当前任务类型
is_suspended = False  # 是否处于人工介入状态

device: Device = None  # 设备连接对象
device_type = "Android"  # 当前设备类型
hierarchy = None  # 层次结构数据

app = FastAPI()

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件服务
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

def save_screenshot():
    action_count = len(action_history)

    # 创建数据目录
    session_base_dir = os.path.dirname(__file__)
    data_base_dir = os.path.join(session_base_dir, 'data')
    app_dir = os.path.join(data_base_dir, current_app_name)
    task_type_dir = os.path.join(app_dir, current_task_type)
    data_dir = os.path.join(task_type_dir, str(currentDataIndex))

    # 复制当前截图到数据目录
    if os.path.exists(screenshot_path):
        screenshot_save_path = os.path.join(data_dir, f'{action_count + 1}.jpg')
        shutil.copy2(screenshot_path, screenshot_save_path)

def get_current_hierarchy_and_screenshot(sleep_time = 0):
    global hierarchy
    time.sleep(sleep_time)
    hierarchy = device.dump_hierarchy()
    
    # with open("hierarchy.xml", "w", encoding="utf-8") as f:
    #     f.write(hierarchy)

    device.screenshot(screenshot_path)
    print(f"操作完成，已重新截图和获取层次结构。总操作数: {len(action_history)}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """返回前端页面"""
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/init_device")
async def init_device(config: DeviceConfig):
    """Initialize device connection
    
    Args:
        config: Device configuration (type and optional adb_endpoint)
        
    Returns:
        Connection status and device information
    """
    global device, device_type
    
    try:
        logger.info(f"Initializing {config.device_type} device...")
        logger.info(f"ADB endpoint: {config.adb_endpoint}")
        
        device = create_device(config.device_type, config.adb_endpoint)
        device_type = config.device_type
        
        logger.info(f"✅ {config.device_type} device initialized successfully")
        device.unlock()
        return {
            "status": "success",
            "message": f"{config.device_type} device initialized successfully",
            "device_type": device_type
        }
    except ModuleNotFoundError as e:
        error_msg = f"Missing dependency: {str(e)}. Please install required packages."
        logger.error(error_msg)
        raise HTTPException(
            status_code=400,
            detail=error_msg
        )
    except Exception as e:
        error_msg = f"Failed to initialize {config.device_type} device: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )

@app.get("/device_status")
async def get_device_status():
    """Get current device status"""
    if device is None:
        return {
            "status": "disconnected",
            "device_type": None
        }
    
    return {
        "status": "connected",
        "device_type": device_type
    }

@app.get("/screenshot")
async def get_screenshot():
    """获取最新截图文件和层次结构信息"""
    try:
        get_current_hierarchy_and_screenshot()
        with open(screenshot_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
        return {
            "status": "success",
            "image_data": f"data:image/jpeg;base64,{image_data}",
            "hierarchy": hierarchy
        }
      
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取截图失败: {str(e)}")

@app.post("/click")
async def handle_click(action: ClickAction):
    """处理点击操作"""
    try:
        # 确保坐标为整数（舍入）
        x = round(action.x)
        y = round(action.y)
        
        # 如果处于suspend状态，只执行操作但不记录
        if is_suspended:
            logger.info(f"Click in suspend mode: ({x}, {y}) - 不记录操作")
            device.click(x, y)
            return {
                "status": "success",
                "message": f"点击操作已执行但未记录 (人工介入模式): ({x}, {y})",
                "action": "click",
                "coordinates": {"x": x, "y": y},
                "suspended": True,
                "action_count": len(action_history)
            }
        
        element_bounds = find_clicked_element(hierarchy, x, y)
        if element_bounds:
            element_bounds = [round(coord) for coord in element_bounds]
        
        get_current_hierarchy_and_screenshot()
        save_screenshot()
        device.click(x, y)
        action_record = {
            "type": "click",
            "position": {"x": x, "y": y},  # 使用嵌套结构以保持一致性
            "position_x": x,  # 也保留扁平结构以向后兼容
            "position_y": y,
            "bounds": element_bounds
        }
        print(action_record)
        action_history.append(action_record)

        return {
            "status": "success", 
            "message": f"点击操作完成: ({x}, {y})",
            "action": "click",
            "coordinates": {"x": x, "y": y},
            "clicked_bounds": element_bounds,
            "action_count": len(action_history)
        }
    
    except Exception as e:
        logger.error(f"点击操作失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"点击操作失败: {str(e)}")

@app.post("/swipe")
async def handle_swipe(action: SwipeAction):
    """处理滑动操作"""
    try:
        # 确保坐标为整数（舍入）
        startX = round(action.startX)
        startY = round(action.startY)
        endX = round(action.endX)
        endY = round(action.endY)
        
        # 如果处于suspend状态，只执行操作但不记录
        if is_suspended:
            logger.info(f"Swipe in suspend mode: ({startX}, {startY}) -> ({endX}, {endY}) - 不记录操作")
            device.swipe(startX, startY, endX, endY, duration=0.1)
            return {
                "status": "success",
                "message": f"滑动操作已执行但未记录 (人工介入模式): ({startX}, {startY}) → ({endX}, {endY})",
                "action": "swipe",
                "start": {"x": startX, "y": startY},
                "end": {"x": endX, "y": endY},
                "suspended": True,
                "action_count": len(action_history)
            }
        
        get_current_hierarchy_and_screenshot()
        save_screenshot()
        device.swipe(startX, startY, endX, endY, duration=0.1)
        action_record = {
            "type": "swipe",
            "press_position": {"x": startX, "y": startY},  # 使用嵌套结构
            "release_position": {"x": endX, "y": endY},  # 使用嵌套结构
            "press_position_x": startX,  # 保留扁平结构以向后兼容
            "press_position_y": startY,
            "release_position_x": endX,
            "release_position_y": endY,
            "direction": action.direction
        }
        print(action_record)
        action_history.append(action_record)

        return {
            "status": "success",
            "message": f"滑动操作完成: ({startX}, {startY}) → ({endX}, {endY}) [{action.direction}]",
            "action": "swipe",
            "start": {"x": startX, "y": startY},
            "end": {"x": endX, "y": endY},
            "direction": action.direction,
            "action_count": len(action_history)
        }
    
    except Exception as e:
        logger.error(f"滑动操作失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"滑动操作失败: {str(e)}")

@app.post("/input")
async def handle_input(action: InputAction):
    """处理文本输入操作"""
    if device is None:
        raise HTTPException(status_code=400, detail="Device not initialized")
    
    try:
        logger.info(f"Text input action received: '{action.text}'")
        
        # 如果处于suspend状态，只执行操作但不记录
        if is_suspended:
            logger.info(f"Input in suspend mode: '{action.text}' - 不记录操作")
            device.input(action.text)
            return {
                "status": "success",
                "message": f"文本输入已执行但未记录 (人工介入模式): '{action.text}'",
                "action": "input",
                "text": action.text,
                "suspended": True,
                "action_count": len(action_history)
            }
        
        get_current_hierarchy_and_screenshot()
        save_screenshot()
        
        # Use the device's input method instead of direct shell access
        logger.info(f"Calling device.input() with text: '{action.text}'")
        device.input(action.text)
        logger.info(f"Device.input() completed successfully")
        
        action_record = {
            "type": "input",
            "text": action.text
        }
        print(action_record)
        action_history.append(action_record)
        
        logger.info(f"Input action recorded successfully")
        
        return {
            "status": "success",
            "message": f"输入操作完成",
            "action": "input",
            "text": action.text,
            "action_count": len(action_history)
        }
    
    except Exception as e:
        logger.error(f"输入操作失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"输入操作失败: {str(e)}")

@app.post("/wait")
async def handle_wait():
    """处理等待操作 - 记录当前页面截图和'wait'动作"""
    if device is None:
        raise HTTPException(status_code=400, detail="Device not initialized")
    
    try:
        logger.info("Wait action triggered")
        get_current_hierarchy_and_screenshot()
        save_screenshot()
        
        action_record = {
            "type": "wait"
        }
        print(action_record)
        action_history.append(action_record)
        
        logger.info(f"Wait action recorded successfully")
        
        return {
            "status": "success",
            "message": "等待操作已记录",
            "action": "wait",
            "action_count": len(action_history)
        }
    
    except Exception as e:
        logger.error(f"等待操作失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"等待操作失败: {str(e)}")

@app.post("/suspend")
async def handle_suspend():
    """处理人工介入操作 - 切换suspend状态"""
    global is_suspended
    
    if device is None:
        raise HTTPException(status_code=400, detail="Device not initialized")
    
    try:
        is_suspended = not is_suspended
        
        if is_suspended:
            logger.info("Suspend mode activated - human intervention started")
            action_record = {
                "type": "suspend",
                "action": "start"
            }
        else:
            logger.info("Suspend mode deactivated - human intervention ended")
            action_record = {
                "type": "suspend",
                "action": "end"
            }
        
        print(action_record)
        action_history.append(action_record)
        
        return {
            "status": "success",
            "message": "人工介入模式" + ("已启动" if is_suspended else "已关闭"),
            "action": "suspend",
            "is_suspended": is_suspended,
            "action_count": len(action_history)
        }
    
    except Exception as e:
        logger.error(f"人工介入操作失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"人工介入操作失败: {str(e)}")

@app.get("/suspend_status")
async def get_suspend_status():
    """获取人工介入模式状态"""
    return {
        "is_suspended": is_suspended
    }

@app.get("/action_history")
async def get_action_history():
    """获取操作历史记录"""
    return {
        "status": "success",
        "total_actions": len(action_history),
        "actions": action_history
    }

@app.post("/save_data")
async def save_current_data():
    """保存当前数据并清空历史记录"""
    global currentDataIndex
    global action_history

    try:
        get_current_hierarchy_and_screenshot()
        save_screenshot()
        action_record = {
            "type": "done"
        }
        action_history.append(action_record)
        action_count = len(action_history)

        app_dir = os.path.join(os.path.dirname(__file__), 'data', current_app_name)
        task_type_dir = os.path.join(app_dir, current_task_type)
        data_dir = os.path.join(task_type_dir, str(currentDataIndex))
        json_file_path = os.path.join(data_dir, 'actions.json')
        
        save_data = {
            "app_name": current_app_name,
            "task_type": current_task_type,
            "task_description": current_task_description,
            "action_count": action_count,
            "actions": action_history
        }
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=4)
  
        action_history.clear()
        global is_suspended
        is_suspended = False  # 重置suspend状态

        # [Info]
        print(f"第 {currentDataIndex} 条数据已保存")
        print(f"应用：{current_app_name} | 任务类型：{current_task_type}")
        print(f"包含 {action_count} 个操作记录")
        print("操作历史记录已清空")
        
        return {
            "status": "success",
            "message": f"第 {currentDataIndex} 条数据已保存",
            "data_index": currentDataIndex,
            "saved_actions": action_count
        }
    except Exception as e:
        print(f"保存数据失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"保存数据失败: {str(e)}")

@app.post("/delete_data")
async def delete_current_data():
    """保存当前数据并清空历史记录"""
    global currentDataIndex

    try:
        app_dir = os.path.join(os.path.dirname(__file__), 'data', current_app_name)
        task_type_dir = os.path.join(app_dir, current_task_type)
        data_dir = os.path.join(task_type_dir, str(currentDataIndex))

        # 删除数据目录
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
    
        action_history.clear()
        global is_suspended
        is_suspended = False  # 重置suspend状态

        return {
            "status": "success",
            "message": f"第 {currentDataIndex} 条数据已删除",
            "data_index": currentDataIndex
        }
    except Exception as e:
        logger.error(f"删除数据失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除数据失败: {str(e)}")


# Device-specific app mappings
def get_app_packages(dev_type: str = "Android"):
    """Get app packages for specific device type"""
    if dev_type == "Android":
        return {
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
            "酷狗音乐": "com.kugou.android",

            "高德地图": "com.autonavi.minimap"
        }
    elif dev_type == "Harmony":
        return {
            "携程": "com.ctrip.harmonynext",
            "飞猪": "com.fliggy.hmos",
            "同城": "com.tongcheng.hmos",
            "饿了么": "me.ele.eleme",
            "知乎": "com.zhihu.hmos",
            "哔哩哔哩": "yylx.danmaku.bili",
            "微信": "com.tencent.wechat",
            "小红书": "com.xingin.xhs_hos",
            "QQ音乐": "com.tencent.hm.qqmusic",
            "高德地图": "com.amap.hmapp",
            "淘宝": "com.taobao.taobao4hmos",
            "微博": "com.sina.weibo.stage",
            "京东": "com.jd.hm.mall",
            "浏览器": "com.huawei.hmos.browser",
            "拼多多": "com.xunmeng.pinduoduo.hos"
        }
    else:
        return {}

@app.post("/set_task_description")
async def set_task_description(task: TaskDescription):
    """设置任务描述"""
    global currentDataIndex
    global current_task_description
    global current_app_name
    global current_task_type
    try:
        current_app_name = task.app_name
        current_task_type = task.task_type
        current_task_description = task.description

        # 创建新的目录结构：data/<应用名称>/<任务类型>/<数据索引>/
        session_base_dir = os.path.dirname(__file__)
        if not os.path.exists(session_base_dir):
            os.makedirs(session_base_dir)

        data_base_dir = os.path.join(session_base_dir, 'data')
        if not os.path.exists(data_base_dir):
            os.makedirs(data_base_dir)
        
        app_dir = os.path.join(data_base_dir, current_app_name)
        if not os.path.exists(app_dir):
            os.makedirs(app_dir)
            
        task_type_dir = os.path.join(app_dir, current_task_type)
        if not os.path.exists(task_type_dir):
            os.makedirs(task_type_dir)

        # 遍历现有数据目录，找到最大的索引
        existing_dirs = [d for d in os.listdir(task_type_dir) if os.path.isdir(os.path.join(task_type_dir, d)) and d.isdigit()]
        if existing_dirs:
            currentDataIndex = max(int(d) for d in existing_dirs) + 1
        else:
            currentDataIndex = 1
        data_dir = os.path.join(task_type_dir, str(currentDataIndex))
        os.makedirs(data_dir)

        print(f"\n{'='*50}")
        print(f"📋 新任务开始")
        print(f"应用名称: {current_app_name}")
        print(f"任务类型: {current_task_type}")
        print(f"任务描述: {current_task_description}")
        print(f"数据目录: data/{current_app_name}/{current_task_type}/{currentDataIndex}/")
        print(f"{'='*50}\n")
        
        # Use device-specific app packages
        app_packages = get_app_packages(device_type)
        package_name = app_packages.get(current_app_name)
        if not package_name:
            raise ValueError(f"App '{current_app_name}' is not registered for device type '{device_type}'.")
        device.start_app(current_app_name)

        return {
            "status": "success", 
            "message": "任务描述已设置",
            "description": current_task_description,
            "app_name": current_app_name,
            "task_type": current_task_type
        }
    except Exception as e:
        logger.error(f"设置任务描述失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"设置任务描述失败: {str(e)}")

if __name__ == "__main__":
    print("启动服务器...")
    print("访问 http://localhost:9000 查看前端页面")
    print("需要先通过 API 初始化设备连接")
    uvicorn.run(app, host="0.0.0.0", port=9000)