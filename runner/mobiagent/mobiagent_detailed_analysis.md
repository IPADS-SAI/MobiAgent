# MobiAgent 核心执行引擎详细分析

## 📋 文件总体架构

这是 **MobiAgent** 的核心执行引擎，用于控制移动设备（Android/Harmony）自动化执行UI任务。文件包含约1738行代码。

整体执行流程：
```
输入任务 → Planner规划 → 启动应用 → 主循环执行 → 保存结果
```

---

## 🔧 核心模块详解

### 1. 导入模块和日志配置（第1-48行）

```python
from openai import OpenAI
import uiautomator2 as u2
from PIL import Image
import xml.etree.ElementTree as ET
from hmdriver2.driver import Driver
```

**核心依赖**：
- **OpenAI**：LLM服务客户端
- **uiautomator2**：Android UI自动化库
- **PIL/cv2**：图像处理
- **ET**：Android XML hierarchy解析
- **hmdriver2**：HarmonyOS设备驱动

**关键常量**：
- `MAX_STEPS = 35`：单个任务最多执行35步，防止死循环
- `factor = 0.5`：截图压缩比例（减少LLM处理时间）

**日志配置**：
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True  # 强制重置，避免重复配置
)
```

---

### 2. 设备抽象基类（第52-100行）

```python
class Device(ABC):
    @abstractmethod
    def start_app(self, app): pass
    @abstractmethod
    def app_stop(self, package_name): pass
    @abstractmethod
    def screenshot(self, path): pass
    @abstractmethod
    def click(self, x, y): pass
    @abstractmethod
    def input(self, text): pass
    @abstractmethod
    def swipe(self, direction): pass
    @abstractmethod
    def keyevent(self, key): pass
    @abstractmethod
    def dump_hierarchy(self): pass
```

**设计模式**：抽象基类定义统一接口，支持多设备适配

---

### 3. AndroidDevice 类（第102-188行）

**核心职责**：通过 uiautomator2 库控制Android设备

#### 初始化
```python
def __init__(self, adb_endpoint=None):
    self.d = u2.connect(adb_endpoint)  # 连接ADB
    self.app_package_names = {
        "携程": "ctrip.android.view",
        "支付宝": "com.eg.android.AlipayGphone",
        "微信": "com.tencent.mm",
        # ... 23个常用APP
    }
```

#### 关键方法

**`start_app(app)`**：
```python
def start_app(self, app):
    package_name = self.app_package_names.get(app)
    self.d.app_start(package_name, stop=True)
    time.sleep(1)
    if not self.d.app_wait(package_name, timeout=10):
        raise RuntimeError(f"Failed to start app '{app}'")
```

**`click(x, y)`**：
```python
def click(self, x, y):
    self.d.click(x, y)
    time.sleep(0.5)  # 等待动画
```

**`input(text)`**：实现跨语言文本输入
```python
def input(self, text):
    # 1. 保存当前输入法
    current_ime = self.d.current_ime()
    
    # 2. 切换到 AdbIME
    self.d.shell(['settings', 'put', 'secure', 'default_input_method', 
                  'com.android.adbkeyboard/.AdbIME'])
    time.sleep(0.5)
    
    # 3. Base64编码文本，通过broadcast发送
    charsb64 = base64.b64encode(text.encode('utf-8')).decode('utf-8')
    self.d.shell(['am', 'broadcast', '-a', 'ADB_INPUT_B64', 
                  '--es', 'msg', charsb64])
    time.sleep(0.5)
    
    # 4. 恢复原输入法
    self.d.shell(['settings', 'put', 'secure', 'default_input_method', 
                  current_ime])
```

**`dump_hierarchy()`**：导出UI树XML结构，用于UI元素定位

---

### 4. HarmonyDevice 类（第190-268行）

**核心职责**：通过 hmdriver2 库控制HarmonyOS设备

**特点**：
- 支持HarmonyOS原生应用和云端应用
- 接口基本同Android，但命令调用不同
- `force_start_app`确保强制启动
- `input_text`直接调用设备API而非ADB broadcast

```python
def start_app(self, app):
    package_name = self.app_package_names.get(app)
    self.d.force_start_app(package_name)
    time.sleep(1.5)

def input(self, text):
    self.d.input_text(text)  # 直接调用
```

---

### 5. 全局客户端初始化（第270-313行）

```python
decider_client = None      # 决策模型客户端
grounder_client = None     # 定位模型客户端
planner_client = None      # 规划模型客户端
experience_rr: ExperienceRR = None
preference_extractor = None

def init(service_ip, decider_port, grounder_port, planner_port, 
         enable_user_profile=False, use_graphrag=False, use_experience_rr=False):
    global decider_client, grounder_client, planner_client
    
    # 初始化三个OpenAI客户端，指向本地vLLM服务
    decider_client = OpenAI(
        api_key="0",
        base_url=f"http://{service_ip}:{decider_port}/v1"
    )
    grounder_client = OpenAI(
        api_key="0",
        base_url=f"http://{service_ip}:{grounder_port}/v1"
    )
    planner_client = OpenAI(
        api_key="0",
        base_url=f"http://{service_ip}:{planner_port}/v1"
    )
```

**三个模型的职责分工**：

| 模型 | 端口 | 职责 |
|------|------|------|
| Planner | 8002 | 解析自然语言任务 → 选择APP + 优化描述 |
| Decider | 8000 | 当前UI + 任务 → 决定下一步动作 |
| Grounder | 8001 | 动作描述 + 截图 → 定位UI元素坐标 |

---

### 6. Pydantic 模型定义（第315-380行）

#### ActionType 枚举
```python
class ActionType(str, Enum):
    CLICK = "click"
    INPUT = "input"
    SWIPE = "swipe"
    DONE = "done"
    STOP = "stop"
    TERMINATE = "terminate"
    WAIT = "wait"
```

#### ActionPlan 结构化输出模型
```python
class ActionPlan(BaseModel):
    reasoning: str = Field(
        description="执行此动作的思考过程"
    )
    action: ActionType = Field(
        description="要执行的下一个动作"
    )
    parameters: Dict[str, str] = Field(
        description="执行动作的参数",
        default_factory=dict
    )
```

**用途**：生成JSON schema给vLLM强制结构化输出

#### GroundResponse 模型
```python
class GroundResponse(BaseModel):
    coordinates: list[int]  # [x, y]
    bbox: list[int]         # [x1, y1, x2, y2]
    bbox_2d: list[int]      # 兼容多命名
```

---

### 7. JSON 解析工具（第382-410行）

```python
def parse_json_response(response_str: str) -> dict:
    """三级降级解析策略"""
    try:
        # 第1级：直接解析
        return json.loads(response_str)
    except json.JSONDecodeError:
        # 第2级：提取 ```json ... ``` 代码块
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', 
                              response_str, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        
        # 第3级：提取任意 { ... }
        json_match = re.search(r'(\{.*?\})', response_str, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        
        raise ValueError("无法找到有效JSON")
```

**健壮性特性**：
- 处理markdown代码块
- 处理多余换行符
- 处理中文省略号

```python
def robust_json_loads(s):
    s = s.strip()
    # 提取 ```json ... ``` 代码块
    codeblock = re.search(r"```json(.*?)```", s, re.DOTALL)
    if codeblock:
        s = codeblock.group(1).strip()
    # 替换中文省略号为英文
    s = s.replace("…", "...")
    # 去除多余换行
    s = s.replace("\r", "").replace("\n", " ")
    return json.loads(s)
```

---

### 8. 截图和坐标转换（第412-481行）

#### get_screenshot()
```python
def get_screenshot(device, device_type="Android"):
    """获取设备截图并编码为base64"""
    screenshot_path = f"screenshot-{device_type}.jpg"
    device.screenshot(screenshot_path)
    
    # 缩放到50%，减少LLM处理时间
    img = Image.open(screenshot_path)
    img = img.resize((int(img.width * factor), int(img.height * factor)), 
                     Image.Resampling.LANCZOS)
    
    # Base64编码
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    screenshot = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return screenshot
```

**作用**：获取缩放后的Base64编码截图，用于LLM视觉输入

#### 坐标转换
```python
def convert_qwen3_coordinates_to_absolute(bbox_or_coords, img_width, img_height, is_bbox=True):
    """Qwen3模型输出范围为0-1000，需要转换为绝对像素坐标"""
    if is_bbox:
        x1, y1, x2, y2 = bbox_or_coords
        x1 = int(x1 / 1000 * img_width)
        x2 = int(x2 / 1000 * img_width)
        y1 = int(y1 / 1000 * img_height)
        y2 = int(y2 / 1000 * img_height)
        return [x1, y1, x2, y2]
    else:
        x, y = bbox_or_coords
        x = int(x / 1000 * img_width)
        y = int(y / 1000 * img_height)
        return [x, y]
```

**重要注意**：不同Grounder模型输出格式不同
- **Qwen3**：返回 0-1000 范围的相对坐标
- **其他模型**：直接返回缩放图像上的像素坐标

---

### 9. 可视化函数（第483-510行）

```python
def create_swipe_visualization(data_dir, image_index, direction):
    """为滑动动作创建箭头可视化"""
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    
    # 计算箭头位置
    center_x, center_y = width // 2, height // 2
    arrow_length = min(width, height) // 4
    
    if direction == "up":
        start_point = (center_x, center_y + arrow_length // 2)
        end_point = (center_x, center_y - arrow_length // 2)
    # ... 其他方向
    
    # 绘制箭头
    cv2.arrowedLine(img, start_point, end_point, (255, 0, 0), 8, tipLength=0.3)
    cv2.putText(img, f"SWIPE {direction.upper()}", (text_x, text_y), 
                font, 1.5, (255, 0, 0), 3)
    cv2.imwrite(swipe_path, img)
```

**用途**：调试，在截图上绘制滑动方向箭头

---

### 10. Android UI树解析（第520-720行）

这是文件中**最复杂的部分**，实现"严格回放"（Strict Replay）的核心。

#### 基础工具函数

**`_parse_android_bounds(bounds_str)`**：
```python
def _parse_android_bounds(bounds_str: str) -> Optional[list[int]]:
    """解析 Android uiautomator bounds 字符串: "[x1,y1][x2,y2]" """
    if not bounds_str:
        return None
    m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
    if not m:
        return None
    return [int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))]
```

**`_bbox_center(bbox)`**：计算边界框中心点
```python
def _bbox_center(bbox: list[int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) // 2, (y1 + y2) // 2
```

**`_bbox_area(b)`**：计算边界框面积
```python
def _bbox_area(b: list[int]) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])
```

**`_point_in_bbox(x, y, b)`**：判断点是否在边界框内
```python
def _point_in_bbox(x: int, y: int, b: list[int]) -> bool:
    return b[0] <= x <= b[2] and b[1] <= y <= b[3]
```

#### UI节点特征提取

**`_extract_android_node_meta(node_attrib)`**：
```python
def _extract_android_node_meta(node_attrib: dict) -> dict:
    """提取UI节点的保守特征集"""
    keys = [
        "resource-id", "text", "content-desc", "class", "package",
        "clickable", "enabled", "selected", "checked", "focused",
        "scrollable", "long-clickable"
    ]
    meta = {k: node_attrib.get(k, "") for k in keys}
    meta["bounds"] = node_attrib.get("bounds", "")
    return meta
```

**用途**：提取节点的稳定特征，用于后续比对验证

#### 核心定位算法

**`_find_android_node_meta_by_bbox(hierarchy_xml, bbox)`**：
```python
def _find_android_node_meta_by_bbox(hierarchy_xml: str, bbox: list[int]) -> Optional[dict]:
    """根据点击bbox查找对应的UI节点"""
    try:
        root = ET.fromstring(hierarchy_xml)
    except Exception as e:
        logging.warning(f"Failed to parse Android hierarchy XML: {e}")
        return None

    cx, cy = _bbox_center(bbox)  # 计算点击中心
    best = None
    best_area = None
    
    for elem in root.iter():
        b = _parse_android_bounds(elem.attrib.get("bounds", ""))
        if not b:
            continue
        if _point_in_bbox(cx, cy, b):  # 中心点在bbox内
            area = _bbox_area(b)
            if best is None or area < best_area:  # 选择最小面积
                best = elem
                best_area = area
    
    if best is None:
        return None
    return _extract_android_node_meta(best.attrib)
```

**算法步骤**：
1. 解析XML树
2. 计算点击点中心坐标
3. 遍历所有UI节点
4. 找到包含该点的最小节点（最精确的匹配）
5. 提取该节点的特征

---

### 11. 多格式Bounds解析（第734-766行）

```python
def _parse_bounds_like(v: Any) -> Optional[list[int]]:
    """支持多种bounds格式"""
    if isinstance(v, list) and len(v) == 4:
        # 列表格式: [x1, y1, x2, y2]
        return [int(v[0]), int(v[1]), int(v[2]), int(v[3])]
    
    if isinstance(v, dict):
        # 字典格式: {left, top, right, bottom}
        keys = {k.lower(): k for k in v.keys()}
        if all(k in keys for k in ["left", "top", "right", "bottom"]):
            return [int(v[keys["left"]]), int(v[keys["top"]]), 
                   int(v[keys["right"]]), int(v[keys["bottom"]])]
    
    if isinstance(v, str):
        # Android格式: "[x1,y1][x2,y2]"
        b = _parse_android_bounds(v)
        if b:
            return b
        # CSV格式: "x1,y1,x2,y2"
        m = re.match(r"\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*", v)
        if m:
            return [int(m.group(1)), int(m.group(2)), 
                   int(m.group(3)), int(m.group(4))]
    
    return None
```

**目的**：兼容Android和HarmonyOS的不同UI树格式

---

### 12. HarmonyOS UI树解析（第768-820行）

```python
def _find_harmony_like_node_meta_by_bbox(hierarchy_json: Any, bbox: list[int]) -> Optional[dict]:
    """递归遍历JSON UI树"""
    cx, cy = _bbox_center(bbox)
    best_node = None
    best_area = None

    def walk(obj: Any):
        nonlocal best_node, best_area
        if isinstance(obj, dict):
            # 尝试从多个可能的字段名提取bounds
            b = None
            for k in ["bounds", "boundingRect", "rect", "bbox"]:
                if k in obj:
                    b = _parse_bounds_like(obj.get(k))
                    if b:
                        break
            
            if b and _point_in_bbox(cx, cy, b):
                area = _bbox_area(b)
                if best_node is None or area < best_area:
                    best_node = obj
                    best_area = area
            
            # 递归遍历子节点
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for it in obj:
                walk(it)

    walk(hierarchy_json)
    if best_node is None:
        return None
    meta = _extract_harmony_like_meta(best_node)
    return meta
```

**特点**：
- 支持多种字段名（bounds/boundingRect/rect）
- 递归遍历树状结构
- 选择最小面积匹配（最精确）

---

### 13. 严格回放检查（第822-828行）

```python
def strict_ui_meta_equal(a: Optional[dict], b: Optional[dict]) -> bool:
    """保守的严格匹配：要求完全相等"""
    if not a or not b:
        return False
    return a == b
```

**设计意图**：
- 在重播历史动作前，必须验证UI元素完全相同
- 防止UI变化导致的误点击
- 若验证失败，放弃回放，重新调用Decider

---

### 14. 主任务执行函数 `task_in_app()`（第830-1102行）

这是**整个文件的核心**，实现完整的任务执行循环。

#### 函数签名和初始化
```python
def task_in_app(app, old_task, task, template, device, data_dir, 
                bbox_flag=True, use_qwen3=True, device_type="Android"):
    history = []           # 当前任务的动作历史
    actions = []           # 执行的所有动作
    reacts = []            # Decider + Grounder的决策过程
    full_history = []      # 用于经验回放的完整历史
    extra_info = []        # 每步动作的额外信息
    
    # 加载提示模板
    if use_qwen3:
        grounder_prompt_template_bbox = load_prompt("grounder_qwen3_bbox.md")
        grounder_prompt_template_no_bbox = load_prompt("grounder_qwen3_coordinates.md")
        decider_prompt_template = load_prompt("decider_v2.md")
    else:
        grounder_prompt_template_bbox = load_prompt("grounder_bbox.md")
        grounder_prompt_template_no_bbox = load_prompt("grounder_coordinates.md")
        decider_prompt_template = load_prompt("decider_v2.md")
```

#### 经验回放逻辑（第865-911行）

```python
# Step 1: 严格条件检查（只有任务完全相同时才启用回放）
rr_enabled = bool(experience_rr and template and old_task == task)

if rr_enabled:
    logging.info("Finding replayable actions (strict same-task mode)")
    # 查询历史经验
    replay_info = experience_rr.query(task, template, enable_cross_task=False)
    
    # replay_info 结构：list[Union[str, MobiAgentAction]]
    # - str: 不可回放的子任务
    # - MobiAgentAction: 可回放的历史动作
    
    for item in replay_info:
        if isinstance(item, str):
            logging.info(f"Non-replayable subtask: {item}")
        elif isinstance(item, MobiAgentAction):
            logging.info(f"Replayable historical action: {item}")
```

#### Decider 模型调用（第913-934行）

```python
def _run_decider(task_text: str, history_text: str, screenshot_b64: str) -> str:
    """运行Decider模型一次"""
    decider_prompt = decider_prompt_template.format(
        task=task_text,
        history=history_text
    )
    logging.info(f"Decider prompt: \n{decider_prompt}")

    decider_start_time = time.time()
    decider_response_obj = decider_client.chat.completions.create(
        model=decider_model,
        messages=[
            {
                "role": "user",
                "content": [
                    # 多模态输入：图像 + 文本
                    {"type": "image_url", 
                     "image_url": {"url": f"data:image/jpeg;base64,{screenshot_b64}"}},
                    {"type": "text", "text": decider_prompt},
                ]
            }
        ],
        temperature=0,  # 确定性输出
        response_format={
            "type": "json_object",
            "schema": ActionPlan.model_json_schema()  # 强制JSON格式
        }
    )
    
    decider_response_str = decider_response_obj.choices[0].message.content
    decider_end_time = time.time()
    logging.info(f"Decider time taken: {decider_end_time - decider_start_time} seconds")
    return decider_response_str
```

**核心特性**：
- **多模态输入**：视觉（Base64截图）+ 文本（任务+历史）
- **结构化输出**：通过`response_format`强制JSON格式
- **Temperature=0**：确保确定性输出
- **时间记录**：用于性能分析

#### 主执行循环（第936-1085行）

```python
while True:
    if len(actions) >= MAX_STEPS:
        logging.info("Reached maximum steps, stopping the task.")
        break
```

**循环内容**：

##### 1️⃣ 获取截图和判断回放
```python
screenshot_resize = get_screenshot(device, device_type)

replay_this_step = False
if replay_info and replay_idx < len(replay_info) and (not executing_subtask):
    replay_item = replay_info[replay_idx]
    if isinstance(replay_item, str):
        # 子任务不可回放
        task = replay_item.rstrip("。") + "，然后结束任务，不要进行其他操作。"
        executing_subtask = True
        history.clear()
    elif isinstance(replay_item, MobiAgentAction):
        # 历史动作可回放
        replay_this_step = True
        decider_response_str = replay_item.model_dump(exclude={"extra_info"})
```

##### 2️⃣ 调用Decider或使用回放
```python
if not replay_this_step:
    decider_response_str = _run_decider(task, history_str, screenshot_resize)

decider_response = robust_json_loads(decider_response_str)
action = decider_response["action"]
```

##### 3️⃣ 保存截图和UI树
```python
# 保存截图
current_image = f"screenshot-{device_type}.jpg"
img = Image.open(img_path)
img.save(os.path.join(data_dir, f"{image_index}.jpg"))

# 保存UI层级树
hierarchy = device.dump_hierarchy()
if device_type == "Android":
    hierarchy_path = os.path.join(data_dir, f"{image_index}.xml")
    with open(hierarchy_path, "w", encoding="utf-8") as f:
        f.write(hierarchy)
else:
    hierarchy_path = os.path.join(data_dir, f"{image_index}.json")
    with open(hierarchy_path, "w", encoding="utf-8") as f:
        json.dump(hierarchy, f, ensure_ascii=False, indent=2)
```

##### 4️⃣ 动作执行分支

**DONE动作**：
```python
if action == "done":
    if replay_info and executing_subtask:
        logging.info(f"Subtask '{task}' completed.")
        history.clear()
        executing_subtask = False
    else:
        logging.info("Task completed.")
        actions.append({"type": "done", "action_index": image_index})
        break
```

**CLICK动作**：

a. 坐标定位：
```python
if replay_grounder_bbox is None:
    # 调用Grounder确定坐标
    reasoning = decider_response["reasoning"]
    target_element = decider_response["parameters"]["target_element"]
    grounder_prompt = grounder_prompt_template_bbox.format(
        reasoning=reasoning, 
        description=target_element
    )
    
    grounder_response_str = grounder_client.chat.completions.create(
        model=grounder_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", 
                     "image_url": {"url": f"data:image/jpeg;base64,{screenshot_resize}"}},
                    {"type": "text", "text": grounder_prompt},
                ]
            }
        ],
        temperature=0,
    ).choices[0].message.content
    
    grounder_response = parse_json_response(grounder_response_str)
```

b. 坐标转换：
```python
if bbox_flag:
    bbox = grounder_response.get("bbox")
    
    # Qwen3坐标转换
    if use_qwen3:
        bbox = convert_qwen3_coordinates_to_absolute(
            bbox, img.width, img.height, is_bbox=True
        )
    else:
        bbox = [int(coord/factor) for coord in bbox]
    
    x1, y1, x2, y2 = bbox
else:
    coordinates = grounder_response["coordinates"]
    if use_qwen3:
        x, y = convert_qwen3_coordinates_to_absolute(
            coordinates, img.width, img.height, is_bbox=False
        )
    else:
        x, y = [int(coord/factor) for coord in coordinates]
```

c. 严格回放验证：
```python
current_bbox = [x1, y1, x2, y2]
current_ui_meta = extract_ui_metadata_from_hierarchy(
    hierarchy, current_bbox, device_type
)

if replay_this_step and replay_expected_ui_meta is not None:
    if not strict_ui_meta_equal(current_ui_meta, replay_expected_ui_meta):
        logging.info(
            "Strict replay aborted (UI metadata mismatch). "
            f"expected={replay_expected_ui_meta}, current={current_ui_meta}"
        )
        replay_info = []  # 放弃回放
        continue  # 重新调用Decider
```

d. 点击执行：
```python
position_x = (x1 + x2) // 2
position_y = (y1 + y2) // 2
device.click(position_x, position_y)

# 保存可视化
img_bounds = Image.open(save_path)
draw_bounds = ImageDraw.Draw(img_bounds)
draw_bounds.rectangle([x1, y1, x2, y2], outline='red', width=5)  # 红框
img_bounds.save(bounds_path)

cv2image = cv2.imread(bounds_path)
cv2.circle(cv2image, (position_x, position_y), 15, (0, 255, 0), -1)  # 绿圆
cv2.imwrite(click_point_path, cv2image)
```

**INPUT动作**：
```python
elif action == "input":
    text = decider_response["parameters"]["text"]
    device.input(text)
    actions.append({
        "type": "input",
        "text": text,
        "action_index": image_index
    })
```

**SWIPE动作**：
```python
elif action == "swipe":
    direction = decider_response["parameters"]["direction"]
    device.swipe(direction.lower(), 0.6)  # 滑动60%屏幕距离
    
    actions.append({
        "type": "swipe",
        "direction": direction.lower(),
        "action_index": image_index
    })
    
    create_swipe_visualization(data_dir, image_index, direction.lower())
```

**WAIT动作**：
```python
elif action == "wait":
    actions.append({
        "type": "wait",
        "action_index": image_index
    })
```

##### 5️⃣ 结果保存（第1077-1085行）

```python
data = {
    "app_name": app,
    "task_type": None,
    "old_task_description": old_task,
    "task_description": task,
    "action_count": len(actions),
    "actions": actions
}

with open(os.path.join(data_dir, "actions.json"), "w", encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
with open(os.path.join(data_dir, "react.json"), "w", encoding='utf-8') as f:
    json.dump(reacts, f, ensure_ascii=False, indent=4)
```

**保存的文件**：

| 文件 | 说明 |
|------|------|
| `actions.json` | 执行的所有动作列表 |
| `react.json` | Decider和Grounder的推理过程 |
| `{image_index}.jpg` | 每步执行时的截图 |
| `{image_index}.xml`或`.json` | 每步的UI层级树 |
| `{image_index}_highlighted.jpg` | 标注了点击坐标的截图 |
| `{image_index}_bounds.jpg` | 标注了边界框的截图 |
| `{image_index}_click_point.jpg` | 标注了点击点的截图 |
| `{image_index}_swipe.jpg` | 标注了滑动方向的截图 |

---

### 15. 用户偏好提取（第1087-1094行）

```python
if preference_extractor and should_extract_preferences(data):
    task_data = {
        'task_description': task,
        'actions': actions,
        'reacts': reacts,
        'app_name': app
    }
    preference_extractor.extract_async(task_data)
    logging.info("Submitted preference extraction task")
```

**作用**：
- 异步提取用户行为偏好
- 用于个性化任务改写
- 不阻塞主任务执行

---

### 16. 经验记录（第1095-1102行）

```python
if experience_rr and template:
    logging.info("Recording experience for future replay")
    experience_rr.record(task, template, full_history, extra_info)
    logging.info("Experience recorded")
```

**作用**：
- 记录这次执行的完整历史
- 供下次相同任务回放
- 提高相同任务的执行效率

---

### 17. Planner 响应解析（第1104-1148行）

```python
def parse_planner_response(response_str: str):
    """解析Planner模型的JSON响应"""
    pattern = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)
    match = pattern.search(response_str)
    
    json_str = match.group(1) if match else response_str.strip()
    data = json.loads(json_str)
    return data

def get_app_package_name(task_description, use_graphrag=False, device_type="Android"):
    """一阶段规划：应用选择 + 任务优化"""
    
    # 1. 本地检索经验
    search_engine = PromptTemplateSearch(default_template_path)
    experience_content = search_engine.get_experience(task_description, 1)
    
    # 2. 选择合适的提示模板
    if device_type == "Android":
        planner_prompt_template = load_prompt("planner_oneshot.md")
    elif device_type == "Harmony":
        planner_prompt_template = load_prompt("planner_oneshot_harmony.md")
    
    # 3. 检索用户偏好（可选）
    user_preferences = {}
    if preference_extractor and preference_extractor.mem:
        user_preferences = retrieve_user_preferences(
            task_description,
            preference_extractor.mem,
            use_graphrag=use_graphrag
        )
    
    # 4. 结合经验 + 偏好优化上下文
    enhanced_context = combine_context(experience_content, user_preferences)
    
    # 5. 构建Prompt
    prompt = planner_prompt_template.format(
        task_description=task_description,
        experience_content=enhanced_context
    )
    
    # 6. 调用Planner
    response_str = planner_client.chat.completions.create(
        model=planner_model,
        messages=[
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
    ).choices[0].message.content
    
    # 7. 解析响应
    response_json = parse_planner_response(response_str)
    
    app_name = response_json.get("app_name")
    package_name = response_json.get("package_name")
    final_desc = response_json.get("final_task_description", task_description)
    
    return app_name, package_name, final_desc, experience_content
```

**流程**：
1. 根据任务描述本地检索历史经验
2. 检索用户偏好（可选GraphRAG增强）
3. 结合经验+偏好优化任务描述
4. 调用Planner获取APP选择和优化后的描述

---

### 18. 主程序入口（第1150-1338行）

#### 命令行参数
```python
parser = argparse.ArgumentParser(description="MobiMind Agent")

# 服务配置
parser.add_argument("--service_ip", type=str, default="localhost")
parser.add_argument("--decider_port", type=int, default=8000)
parser.add_argument("--grounder_port", type=int, default=8001)
parser.add_argument("--planner_port", type=int, default=8002)

# 功能开关
parser.add_argument("--user_profile", choices=["on", "off"], default="off")
parser.add_argument("--use_graphrag", choices=["on", "off"], default="off")
parser.add_argument("--use_experience", action="store_true", default=False)
parser.add_argument("--use_experience_rr", action="store_true", default=False)

# 设备配置
parser.add_argument("--device", choices=["Android", "Harmony"], default="Android")
parser.add_argument("--device_endpoint", default=None)

# 数据配置
parser.add_argument("--data_dir", default=None)
parser.add_argument("--task_file", default=None)
```

#### 初始化流程
```python
# 1. 初始化系统
enable_user_profile = (args.user_profile == "on")
use_graphrag = (args.use_graphrag == "on")
init(
    args.service_ip, args.decider_port, args.grounder_port, args.planner_port,
    enable_user_profile=enable_user_profile,
    use_graphrag=use_graphrag,
    use_experience_rr=args.use_experience_rr
)

# 2. 连接设备
if args.device == "Android":
    device = AndroidDevice(args.device_endpoint)
else:
    device = HarmonyDevice(args.device_endpoint)

# 3. 读取任务列表
with open(args.task_file or "task.json", "r") as f:
    task_list = json.load(f)
```

#### 任务执行循环
```python
for task in task_list:
    # 创建数据目录
    existing_dirs = [d for d in os.listdir(data_base_dir) if d.isdigit()]
    data_index = max(int(d) for d in existing_dirs) + 1 if existing_dirs else 1
    data_dir = os.path.join(data_base_dir, str(data_index))
    os.makedirs(data_dir)
    
    # 规划任务
    app_name, package_name, planner_task_description, template = get_app_package_name(
        task, use_graphrag=use_graphrag, device_type=current_device_type
    )
    logging.info(f"Planner result - App: {app_name}, Package: {package_name}")
    
    # 选择任务描述
    if use_experience:
        final_task = planner_task_description  # 使用优化后的描述
    else:
        final_task = task  # 使用原始描述
    
    # 执行任务
    device.app_start(package_name)
    task_in_app(app_name, task, final_task, template, device, data_dir, 
                True, True, current_device_type)
    device.app_stop(package_name)

# 等待异步操作完成
if preference_extractor and hasattr(preference_extractor, 'executor'):
    logging.info("Waiting for all preference extraction tasks to complete...")
    preference_extractor.executor.shutdown(wait=True)
```

---

## 📊 完整执行流程图

```
┌─────────────────────────┐
│  输入：自然语言任务      │
└──────────┬──────────────┘
           │
           ▼
    ┌──────────────────┐
    │  [Planner模型]   │
    │  应用选择+优化   │
    └──────┬───────────┘
           │
           ▼
    ┌──────────────────┐
    │   启动目标应用   │
    └──────┬───────────┘
           │
           ▼
  ┌────────────────────────────────────┐
  │  主循环（最多35步）                │
  │  ┌──────────────────────────────┐  │
  │  │ 1. 获取当前截图             │  │
  │  │ 2. 检查是否可以回放历史      │  │
  │  │    ├─ 若可回放               │  │
  │  │    │  ├─ 使用历史动作        │  │
  │  │    │  └─ 严格验证UI元素      │  │
  │  │    └─ 若不可回放             │  │
  │  │       └─ 调用[Decider]       │  │
  │  │          决策下一步动作      │  │
  │  │                              │  │
  │  │ 3. 解析动作类型              │  │
  │  │    ├─ done：任务完成         │  │
  │  │    ├─ click：                │  │
  │  │    │  └─ [Grounder]定位坐标  │  │
  │  │    │     └─ 执行点击         │  │
  │  │    ├─ input：输入文本         │  │
  │  │    ├─ swipe：滑动屏幕         │  │
  │  │    └─ wait：等待             │  │
  │  │                              │  │
  │  │ 4. 保存执行状态              │  │
  │  │    ├─ 截图+UI树             │  │
  │  │    ├─ 可视化标注             │  │
  │  │    ├─ 用户偏好提取(异步)     │  │
  │  │    └─ 经验记录(用于回放)     │  │
  │  └──────────────────────────────┘  │
  └────────────┬───────────────────────┘
               │
               ▼
   ┌───────────────────────────────┐
   │ 输出：actions.json + react.json│
   │      可视化截图 + UI树        │
   └───────────────────────────────┘
```

---

## 🎯 关键技术亮点

| 特性 | 说明 |
|------|------|
| **三模型协同** | Planner（规划）+ Decider（决策）+ Grounder（定位） |
| **多设备支持** | Android + HarmonyOS，通过抽象基类统一接口 |
| **严格回放机制** | 通过UI元素特征验证，确保重复任务可靠执行 |
| **多格式兼容** | 支持Android XML和HarmonyOS JSON的UI树解析 |
| **可视化调试** | 自动保存带标注的截图，便于问题诊断 |
| **用户个性化** | 异步提取用户偏好，支持GraphRAG增强 |
| **结构化输出** | Pydantic + vLLM JSON Schema强制约束 |
| **跨语言输入** | ADB IME + Base64解决Unicode文本输入 |
| **智能坐标转换** | 自动适配不同Grounder模型的输出格式 |
| **性能记录** | 记录模型推理时间，便于性能分析 |

---

## 💡 使用示例

### 基础执行
```bash
python mobiagent.py \
    --service_ip localhost \
    --decider_port 8000 \
    --grounder_port 8001 \
    --planner_port 8002 \
    --device Android
```

### 启用所有功能
```bash
python mobiagent.py \
    --service_ip localhost \
    --decider_port 8000 \
    --grounder_port 8001 \
    --planner_port 8002 \
    --user_profile on \
    --use_graphrag on \
    --use_experience on \
    --use_experience_rr on \
    --device Android
```

### HarmonyOS设备
```bash
python mobiagent.py \
    --device Harmony \
    --device_endpoint "127.0.0.1:7788" \
    --service_ip localhost
```

### 自定义数据目录和任务文件
```bash
python mobiagent.py \
    --device Android \
    --data_dir /path/to/data \
    --task_file /path/to/task.json
```

### 清除所有用户记忆
```bash
python mobiagent.py \
    --user_profile on \
    --clear_memory
```

---

## 🔗 核心类和函数索引

### 设备类
- `Device` - 抽象基类
- `AndroidDevice` - Android设备实现
- `HarmonyDevice` - HarmonyOS设备实现

### 主要函数
- `init()` - 初始化系统
- `task_in_app()` - 主任务执行
- `get_app_package_name()` - 应用规划
- `get_screenshot()` - 截图和Base64编码
- `convert_qwen3_coordinates_to_absolute()` - 坐标转换
- `parse_json_response()` - JSON解析
- `robust_json_loads()` - 健壮JSON解析

### UI树解析函数
- `_parse_android_bounds()` - Android bounds解析
- `_find_android_node_meta_by_bbox()` - Android节点查找
- `_find_harmony_like_node_meta_by_bbox()` - HarmonyOS节点查找
- `_extract_ui_metadata_from_hierarchy()` - 统一UI元数据提取
- `strict_ui_meta_equal()` - 严格UI匹配

### 数据模型
- `ActionType` - 动作类型枚举
- `ActionPlan` - 动作计划结构
- `GroundResponse` - 定位响应结构

---

## 📝 总结

这个文件是一个**高度集成的移动设备自动化框架**，具有以下特点：

1. **多层级决策**：Planner → Decider → Grounder，逐步精化
2. **严格回放机制**：通过UI特征验证实现可靠的经验重用
3. **多设备支持**：统一接口适配Android和HarmonyOS
4. **鲁棒性设计**：多级降级解析、异常恢复、性能监控
5. **可观测性强**：详细的可视化和日志，便于调试
6. **可扩展性好**：抽象设计，易于添加新功能

这是一个**生产级的自动化框架**，融合了LLM、计算机视觉、UI树解析和经验学习等前沿技术！
