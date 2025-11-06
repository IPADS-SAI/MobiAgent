# Manual Data Collection Tool - Device Support

设备数据收集工具现已支持 Android 和 Harmony 两种设备类型。

## 功能概述

本工具已重构为支持多设备类型，主要改进包括：

### 设备抽象层 (`device.py`)
- **设备基类** (`Device`): 定义了所有设备操作的抽象接口
- **Android设备** (`AndroidDevice`): 使用 `uiautomator2` 库进行Android设备操作
- **Harmony设备** (`HarmonyDevice`): 使用 `hmdriver2` 库进行Harmony设备操作
- **工厂函数** (`create_device`): 根据设备类型创建相应的设备实例

### 后端服务 (`server.py`)
- 支持设备初始化API: `/init_device`
- 支持设备状态查询API: `/device_status`
- 自动根据设备类型加载相应的应用包名映射
- 所有设备操作均通过统一的Device接口进行

### 前端界面
- 新增设备配置区域，支持选择设备类型
- 支持输入Android设备的ADB端点（可选，用于远程连接）
- 实时显示设备连接状态
- 连接设备后才能开始数据收集

## 使用步骤

### 1. 启动服务器
```bash
cd /home/zhaoxi/ipads/llm-agent/MobiAgent/collect/manual
python server.py
```

服务器启动后，访问 `http://localhost:9000`

### 2. 连接设备

#### Android设备
1. 在前端选择设备类型：**Android (uiautomator2)**
2. **ADB端点** (可选):
   - 本地连接：留空
   - 远程连接：输入 `192.168.1.100:5555` (示例)
3. 点击 **🔗 连接设备** 按钮

#### Harmony设备
1. 在前端选择设备类型：**Harmony (hmdriver2)**
2. ADB端点输入框可输入设备serial,若为空则默认选择第一个设备
3. 点击 **🔗 连接设备** 按钮

### 3. 开始数据收集
1. 设备连接成功后，状态显示为 ✅
2. 点击 **🚀 开始收集**
3. 选择应用和任务类型
4. 输入任务描述
5. 开始进行点击、滑动、输入等操作

## 文件结构

```
collect/manual/
├── device.py              # 新增：设备抽象层，支持Android和Harmony
├── server.py              # 修改：添加设备初始化和管理功能
├── static/
│   ├── index.html         # 修改：添加设备选择UI
│   ├── css/style.css      # 修改：添加设备选择样式
│   └── js/script.js       # 修改：添加设备初始化和管理逻辑
└── README.md              # 本文件
```

## 设备特定的应用包名

### Android应用映射
支持应用包括：微信、QQ、微博、饿了么、美团、bilibili、爱奇艺、腾讯视频、优酷、淘宝、京东、携程、同城、飞猪、去哪儿、华住会、知乎、小红书、QQ音乐、网易云音乐、酷狗音乐、高德地图等。

### Harmony应用映射
支持应用包括：携程、飞猪、同城、饿了么、知乎、哔哩哔哩、微信、小红书、QQ音乐、高德地图、淘宝、微博、京东、浏览器、拼多多等。

## API参考

### 初始化设备
**请求**：
```
POST /init_device
Content-Type: application/json

{
    "device_type": "Android",
    "adb_endpoint": null
}
```

**响应**（成功）：
```json
{
    "status": "success",
    "message": "Android device initialized successfully",
    "device_type": "Android"
}
```

### 获取设备状态
**请求**：
```
GET /device_status
```

**响应**：
```json
{
    "status": "connected",
    "device_type": "Android"
}
```

或

```json
{
    "status": "disconnected",
    "device_type": null
}
```

## 常见问题

### Q: 如何远程连接Android设备？
A: 在Android设备上运行 `adb tcpip 5555`，然后在前端输入设备IP和端口（如 `192.168.1.100:5555`）。

### Q: Harmony设备需要输入ADB端点吗？
A: 不需要，Harmony设备使用 `hmdriver2` 库，会自动检测设备。

### Q: 如果应用不在预设列表中怎么办？
A: 可以在 `device.py` 中的 `app_package_names` 字典中添加应用包名映射。

### Q: 如何添加新的设备类型支持？
A: 
1. 在 `device.py` 中继承 `Device` 基类
2. 实现所有抽象方法
3. 在 `create_device` 工厂函数中添加新的设备类型
4. 在前端添加相应的UI选项

## 技术细节

### 设备抽象设计
所有设备操作通过统一的 `Device` 接口进行，支持的操作包括：
- `start_app(app)`: 启动应用
- `app_stop(package_name)`: 停止应用
- `screenshot(path)`: 截图
- `click(x, y)`: 点击
- `input(text)`: 输入文本
- `swipe(start_x, start_y, end_x, end_y, duration)`: 滑动
- `keyevent(key)`: 按键事件
- `dump_hierarchy()`: 获取UI层次结构

### 应用包名映射
根据设备类型动态加载相应的应用包名映射，确保应用兼容性。

## 扩展建议

1. **添加设备连接历史**：记录常用设备连接，加快下次连接
2. **设备连接测试**：在连接前进行测试连接，验证连接性
3. **多设备支持**：同时支持多个设备连接和操作
4. **设备监控**：显示设备电量、屏幕状态等信息
5. **更多应用支持**：根据需求添加更多应用的包名映射

## 依赖要求

- `fastapi`: 后端框架
- `pydantic`: 数据验证
- `uiautomator2`: Android自动化测试库
- `hmdriver2`: Harmony设备驱动库


