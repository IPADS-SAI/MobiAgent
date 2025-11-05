# 设备配置指南

本文档详细说明如何配置和使用支持多设备类型的手动数据收集工具。

## 快速开始

### 前置要求

1. **Python环境** (3.7+)
2. **必要的Python包**：
   ```bash
   pip install fastapi uvicorn pydantic uiautomator2 hmdriver2
   ```

3. **设备连接**：
   - **Android设备**：需要安装ADB驱动，并开启USB调试模式
   - **Harmony设备**：需要配置hmdriver2环境

### 启动服务

```bash
cd /home/zhaoxi/ipads/llm-agent/MobiAgent/collect/manual
python server.py
```

访问 `http://localhost:9000` 打开前端界面。

## 设备配置详解

### 📱 Android设备配置

#### 本地连接（最常见）

1. 连接Android设备到计算机（USB）
2. 开启设备的**开发者模式**和**USB调试**
3. 前端界面选择：
   - 设备类型：`Android (uiautomator2)`
   - ADB端点：**留空**（自动检测）
4. 点击 **🔗 连接设备**

**验证连接**：
```bash
# 在终端运行
adb devices
# 应该看到你的设备列表
```

#### 远程连接（网络连接）

1. 确保Android设备和电脑在同一网络
2. 在**Android设备**的终端中运行：
   ```bash
   adb tcpip 5555
   ```
3. 查看设备IP地址（设置 → 关于手机 → IP地址）
4. 前端界面配置：
   - 设备类型：`Android (uiautomator2)`
   - ADB端点：`192.168.1.100:5555`（替换为实际IP）
5. 点击 **🔗 连接设备**

**验证连接**：
```bash
# 在终端运行
adb connect 192.168.1.100:5555
adb devices
```

#### 常见问题

| 问题 | 解决方案 |
|------|---------|
| 设备不被识别 | 检查USB驱动是否安装，运行 `adb kill-server && adb start-server` |
| USB调试未启用 | 进入设备设置 → 开发者选项 → USB调试 |
| 连接超时 | 检查网络连接，重启ADB服务 |
| 权限被拒绝 | 在设备上确认USB调试权限弹窗 |

### 🎯 Harmony设备配置

#### 前置条件

1. Harmony设备已连接
2. 已安装hmdriver2库：
   ```bash
   pip install hmdriver2
   ```

#### 连接步骤

1. 确保Harmony设备通过USB连接或网络连接
2. 前端界面选择：
   - 设备类型：`Harmony (hmdriver2)`
   - ADB端点：**自动禁用**（不需要）
3. 点击 **🔗 连接设备**

#### hmdriver2环境配置

hmdriver2通常需要特定的环境配置。如果连接失败，请参考：
- [hmdriver2 官方文档](https://github.com/openharmony-sig/hmdriver2)
- 确保设备驱动已正确安装

## 代码结构详解

### device.py - 设备抽象层

```python
# 基类定义
class Device(ABC):
    """所有设备的基类"""
    def start_app(self, app): pass
    def screenshot(self, path): pass
    def click(self, x, y): pass
    # ... 其他方法

# Android实现
class AndroidDevice(Device):
    """使用uiautomator2的Android实现"""
    
# Harmony实现
class HarmonyDevice(Device):
    """使用hmdriver2的Harmony实现"""

# 工厂函数
def create_device(device_type: str, adb_endpoint: str = None) -> Device:
    """根据类型创建相应设备"""
```

### server.py - 后端服务

**关键API**：

```python
@app.post("/init_device")
async def init_device(config: DeviceConfig):
    """初始化设备连接"""
    global device, device_type
    device = create_device(config.device_type, config.adb_endpoint)
    # ...

@app.get("/device_status")
async def get_device_status():
    """获取当前设备状态"""
    return {"status": "connected", "device_type": "Android"}
```

### 前端交互流程

```
用户选择设备类型和ADB端点
        ↓
点击 "🔗 连接设备" 按钮
        ↓
调用 POST /init_device API
        ↓
后端创建设备实例
        ↓
返回连接状态
        ↓
前端显示连接结果 ✅/❌
        ↓
启用 "🚀 开始收集" 按钮
```

## 添加新的应用支持

### 步骤1：查询应用包名

**Android**：
```bash
adb shell pm list packages | grep 应用名
# 例如：find WeChat
# com.tencent.mm
```

**Harmony**：
```bash
# 从Harmony应用商城查询或设备开发者文档
```

### 步骤2：修改device.py

在 `AndroidDevice.__init__()` 或 `HarmonyDevice.__init__()` 中添加：

```python
self.app_package_names = {
    # 现有应用...
    "新应用名": "com.example.newapp",  # 添加这一行
}
```

### 步骤3：修改server.py（可选）

如果需要在前端选择框中显示新应用，修改 `static/index.html`：

```html
<select id="appName">
    <!-- 现有应用... -->
    <option value="新应用名">新应用名</option>
</select>
```

## 扩展开发指南

### 添加新设备类型

1. 在 `device.py` 中创建新类：

```python
class NewDeviceType(Device):
    def __init__(self):
        # 初始化设备连接
        pass
    
    # 实现所有抽象方法
    def start_app(self, app): ...
    def screenshot(self, path): ...
    # ... 等等
```

2. 在 `create_device()` 工厂函数中添加支持：

```python
def create_device(device_type: str, adb_endpoint: str = None) -> Device:
    if device_type == "Android":
        return AndroidDevice(adb_endpoint)
    elif device_type == "Harmony":
        return HarmonyDevice()
    elif device_type == "NewDevice":  # 添加这个
        return NewDeviceType()
    else:
        raise ValueError(f"Unsupported device type: {device_type}")
```

3. 在前端添加UI选项：

```html
<select id="deviceType">
    <option value="Android">Android</option>
    <option value="Harmony">Harmony</option>
    <option value="NewDevice">NewDevice</option>
</select>
```

### 自定义设备操作

如果某个操作在新设备上有特殊实现，可以在对应的类中重写：

```python
class CustomDevice(Device):
    def swipe(self, start_x, start_y, end_x, end_y, duration=0.1):
        """自定义swipe实现"""
        # 特殊逻辑
        pass
```

## 性能优化建议

1. **连接缓存**：记住上次成功的连接参数
2. **超时设置**：为长时间操作设置合理的超时
3. **批量操作**：使用事务或批处理减少往返次数
4. **截图优化**：缩小截图尺寸以加快传输速度

## 调试技巧

### 启用详细日志

在 `server.py` 顶部修改：
```python
logging.basicConfig(level=logging.DEBUG)  # 改为DEBUG
```

### 测试设备连接

运行示例脚本：
```bash
python example_devices.py android   # 测试Android
python example_devices.py harmony   # 测试Harmony
python example_devices.py multi     # 同时测试
```

### 检查API响应

在浏览器控制台运行：
```javascript
// 检查设备状态
fetch('/device_status').then(r => r.json()).then(console.log)

// 初始化设备
fetch('/init_device', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({device_type: 'Android'})
}).then(r => r.json()).then(console.log)
```

## 相关文件参考

- `device.py` - 设备抽象层实现
- `server.py` - FastAPI后端服务
- `static/index.html` - 前端界面
- `static/js/script.js` - 前端交互逻辑
- `example_devices.py` - 使用示例
- `test_device.py` - 单元测试

## 许可和支持

如有问题，请参考：
1. [MobiAgent项目README](../../../README.md)
2. [uiautomator2文档](https://github.com/openatom-cloud/uiautomator2)
3. [hmdriver2文档](https://github.com/openharmony-sig/hmdriver2)
