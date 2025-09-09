# A.I. Chatbot README

## 主要功能

### AI聊天功能
- **智能对话**: 支持与AI进行自然语言对话
- **实时响应**: 使用HTTP API与AI服务进行实时通信
- **消息历史**: 完整的聊天记录管理和显示
- **多媒体支持**: 支持文本输入和屏幕截图分析

### 设备自动化控制
- **点击操作**: 自动点击屏幕指定位置或元素
- **文本输入**: 自动输入文本内容
- **滑动手势**: 支持各方向的滑动操作
- **应用启动**: 自动打开指定应用程序
- **任务管理**: 支持任务完成和异常终止控制

### 交互界面
- **现代化UI**: 采用Material Design设计风格
- **响应式布局**: 适配不同屏幕尺寸
- **实时反馈**: Toast提示和状态指示
- **便捷操作**: 清空、分享、终止等快捷按钮

## 技术架构

### 开发环境
- **IDE**: Android Studio
- **语言**: Java
- **最低SDK**: Android 8.0 (API 26)
- **目标SDK**: Android 14 (API 34)
- **编译工具**: Gradle 8.3.0

### 核心依赖
```gradle
// 网络请求
implementation 'com.squareup.okhttp3:okhttp:4.10.0'

// JSON处理
implementation 'com.google.code.gson:gson:2.10.1'
implementation 'com.fasterxml.jackson.core:jackson-databind:2.15.2'

// OpenAI集成
implementation 'com.theokanning.openai-gpt3-java:service:0.18.2'

// UI组件
implementation 'androidx.appcompat:appcompat:1.6.0'
implementation 'com.google.android.material:material:1.8.0'
```

### 系统权限
```xml
<!-- 网络通信 -->
<uses-permission android:name="android.permission.INTERNET" />

<!-- 无障碍服务 -->
<uses-permission android:name="android.permission.BIND_ACCESSIBILITY_SERVICE" />
<uses-permission android:name="android.permission.SYSTEM_ALERT_WINDOW" />

<!-- 屏幕录制/截图 -->
<uses-permission android:name="android.permission.FOREGROUND_SERVICE_MEDIA_PROJECTION" />
<uses-permission android:name="android.permission.CAPTURE_VIDEO_OUTPUT" />

<!-- 系统控制 -->
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.VIBRATE" />
<uses-permission android:name="android.permission.WAKE_LOCK" />
```

## 安装和使用

### 环境要求
1. Android 8.0 (API 26) 或更高版本
2. 启用"无障碍服务"权限
3. 稳定的网络连接

### 安装步骤
1. 克隆项目到本地
```bash
git clone https://github.com/your-username/AI-Chatbot-AndroidStudio.git
```

2. 在Android Studio中打开项目

3. 编译并安装到设备
```bash
./gradlew assembleDebug
```

### 权限设置
1. **无障碍服务**
2. **通知权限**

## 支持的命令

### 自动化控制命令
| 命令 | 功能 | 参数示例 |
|------|------|----------|
| `click` | 点击屏幕指定位置 | `{"target_element": "登录按钮", "bbox": "100,200,300,250"}` |
| `input` | 输入文本内容 | `{"text": "Hello World"}` |
| `swipe` | 滑动操作 | `{"direction": "up", "duration": "500"}` |
| `open_app` | 启动应用 | `{"package_name": "com.tencent.mm"}` |
| `wait` | 等待操作 | `{"duration": "2000"}` |

### 控制命令
| 命令 | 功能 | 说明 |
|------|------|------|
| `done` | 任务完成 | 正常结束当前任务 |
| `terminate` | 异常终止 | 强制中断当前任务 |

## 配置说明

### 服务器IP配置
在 `MainActivity.java` 中配置AI API相关参数：
```java
Request request = new Request.Builder()
                 .url("http://ip:port/version")
                 .post(body)
                 .addHeader("Content-Type", "application/json; charset=utf-8")
                 .build();
```

### 无障碍服务配置
在 `accessibility_service_config.xml` 中配置服务权限：
```xml
<accessibility-service
    android:accessibilityEventTypes="typeAllMask"
    android:accessibilityFlags="flagDefault"
    android:accessibilityFeedbackType="feedbackGeneric"
    android:canRetrieveWindowContent="true" />
```

## Acknowledgement

本项目基于[AI-Chatbot](https://github.com/aTh1ef/AI-Chatbot-AndroidStudio.git)改造而来，感谢`aTh1ef`的开源代码.
