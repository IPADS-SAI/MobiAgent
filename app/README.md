# A.I. Chatbot README

## 开发环境
- **IDE**: Android Studio
- **语言**: Java
- **最低SDK**: Android 8.0 (API 26)
- **目标SDK**: Android 14 (API 34)
- **编译工具**: Gradle 8.3.0

## 核心依赖
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

在Android Studio中打开`app`文件夹项目，连接到对应`Android`设备后`Build`,`Run`

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
