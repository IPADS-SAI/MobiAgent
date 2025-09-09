# MobiAgent README

## Development Environment
- **IDE**: Android Studio
- **Language**: Java
- **Minimum SDK**: Android 8.0 (API 26)
- **Target SDK**: Android 14 (API 34)
- **Build Tool**: Gradle 8.3.0

## Core Dependencies
```gradle
// Network requests
implementation 'com.squareup.okhttp3:okhttp:4.10.0'

// JSON processing
implementation 'com.google.code.gson:gson:2.10.1'
implementation 'com.fasterxml.jackson.core:jackson-databind:2.15.2'

// OpenAI integration
implementation 'com.theokanning.openai-gpt3-java:service:0.18.2'

// UI components
implementation 'androidx.appcompat:appcompat:1.6.0'
implementation 'com.google.android.material:material:1.8.0'
```

### System Permissions
```xml
<!-- Network communication -->
<uses-permission android:name="android.permission.INTERNET" />

<!-- Accessibility service -->
<uses-permission android:name="android.permission.BIND_ACCESSIBILITY_SERVICE" />
<uses-permission android:name="android.permission.SYSTEM_ALERT_WINDOW" />

<!-- Screen recording/screenshot -->
<uses-permission android:name="android.permission.FOREGROUND_SERVICE_MEDIA_PROJECTION" />
<uses-permission android:name="android.permission.CAPTURE_VIDEO_OUTPUT" />

<!-- System control -->
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.VIBRATE" />
<uses-permission android:name="android.permission.WAKE_LOCK" />
```

## Installation and Usage

### Requirements
1. Android 8.0 (API 26) or higher
2. Enable "Accessibility Service" permission
3. Stable network connection

### Installation Steps

Open the `app` folder project in Android Studio, connect to the corresponding `Android` device, then `Build` and `Run`.

### Permission Settings
1. **Accessibility Service**
2. **Notification Permission**

## Supported Commands

### Automation Control Commands
| Command | Function | Parameter Example |
|---------|----------|-------------------|
| `click` | Click on specified screen location | `{"target_element": "Login Button", "bbox": "100,200,300,250"}` |
| `input` | Input text content | `{"text": "Hello World"}` |
| `swipe` | Swipe operation | `{"direction": "up", "duration": "500"}` |
| `open_app` | Launch application | `{"package_name": "com.tencent.mm"}` |
| `wait` | Wait operation | `{"duration": "2000"}` |

### Control Commands
| Command | Function | Description |
|---------|----------|-------------|
| `done` | Task completion | Normal termination of current task |
| `terminate` | Abnormal termination | Force interruption of current task |

## Configuration

### Server IP Configuration
Configure AI API related parameters in `MainActivity.java`:
```java
Request request = new Request.Builder()
                 .url("http://ip:port/version")
                 .post(body)
                 .addHeader("Content-Type", "application/json; charset=utf-8")
                 .build();
```

### Accessibility Service Configuration
Configure service permissions in `accessibility_service_config.xml`:
```xml
<accessibility-service
    android:accessibilityEventTypes="typeAllMask"
    android:accessibilityFlags="flagDefault"
    android:accessibilityFeedbackType="feedbackGeneric"
    android:canRetrieveWindowContent="true" />
```

## Acknowledgement

This project is based on the transformation of [AI-Chatbot](https://github.com/aTh1ef/AI-Chatbot-AndroidStudio.git). Thanks to `aTh1ef` for the open source code.