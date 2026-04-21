# MobiAgent Mobile Standalone 部署指南

## 目录结构

```
mobile_mobi/
├── mobiagent_mobile_standalone.py   # 主脚本（单文件，提示词已内联）
├── task.json                         # 任务列表示例
└── DEPLOY.md                         # 本文档
```

## 依赖

**Python 包**（仅 2 个）：
```bash
pip install openai Pillow
```

**系统工具**：
- `adb`（android-tools）
- ADB Keyboard APK（中文输入）：https://github.com/senzhk/ADBKeyBoard/releases

不需要 `uiautomator2`，不需要在手机上安装任何后端 APK。

---

## Termux 快速开始

### 1. 安装 Termux

从 F-Droid 安装（不要用 Google Play 版）：https://f-droid.org/packages/com.termux/

### 2. 安装依赖

```bash
pkg update && pkg upgrade -y
pkg install -y python android-tools libjpeg-turbo
pip install openai Pillow
```

### 3. 开启 ADB 调试

手机设置 → 开发者选项 → 开启 **USB 调试** 和 **无线调试**

### 4. ADB 自连接（手机控制自身）

**方法 A：无线调试配对（Android 11+，无需 PC）**

```bash
# 开发者选项 → 无线调试 → 使用配对码配对设备
adb pair <IP>:<配对端口>   # 输入配对码
adb connect 127.0.0.1:<调试端口>
adb devices                # 验证：应显示 device
```

**方法 B：USB 辅助（需要 PC 一次）**

```bash
# PC 上执行（手机 USB 连接时）：
adb tcpip 5555
# 断开 USB，在 Termux 中：
adb connect 127.0.0.1:5555
```

### 5. 安装 ADB Keyboard

下载 APK 安装后，验证：
```bash
adb shell ime list -s | grep adbkeyboard
# 应输出：com.android.adbkeyboard/.AdbIME
```

### 6. 运行

```bash
cd ~/mobi

# 单条任务
python mobiagent_mobile_standalone.py \
  --service_ip 192.168.1.100 \
  --decider_port 8000 \
  --task "在微信给小赵发消息：明天开会"

# 从 task.json 批量执行
python mobiagent_mobile_standalone.py \
  --service_ip 192.168.1.100 \
  --decider_port 8000 \
  --task task.json \
  --data_dir ./data
```

---

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--service_ip` | `localhost` | Decider 模型服务 IP |
| `--decider_port` | `8000` | Decider 服务端口 |
| `--decider_model` | `""` | 模型名称（留空自动获取） |
| `--device_endpoint` | `127.0.0.1:5555` | ADB 端点 |
| `--task` | 必填 | 任务描述或 task.json 路径 |
| `--data_dir` | `./data` | 截图和结果保存目录 |

## Decider 模型服务

需要在局域网服务器上运行 vLLM（手机和服务器同一局域网）：

```bash
vllm serve IPADS-SAI/MobiMind-1.5-4B --port 8000 --host 0.0.0.0
```

## 执行结果

```
data/
└── 1/
    ├── 1.jpg        # 第1步截图（缩放 0.5x）
    ├── 1.xml        # UI hierarchy
    ├── actions.json # 完整动作序列
    └── react.json   # 模型推理过程
```

## 常见问题

**`adb devices` 显示 unauthorized**：开发者选项中撤销 USB 调试授权后重新连接，手机弹出授权对话框点允许。

**中文输入失败**：确认 ADB Keyboard 已安装，`adb shell ime list -s | grep adbkeyboard` 有输出。

**ADB 连接失败**：确认无线调试已开启，端口号与 `adb connect` 中一致。
