
# 在 Android 上本地运行MobiAgent

在 Android 设备上使用 [Termux](https://termux.dev/) 环境本地运行量化后的大语言模型（使用 AWQ 量化），支持文本与图像理解。

## 📱 适配设备

- 一台 Android 设备（建议 Android 10+，RAM ≥ 12GB 以获得较好体验, 高通 8gen3 或者 8elite）
- 安装 [Termux](https://github.com/termux/termux-app)（推荐从 [F-Droid](https://f-droid.org/packages/com.termux/) 安装以获得最新版）


---

## 🔗 模型地址

- **ModelScope 模型页面**：  
  https://www.modelscope.cn/models/doulujiyao/mnn_mobi_awq_text_image_bit8_group32_1


---

## 🛠️ 安装与配置

1. **启动 Termux 并更新包索引**：
   ```bash
   pkg update && pkg upgrade -y
   ```

2. **安装必要依赖**：
   ```bash
   pkg install rust python git -y
   pip install --upgrade pip
   pip install --only-binary=all pillow
   pkg install libxml2 libxslt python-lxml -y
   pip install --only-binary=all uiautomator2
   ```

3. **手动将模型与代码放入设备**：
   ```bash
   scp 或者 abd 把 /local_server 和 /model 权重传输到手机上
   ```

4. **进入本地服务目录并启动模型服务器**：
   ```bash
   export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64:/your/local_server:$LD_LIBRARY_PATH
   
   cd local_server
   ./llm_demo ../mnn_mobi_awq_text_image_bit8_group32_1/config.json --server 8080
   ```

   > 💡 确保 `llm_demo` 有执行权限。若没有，请先运行：
   > ```bash
   > chmod +x llm_demo
   > ```

5. **运行mobiagent**：

    ```bash
    python mobiagent.py --service_ip --decider_port --grounder_port --planner_port --local_planner --local_decider --local_grounder
    ```

---

## 📂 目录结构说明

```
.
├── mnn_mobi_awq_text_image_bit8_group32_1/  # 量化后的 MNN 模型目录
│   └── config.json                          # 模型配置文件
├── local_server/
│   └── llm_demo                             # 模型推理服务可执行文件（ARM64 编译）
└── mobiagent.py
```

> 本方案使用 8-bit AWQ 量化 + Group 32 的 MNN 模型，适合在移动端高效推理。

---

## ⚠️ 注意事项

- 首次运行可能较慢（需加载模型到内存）。
- 某些 Android 厂商会限制后台应用的 CPU 或内存使用，请在开发者选项中关闭电池优化。
- 确保设备有足够存储空间（模型通常 >1GB）。

---

## 🔍 调试建议

- 若启动失败，请检查：
  - `llm_demo` 是否为 ARM64 架构编译
  - Termux 是否授予存储权限（运行 `termux-setup-storage`）

---

