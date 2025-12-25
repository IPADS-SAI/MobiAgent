# Agent Runner

[English](README.md) | **中文**

## MobiAgent Runner

**支持功能**
1. 论坛文章视频类（小红书，b站，知乎等）
- 关注xx，进入主页
- 搜索，打开，播放
- 在用户主页搜索，打开，播放
- 点赞，收藏，评论，转发

2. 社交软件类（微信QQ等）
- 发消息，打电话，打视频，查找聊天内容
- @某人+发消息
- 打开小程序，打开朋友圈（打开朋友圈评论我们这个框架肯定可以）

3. 购物类（淘宝，京东等）
- 搜索，按照价格销量等排序搜索，打开搜索结果
- 加入购物车和下单，选择对应规格加入购物车和下单
- 关注店铺

4. 外卖类（饿了么，美团）
- 点外卖，包括选择规格和数量

5. 旅游类（飞猪，去哪儿，携程，同城，华住会）
- 查询酒店价格（地点，地标附近，指定酒店，日期）
- 预定酒店（地点，地标附近，指定酒店，日期，房间类型）
- 购买火车票飞机票（和设定始发地和目的地，以及日期时间段）

6. 地图类（高德）
- 导航，打车（始发地，目的地可以更改）

7. 听歌类（网易云，QQ音乐）
- 搜索歌曲，歌手，乐队
- 搜索并播放

### 模型部署
下载好 `decider`、`grounder` 和 `planner` 三个模型后，使用 vLLM 部署模型推理服务：

**默认端口部署**
```bash
vllm serve IPADS-SAI/MobiMind-Decider-7B --port <decider port>
vllm serve IPADS-SAI/MobiMind-Grounder-3B --port <grounder port>
vllm serve Qwen/Qwen3-4B-Instruct --port <planner port>
```

**注意事项**
- 确保部署的服务端口与后续启动 MobiMind-Agent 时指定的端口参数一致
- 如果使用非默认端口，需要在启动 Agent 时通过 `--decider_port`、`--grounder_port`、`--planner_port` 参数指定对应端口

### 设置任务
在 `runner/mobiagent/task.json` 中写入要测试的任务列表

### 项目启动

**基本启动**（使用默认配置）
```bash
python -m runner.mobiagent.mobiagent
```

**自定义配置启动**
```bash
python -m runner.mobiagent.mobiagent --service_ip <服务IP> --decider_port <决策服务端口> --grounder_port <定位服务端口> --planner_port <规划服务端口>  --device Android --data_dir data --task_file task.json
```

**参数说明**

- `--service_ip <str>`：服务 IP，默认 `localhost`。用于连接 decider/grounder/planner 服务的主机地址。
- `--decider_port <int>`：decider 服务端口，默认 `8000`。
- `--grounder_port <int>`：grounder 服务端口，默认 `8001`。
- `--planner_port <int>`：planner 服务端口，默认 `8002`。
- `--user_profile on|off`：是否启用用户画像/偏好记忆（Mem0）。默认：`off`。此参数接受字符串 `on` 或 `off`，启用后会初始化偏好提取器，在任务完成后可异步提取并写入偏好记忆。
- `--use_graphrag on|off`：是否使用 GraphRAG（Neo4j）进行偏好检索。默认：`off`。若启用，检索优先使用 GraphRAG（由命令行决定，不再依赖环境变量）。
- `--clear_memory`：若指定，程序会在启动时清空当前用户（`default_user`）的所有记忆并退出（需要 `--user_profile on` 且 memory client 已正确初始化）。
- `--device Android|Harmony`：连接的设备类型，默认 `Android`。
- `--use_qwen3 on|off`：是否使用 Qwen3VL 作为基模的模型（例如`MobiMind-Mixed-4B`）。默认：`on`。
- `--use_experience on|off`：是否使用经验改写功能。默认：`off`。
- `--data_dir <path>`：结果数据保存目录，默认为脚本目录下的 `data/`（若不存在会自动创建）。
- `--task_file <path>`：任务列表文件路径，默认为脚本目录下的 `task.json`。


### 用户画像与偏好记忆（Mem0/GraphRAG）

MobiAgent 集成了用户偏好记忆系统（Mem0），用于在规划阶段为 LLM 提供个性化上下文。

- 开关与模式：
  - `--user_profile on|off`：是否启用用户画像（默认 on）。
  - `--use_graphrag on|off`：是否使用 GraphRAG 检索（默认 off）。该参数为单一真相源，优先级最高；不再读取环境变量或配置文件来决定是否启用 GraphRAG。
  - `--clear_memory`：启动后立即清空当前用户（`default_user`）的所有记忆并退出。

- 使用示例：
  - 仅使用向量检索：
    ```bash
    python -m runner.mobiagent.mobiagent \
      --service_ip 123.60.91.241 --decider_port 8000 --grounder_port 8000 --planner_port 8080 \
      --user_profile on --use_graphrag off
    ```
  - 使用 GraphRAG（Neo4j）：
    ```bash
    python -m runner.mobiagent.mobiagent \
      --service_ip 123.60.91.241 --decider_port 8000 --grounder_port 8000 --planner_port 8080 \
      --user_profile on --use_graphrag on
    ```
  - 清空记忆并退出：
    ```bash
    python -m runner.mobiagent.mobiagent \
      --service_ip 123.60.91.241 --decider_port 8000 --grounder_port 8000 --planner_port 8080 \
      --user_profile on --use_graphrag on --clear_memory
    ```

- 行为说明：
  - 偏好“提取”在任务成功完成后异步进行（使用当前 planner LLM 进行语义分析），将偏好以自然语言形式写入 Mem0。
  - 偏好“检索”阶段不再做本地正则/结构化解析，直接返回检索到的“原始文本列表”。
  - 在 `mobiagent.py` 中，这些原文偏好通过 `combine_context(...)` 直接附加到经验模板后，作为 LLM 的增强上下文。

- .env 关键变量（用于底层组件初始化）：
  - LLM（OpenAI 兼容网关）
    - `OPENAI_API_KEY`
    - `OPENAI_BASE_URL`（如使用自建网关/OpenRouter 兼容端点）
  - GraphRAG（Neo4j）
    - `NEO4J_URL`（例如：`neo4j+s://<host>:<port>`）
    - `NEO4J_USERNAME`
    - `NEO4J_PASSWORD`
  - 向量存储/Embedding（Mem0 组件使用）
    - `EMBEDDING_MODEL`（如：`BAAI/bge-small-zh`）
    - `EMBEDDING_MODEL_DIMS`（与模型维度一致，如：`384`）
    - `MILVUS_URL`（如：`http://localhost:19530`）

### 环境准备（Milvus + Neo4j）

本项目支持两种偏好检索后端：
- 向量检索（Milvus）
- GraphRAG（Neo4j）

两者可以独立或同时部署。是否启用 GraphRAG 由命令行 `--use_graphrag` 决定。

#### 1) Milvus（矢量数据库）

```bash
# Download the installation script
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
# Start the Docker container
bash standalone_embed.sh start
```

必需的 .env：
```bash
MILVUS_URL=http://localhost:19530
EMBEDDING_MODEL=BAAI/bge-small-zh(从huggingface下载)
EMBEDDING_MODEL_DIMS=384(与模型维度一致)
```

> 说明：`EMBEDDING_MODEL` 与 `EMBEDDING_MODEL_DIMS` 必须匹配；如更换模型，请同步修改维度。

#### 2) Neo4j（GraphRAG）

最简本地启动（官方镜像）：
```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/testpassword \
  neo4j:5.23.0
```

Web 控制台：访问 http://localhost:7474 （默认用户 `neo4j`，密码 `testpassword`）。

必需的 .env：
```bash
NEO4J_URL=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=testpassword
```

> 如使用 Neo4j Aura 或自签名证书，请将 `NEO4J_URL` 改为 `neo4j+s://<your-host>`，并确保网络连通与证书策略匹配。

### 启用与验证

向量检索（Milvus）：
```bash
python -m runner.mobiagent.mobiagent \
  --service_ip 127.0.0.1 --decider_port 8000 --grounder_port 8000 --planner_port 8080 \
  --user_profile on --use_graphrag off
```

GraphRAG（Neo4j）：
```bash
python -m runner.mobiagent.mobiagent \
  --service_ip 127.0.0.1 --decider_port 8000 --grounder_port 8000 --planner_port 8080 \
  --user_profile on --use_graphrag on
```

清空记忆：
```bash
python -m runner.mobiagent.mobiagent \
  --service_ip 127.0.0.1 --decider_port 8000 --grounder_port 8000 --planner_port 8080 \
  --user_profile on --use_graphrag on --clear_memory
```

## UI-TARS Runner

本节基于仓内 `runner/UI-TARS-agent`进行介绍，支持将UI-TARS模型接入MobiAgent框架，提供一致的快速启动、模型部署、真实移动端设备接入与数据收集。

### 快速准备
1. 安装依赖（在项目跟目录安装过即可）：
```bash
pip install -r requirements.txt
```
1. 准备 Android 设备并启用 USB 调试：
```bash
adb devices
```
1. （可选）安装 ADB 键盘用于文本输入，未安装则无法执行需要进行文本输入的操作：
```bash
adb install -r ADBKeyboard.apk
```

### 模型部署（vLLM）
下载[ByteDance-Seed/UI-TARS-7B-SFT](https://huggingface.co/ByteDance-Seed/UI-TARS-7B-SFT)或者[ByteDance-Seed/UI-TARS-7B-DPO](https://huggingface.co/ByteDance-Seed/UI-TARS-7B-DPO)到本地，推荐的启动示例指令：
```bash
# 使用 vLLM 以 OpenAI-like 接口启动模型服务
python -m vllm.entrypoints.openai.api_server \
    --model UI-TARS-7B-SFT \
    --served-model-name UI-TARS-7B-SFT \
    --host 0.0.0.0 \
    --port 8000
```

常见配置说明：
- 默认模型地址示例（框架内默认）：`http://192.168.12.152:8000/v1`
- 模型名称：`UI-TARS-7B-SFT`（按实际模型名替换）
- 若使用 vLLM 原生命令 `vllm serve`，请根据你部署的 vLLM 版本调整参数。

### 运行示例（Runner / 执行脚本）
仓内提供多种运行脚本,进入目录`MobiAgent/runner/UI-TARS-agent`执行下述任务脚本：
- 单任务（交互式调试）：
```bash
python quick_start.py
```
- 测试少量任务：
```bash
python test_batch_executor.py
```
- 批量任务（生产/采样）：
```bash
python batch_task_executor.py --config auto_task.json
```

USAGE_GUIDE 中说明：先小批量（2~5 个）做 smoke 测试，再扩大执行规模。

框架包含以下预定义任务：

    淘宝购物: 打开淘宝应用，搜索'手机壳'，查看搜索结果
    微信聊天: 打开微信，找到好友列表，查看最近的聊天记录
    系统设置: 打开系统设置，找到WiFi设置，查看当前连接的WiFi信息
    网页浏览: 打开浏览器，搜索'UI-TARS'相关信息
    短视频: 打开抖音或B站，浏览视频内容
    地图导航: 打开地图应用，搜索附近的餐厅
    音乐播放: 打开音乐应用，搜索并播放一首歌曲

支持的操作

  - `click(point)` - 点击指定坐标
  - `long_press(point)` - 长按指定坐标
  - `type(content)` - 输入文本
  - `scroll(point, direction)` - 滚动(上/下/左/右)
  - `drag(start_point, end_point)` - 拖拽操作
  - `wait` 默认等待1s
  - `press_home()` - 按Home键
  - `press_back()` - 按返回键
  - `finished(content)` - 任务完成

### Runner 与模型地址配置
框架通过 `ExecutionConfig` / 启动参数指定模型服务地址，例如：
```python
config = ExecutionConfig(
    model_base_url="http://192.168.12.152:8000/v1",
    model_name="UI-TARS-7B-SFT",
    max_steps=30,
    step_delay=2.0
)
```
或者在启动脚本中通过命令行参数或环境变量设置 `model_base_url`。

### 数据保存与格式
执行产生的数据目录结构与格式遵循 `USAGE_GUIDE.md`：
- 数据目录示例：
```
data_example/          # 生产数据
test_data_example/     # 测试数据
├── bilibili/
│   ├── type1/
│   │   ├── 1/
│   │   │   ├── task_data.json      # 任务执行数据（原格式）
│   │   │   ├── actions.json        # 操作记录（参考淘宝格式）
│   │   │   ├── 1/screenshot_1.jpg  # 第1步截图
│   │   │   ├── 1/hierarchy_1.xml   # 第1步XML
│   │   │   └── ...
│   │   │   └── 1.jpg
```

- `task_data.json`（仓内样例）包含字段：`task_description`, `app_name`, `task_type`, `task_index`, `package_name`, `execution_time`, `action_count`, `actions`, `success`。

- `actions.json` 示例格式包含每步操作的 `type`, 坐标/bounds, `text`（若是输入操作）等。
- `react.json` 示例格式包含每步操作的 `reasoning`, 操作的动作记录等。

将每个任务的数据保存在以任务类型/任务索引为子目录的结构中，`DataManager`（位于 `ui_tars_automation/data_manager.py`）负责保存截图、XML 和动作记录。

推荐同时采集的最小日志项：
- 对应步骤的截图与 UI hierarchy XML
- 用户/自动标注的 操作原因reasoning

### 调试、监控与日志
- 日志文件：`batch_execution.log`, `test_batch_execution.log`, `automation.log`（按脚本输出位置）
- 实时监控：脚本执行时会在终端打印当前任务、步骤、成功/失败状态。
- 调试技巧：先运行 `test_single_task.py` 或 `test_batch_executor.py`；查看保存的截图与 `execution_log_*.json` 来定位失败步骤。

### 故障排查要点
- 设备未连接：`adb devices`，重启 adb 服务：`adb kill-server && adb start-server`。
- 应用未启动或包名错误：确认包名并尝试手动启动。
- 模型调用失败：检查 `model_base_url`、网络连通性和模型服务是否在监听端口（查看 vLLM 日志）。
- 执行卡住：检查步骤超时、坐标匹配失败、或设备性能问题。

### 安全与注意事项
- 请提前确认用户隐私与合规性：截图或录屏可能包含敏感信息，收集前应取得用户同意并对敏感字段做脱敏处理。
- 数据量大：长时间批量执行会产生大量截图，请确保目标机器有足够存储空间并定期清理。


