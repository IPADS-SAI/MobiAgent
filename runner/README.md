# Agent Runner

**English** | [中文](README_zh.md)

## MobiAgent Runner

Supported capabilities

1. Content apps (Xiaohongshu, Bilibili, Zhihu, etc.)
- Follow a user and enter their profile
- Search, open, and play
- Search within a user profile, open, and play
- Like, favorite, comment, and share

2. Social apps (WeChat, QQ, etc.)
- Send messages, make calls/video calls, find chat content
- Mention someone and send a message
- Open mini programs, open Moments (commenting in Moments is supported by this framework)

3. Shopping apps (Taobao, JD, etc.)
- Search, sort results by price/sales, open search results
- Add to cart and place orders; select correct spec to add to cart/order
- Follow shops

4. Food delivery (Ele.me, Meituan)
- Order food, including selecting specification and quantity

5. Travel (Fliggy, Qunar, Ctrip, Tongcheng, Huazhu)
- Check hotel prices (by location, landmark vicinity, specific hotel, dates)
- Book hotels (by location, landmark vicinity, specific hotel, dates, room type)
- Purchase train/flight tickets (with origin, destination, and time range)

6. Maps (Amap/Gaode)
- Navigation and ride-hailing (origin and destination can be changed)

7. Music (NetEase Cloud Music, QQ Music)
- Search songs, singers, bands
- Search and play

### Model Deployment
After downloading the three models (decider, grounder, and planner), deploy them with vLLM:

Default ports deployment

```bash
vllm serve IPADS-SAI/MobiMind-Decider-7B --port <decider port>
vllm serve IPADS-SAI/MobiMind-Grounder-3B --port <grounder port>
vllm serve Qwen/Qwen3-4B-Instruct --port <planner port>
```

Notes
- Ensure the service ports match the ports passed when launching MobiMind-Agent later.
- If using non-default ports, specify them via `--decider_port`, `--grounder_port`, and `--planner_port` when starting the Agent.

### Configure Tasks
Write the tasks to test in `runner/mobiagent/task.json`.

### Project Start

Basic launch:

```bash
python -m runner.mobiagent.mobiagent \
  --service_ip <Service IP> \
  --decider_port <Decider Service Port> \
  --grounder_port <Grounder Service Port> # Same as Decider Port, be ignored when using --e2e flag \
  --planner_port <Planner Service Port>
```

Parameters：

- `--service_ip <str>`: Service IP, default `localhost`. Used to connect to decider/grounder/planner services.
- `--decider_port <int>`: Decider service port, default `8000`.
- `--grounder_port <int>`: Grounder service port, default `8001`.
- `--planner_port <int>`: Planner service port, default `8002`.
- `--user_profile on|off`: Enable user profile/preference memory (Mem0). Default: `off`. Accepts string `on` or `off`. When enabled, initializes the preference extractor and can asynchronously extract and write preference memories after task completion.
- `--use_graphrag on|off`: Use GraphRAG (Neo4j) for preference retrieval. Default: `off`. When enabled, retrieval prioritizes GraphRAG (determined by command-line, no longer relies on environment variables).
- `--clear_memory`: When specified, the program will clear all memories for the current user (`default_user`) at startup and exit (requires `--user_profile on` and properly initialized memory client).
- `--device Android|Harmony`: Device type, default `Android`.
- `--use_qwen3 on|off`: Whether to use Qwen3VL-based models (e.g., `MobiMind-Mixed-4B`). Default: `on`.
- `--use_experience on|off`: Whether to use experience rewriting. Default: `off`.
- `--e2e`: Use end-to-end model (e.g., [MobiMind-1.5](https://huggingface.co/IPADS-SAI/MobiMind-1.5-4B-1220)) and elinimates grounder calls (default: `false`)
- `--data_dir <path>`: Result data save directory, defaults to `data/` under the script directory (will be created automatically if it doesn't exist).
- `--task_file <path>`: Task list file path, defaults to `task.json` under the script directory.

### User Profile & Preference Memory (Mem0/GraphRAG)

MobiAgent integrates a user preference memory system (Mem0) to provide personalized context for planning.

- Switches and modes:
  - `--user_profile on|off`: enable/disable user profile memory (default on).
  - `--use_graphrag on|off`: enable GraphRAG retrieval (default off). This is the single source of truth; do not rely on env/config to toggle this.
  - `--clear_memory`: clear all memories for the current user (`default_user`) and exit.

- Behavior:
  - Preference extraction runs asynchronously after a successful task; preferences are stored as natural-language texts in Mem0.
  - Retrieval returns a list of raw preference texts (no local regex/struct parsing). These texts are appended to experience templates via `combine_context(...)` as enhanced context for the LLM.

- Required .env (for underlying components):
  - LLM (OpenAI-compatible gateway)
    - `OPENAI_API_KEY`
    - `OPENAI_BASE_URL` (e.g., your vLLM/OpenRouter-compatible endpoint)
  - GraphRAG (Neo4j)
    - `NEO4J_URL` (e.g., `neo4j+s://<host>:<port>` or `neo4j://localhost:7687`)
    - `NEO4J_USERNAME`
    - `NEO4J_PASSWORD`
  - Vector store / Embedding (used by Mem0 components)
    - `EMBEDDING_MODEL` (e.g., `BAAI/bge-small-zh`)
    - `EMBEDDING_MODEL_DIMS` (must match the model, e.g., `384`)
    - `MILVUS_URL` (e.g., `http://localhost:19530`)

### Environment Setup (Milvus + Neo4j)

This project supports two preference backends:
- Vector search (Milvus)
- GraphRAG (Neo4j)

You can deploy either or both. Whether to use GraphRAG is decided by `--use_graphrag`.

#### 1) Milvus (Vector DB)

Use the official embedded startup script for a standalone Docker deployment:
```bash
# Download the script
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
# Start the container
bash standalone_embed.sh start
```

Required .env:
```bash
MILVUS_URL=http://localhost:19530
EMBEDDING_MODEL=BAAI/bge-small-zh  # download from Hugging Face
EMBEDDING_MODEL_DIMS=384           # must match the model
```

#### 2) Neo4j (GraphRAG)

Minimal local start (official image):
```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/testpassword \
  neo4j:5.23.0
```

Web console: open http://localhost:7474 (user `neo4j`, password `testpassword`).

Required .env:
```bash
NEO4J_URL=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=testpassword
```

> For Neo4j Aura or TLS endpoints, use `neo4j+s://<your-host>` and ensure connectivity and certificates are properly configured.

### Enable & Verify

Vector retrieval (Milvus):
```bash
python -m runner.mobiagent.mobiagent \
  --service_ip 127.0.0.1 --decider_port 8000 --grounder_port 8000 --planner_port 8080 \
  --user_profile on --use_graphrag off
```

GraphRAG (Neo4j):
```bash
python -m runner.mobiagent.mobiagent \
  --service_ip 127.0.0.1 --decider_port 8000 --grounder_port 8000 --planner_port 8080 \
  --user_profile on --use_graphrag on
```

Clear all memories and exit:
```bash
python -m runner.mobiagent.mobiagent \
  --service_ip 127.0.0.1 --decider_port 8000 --grounder_port 8000 --planner_port 8080 \
  --user_profile on --use_graphrag on --clear_memory
```

## UI-TARS Runner

This section is based on `runner/UI-TARS-agent` in the repo. It integrates the UI-TARS model into the MobiAgent framework, providing consistent quick start, model deployment, real-device connection, and data collection.

### Quick Preparation
1. Install dependencies (if already installed at the repo root, skip):
```bash
pip install -r requirements.txt
```
2. Prepare an Android device and enable USB debugging:
```bash
adb devices
```
3. (Optional) Install ADB Keyboard for text input. Without it, actions requiring text input cannot be executed:
```bash
adb install -r ADBKeyboard.apk
```

### Model Deployment (vLLM)
Download either [ByteDance-Seed/UI-TARS-7B-SFT](https://huggingface.co/ByteDance-Seed/UI-TARS-7B-SFT) or [ByteDance-Seed/UI-TARS-7B-DPO](https://huggingface.co/ByteDance-Seed/UI-TARS-7B-DPO). Recommended launch example:

```bash
# Start vLLM with an OpenAI-like API
python -m vllm.entrypoints.openai.api_server \
    --model UI-TARS-7B-SFT \
    --served-model-name UI-TARS-7B-SFT \
    --host 0.0.0.0 \
    --port 8000
```

Common configuration notes:
- Default model base URL (used by default in this framework): `http://192.168.12.152:8000/v1`
- Model name: `UI-TARS-7B-SFT` (replace as needed)
- If using native `vllm serve`, adjust parameters to match your vLLM version.

### Run Examples (Runner / Scripts)
In `MobiAgent/runner/UI-TARS-agent`, run the following scripts:
- Single task (interactive debug):
```bash
python quick_start.py
```
- Small set of tasks:
```bash
python test_batch_executor.py
```
- Batch tasks (production/sampling):
```bash
python batch_task_executor.py --config auto_task.json
```

As noted in USAGE_GUIDE: first run a small smoke test (2–5 tasks), then scale up.

Predefined example tasks:

- Taobao shopping: Open Taobao, search for 'phone case', view results
- WeChat chat: Open WeChat, find the friends list, view recent chat history
- System settings: Open Settings, find Wi-Fi settings, check current Wi-Fi info
- Web browsing: Open a browser, search for information about 'UI-TARS'
- Short video: Open Douyin or Bilibili, browse videos
- Map navigation: Open a maps app, search for nearby restaurants
- Music playback: Open a music app, search and play a song

Supported actions

- `click(point)` - Tap at the coordinate
- `long_press(point)` - Long press at the coordinate
- `type(content)` - Input text
- `scroll(point, direction)` - Scroll (up/down/left/right)
- `drag(start_point, end_point)` - Drag
- `wait` - Waits 1s by default
- `press_home()` - Press Home button
- `press_back()` - Press Back button
- `finished(content)` - Task finished

### Runner and Model Endpoint Configuration
Specify the model service in `ExecutionConfig` or via CLI/env vars in your startup script, e.g.:

```python
config = ExecutionConfig(
    model_base_url="http://192.168.12.152:8000/v1",
    model_name="UI-TARS-7B-SFT",
    max_steps=30,
    step_delay=2.0
)
```

### Data Saving and Format
The data directory structure and formats follow `USAGE_GUIDE.md`:
- Example directories:
```
data_example/          # Production data
test_data_example/     # Test data
├── bilibili/
│   ├── type1/
│   │   ├── 1/
│   │   │   ├── task_data.json      # Task execution data (original format)
│   │   │   ├── actions.json        # Action records (refer to Taobao format)
│   │   │   ├── 1/screenshot_1.jpg  # Screenshot at step 1
│   │   │   ├── 1/hierarchy_1.xml   # XML at step 1
│   │   │   └── ...
│   │   │   └── 1.jpg
```

- `task_data.json` (examples in repo) includes: `task_description`, `app_name`, `task_type`, `task_index`, `package_name`, `execution_time`, `action_count`, `actions`, `success`.
- `actions.json` example contains per-step `type`, coordinates/bounds, `text` (for input), etc.
- `react.json` example contains per-step `reasoning` and the action record.

Save each task under directories grouped by task type/index. `DataManager` (in `ui_tars_automation/data_manager.py`) handles saving screenshots, XMLs, and action logs.

Recommended minimal logs to collect:
- Screenshot and UI hierarchy XML for each step
- User or auto-labeled reasoning for each action

### Debugging, Monitoring, and Logs
- Log files: `batch_execution.log`, `test_batch_execution.log`, `automation.log` (by script output location)
- Real-time monitoring: scripts print current task, step, and success/failure to the terminal.
- Debugging tips: start with `test_single_task.py` or `test_batch_executor.py`; review saved screenshots and `execution_log_*.json` to locate failing steps.

### Troubleshooting
- Device not connected: run `adb devices`. Restart ADB if needed: `adb kill-server && adb start-server`.
- App not started or wrong package: confirm the package name and try launching manually.
- Model call failed: check `model_base_url`, network connectivity, and whether the model server is listening (see vLLM logs).
- Execution stuck: check step timeouts, coordinate mismatches, or device performance issues.

### Security and Notes
- Privacy and compliance: screenshots or recordings may contain sensitive info. Obtain consent before collection and mask sensitive fields.
- Large data volume: long batch runs produce many screenshots. Ensure sufficient storage and clean up regularly.
