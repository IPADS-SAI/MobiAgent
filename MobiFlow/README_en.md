# MobiFlow: A DAG-based Benchmarking and Verification Framework for Mobile Agents

An offline verification framework: given a complete execution trace (key frames/events) of a task, MobiFlow reads the task's DAG configuration and checks whether there exists a "satisfying path" that meets dependency and ordering constraints, thus determining whether the execution achieved the task goal.

## Features

### Core Verification Capabilities
- Multi-level condition checks: support text matching, regular expressions, UI state, XML parsing, icon detection, OCR recognition, and LLM reasoning
- Icon detection and recognition: OpenCV template-matching based icon detection with multi-scale matching and similarity threshold control, for fast UI element recognition
- Progressive escalation strategy: automatically escalate to a more advanced method when a simpler one is not applicable
- Multi-path DAG verification: verify dependencies between task nodes via DAG, supporting both AND and OR branching semantics
- Dynamic condition matching: extract conditions from the task description and verify accordingly

### Advanced Verification
- Dual-semantics dependency system:
  - deps field: AND semantics, all predecessor nodes must be completed
  - next field: OR semantics, multiple branch choices are supported
- Path-aware verification: intelligent frame allocation to avoid cross-path frame conflicts
- Constraint checks: automatically detect deps/next configuration conflicts and warn
- Path analysis logs: display all possible successful paths before verification for easier debugging and configuration validation
- On-screen text recognition: read current screen text and match key nodes from article elements
- Smart icon recognition: `icons_match` checker supports template matching for fast UI state verification
- Multimodal LLM verification: combine screenshots and context for LLM-based verification
- Flexible configuration system: define verification conditions and modes via YAML configuration files

## Directory Structure

- `avdag/` Core library
  - `conditions.py` Condition checkers and registry (includes dynamic matching and icon detection checkers)
  - `dag.py` DAG definition, topological sort, path analysis, and constraint checks
  - `verifier.py` Core verification logic (path-aware frame allocation and multi-path verification)
  - `loader.py` Load task configs from YAML/JSON (supports deps and next fields)
  - `trace_loader.py` Load traces from a directory structure
  - `types.py` Core types (Frame/Result/VerifierOptions, NodeSpec supports next)
- `tools/` Utilities
  - `Icon_detection/` Icon detection toolkit
    - `icon_detector.py` Multi-scale template matching based core detector (OpenCV)
    - `icon_detection.py` High-level icon detection service
    - `config.py` Icon detection configuration
    - `test_*.py` Tests for icon detection
- `task_configs/icons/` Icon resources
  - `weixin/` WeChat icon templates
  - `bilibili/` Bilibili icon templates
  - `xiecheng/` Ctrip icon templates
- `task_configs/` Task configs and icon resources (examples are under this directory)
  - `task_configs/*.json` Example task configs in JSON
  - `task_configs/icons/` Icon resource templates
- `docs/` Usage and design docs (includes multi-path, checker modes, OCR/LLM guides)
- `tests/` Unit tests
  - `test_dependency_validation.py` Dependency constraint checks
  - `test_next_paths.py` OR semantics for next
  - `test_path_analysis.py` Path analysis logs
- Test scripts
  - `test_dynamic_filter.py` Dynamic filter condition validation
  - `test_filter_verification.py` Filter verification
  - `test_image_verification.py` Image verification

## Install Dependencies

Install project-level dependencies from the repo root:

```bash
pip install -r requirements.txt
```

Optionally install an OCR helper with Paddle-based detection:

```bash
# Install Tesseract OCR (Linux example)
sudo apt-get install tesseract-ocr

# Install Chinese language pack
sudo apt-get install tesseract-ocr-chi-sim

# Verify installation
tesseract --version
```

### Extra Dependencies for Icon Detection

```bash
pip install opencv-python numpy
```

## Quick Start

1) Explore sample configs in this directory: `task_configs/` contains task configs and icon resources.

2) Run the minimal demo (using a config under `task_configs` as an example):

```bash
python -m avdag.verifier task_configs/taobao.json trace_folder/
```

The output includes:
- Path analysis log (all possible successful paths)
- Success indicator (whether a satisfying path exists under constraints)
- Satisfied nodes and their matched frame indices
- A time-ordered satisfying sequence (a linearized trace)

Example output:
```
[INFO] === DAG Path Analysis ===
[INFO] Found 2 possible successful paths:
  Path 1: activate_search -> input_keyword -> results_page -> open_profile -> follow_author
  Path 2: activate_search -> input_keyword -> results_page -> follow_author
[INFO] === End of Path Analysis ===
```

## Configuration Format (YAML/JSON)

You can use the tools under `MobiFlow/auto_rules` to automatically generate task configs with LLM analysis, or write them manually as below.

### Basic Example

```yaml
task_id: shop_search
nodes:
  - id: open_app
    name: Open the shopping app
    condition:
      type: text_match
      params:
        any: ["Taobao opened", "Homepage"]
  - id: search_page
    deps: [open_app]  # AND semantics: must wait for open_app
    condition:
      type: ui_flag
      params:
        key: screen
        equals: search
  - id: search_keyword
    deps: [search_page]
    condition:
      type: regex_match
      params:
        pattern: ".*iPhone 15.*"
  - id: result_list
    deps: [search_keyword]
    condition:
      type: text_match
      params:
        any: ["results", "items", "products"]
# Success condition (optional). If omitted, any "sink" node (no outgoing edges) being satisfied implies success.
success:
  any_of: [result_list]
```

### Multi-path Example

```yaml
task_id: bilibili_search_follow
nodes:
  - id: activate_search
    name: Activate search
    condition:
      type: text_match
      params:
        any: ["搜索", "search"]
    next: [input_keyword]  # OR semantics: can go to input_keyword
  
  - id: input_keyword
    name: Input keyword
    condition:
      type: text_match
      params:
        any: ["关键词", "搜索词"]
    next: [results_page]
  
  - id: results_page
    name: Results page
    condition:
      type: text_match
      params:
        any: ["搜索结果", "结果页"]
    next: [follow_author, open_profile]  # OR semantics: two choices
  
  - id: open_profile
    name: Open user profile
    condition:
      type: text_match
      params:
        any: ["用户主页", "个人页面"]
    next: [follow_author]
  
  - id: follow_author
    name: Follow author
    condition:
      type: text_match
      params:
        any: ["关注", "已关注"]

success:
  any_of: [follow_author]
```

### Icon Detection Config Example

```yaml
task_id: wechat_send_message
app_id: com.tencent.mm
description: Send a message to a contact or group in WeChat
nodes:
  - id: find_contact_entry
    name: Find contact or group
    condition:
      type: escalate
      params:
        icons:
          all: ["icon_001_通讯录", "icon_002_微信", "icon_000_我"]  # must detect all icons
        ocr:
          all: ["微信", "通讯录", "发现", "我"]
        llm:
          prompt: Is the current page WeChat home, Contacts, or Search?
          expected_true: true
    next: [send_message_success]

  - id: send_message_success
    name: Message sent successfully
    condition:
      type: juxtaposition  # requires icons and OCR both pass
      params:
        icons:
          any: ["icon_001_回车", "icon_002_发送"]  # match any sending-related icon
          threshold: 0.85  # custom similarity threshold
        ocr:
          all: ["发送"]

success:
  any_of: [send_message_success]
```

### Configuration Notes

- `nodes[].deps`: predecessors (AND) — all listed nodes must complete first
- `nodes[].next`: successors (OR) — any listed successor may follow
- `condition`: checker type specified by `type`, with parameters in `params`
- `success`:
  - `any_of: [node_id...]` success if any of these nodes are satisfied
  - `all_of: [node_id...]` success only if all these nodes are satisfied
  - If omitted, default checks whether any sink node (no outgoing edges) is satisfied

#### Dependency Semantics

- deps (AND): strict precedence; all listed predecessors must be satisfied first
- next (OR): flexible branching; any listed successor may be chosen
- Constraint check: when a node has both deps and is targeted by other nodes' next, a warning is issued; deps takes precedence

#### Path Analysis

Before verification, all possible successful paths are printed:
```
[INFO] Found 2 possible successful paths:
  Path 1: activate_search -> input_keyword -> results_page -> open_profile -> follow_author
  Path 2: activate_search -> input_keyword -> results_page -> follow_author
```

### Supported Checker Types

#### Basic Checkers
- `text_match`: text match
- `regex_match`: regular expression match
- `ui_flag`: UI state check
- `xml_text_match`: XML text match
- `action_match`: action type match

#### Image-based Checker
- `icons_match`: icon detection via OpenCV template matching

#### Advanced Checkers
- `escalate`: try multiple checkers in a configured escalating order
- `juxtaposition`: parallel checks; all configured checks must pass
- `dynamic_match`: dynamic condition matching based on the task description

## Trace Data (Frames)

Two trace formats are supported:

### 1) JSON format (simple)

Collect key frames/events into a time-ordered array:

```json
{
  "timestamp": 1723456789.123,
  "text": "Taobao opened, entered search page",
  "ui": {"screen": "search"},
  "payload": {"extra": "extensible"}
}
```

### 2) Directory format (mobile automation)

A directory containing screenshots, XMLs, and action logs:

```
trace_folder/
├── 1.jpg          # screenshot
├── 1.xml          # UI layout XML
├── 2.jpg
├── 2.xml
├── ...
├── actions.json   # action sequence
└── react.json     # reasoning logs
```

Each frame may include:
- `image`: screenshot file path
- `screenshot`: screenshot data (numpy array or file path, for icon detection)
- `xml_text`: UI layout XML text
- `reasoning`: reasoning text
- `action`: executed action info
- `task_description`: task description (for dynamic matching)
- `text`: aggregated textual info (for simple matching)
- `app_id`: app package name (for icon resource path resolution)

Built-in checkers will look for info in these fields; you can also register custom checkers (see below).

## Custom Checker

```python
from avdag.conditions import register_condition, ConditionChecker

@register_condition("my_checker")
class MyChecker(ConditionChecker):
    def check(self, frame: dict, params: dict, options=None) -> bool:
        # read frame/params and perform any logic
        return frame.get("payload", {}).get("flag") == params.get("flag")
```

Then use it in configs:

```yaml
condition:
  type: my_checker
  params:
    flag: true
```

## Running Tests / Programmatic Usage

Use the following example to run a verification or execute provided test scripts:

### Custom Verification Options

```python
from avdag.verifier import make_llm_options, verify_task_folder

# Configure LLM verification
opts = make_llm_options(
    api_key="your-api-key",
    base_url="https://api.openai.com/v1",
    model="gpt-4o",
    force_llm=True  # force using LLM verification
)

# Run verification
result = verify_task_folder("task.yaml", "trace_folder", opts)
```

## Design Notes

### Core Algorithms
- Multi-path DAG verification: supports AND (deps) and OR (next) branching
- Path-aware verification: intelligent frame allocation to avoid cross-path conflicts
- Constraint checks: a node's matched frame index must be later than all its dependencies
- Verification flow:
  - Path analysis: print all possible successful paths
  - Candidate collection: collect candidate frame index sets per node (based on reachability)
  - Topological verification: compute minimum feasible index per node
  - Success decision: success if any/all target nodes in `success` have feasible indices
  - Result output: provide a linearized satisfying sequence based on matched indices

### Use Cases
- Mobile multi-path tasks: apps like Bilibili, Taobao with diverse user flows
- UI state recognition: quickly detect app states like WeChat home or chat window using icons
- Complex branching logic: tasks where users may choose different paths
- Filter condition verification: validate correct filtering operations per task requirements
- Multimodal verification chain: icons + OCR + LLM in escalating strategy
- Config debugging: quickly locate config issues via path analysis logs
- Human-in-the-loop review: mark uncertain cases for manual inspection

## Icon Detection: Detailed Notes

### Principles
Icon detection is powered by OpenCV template matching with multi-scale search and configurable similarity thresholds, enabling fast and accurate UI icon recognition on mobile screenshots.

### Key Features
- Multi-scale template matching: supports 0.5× to 2.0× scaling to adapt to different resolutions
- Smart threshold control: configurable similarity threshold for precision/recall balance
- Non-maximum suppression: remove duplicates and improve accuracy
- Smart resource path resolution: resolve icon resources based on app package name
- Batch detection optimization: detect multiple icons simultaneously

### Configuration Parameters

#### icons checker params
```yaml
icons:
  any: ["icon_001_Contacts", "icon_002_WeChat"]  # match any icon
  all: ["icon_001_Enter", "icon_002_Send"]       # must match all icons
  threshold: 0.85                                   # optional similarity threshold
```

#### Icon resources layout
```
task_configs/icons/
├── weixin/
│   ├── icon_001_通讯录.jpg
│   ├── icon_002_微信.jpg
│   └── icon_000_我.jpg
├── bilibili/
└── taobao/
```

### Position in Escalation Strategy
In the `escalate` checker, icon detection is placed after UI and before OCR/LLM:
```
text → regex → ui → icons → ocr → llm
```
This ensures:
1) Fast text/UI checks first
2) Icon detection provides visual confirmation
3) OCR handles complex text recognition
4) LLM performs the final intelligent judgment

### Performance Optimization
- Icon caching: loaded templates are cached to avoid repeated reads
- Early termination: in escalation mode, return early when icon detection succeeds
- Size pre-check: avoid oversized scales to speed up detection

### Recommendations
1) Icon quality: use clear, distinctive templates
2) Threshold tuning: adjust similarity, typically 0.8–0.9 works well
3) Naming convention: adopt a consistent naming scheme for easier management
4) Combine with other checkers for robustness

---

If you need to adapt data collected from real devices (OCR, UI dump, strict LLM callbacks, etc.), you can convert them into the `frames` array defined above for offline verification; you can also integrate more complex logic via custom checkers.
