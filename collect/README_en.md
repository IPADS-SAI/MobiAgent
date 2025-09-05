# Data Collection and Annotation Tools

## Data Collection

### Data Format

Using manual/automatic tools, capture a screenshot before each action and record action metadata in a single `actions.json`. The general action schema is as follows:
```
{
  "app_name": string,
  "task_description": ["The description of the task list."],
  "action_count": "The count of the actions.",
  "actions": [
    {
      "type": "The type of the action",
      "parameters": "etc."
    },
    {
      "type": "click",
      "position_x": "x-coordinate of click",
      "position_y": "y-coordinate of click action",
      "bounds": "the bound of the clicked element"
    },
    {
      "type": "swipe",
      "press_position_x": "x-coordinate of press",
      "press_position_y": "y-coordinate of press",
      "release_position_x": "x-coordinate of release",
      "release_position_y": "y-coordinate of release",
      "direction": "The direction of the user's swipe gesture. UP: swipe finger upward to scroll content up and reveal content below. DOWN: swipe finger downward to scroll content down and reveal content above. LEFT: swipe finger leftward to scroll content left. RIGHT: swipe finger rightward to scroll content right."
    },
    {
      "type": "input",
      "text": "The text to input"
    },
    { "type": "done" },
    { "type": "wait" }
  ]
}
```

### Manual Data Collection

Start server:
```bash
python -m collect.manual.server
```
After startup, open http://localhost:9000 to access the web UI.

Operations:
1) Start Collection: click "Start Collection" in the web UI.
2) Configure App Info: fill in the popup dialog
   - App Name: e.g., "Eleme", "WeChat", "Taobao"
   - Task Type: e.g., "type1", "type2" (refer to your task doc)
3) Enter Task Description: describe the task clearly for later analysis and training
4) Execute Actions on the phone screenshot in the web UI:
   - Click: click the target position directly
   - Swipe: hold left mouse button and drag within screen area
   - Text Input: click "Text Input" and enter text in the dialog
5) Save Data:
   - Next Data: continue collecting more samples of the same type
   - Finish: end current session and save data
   - Delete Task: discard current data (for mistakes or invalid samples)

Data layout:
```
data/
├── <app_name>/
│   ├── <task_type>/
│   │   ├── 1/
│   │   │   ├── 1.jpg          # screenshot before step 1
│   │   │   ├── 2.jpg          # screenshot before step 2
│   │   │   ├── ...
│   │   │   └── actions.json   # action records and task info
│   │   ├── 2/
│   │   │   └── ...            # the second sample
│   │   └── ...
│   └── <other_task_type>/
└── <other_app_name>/
```
Each sample includes:
- Screenshot sequence: UI state before each step
- actions.json: full action sequence, task description, and app info

### Automatic Data Collection

First write your task list in `collect/auto/task.json` as a string array:
```json
[
  "Search iPhone on Taobao",
  "Send 'hello' to Zhang San on WeChat",
  "Follow the uploader Li Si on Bilibili"
]
```

Run the automatic collector:
```bash
python -m collect.auto.server --model <model_name> --api_key <API_key> --base_url <API_base_URL> [--max_steps <max_steps>]
```

Required params:
- `--model`: LLM model name
- `--api_key`: API key
- `--base_url`: API base URL

Optional params:
- `--max_steps`: max execution steps per task (default: 15)

Workflow:
1) Read tasks from `task.json`
2) For each task:
   - The agent chooses and launches the corresponding app
   - Executes actions (click, swipe, input, etc.) automatically
   - Takes a screenshot before every step and records metadata
   - Stops upon reaching max steps or finishing the task
3) Saves data automatically

Storage:
- Raw logs in `collect/auto/data_log/`
- Normalized data in `collect/auto/data/`
- Structure matches manual collection (screenshots + actions.json)

## Data Annotation

The annotation module converts raw action data into visually annotated data, providing richer context for LLMs to yield more accurate reasoning.

### Visual Annotation Format

Operation overlay:
- Each step's operation is labeled in red text at the top of the screenshot
- Auxiliary overlays:
  - Click: red circle at the click position
  - Swipe: red arrow from start to end

Data generation:
The annotated screenshots and task description are sent to the LLM to produce `react.json`, containing the reasoning and action decisions:

```json
[
  {
    "reasoning": "Reasoning for choosing this operation type",
    "function": {
      "name": "click",
      "parameters": {
        "target_element": "High-level semantic description of target"
      }
    }
  },
  {
    "reasoning": "Reasoning for swipe operation",
    "function": {
      "name": "swipe",
      "parameters": {
        "direction": "UP, DOWN, LEFT, RIGHT"
      }
    }
  },
  {
    "reasoning": "Reasoning for text input",
    "function": {
      "name": "input",
      "parameters": {
        "text": "Text to input"
      }
    }
  },
  {
    "reasoning": "Basis for task completion",
    "function": {
      "name": "done",
      "parameters": {}
    }
  },
  {
    "reasoning": "Reason for waiting",
    "function": {
      "name": "wait",
      "parameters": {}
    }
  }
]
```

### Run Auto Annotation

Command:
```bash
python -m collect.annotate --data_path <data_path> --model <model_name> --api_key <API_key> --base_url <API_base_URL>
```

Parameters:
- `--data_path`: path to raw trajectory data (default: `data` under current dir)
- `--model`: LLM model name (required)
- `--api_key`: model service API key (required)
- `--base_url`: model service base URL (required)

Process:
1) Load screenshots and `actions.json` from the data directory
2) Add visual annotations to screenshots per action info
3) Send annotated data to the LLM for reasoning
4) Generate `react.json` with step-by-step reasoning
5) Save the complete annotated dataset for training

Storage:
```
dir/
├── 1.jpg          # screenshot before step 1
├── 2.jpg          # screenshot before step 2
├── ...
└── actions.json   # action records and task info
└── react.json     # annotated reasoning data
```

## Data Construction

The construction module converts annotated data into training-ready formats, supporting SFT dataset generation.

### Command

```bash
python -m collect.construct_sft --data_path <raw_data_path> --ss_data_path <single_step_data_path> --unexpected_img_path <unexpected_img_path> --out_path <output_path> [--factor <scale_factor>] [--train_ratio <train_ratio>]
```

### Parameters

Required:
- `--data_path`: raw trajectory data path (default: `data`)
- `--ss_data_path`: single-step data path (default: `ss_data`)
- `--unexpected_img_path`: unexpected image data path (default: `unexpected_img`)
- `--out_path`: output path for training dataset (default: `output`)

Optional:
- `--factor`: image downscale factor (default: `0.5`)
- `--train_ratio`: train/val split ratio (default: `0.9`)
