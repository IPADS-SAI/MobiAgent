<div align="center">
  <picture>
    <img alt="MobiAgent" src="assets/logo.png" width=10%>
  </picture>
</div>

<h3 align="center">
MobiAgent: A Systematic Framework for Customizable Mobile Agents
</h3>

<p align="center">
| <a href="https://arxiv.org/abs/2509.00531"><b>Paper</b></a> | <a href="https://huggingface.co/collections/IPADS-SAI/mobimind-68b2aad150ccafd9d9e10e4d"><b>Huggingface</b></a> | <a href="https://github.com/IPADS-SAI/MobiAgent/releases/tag/v1.0"><b>App</b></a> |
</p> 

<p align="center">
 <strong>English</strong> | <a href="README_zh.md">中文</a>
</p> 

---

## About

**MobiAgent** is a powerful and customizable mobile agent system including:

* **An agent model family**: MobiMind
* **An agent acceleration framework**: AgentRR
* **An agent benchmark**: MobiFlow

**System Architecture:**

<div align="center">
<p align="center">
  <img src="assets/arch.png" width="100%"/>
</p>
</div>

## News
- `[2025.11.03]` ✅ Added multi-task execution module support and user preference support. For details about multi-task usage and configuration, see [here](runner/mobiagent/multi_task/README.md). 
- `[2025.11.03]` 🧠 Introduced a user profile memory system: async preference extraction with LLM, raw-text preference storage and retrieval, optional GraphRAG via Neo4j. Preferences are retrieved as original texts and appended to experience prompts to personalize planning, see [here](runner/mobiagent/README.md).
- `[2025.10.31]` 🔥We've updated the MobiMind-Mixed model based on Qwen3-VL-4B-Instruct! Download it at [MobiMind-Mixed-4B-1031](https://huggingface.co/IPADS-SAI/MobiMind-Mixed-4B-1031), and add `--use_qwen3` flag when running dataset creation and agent runner scripts.
- `[2025.9.30]` 🚀 added a local experience retrieval module, supporting experience query based on task description, enhancing the intelligence and efficiency of task planning!
- `[2025.9.29]` We've open-sourced a mixed version of MobiMind, capable of handling **both Decider and Grounder tasks**! Feel free to download and try it at [MobiMind-Mixed-7B](https://huggingface.co/IPADS-SAI/MobiMind-Mixed-7B).
- `[2025.8.30]` We've open-sourced the MobiAgent!

## Evaluation Results

<div align="center">
<p align="center">
  <img src="assets/result1.png" width="30%" style="margin-right: 15px;"/>
  <img src="assets/result2.png" width="30%" style="margin-right: 15px;"/>
  <img src="assets/result3.png" width="30%"/>
</p>
</div>

<div align="center">
<p align="center">
  <img src="assets/result_agentrr.png" width="60%"/>
</p>
</div>

## Demo

**Mobile App Demo**:
<div align="center">
  <video src="https://github.com/user-attachments/assets/3a6539ea-34a5-4073-93aa-18986ca065ff"/>
</div>

**AgentRR Demo** (Left: first task; Right: subsequent task)
<div align="center">
  <video src="https://github.com/user-attachments/assets/ef5268a2-2e9c-489c-b8a7-828f00ec3ed1"/>
</div>

**Multi Task Demo**

task: `帮我在小红书找一下推荐的最畅销的男士牛仔裤，然后在淘宝搜这一款裤子，把淘宝中裤子品牌、名称和价格用微信发给小赵`
<div align="center">
  <video src="https://github.com/user-attachments/assets/92fdf23c-71d6-4c67-b02a-c3fa13fcc0e7"/>
</div>

## Project Structure

- `agent_rr/` - Agent Record & Replay framework
- `collect/` - Data collection, annotation, processing and export tools
- `runner/` - Agent executor that connects to phone via ADB, executes tasks, and records execution traces
- `MobiFlow/` - Agent evaluation benchmark based on milestone DAG
- `app/` - MobiAgent Android app
- `deployment/` - Service deployment for MobiAgent mobile application

## Quick Start

### Use with MobiAgent APP

If you would like to try MobiAgent directly with our APP, please download it in [Download Link](https://github.com/IPADS-SAI/MobiAgent/releases/tag/v1.0) and enjoy yourself!

### Use with Python Scripts

If you would like to try MobiAgent with python scripts which leverage Android Debug Bridge (ADB) to control your phone, please follow these steps:

#### Environment Setup

Create virtual environment, e.g., using conda:

```bash
conda create -n MobiMind python=3.10
conda activate MobiMind
```

Simplest environment setup (in case you want to run the agent runner alone, and do not want heavy dependencies like torch to be installed):

```bash
# Install simplest dependencies
pip install -r requirements_simple.txt
```

Full environment setup (in case you want to run the full pipeline): 

```bash
pip install -r requirements.txt

# Download OmniParser model weights
for f in icon_detect/{train_args.yaml,model.pt,model.yaml} ; do huggingface-cli download microsoft/OmniParser-v2.0 "$f" --local-dir weights; done

# Download embedding model utils
huggingface-cli download BAAI/bge-small-zh --local-dir ./utils/experience/BAAI/bge-small-zh

# Install OCR utils
sudo apt install tesseract-ocr tesseract-ocr-chi-sim  # chinese optional

# If you need GPU acceleration for OCR, install paddlepaddle-gpu according to your CUDA version
# For details, refer to https://www.paddlepaddle.org.cn/install/quick, for example CUDA 11.8:
python -m pip install paddlepaddle-gpu==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

```

#### Mobile Device Setup

- Download and install [ADBKeyboard](https://github.com/senzhk/ADBKeyBoard/blob/master/ADBKeyboard.apk) on your Android device
- Enable Developer Options on your Android device and allow USB debugging
- Connect your phone to the computer using a USB cable

#### Model Deployment

After downloading the `decider`, `grounder`, and `planner` models, use vLLM to deploy model inference services:

```bash
vllm serve IPADS-SAI/MobiMind-Decider-7B --port <decider port>
vllm serve IPADS-SAI/MobiMind-Grounder-3B --port <grounder port>
vllm serve Qwen/Qwen3-4B-Instruct --port <planner port>
```


#### Launch Agent Runner

Write the list of tasks that you would like to test in `runner/mobiagent/task.json`, then launch agent runner:

```bash
python -m runner.mobiagent.mobiagent --service_ip <Service IP> --decider_port <Decider Service Port> --grounder_port <Grounder Service Port> --planner_port <Planner Service Port>
```

Parameters:

- `--service_ip`: Service IP (default: `localhost`)
- `--decider_port`: Decider service port (default: `8000`)
- `--grounder_port`: Grounder service port (default: `8001`)
- `--planner_port`: Planner service port (default: `8002`)

The runner automatically controls the device and invoke agent models to complete the pre-defined tasks.

## Detailed Sub-module Usage

For detailed usage instructions, see the `README.md` files in each sub-module directory.

## Citation

If you find MobiAgent useful in your research, please feel free to cite our [paper](https://arxiv.org/abs/2509.00531):

```
@misc{zhang2025mobiagentsystematicframeworkcustomizable,
  title={MobiAgent: A Systematic Framework for Customizable Mobile Agents}, 
  author={Cheng Zhang and Erhu Feng and Xi Zhao and Yisheng Zhao and Wangbo Gong and Jiahui Sun and Dong Du and Zhichao Hua and Yubin Xia and Haibo Chen},
  year={2025},
  eprint={2509.00531},
  archivePrefix={arXiv},
  primaryClass={cs.MA},
  url={https://arxiv.org/abs/2509.00531}, 
}
```

## Acknowledgements
We gratefully acknowledge the open-source projects like MobileAgent, UI-TARS, and Qwen-VL, etc. We also thank the National Innovation Institute of High-end Smart Appliances for their support of this project.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=IPADS-SAI/MobiAgent&type=Date)](https://www.star-history.com/#IPADS-SAI/MobiAgent&Date)
