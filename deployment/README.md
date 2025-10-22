# MobiAgent Server

## Deploy MobiMind Models with vLLM

```bash
vllm serve IPADS-SAI/MobiMind-Decider-7B --port <decider port>
vllm serve IPADS-SAI/MobiMind-Grounder-3B --port <grounder port>
vllm serve Qwen/Qwen3-4B-Instruct --port <planner port>
```

## Run Server

```bash
python -m mobiagent_server.server \
    --port <server port> \
    --decider_url <base url for decider model> \
    --grounder_url <base url for grounder model> \
    --planner_url <base url for planner model> \
    --use_qwen3 <whether to use MobiMind Qwen3-VL model series>
```

Then you can set MobiAgent Server IP and port in the MobiAgent App, and start exploration!