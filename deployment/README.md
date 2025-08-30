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
    --service_ip <vllm service IP> \
    --port <server port> \
    --decider_port <vllm decider service port> \
    --grounder_port <vllm grounder service port> \
    --planner_port <vllm planner service port> \
```

Then you can set MobiAgent Server IP and port in the MobiAgent App, and start exploration!