#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_PYTHON_BIN=$(command -v python 2>/dev/null || true)
PYTHON_BIN=${PYTHON_BIN:-${DEFAULT_PYTHON_BIN:-/home/reck/Utils/anaconda3/envs/MobiMind/bin/python}}
MODEL_ID=${PROFILE_MEM_LLM_MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}
MODEL_DIR=${PROFILE_MEM_LLM_MODEL_DIR:-$SCRIPT_DIR/models/llm/Qwen2.5-0.5B-Instruct}
SERVED_MODEL_NAME=${PROFILE_MEM_SERVED_MODEL_NAME:-gpt-4o-mini}
HOST=${PROFILE_MEM_LLM_HOST:-127.0.0.1}
PORT=${PROFILE_MEM_LLM_PORT:-18001}
PID_FILE="$SCRIPT_DIR/openai-llm-server.pid"
LOG_FILE="$SCRIPT_DIR/openai-llm-server.log"

server_url() {
    printf 'http://%s:%s' "$HOST" "$PORT"
}

is_running() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" >/dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

download_model() {
    "$PYTHON_BIN" "$SCRIPT_DIR/local_openai_llm_server.py" \
        --model-id "$MODEL_ID" \
        --model-dir "$MODEL_DIR" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --download-only
}

start_service() {
    if is_running; then
        echo "Service already running (pid $(cat "$PID_FILE"))."
        status_service
        return 0
    fi

    download_model

    nohup "$PYTHON_BIN" "$SCRIPT_DIR/local_openai_llm_server.py" \
        --model-id "$MODEL_ID" \
        --model-dir "$MODEL_DIR" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --host "$HOST" \
        --port "$PORT" \
        >"$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE"

    for _ in $(seq 1 180); do
        if curl -fsS "$(server_url)/healthz" >/dev/null 2>&1; then
            echo "Local OpenAI-compatible LLM service is ready at $(server_url)/v1"
            return 0
        fi

        if ! kill -0 "$(cat "$PID_FILE")" >/dev/null 2>&1; then
            echo "Local LLM service exited during startup. Recent logs:"
            tail -n 80 "$LOG_FILE" || true
            rm -f "$PID_FILE"
            return 1
        fi

        sleep 1
    done

    echo "Timed out waiting for the local LLM service. Recent logs:"
    tail -n 80 "$LOG_FILE" || true
    return 1
}

stop_service() {
    if ! is_running; then
        echo "Service is not running."
        rm -f "$PID_FILE"
        return 0
    fi

    kill "$(cat "$PID_FILE")"
    wait "$(cat "$PID_FILE")" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "Service stopped."
}

status_service() {
    if ! is_running; then
        echo "Service is not running."
        return 1
    fi

    echo "Service running (pid $(cat "$PID_FILE"))"
    echo "Endpoint: $(server_url)/v1"
    curl -fsS "$(server_url)/v1/models"
    echo
}

case "${1:-}" in
    download)
        download_model
        ;;
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        stop_service || true
        start_service
        ;;
    status)
        status_service
        ;;
    *)
        echo "Usage: bash manage_openai_llm_service.sh download|start|stop|restart|status"
        exit 1
        ;;
esac