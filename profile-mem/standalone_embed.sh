#!/usr/bin/env bash

# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_VOLUME_DIR="$SCRIPT_DIR/volumes/milvus"
FALLBACK_VOLUME_DIR="$SCRIPT_DIR/volumes-user/milvus"
VOLUME_DIR="$DEFAULT_VOLUME_DIR"
EMBED_ETCD_CONFIG="$SCRIPT_DIR/embedEtcd.yaml"
USER_CONFIG="$SCRIPT_DIR/user.yaml"

docker_cmd() {
    docker "$@"
}

prepare_volume_dir() {
    if ! mkdir -p "$DEFAULT_VOLUME_DIR" 2>/dev/null || [ ! -w "$DEFAULT_VOLUME_DIR" ]; then
        VOLUME_DIR="$FALLBACK_VOLUME_DIR"
    fi

    mkdir -p "$VOLUME_DIR"
    chmod 777 "$VOLUME_DIR"
}

write_config_files() {
    cat << EOF > "$EMBED_ETCD_CONFIG"
listen-client-urls: http://0.0.0.0:2379
advertise-client-urls: http://0.0.0.0:2379
quota-backend-bytes: 4294967296
auto-compaction-mode: revision
auto-compaction-retention: '1000'
EOF

    cat << EOF > "$USER_CONFIG"
# Extra config to override default milvus.yaml
EOF
}

run_embed() {
    write_config_files
    prepare_volume_dir

    if [ ! -f "$EMBED_ETCD_CONFIG" ]
    then
        echo "embedEtcd.yaml file does not exist. Please try to create it in the current directory."
        exit 1
    fi

    if [ ! -f "$USER_CONFIG" ]
    then
        echo "user.yaml file does not exist. Please try to create it in the current directory."
        exit 1
    fi
    
    docker_cmd run -d \
        --name milvus-standalone \
        --security-opt seccomp:unconfined \
        -e ETCD_USE_EMBED=true \
        -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
        -e ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml \
        -e COMMON_STORAGETYPE=local \
        -e DEPLOY_MODE=STANDALONE \
        -v "$VOLUME_DIR":/var/lib/milvus \
        -v "$EMBED_ETCD_CONFIG":/milvus/configs/embedEtcd.yaml \
        -v "$USER_CONFIG":/milvus/configs/user.yaml \
        -p 19530:19530 \
        -p 9091:9091 \
        -p 2379:2379 \
        --health-cmd="curl -f http://localhost:9091/healthz" \
        --health-interval=30s \
        --health-start-period=90s \
        --health-timeout=20s \
        --health-retries=3 \
        milvusdb/milvus:v3.0-beta \
        milvus run standalone  1> /dev/null
}

wait_for_milvus_running() {
    echo "Wait for Milvus Starting..."
    local start_time
    start_time=$(date +%s)
    while true
    do
        health_status=$(docker_cmd inspect --format '{{if .State.Running}}{{if .State.Health}}{{.State.Health.Status}}{{else}}running{{end}}{{else}}{{.State.Status}}{{end}}' milvus-standalone 2>/dev/null)
        if [ "$health_status" = "healthy" ]
        then
            echo "Start successfully."
            echo "To change the default Milvus configuration, add your settings to the user.yaml file and then restart the service."
            break
        fi

        if [ "$health_status" = "exited" ] || [ "$health_status" = "dead" ]
        then
            echo "Milvus container exited before becoming healthy. Recent logs:"
            docker_cmd logs --tail 50 milvus-standalone
            exit 1
        fi

        if [ $(( $(date +%s) - start_time )) -ge 180 ]
        then
            echo "Timed out waiting for Milvus to become healthy. Current status: ${health_status:-unknown}"
            docker_cmd logs --tail 50 milvus-standalone
            exit 1
        fi

        sleep 1
    done
}

start() {
    res=`docker_cmd ps|grep milvus-standalone|grep healthy|wc -l`
    if [ $res -eq 1 ]
    then
        echo "Milvus is running."
        exit 0
    fi

    res=`docker_cmd ps -a|grep milvus-standalone|wc -l`
    if [ $res -eq 1 ]
    then
        docker_cmd start milvus-standalone 1> /dev/null
    else
        run_embed
    fi

    if [ $? -ne 0 ]
    then
        echo "Start failed."
        exit 1
    fi

    wait_for_milvus_running
}

stop() {
    docker_cmd stop milvus-standalone 1> /dev/null

    if [ $? -ne 0 ]
    then
        echo "Stop failed."
        exit 1
    fi
    echo "Stop successfully."

}

delete_container() {
    res=`docker_cmd ps|grep milvus-standalone|wc -l`
    if [ $res -eq 1 ]
    then
        echo "Please stop Milvus service before delete."
        exit 1
    fi
    docker_cmd rm milvus-standalone 1> /dev/null
    if [ $? -ne 0 ]
    then
        echo "Delete milvus container failed."
        exit 1
    fi
    echo "Delete milvus container successfully."
}

delete() {
    read -p "Please confirm if you'd like to proceed with the delete. This operation will delete the container and data. Confirm with 'y' for yes or 'n' for no. > " check
    if [ "$check" == "y" ] ||[ "$check" == "Y" ];then
        delete_container
        rm -rf "$SCRIPT_DIR/volumes"
        rm -rf "$SCRIPT_DIR/volumes-user"
        rm -rf "$EMBED_ETCD_CONFIG"
        rm -rf "$USER_CONFIG"
        echo "Delete successfully."
    else
        echo "Exit delete"
        exit 0
    fi
}

upgrade() {
    read -p "Please confirm if you'd like to proceed with the upgrade. The default will be to the latest version. Confirm with 'y' for yes or 'n' for no. > " check
    if [ "$check" == "y" ] ||[ "$check" == "Y" ];then
        res=`docker_cmd ps -a|grep milvus-standalone|wc -l`
        if [ $res -eq 1 ]
        then
            stop
            delete_container
        fi

        curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed_latest.sh && \
        bash standalone_embed_latest.sh start 1> /dev/null && \
        echo "Upgrade successfully."
    else
        echo "Exit upgrade"
        exit 0
    fi
}

case $1 in
    restart)
        stop
        start
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    upgrade)
        upgrade
        ;;
    delete)
        delete
        ;;
    *)
        echo "please use bash standalone_embed.sh restart|start|stop|upgrade|delete"
        ;;
esac
