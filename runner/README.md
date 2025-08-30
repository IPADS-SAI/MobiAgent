# Agent Runner

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
python -m runner.mobiagent.mobiagent --service_ip <服务IP> --decider_port <决策服务端口> --grounder_port <定位服务端口> --planner_port <规划服务端口>
```

**参数说明**
- `--service_ip`：服务IP（默认：`localhost`）
- `--decider_port`：决策服务端口（默认：`8000`）
- `--grounder_port`：定位服务端口（默认：`8001`）
- `--planner_port`：规划服务端口（默认：`8002`）

## UI-TARS Runner

<!-- TODO: UI-TARS Runner REAME here -->