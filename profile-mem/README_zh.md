# profile-mem 使用说明

这个目录下提供了一套本地 profile memory 工具，用来把用户偏好等个人信息存到 Mem0 + Milvus 中，并进行查询、插入、更新、删除、批量清空。

## 1. 前置条件

在使用下面的脚本前，建议先确认以下服务已经可用：

```bash
# 启动 Milvus
bash profile-mem/standalone_embed.sh start

# 启动本地 OpenAI 兼容 LLM 服务
bash profile-mem/manage_openai_llm_service.sh start
```

然后确认 `runner/mobiagent/.env` 已配置好至少这些变量：

```bash
MILVUS_URL=http://127.0.0.1:19530
EMBEDDING_MODEL=/absolute/path/to/embedding/model
EMBEDDING_MODEL_DIMS=512
MEM0_COLLECTION_NAME=mobiagent_local
OPENAI_API_KEY=local-openai-key
OPENAI_BASE_URL=http://127.0.0.1:18001/v1
```

默认情况下，下面的管理脚本会读取 `runner/mobiagent/.env`。

## 2. 查询当前存了哪些个人信息

```bash
/home/reck/Utils/anaconda3/envs/MobiMind/bin/python profile-mem/manage_profile_memory.py list
```

常用参数：

```bash
# 只看前 10 条
/home/reck/Utils/anaconda3/envs/MobiMind/bin/python profile-mem/manage_profile_memory.py list --limit 10

# 用语义查询缩小范围
/home/reck/Utils/anaconda3/envs/MobiMind/bin/python profile-mem/manage_profile_memory.py list --query "外卖偏好"

# 以 JSON 输出
/home/reck/Utils/anaconda3/envs/MobiMind/bin/python profile-mem/manage_profile_memory.py --json list
```

输出里最重要的是每条记录的 `id`。删除单条记录时建议先 `list`，再按 `id` 删除。

## 3. 插入一条个人信息

```bash
/home/reck/Utils/anaconda3/envs/MobiMind/bin/python profile-mem/manage_profile_memory.py add "用户偏好饮料：少冰，七分糖"
```

也可以顺手补充元数据：

```bash
/home/reck/Utils/anaconda3/envs/MobiMind/bin/python profile-mem/manage_profile_memory.py add \
  "用户偏好外卖：更看重配送速度和评分" \
  --type preference \
  --task-type takeaway
```

默认写入参数：

- `user_id=default_user`
- `infer=False`
- `source=manual_cli`

如果你想操作别的用户，可以加 `--user-id`：

```bash
/home/reck/Utils/anaconda3/envs/MobiMind/bin/python profile-mem/manage_profile_memory.py --user-id test_user add "用户偏好品牌：优先小米"
```

## 4. 更新一条个人信息

先查出要修改的记录 id：

```bash
/home/reck/Utils/anaconda3/envs/MobiMind/bin/python profile-mem/manage_profile_memory.py list
```

然后按 id 更新文本内容：

```bash
/home/reck/Utils/anaconda3/envs/MobiMind/bin/python profile-mem/manage_profile_memory.py update --id <memory_id> --text "用户偏好饮料：少冰，少糖"
```

例如：

```bash
/home/reck/Utils/anaconda3/envs/MobiMind/bin/python profile-mem/manage_profile_memory.py update --id 12345678-abcd-efgh-ijkl-1234567890ab --text "用户偏好饮料：少冰，少糖"
```

建议更新后再执行一次 `list`，确认修改结果。

## 5. 删除一条个人信息

先查出记录 id：

```bash
/home/reck/Utils/anaconda3/envs/MobiMind/bin/python profile-mem/manage_profile_memory.py list
```

然后按 id 删除：

```bash
/home/reck/Utils/anaconda3/envs/MobiMind/bin/python profile-mem/manage_profile_memory.py delete --id <memory_id>
```

例如：

```bash
/home/reck/Utils/anaconda3/envs/MobiMind/bin/python profile-mem/manage_profile_memory.py delete --id 12345678-abcd-efgh-ijkl-1234567890ab
```

## 6. 批量删除当前用户的所有个人信息

如果你想清空某个 `user_id` 下的所有记录，可以使用 `delete-all`。

注意：这个操作有破坏性，必须显式加 `--yes` 才会真正执行。

```bash
/home/reck/Utils/anaconda3/envs/MobiMind/bin/python profile-mem/manage_profile_memory.py delete-all --yes
```

如果要清空别的用户：

```bash
/home/reck/Utils/anaconda3/envs/MobiMind/bin/python profile-mem/manage_profile_memory.py --user-id test_user delete-all --yes
```

建议先执行一次 `list` 确认范围，再执行 `delete-all --yes`。

## 7. 指定别的环境文件

如果你不想使用默认的 `runner/mobiagent/.env`，可以显式指定：

```bash
/home/reck/Utils/anaconda3/envs/MobiMind/bin/python profile-mem/manage_profile_memory.py \
  --env-file /path/to/your/.env \
  list
```

## 8. 推荐操作流程

推荐按这个顺序使用：

1. 启动 Milvus 和本地 LLM 服务。
2. 执行 `list` 查看当前已存的个人信息。
3. 执行 `add` 手动补充一条偏好。
4. 再执行一次 `list`，确认写入结果。
5. 如果文本写得不准确，可以用 `update --id ... --text ...` 修改对应记录。
6. 如果有误，再用 `delete --id ...` 删除对应记录。
7. 如果要清空某个用户的所有记录，再使用 `delete-all --yes`。

## 9. 常见问题

如果报下面几类错误，通常按这个方向检查：

- `Missing required environment variable`：说明 `.env` 里缺少必需配置。
- `Embedding model path does not exist`：说明 `EMBEDDING_MODEL` 指向的本地模型目录不对。
- `Current mem0 client does not expose an update method`：说明当前环境里的 `mem0` 版本不支持更新接口。
- `delete-all is destructive`：说明你还没有加 `--yes` 确认参数。
- 连不上 Milvus：先执行 `bash profile-mem/standalone_embed.sh start`。
- 连不上本地 LLM：先执行 `bash profile-mem/manage_openai_llm_service.sh status`。
