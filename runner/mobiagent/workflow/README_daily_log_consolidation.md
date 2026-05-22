# Daily Log Consolidation

这个脚本用于把 `daily-log/YYYY-MM-DD/` 下面按日期分散存放的原始总结，整理成按 task 聚合的结构化画像文件。

目标不是简单拼接，而是做三件事：

1. 把同一个 task 在不同日期下的记录聚合到一个文件里。
2. 把单条原始 entry 中可能混在一起的多条事实拆开。
3. 结合最近已经记录过的事实做增量去重，避免重复写入。

## 输入约定

- 原始输入目录默认是 `runner/mobiagent/workflow/test-runs/daily-log/`
- 每个日期一个子目录，例如 `2026-05-22/`
- 每个原始文件名默认按 `package__task_key__[persona].md` 解析
- 当前默认严格按文件名聚合，不自动合并 `*_v2` 这样的 task 版本

也就是说：

- `com.tencent.wechat__01_basic_gui_task_weixin__任姚丹珺.md` 和
- `com.tencent.wechat__01_basic_gui_task_weixin_v2__任姚丹珺.md`

默认会被视为两个不同的聚合目标。

## 输出约定

输出目录默认是 `runner/mobiagent/workflow/profile-consolidation/outputs/`。

脚本会按 `YYYY-MM` 自动创建月度子目录，例如：

- `runner/mobiagent/workflow/profile-consolidation/outputs/2026-05/`

每个输出文件仍然按原始文件名命名，但会放在对应月份目录下面，例如：

- `2026-05/com.tencent.wechat__01_basic_gui_task_weixin__任姚丹珺.md`

输出现在拆成两个文件：

1. 主 Markdown 文件：只保留人可读表格
2. 同名 sidecar 状态文件：`<文件名>.state.json`

其中：

- Markdown 内容用于人工审阅
- sidecar 状态文件用于让脚本可靠地读取、增量更新、避免破坏结构

可见的 Markdown 正文现在只保留一个表格。

表格列包括：

- 序号
- 记录日期
- 最近来源日期
- 内容
- 来源条目
- 去重状态

## 模型参与的两个阶段

脚本默认通过 `service_ip + model_port` 连接一个 OpenAI-compatible 服务，并调用目标端口上的模型完成两阶段处理。

`--model_name` 现在是可选的：

- 如果你显式传了 `--model_name`，脚本就用这个名字。
- 如果你不传，脚本会自动请求目标端口的 `/v1/models`，并使用返回列表中的第一个模型 ID。

这适合直接在一个端口上部署单模型 vLLM 服务的场景。

### 阶段一：事实拆分与规范化

输入是一条原始 daily-log entry，例如一段聊天总结或一条账单总览。

模型需要把它拆成 1 到多条更适合沉淀到个人画像的事实，每条事实至少包含：

- `summary`
- `normalized_fact`
- `event_date`
- `tags`
- `model_notes`

这一步解决的问题是：一条原始记录里可能混着多件事，而且原文表述不稳定，不适合直接拿来去重。

### 阶段二：近邻去重判断

对每条候选事实，脚本不会扫描整个月的全部记录，而是先把候选范围限制在：

- 近 7 天内的记录
- 或最近 100 条记录

两者中条数更小的那一组。

然后才会从这组记录里进一步挑出最多 `dedup_window` 条最相关候选，发给模型做判断。默认 `dedup_window = 5`。

模型返回以下四种决策之一：

- `new`
- `duplicate`
- `merge_with_existing`
- `skip_ambiguous`

脚本只负责读取已有文件、组织上下文、接收模型决策、并以稳定格式写入最终结果。

去重的基准不只是本次输入的原始 daily-log，还包括已经存在于 `profile-consolidation/outputs/YYYY-MM/` 里的当月历史聚合结果。每次运行时，脚本都会先读取目标月份目录下、同名 Markdown 对应的 sidecar 状态文件，再决定是新增、合并还是跳过。

也就是说：

- 生成和语义判断交给模型
- 最终文件结构、读取、更新、追加交给代码

这样做的原因是，去重判断需要语义能力，但最终产物必须可重复执行、可增量追加、且格式稳定。

## 为什么最终写入必须由代码负责

如果把最终文件整个交给模型重写，会有几个问题：

1. 文件格式不稳定，表格和详情块容易漂移。
2. 重复运行时幂等性差，容易出现重复记录或编号跳变。
3. 很难保证 source refs、record_id、metadata 等字段持续一致。

因此当前设计是：

- 模型只输出结构化 JSON 决策
- 代码根据这个决策更新固定结构的 Markdown 文件和 sidecar 状态文件

## 命令示例

### 1. 处理单个 task 文件

```bash
python -m runner.mobiagent.workflow.consolidate_daily_logs \
  --start_date 2026-05-20 \
  --end_date 2026-05-22 \
  --file_name com.tencent.wechat__01_basic_gui_task_weixin__任姚丹珺.md \
  --service_ip 166.111.53.96 \
  --model_port 8000
```

### 2. 批量处理一类文件

```bash
python -m runner.mobiagent.workflow.consolidate_daily_logs \
  --start_date 2026-05-20 \
  --end_date 2026-05-22 \
  --file_glob 'com.tencent.wechat__*.md' \
  --service_ip 166.111.53.96 \
  --model_port 8000
```

### 3. Dry run

`--dry_run` 会执行解析、拆分、去重判断，但不写目标文件。

```bash
python -m runner.mobiagent.workflow.consolidate_daily_logs \
  --start_date 2026-05-20 \
  --end_date 2026-05-22 \
  --file_name com.tencent.wechat__01_basic_gui_task_weixin__任姚丹珺.md \
  --service_ip 166.111.53.96 \
  --model_port 8000 \
  --dry_run
```

### 4. 不调用模型，只做结构验证

`--disable_model` 主要用于调试脚本本身。这个模式下：

- 一条原始 entry 默认只会生成一条候选事实
- 去重只用非常保守的字符串精确匹配

所以它适合验证脚本结构、解析逻辑、输出格式，不适合作为正式画像整理结果。

```bash
python -m runner.mobiagent.workflow.consolidate_daily_logs \
  --start_date 2026-05-20 \
  --end_date 2026-05-22 \
  --file_name com.tencent.wechat__01_basic_gui_task_weixin__任姚丹珺.md \
  --dry_run \
  --disable_model
```

## 关键参数

- `--input_root`: 原始 daily-log 根目录
- `--output_dir`: 聚合结果目录
- `--start_date`: 开始日期
- `--end_date`: 结束日期
- `--file_name`: 精确指定一个原始文件名
- `--file_glob`: 用 glob 批量选择原始文件
- `--service_ip`: 模型服务 IP
- `--model_port`: OpenAI-compatible 端口
- `--model_name`: 可选，显式指定模型名称；如果不传，脚本会自动取 `/v1/models` 返回的第一个模型
- `--dedup_window`: 去重时发送给模型的最近记录数量，默认 5
- `--max_entries`: 限制本次最多处理多少条原始 entry
- `--dry_run`: 不落盘
- `--disable_model`: 禁用模型，仅做调试

## 当前实现策略

当前版本采用以下规则：

1. 聚合顺序固定为日期升序、文件内 entry 编号升序。
2. 输出按月写入 `outputs/YYYY-MM/`，每次读取、去重、更新都只作用在对应月份目录下的文件中。
3. 去重候选范围先限制为“近 7 天 / 最近 100 条”中较小的集合，而不是整个月全部记录。
4. 在这个近窗范围里，再选最多 `dedup_window` 条最相关记录送给模型判断，默认 `dedup_window=5`。
5. 如果模型认为是 `duplicate` 或 `merge_with_existing`，脚本会更新已有记录的来源引用和补充信息，而不是重复新增。
6. 如果模型无法稳定判断，则跳过该事实，并在日志中记录。

## 建议的使用方式

推荐先这样跑：

1. 先用 `--dry_run` 看拆分和去重统计是否合理。
2. 再对单个 task 做真实写入。
3. 检查输出文件结构是否符合预期。
4. 最后再放大到一个文件 glob 或更长日期范围。
