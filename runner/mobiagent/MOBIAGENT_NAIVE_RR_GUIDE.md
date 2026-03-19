# MobiAgent `mobiagent_naive_rr.py` 全量技术说明（Graph-First Replay）

本文面向第一次接触本项目的开发同学，目标是回答三个问题：

1. 这个文件在系统里负责什么。
2. 它如何把轨迹增量沉淀成可复用的有向图（template graph）。
3. 新任务到来时，它如何“尽可能复用、复用失败如何回退”。

本文基于当前分支现行实现（`graph-first replay`），并在附录说明旧版 `ExperienceRR` 逻辑。

---

## 1. 这是什么

`runner/mobiagent/mobiagent_naive_rr.py` 是一个 Android 任务执行器，负责：

- 连接设备，执行“截图 -> 理解 -> 动作 -> 记录”循环。
- 维护每步轨迹（截图、XML、动作、可视化结果）。
- 优先使用 `template_graph_builder.py` 的图检索做无模型复用（graph replay）。
- 复用失败时，单步回退到 `decider + grounder` 模型链路。
- 任务结束后把本次动作写回 template graph，持续增量学习。

在系统中，它是“运行时控制面 + 轨迹生产者 + 图增量更新入口”。

---

## 2. 快速上手

## 2.1 运行前准备

- Android 设备可通过 `uiautomator2` 连接。
- 三个推理服务可访问：
  - Decider（默认 `8000`）
  - Grounder（默认 `8001`）
  - Planner（默认 `8002`，当前主流程不强依赖）
- 已安装 Python 依赖（`openai`, `Pillow`, `uiautomator2` 等）。

## 2.2 最小运行命令（单任务）

```bash
python runner/mobiagent/mobiagent_naive_rr.py \
  --service_ip localhost \
  --decider_port 8000 \
  --grounder_port 8001 \
  --planner_port 8002 \
  --device Android \
  --app 饿了么 \
  --task "打开饿了么，搜索茶百道并加入购物车" \
  --data_dir runner/mobiagent/data-Android/3.19test \
  --template_graph on \
  --template_graph_dir runner/mobiagent/data-Android/3.19test/_template_graph
```

## 2.3 关键 CLI 参数

- `--task` / `--task_file`: 单任务文本 / 任务列表 JSON（字符串数组）。
- `--data_dir`: 任务产物目录根（必填）。
- `--app`: 指定应用名；不填时尝试从任务文本中提取。
- `--template_graph`: `on/off`，是否启用图能力（默认 `on`）。
- `--template_graph_dir`: 图文件目录（默认 `<data_dir>/_template_graph`）。
- `--use_qwen3`: grounder 坐标体系开关（默认开启）。
- `--no_compress`: 是否关闭截图压缩（默认压缩）。

## 2.4 典型产物目录

每条任务在 `<data_dir>/task_<task_hash>/` 下落盘：

- `01_screenshot.jpg`, `02_screenshot.jpg`, ...
- `01.xml`, `02.xml`, ...
- `*_click_visualization.jpg`, `*_bounds.jpg`
- `actions_record.json`（本次运行总结果）
- `task_<hash>_experience.json`（当前 run 的动作审计记录）

图文件在 `--template_graph_dir` 下，按 app 名保存：

- `<safe_app_name>.graph.json`（例如 `app_dc47280b.graph.json`）

---

## 3. 执行主流程（时序）

主入口在 `main()`，核心执行在 `execute_task_with_rr(...)`。

## 3.1 启动阶段

1. 初始化 LLM 客户端（decider / grounder / planner）。
2. 连接 Android 设备。
3. 解析任务（单条或列表），为每条任务创建独立目录 `task_<md5_8>`。
4. 启动 app，进入执行循环。

## 3.2 每步循环（`while step < MAX_STEPS`）

每一步都按这个顺序：

1. 截图并落盘。
2. dump 当前 UI XML 并落盘。
3. 计算 XML 页面签名；若连续 `NO_PROGRESS_XML_STREAK=3` 次不变，强制结束（`stop_reason=no_progress_xml`）。
4. 收集图运行时 hint（候选节点、预测下一步、中断检测、低置信 frontier）。
5. 尝试 graph replay（无模型）：
   - 调 `search_replay_actions` 拿候选动作。
   - 逐候选执行 `try_execute_graph_replay_candidate`。
   - 成功则写动作并直接进入下一步；失败仅记录拒绝原因，不影响后续步骤继续尝试图复用。
6. 若本步没有任何候选成功，回退到模型链路：
   - decider 出动作意图
   - 对 click/click_input 由 grounder 出 bbox
   - 在执行 `click/click_input/input/swipe` 前执行 TOCTOU 截图一致性校验（aHash，相似度阈值 `0.95`）
   - 若校验失败：同一步最多重试 `2` 次（重抓图 -> 刷新本步 XML -> 重跑模型链路）
   - 超过重试上限仍不一致：本步跳过动作，进入下一步
   - 校验通过后才执行动作并落盘记录
7. `done` 动作则结束循环。

## 3.3 结束阶段

1. 调 `TemplateGraphBuilder.update_from_run(...)` 把本次 `actions` 写回图。
2. 写 `actions_record.json`。
3. 写本次 `task_<hash>_experience.json`（仅审计用途，当前不参与 runtime replay）。

---

## 4. 数据模型设计

## 4.1 `actions`（运行时动作序列）

每步至少包含：

- `step`, `action`, `screenshot_file`, `xml_file`

按动作类型附加：

- `click/click_input`: `target`, `position`, `bbox`, `ui_meta`, ...
- `input`: `text`
- `swipe`: `direction`
- 模型链路执行成功的 `click/click_input/input/swipe` 会额外带：
  - `toctou_check: {score, threshold, attempt, check_image_file}`
- graph replay 命中动作会额外有：
  - `replayed: true`
  - `replay_source: "template_graph"`
  - `graph_replay: {edge_id, from_node_id, to_node_id, action_sig, ...}`

## 4.2 `actions_record.json` 顶层结构（当前版本）

```json
{
  "app_name": "...",
  "task_description": "...",
  "mode": "auto",
  "graph_replay": { ...统计... },
  "stop_reason": "...",
  "action_count": 6,
  "actions": [ ... ],
  "template_graph": {
    "enabled": true,
    "graph_file": "...",
    "update_summary": { ... },
    "runtime_hints": [ ... ],
    "error": "..."  // 可选
  }
}
```

`graph_replay` 字段含义：

- `steps_attempted`: 尝试图复用的步数
- `candidate_count`: 累计纳入尝试窗口的候选数
- `candidates_tried`: 实际尝试执行的候选数
- `steps_replayed` / `actions_replayed`: 命中复用步数/动作数
- `step_fallback_to_decider`: 本步图复用失败后回退次数
- `rejected_reasons`: 拒绝原因计数（如 `ui_meta_not_matched`）
- `executed_edges` / `executed_from_nodes` / `executed_sources`: 命中来源统计
- `done_by_graph_replay`: 是否由图候选直接 done

## 4.3 Template Graph JSON 结构

图文件核心是 `meta + nodes + edges`。

`meta`：

- `run_count`: 累计写图 run 次数
- `node_seq`, `edge_seq`: 自增 ID 计数器
- `updated_at`: 最近更新时间

`nodes[*]` 关键字段：

- `node_id`, `type`（`main/overlay_popup/interruption/terminal`）
- `visit_count`
- `confidence`
- `exact_match_count/fuzzy_match_count/uncertain_match_count`
- `signatures.strict/coarse`
- `feature_stats`：
  - `coarse_token_counts`
  - `package_counts`
  - `keyword_counts`
  - `ui_meta_counts`
- `context_counts`：
  - `task_hash/task_tokens/action_name/replayed/stop_reason`
- `examples`（样例 step/xml/screenshot 指针）

`edges[*]` 关键字段：

- `edge_id`, `from`, `to`
- `action_sig`, `action_type`
- `count`, `probability`, `confidence`
- `transition_type`（`normal/interrupt/popup_enter/popup_exit/recovery`）
- `context_counts`（含 `replayed` 统计）
- `action_payload`（当前主载荷）
- `action_payload_counts`（载荷频次，用于主载荷选举）

---

## 5. 节点与边如何生成

## 5.1 从动作到状态转移

`_build_transitions` 会把 `actions` 转为同长度的转移列表：

- `from_obs`：当前 action 所在页面 XML
- `to_obs`：
  - 若当前 action 是 `done`（或强制终止），转到 terminal 观测；
  - 否则用下一条 action 的 XML；
  - 若没有下一条，则转到 `trace_end` terminal。

这意味着一次 run 的转移数通常等于 `len(actions)`。

## 5.2 节点匹配/创建规则

`_resolve_or_create_node` 规则：

1. strict 命中：`strict_signature` 完全一致，直接归并。
2. 否则 fuzzy：基于 `_fuzzy_similarity` 算分，阈值：
   - `MID_MATCH_THRESHOLD = 0.62`（低于则新建）
   - `HIGH_MATCH_THRESHOLD = 0.82`（`0.62~0.82` 记为 `soft`）
3. 特殊防误并：
   - popup/interruption 与 main 之间，若分数 `<0.97` 倾向拆成分支，不强行归并。
4. 仍不满足则新建节点。

## 5.3 边更新规则

`_upsert_edge` 的键是 `(from_node_id, action_sig, to_node_id)`：

- 命中已有边：`count += 1`
- 否则新建边

并且每次更新：

- `transition_type_counts` 与主 `transition_type`
- `context_counts`（包含 `replayed=True/False`）
- `action_payload/action_payload_counts`
- 同组边（同 `from + action_sig`）的概率与置信度

---

## 6. 复用机制（核心）

复用入口：`TemplateGraphBuilder.search_replay_actions(current_xml, mode="conservative")`

## 6.1 固定保守模式

当前实现会把非 `conservative` 模式强制回退到 `conservative`。

## 6.2 候选检索流程

1. 先 `infer_current_node`（top-3）得到当前状态候选节点。
2. 节点筛选：
   - strict 命中直接保留（`node_score=1.0`）
   - fuzzy 命中需同时满足：
     - `node_score >= 0.90`
     - `node_confidence >= 0.60`
3. 若没有任何节点通过筛选，启用多起点回退：
   - `get_root_nodes()` 找所有入度为 0 的非 terminal 节点
   - 取 top-3，赋默认 `node_score=0.58`，来源标记 `root_fallback`
4. 对每个候选节点调用 `predict_next(top-3)`，再过滤边：
   - `edge_confidence >= 0.60`
   - `edge_probability >= 0.34`
   - 边必须有可解析 `action_payload`
   - `action_type` 必须在 `{click, click_input, input, swipe, done}`
5. 候选动作打分并排序，最多返回 6 个：

```text
replay_score = 0.50 * node_score + 0.30 * edge_confidence + 0.20 * edge_probability
```

排序键：`replay_score desc -> edge_probability desc -> edge_confidence desc`

## 6.3 每步执行策略

在 `execute_task_with_rr` 中：

- 每步最多尝试 `GRAPH_REPLAY_MAX_CANDIDATES_PER_STEP = 6` 个候选。
- 一旦某候选成功，本步直接结束并进入下一步。
- 本步所有候选都失败，才调用 decider/grounder。

这就是“尽可能复用 + 单步回退”的关键语义。

---

## 7. 匹配机制细节

## 7.1 状态匹配（节点）

观测提取 `_extract_xml_features`：

- `strict_signature`：对包含 `bounds` 的 token 序列做 MD5
- `coarse_signature`：去重后的不含 `bounds` token 序列做 MD5
- `coarse_tokens`, `package_counts`, `keyword_counts`

`_fuzzy_similarity` 公式：

```text
score = 0.65 * jaccard(coarse_tokens)
      + 0.25 * package_overlap
      + 0.10 * ui_meta_hit
```

## 7.2 点击 guard（graph replay）

`try_execute_graph_replay_candidate` 对 click 类动作的确定性校验：

1. 若 payload 含 `ui_meta`：
   - 用 `find_android_point_by_ui_meta` 在当前 XML 精确匹配。
   - 匹配优先级：`matched_keys` 更高优先；相同则面积更小优先。
   - 匹配失败拒绝：`ui_meta_not_matched`。
2. 若无可用 `ui_meta`，尝试 `position` 回退。
3. `click_input` 必须有 `text`，否则拒绝：`missing_click_input_text`。

常见拒绝原因：

- `ui_meta_not_matched`
- `invalid_click_position`
- `missing_click_input_text`
- `missing_input_text`
- `unsupported_action:<type>`

## 7.3 页面无进展检测

每步计算 `build_xml_page_signature`，若连续 3 次同签名触发强制停止：

- `stop_reason = no_progress_xml`

## 7.4 TOCTOU 执行前一致性 guard（模型链路）

- 仅作用于 decider/grounder 路径，不影响 graph replay 路径。
- 比对方式：对“模型参考截图”和“执行前新截图”做 aHash 相似度比较。
- 关键参数：
  - `TOCTOU_HASH_SIZE = 16`
  - `TOCTOU_IMAGE_SIM_MIN = 0.95`
  - `TOCTOU_MAX_RETRY_PER_STEP = 2`
- 失败处理：同一步重试（最多 2 次），超过上限则跳过本步动作，不终止任务。
- 可观测性：日志含 `[TOCTOU]` 前缀，记录 `score/threshold/attempt/reason`。

---

## 8. 回退与终止策略

## 8.1 回退策略

- 复用失败不“全局禁用 replay”。
- 仅当前步回退模型决策，下一步仍继续尝试 graph replay。

## 8.2 终止条件

- decider 返回 `done`
- graph replay 候选直接 `done`（`stop_reason=graph_replay_done`）
- no-progress guard
- 达到 `MAX_STEPS=15`（`max_steps_reached`）
- 截图/编码/模型解析等异常导致提前退出
- TOCTOU 重试耗尽不会终止任务，只会“跳过本步动作并进入下一步”

## 8.3 Ctrl+C（用户中断）

当前实现中，`KeyboardInterrupt` 在 `main()` 捕获后直接退出。  
如果中断发生在 `execute_task_with_rr` 收尾前，通常不会执行：

- `update_from_run`（图更新）
- `actions_record.json` 写盘
- `experience` 保存

但已生成的截图/XML等中间文件会保留在任务目录。

---

## 9. 日志与可观测性

## 9.1 关键日志行

- 图候选检索失败：
  - `[GRAPH-REPLAY] candidate search failed ...`
- 本步命中复用：
  - `[GRAPH-REPLAY] accepted step=... action=... edge=... from=...`
- 本步回退：
  - `[GRAPH-REPLAY] no accepted candidate at step ..., fallback to decider`
- 图更新结果：
  - `[GRAPH] Updated graph: ... (nodes+=..., edges+=...)`

## 9.2 如何判断“复用了哪些部分”

优先看 `actions_record.json`：

1. 顶层 `graph_replay` 统计（命中步数、回退步数、拒绝原因）。
2. `actions[*]` 中 `replayed=true` 的动作：
   - `replay_source = "template_graph"`
   - `graph_replay.edge_id/from_node_id/to_node_id`

命中样例（简化）：

```json
{
  "step": 1,
  "action": "click_input",
  "replayed": true,
  "replay_source": "template_graph",
  "graph_replay": {
    "edge_id": "e16",
    "from_node_id": "n1",
    "to_node_id": "n2",
    "search_source": "root_fallback",
    "replay_score": 0.7
  }
}
```

回退样例（顶层统计）：

```json
{
  "graph_replay": {
    "steps_attempted": 6,
    "steps_replayed": 1,
    "step_fallback_to_decider": 5,
    "rejected_reasons": {
      "ui_meta_not_matched": 5
    }
  }
}
```

## 9.3 最小可复制排障流程

1. 看 `actions_record.json.graph_replay.steps_replayed` 是否为 0。
2. 看 `rejected_reasons` 主因（常见是 `ui_meta_not_matched`）。
3. 找 `actions[*].graph_replay.edge_id`，回查 graph 文件对应 edge。
4. 看 edge 是否有 `action_payload`（旧边无 payload 无法复用）。
5. 看 `template_graph.update_summary` 是否持续 `edges_created=0`（可能没有有效学习）。
6. 看日志是否长期 `fallback to decider`。
7. 必要时核查当前 XML 与 payload 的 `ui_meta` 是否一致。

---

## 10. 常见问题与调参建议

## 10.1 误复用 vs 漏复用

- 误复用高：提高 `edge_confidence/probability` 阈值，增加 `ui_meta` 约束。
- 漏复用高：适度降低 `node_score` 或 `edge_confidence`，但会提高误复用风险。

## 10.2 图冷启动

- 冷启动初期没有足够边，`steps_replayed` 低是正常现象。
- 先让模型跑出几条稳定轨迹，再观察复用率。

## 10.3 旧图兼容

- 旧边没有 `action_payload` 时会被检索层跳过。
- 新 run 会逐步补齐 payload（边命中后写入 `action_payload_counts` 并选主载荷）。

## 10.4 参数影响（当前默认）

| 参数 | 默认值 | 作用 | 增大效果 | 减小效果 |
|---|---:|---|---|---|
| `SEARCH_CONSERVATIVE_NODE_SCORE_MIN` | 0.90 | 节点 fuzzy 门槛 | 更保守 | 更激进 |
| `SEARCH_CONSERVATIVE_EDGE_CONFIDENCE_MIN` | 0.60 | 边可靠度门槛 | 更保守 | 更激进 |
| `SEARCH_CONSERVATIVE_EDGE_PROBABILITY_MIN` | 0.34 | 分支概率门槛 | 更保守 | 更激进 |
| `GRAPH_REPLAY_MAX_CANDIDATES_PER_STEP` | 6 | 每步尝试上限 | 更高命中、也更慢 | 更快、可能漏命中 |
| `TOCTOU_IMAGE_SIM_MIN` | 0.95 | 执行前截图相似度门槛 | 更严格、更易触发重试/跳步 | 更宽松、误操作风险上升 |
| `TOCTOU_MAX_RETRY_PER_STEP` | 2 | 同一步 TOCTOU 最大重试次数 | 更稳健但更慢 | 更快但更容易因漂移丢动作 |
| `TOCTOU_HASH_SIZE` | 16 | aHash 比对粒度（`N x N`） | 更细粒度、更敏感 | 更粗粒度、更容错 |

---

## 附录A：旧 `ExperienceRR` 机制（只读）

当前 runtime 已切换为 graph-first，但 `ExperienceRR` 类仍保留，主要用于：

- 当前 run 的审计记录写盘（`save_experience`）。
- 兼容旧逻辑代码资产。

历史机制（旧）要点：

1. 先按任务文本检索历史经验（3阶段）：
   - 编辑距离/关键词（阈值 `0.70/0.30`）
   - BM25（阈值 `0.45`）
   - 向量相似度（阈值 `0.58`）
2. 再做 replay 可用性门控（长度、重复率、XML 完整性等）。
3. 线性 `replay_idx` 顺序复放，一旦某步失败常会导致后续不再复放。

与当前 graph-first 对比：

| 维度 | 旧 Experience Replay | 当前 Graph Replay |
|---|---|---|
| 复用单位 | 线性历史轨迹 | 当前状态出边动作 |
| 步间依赖 | 强（前一步失败影响后续） | 弱（每步独立） |
| 多起点 | 不支持 | 支持（root fallback） |
| 失败策略 | 常见全局降级 | 本步回退，下一步继续尝试 |
| 数据来源 | 历史经验文件 | template graph |

---

## 附录B：关键函数索引

## B.1 `mobiagent_naive_rr.py`

- 启动与流程：
  - `main`
  - `execute_task_with_rr`
- 图复用执行：
  - `try_execute_graph_replay_candidate`
  - `find_android_point_by_ui_meta`
- 模型链路：
  - `get_decider_action`
  - `get_grounder_coords`
- 数据与工具：
  - `dump_hierarchy_for_step`
  - `get_screenshot_b64`
  - `compute_image_ahash_similarity`
  - `evaluate_toctou_pre_action`
  - `parse_json_response`
  - `visualize_click`
  - `visualize_model_bbox`
- 设备封装：
  - `AndroidDevice`
- 旧机制相关（附录）：
  - `ExperienceRR`
  - `evaluate_replay_xml_match`（当前 graph-first 主流程不使用）

## B.2 `template_graph_builder.py`

- 图主流程：
  - `update_from_run`
  - `_build_transitions`
  - `_resolve_or_create_node`
  - `_upsert_edge`
- 运行时查询：
  - `infer_current_node`
  - `predict_next`
  - `get_root_nodes`
  - `search_replay_actions`
- 观测与特征：
  - `_extract_xml_features`
  - `_fuzzy_similarity`
- 置信度与统计：
  - `_calc_node_confidence`
  - `_refresh_edge_group_confidence`
- 动作载荷学习：
  - `_build_replay_payload`
  - `_update_edge_payload`
  - `_edge_primary_payload`

---

## 备注

- 本文描述的是“当前代码行为”，不是理论设计草案。
- 如果你准备二次开发，建议先从：
  - `search_replay_actions`（检索策略）
  - `try_execute_graph_replay_candidate`（执行 guard）
  - `update_from_run`（图增量学习）
  三个点入手。

---
- 怎么看“新轨迹是否复用了、复用了哪些部分”
看每个 run 的 actions_record.json，重点两层：
1. 顶层统计 graph_replay
  - 例：task_5c37a31e 中 steps_replayed=1、step_fallback_to_decider=5、rejected_reasons={"ui_meta_not_matched":5}
  - 例：task_afc88f70 中 steps_replayed=0
2. 每条 action 级别
  - 被复用的 action 会有：
    - replayed: true
    - replay_source: "template_graph"
    - graph_replay.edge_id/from_node_id/to_node_id/...
LOG中：
[GRAPH-REPLAY] accepted step=1 action=click_input edge=e16 from=n1
