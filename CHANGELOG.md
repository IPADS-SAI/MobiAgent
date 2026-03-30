# CHANGELOG

## 2026-03-30

### OPT-2：增强回溯验证 + 全路径重播恢复

**文件**：`MobiAgent/runner/mobiagent/auto-search.py`

#### 问题
原有回溯逻辑存在两个缺陷：
1. 只用单一文本指纹（SHA1）判断是否回到正确页面，结构相同但内容不同的页面会碰撞误判
2. 验证失败后只重播 `current_path_actions[-2]` 一步，不验证前置状态也不验证执行结果

#### 修改内容

**修改 1**（第 1753-1760 行）：候选执行前新增基准状态采集

- 新增 `get_screenshot()` 调用，更新临时截图文件
- 新增 `pre_struct_fp`：结构指纹（仅 clickable 元素）
- 新增 `pre_dhash`：视觉 dHash（来自 `_compute_dhash_hex`）
- 用于回溯后的三重比对基准

**修改 2**（第 1888-1952 行）：替换回溯验证+恢复逻辑

旧逻辑：
- 单一文本指纹比对
- 失败则重播 `[-2]` 一步（无前置验证，无结果验证）

新逻辑：
- **三重验证**：文本指纹 + 结构指纹 + 视觉 dHash，2/3 通过视为回溯成功
- **全路径重播恢复**：验证失败时，`start_app()` 回根页面，循环重播 `current_path_actions[:-1]`，最多重试 2 次
- 每次重播后再次三重验证
- 全部失败则 `break` 跳出候选循环，防止后续兄弟候选在错误页面执行污染数据

#### 复用的已有函数（无新增函数）

| 函数 | 原位置 |
|------|--------|
| `_hierarchy_fingerprint()` | 第 575 行 |
| `_compute_hierarchy_struct_fingerprint()` | 第 705 行 |
| `_compute_dhash_hex()` | 第 646 行 |
| `get_screenshot()` | mobiagent.py 第 485 行 |
| `replay_action_record()` | 第 499 行 |

---

### OPT-3：自适应页面变化相似度阈值

**文件**：`MobiAgent/runner/mobiagent/auto-search.py`

#### 问题
候选广度循环中判断页面是否跳转的阈值硬编码为 `0.9`，无法适应不同复杂度的页面：
- **feed 流页面**（外卖首页、推荐列表等，50+ 元素）：内容自动刷新、时间戳变化导致相似度正常波动到 0.85 左右，频繁误触发候选重生成，浪费 Explorer API 调用
- **简单弹窗**（2-5 个按钮）：任何真实跳转相似度变化都很大，0.9 阈值反而合适

#### 修改内容

**新增函数** `_compute_adaptive_similarity_threshold(hierarchy_text) -> float`（第 574 行附近）：
- 统计页面 clickable 元素数量，复用已有的 `_collect_struct_tokens_from_xml/json`
- 元素数 ≤5 时阈值 0.93，元素数 ≥30 时阈值 0.70，中间线性插值：
  `threshold = max(0.70, 0.95 - 0.01 * min(element_count, 25))`

**替换第 1710 行**：
- 旧：`if similarity < 0.9`
- 新：`page_change_threshold = _compute_adaptive_similarity_threshold(base_hierarchy_text)` + `if similarity < page_change_threshold`

**更新日志格式**：日志中同步打印实际使用的阈值，便于调试观察。

#### 复用的已有函数（无新增函数）

| 函数 | 原位置 |
|------|--------|
| `_collect_struct_tokens_from_xml()` | 第 662 行 |
| `_collect_struct_tokens_from_json()` | 第 680 行 |

---

### OPT-5：候选动作执行前语义去重

**文件**：`MobiAgent/runner/mobiagent/auto-search.py`

#### 问题
Explorer 模型经常返回语义近似的候选（如"点击设置"与"进入设置界面"），浪费 Decider 调用和 DFS 分支预算。

#### 修改内容

**新增函数** `_deduplicate_candidates(candidates, already_explored, sim_threshold=0.75)`：
- 过滤与 `already_explored`（当前页已探索任务）相似度 >0.8 的候选
- 过滤候选列表内部相似度 >0.75 的重复项，保留 rank 更高者
- 使用 `difflib.SequenceMatcher`，无需外部依赖

**调用位置**：
- Explorer 返回后立即调用
- 动态重生成候选后也调用

---

### OPT-6：已探索路径注入 Explorer Prompt + 路径历史替代全局历史

**文件**：`MobiAgent/runner/mobiagent/auto-search.py`

#### 问题
1. Explorer 收到全局 `actions`（混入所有已回溯分支），无法判断当前设备处于哪条路径，生成候选语义脱节
2. Explorer 不知道当前页面哪些操作已被探索，重生成时重复已执行的动作

#### 修改内容

**`build_explorer_prompt`** 新增 `already_explored: Optional[List[str]]` 参数：
- 若非空，在 prompt 末尾追加"已在当前页面完成探索的操作（请勿重复生成）"段落

**`call_explorer_model`** 新增 `already_explored` 参数并透传给 `build_explorer_prompt`

**`explore_dfs`** 新增 `visited_tasks: Optional[Dict[str, set]]` 参数：
- 每次候选执行后：`visited_tasks[current_page_fp].add(task)`
- DFS 入口：`action_history` 改为 `path_actions`（当前路径）而非全局 `actions`
- Explorer 调用时传入 `already_explored_list = list(visited_tasks.get(fp, set()))`
- 递归调用时透传 `visited_tasks`

**`main()`** 初始化 `visited_tasks: Dict[str, set] = {}`

---

### OPT-9：Explorer 响应缓存

**文件**：`MobiAgent/runner/mobiagent/auto-search.py`

#### 问题
回溯返回同一页面后，下一个兄弟候选重新调用 Explorer API，但页面完全相同，浪费 API 费用和延迟。

#### 修改内容

**新增类** `ExplorerCache`：
- Cache key：`struct_fp | depth | breadth | hash(already_explored)`
- TTL：300 秒（防止动态内容污染）
- `get()` 返回缓存候选并过滤已执行任务；`put()` 存入新结果

**`explore_dfs`** 新增 `explorer_cache: Optional[ExplorerCache]` 参数：
- Explorer 调用前先查缓存，命中则跳过 API 调用
- 未命中时调用后写入缓存

**`main()`** 初始化 `explorer_cache = ExplorerCache(ttl_sec=300.0)`

---

### OPT-10：设备状态缓存

**文件**：`MobiAgent/runner/mobiagent/auto-search.py`

#### 问题
`get_screenshot()` 和 `get_hierarchy_text()` 在 DFS 多处被重复调用，`dump_hierarchy()` 实际耗时 0.5-2 秒。

#### 修改内容

**新增类** `ScreenStateCache`：
- `capture(device, device_type, force)` 在 staleness 阈值（0.3 秒）内返回缓存值，否则重新采集
- `invalidate()` 动作执行后立即失效

**`explore_dfs`** 新增 `screen_cache: Optional[ScreenStateCache]` 参数：
- DFS 入口的截图+层级采集改为 `screen_cache.capture(force=True)`
- 动作执行后调用 `screen_cache.invalidate()`
- 递归调用时透传

**`main()`** 初始化 `screen_cache = ScreenStateCache(staleness_sec=0.3)`
