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



### 4.6
#### 问题1
经常小箭头返回导致死循环
#### 修改内容
修改prompt，告诉模型尽量不点击小箭头

#### 问题2
更改手机和模型后框识别有无
#### 修改内容
bat中新增4个可调参数
`BBOX_IOU_THRESHOLD`	IoU 门槛降低，更容易匹配到 XML 元素
`BBOX_CENTER_DIST_RATIO`	中心距容忍范围扩大近 2 倍
`BBOX_AREA_RATIO_MIN`	允许匹配更小的元素
`BBOX_AREA_RATIO_MAX`	允许匹配更大的元素

#### 问题3
对于广告弹窗的处理
#### 修改内容
- `build_explorer_prompt()` 新增第 11 条：要求模型检测广告/弹窗并返回 `popup: {detected, close_point}` 字段，坐标格式 0-1000
- `call_explorer_model()` 新增解析 `popup_info` 并作为第二返回值
- `explore_dfs()` 新增弹窗自动关闭循环：有 `close_point` 则点击，否则按返回键；关闭后重新调用 Explorer 验证，最多重试 `popup_dismiss_max_attempts` 次（默认 2，可通过 bat 参数配置）
1、点击关闭
2、等待稳定
3、重新explorer检查是否还有广告
4、若有再重试

#### 新问题 广告类型多种多样，可能有的必须要等待一定时间在解决


#### 问题4
对于不同页面的判断，比如美团的待付款 待收款 多个选项界面非常相似，且会陷入循环 回溯后同一个界面重新进入后选项变了
#### 修改内容
回溯验证三重指纹中的文本指纹（`fp_ok`）改用稳定文本指纹：
- 新增 `_stable_text_fingerprint()`：只提取 resource-id 或 class 包含 `title/tab/nav/toolbar/header/bottom/action_bar` 等关键词的固定 UI 元素文字计算 SHA1，忽略推荐内容、商家名称等动态文字
- 提取不到稳定元素时自动退化为原全文本指纹 `_hierarchy_fingerprint()`
- 替换回溯验证（第 2232 行）和全路径重播验证（第 2270 行）中的 `fp_ok` / `r_fp_ok` 计算
- 效果：动态 feed 页面（美团首页等）回溯后内容刷新不再导致 `fp_ok = False` 误判；待付款/待收货等 Tab 因标题文字不同仍能正确区分


#### 问题5
一些加载页面，模型立刻判断检测，导致问题
#### 修改内容
动作执行后新增两段式等待逻辑（`_wait_for_page_loaded`）：
1. 先固定等待 `PAGE_LOAD_WAIT_SEC`（默认 1.5s），让页面开始渲染
2. 再循环最多 `PAGE_LOAD_STABLE_MAX_POLLS` 次（默认 6 次，每次间隔 0.5s），将截图发给 VLM 判断页面是否仍处于加载状态（spinner、骨架屏、空白内容区、"加载中"文字等）；VLM 回答 NO 或调用失败时立即退出循环，不阻塞流程
两个参数均可在 bat 文件中配置，适配不同 App 和手机性能

