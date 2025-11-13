from collections import defaultdict
from unittest import result
from pydantic import BaseModel
from typing import Any, Optional, Union
from openai import OpenAI
import json, logging, os

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())

BINDING_PROMPT = """
## 角色定义

你是一个经验重用系统智能体，负责将一个手机使用任务的各个子任务和相应动作序列进行绑定。

## 输入格式

输入包含一个任务拆分的子任务序列，以及这个任务实际执行的动作序列。

一个子任务是一个高层次的自然语言描述；一个动作是实际可在手机上执行的低层次描述，为一个JSON对象，包含以下字段：

- `reasoning`: 产生该动作的推理过程描述。
- `action`: 动作类别，可选项：`click`, `input`, `scroll`, `wait`, `done`。
- `parameters`: 动作所需的参数，例如点击位置、输入内容等。

一个子任务可能对应动作序列中的一到多步操作，你需要为每个子任务找到对应的动作序列范围。

## 输出格式

请**按照接下来给定的子任务顺序**输出一个JSON列表，列表中的每一项代表一个绑定关系，包含以下字段：

- `subtask`：重复一遍子任务的自然语言描述，确保你正确理解了子任务内容，并能够选择正确的索引。
- `subtask_index`: 子任务在子任务序列中的索引，索引从1开始，每个子任务的索引已经在输入中给出。
- `start_action`：重复一遍起始动作，确保你能够正确选择起始动作的索引。
- `start_action_index`: 起始动作在动作序列中的索引，索引从1开始。
- `end_action`：重复一遍结束动作，确保你能够正确选择结束动作的索引。
- `end_action_index`: 结束动作在动作序列中的索引，索引从1开始。

## 子任务序列

{subtasks}

## 实际动作序列

{actions}

## 要求

你的输出需要严格遵守以下要求：

1. 确保上一个子任务的结束动作索引+1等于下一个子任务的起始动作索引
2. 确保索引从1开始，从最后一个子任务/动作的索引结束
3. 确保按照输入顺序输出所有子任务
4. 确保每个子任务至少对应一个动作
"""


"""
## 示例

假设有以下子任务序列：

1. 进入旅游应用的火车票分区
2. 输入目的地为“北京”
3. 执行搜索

对应的动作序列为：

1. {{"reasoning": "我需要点击屏幕左上角的火车票按钮，进入火车票分区", "action": "click", "parameters": {{"target_element": "火车票按钮"}}}}
2. {{"reasoning": "我需要点击目的地输入框，以便开始输入我的目的地", "action": "click", "parameters": {{"target_element": "目的地输入框"}}}}
3. {{"reasoning": "我需要输入目的地“北京”", "action": "input", "parameters": {{ "text": "北京"}}}}
4. {{"reasoning": "我需要点击搜索按钮，执行搜索操作", "action": "click", "parameters": {{"target_element": "搜索按钮"}}}}

那么正确的输出应该是：

[
  {{
    "subtask": "进入旅游应用的火车票分区",
    "subtask_index": 1,
    "start_action": {{"reasoning": "我需要点击屏幕左上角的火车票按钮，进入火车票分区", "action": "click", "parameters": {{"target_element": "火车票按钮"}}}},
    "start_action_index": 1,
    "end_action": {{"reasoning": "我需要点击屏幕左上角的火车票按钮，进入火车票分区", "action": "click", "parameters": {{"target_element": "火车票按钮"}}}},
    "end_action_index": 1
  }},
  {{
    "subtask": "输入目的地为“北京”",
    "subtask_index": 2,
    "start_action": {{"reasoning": "我需要点击目的地输入框，以便开始输入我的目的地", "action": "click", "parameters": {{"target_element": "目的地输入框"}}}},
    "start_action_index": 2,
    "end_action": {{"reasoning": "我需要输入目的地“北京”", "action": "input", "parameters": {{ "text": "北京"}}}},
    "end_action_index": 3
  }},
  {{
    "subtask": "执行搜索",
    "subtask_index": 3,
    "start_action": {{"reasoning": "我需要点击搜索按钮，执行搜索操作", "action": "click", "parameters": {{"target_element": "搜索按钮"}}}},
    "start_action_index": 4,
    "end_action": {{"reasoning": "我需要点击搜索按钮，执行搜索操作", "action": "click", "parameters": {{"target_element": "搜索按钮"}}}},
    "end_action_index": 4
  }}
]
"""

class MidLevelSequence(BaseModel):
    template_hash: int
    sequence: list[str]

    @classmethod
    def from_experience(cls, final_desc: str, template: str) -> "MidLevelSequence":
        template_hash = hash(template)
        lines = final_desc.strip().split("\n")
        idx = 1
        sequence = []
        for line in lines:
            if not line.startswith(f"{idx}."):
                continue
            subtask = line[len(f"{idx}."):].strip()
            # replace chinese quotes with english quotes
            subtask = subtask.replace("“", "\"").replace("”", "\"")
            sequence.append(subtask)
            idx += 1
        return cls(template_hash=template_hash, sequence=sequence)
    
    def __str__(self):
        # skip first open_app
        # filtered_sequence = self.sequence[1:] if len(self.sequence) > 1 else self.sequence
        return "\n".join([f"{idx}. {subtask}" for idx, subtask in enumerate(self.sequence, 1)])

    def __len__(self):
        return len(self.sequence)
    
    def __getitem__(self, index):
        return self.sequence[index]

class MobiAgentAction(BaseModel):
    reasoning: str
    action: str
    parameters: dict[str, Any]
    extra_info: Optional[dict[str, Any]] = None

class LowLevelSequence(BaseModel):
    template_hash: int
    sequence: list[MobiAgentAction]

    @classmethod
    def from_history(cls, template_hash: int, history: list[str], extra_info: list[Optional[dict[str, Any]]]) -> "LowLevelSequence":
        sequence = []
        for h, extra in zip(history, extra_info):
            h = json.loads(h)
            action = MobiAgentAction(
                reasoning=h.get("reasoning"),
                action=h.get("action"),
                parameters=h.get("parameters"),
                extra_info=extra,
            )
            sequence.append(action)
        return cls(template_hash=template_hash, sequence=sequence)
    
    def __str__(self):
        filtered_sequence = self.sequence
        return "\n".join([f"{idx}. {action.model_dump_json(exclude={'extra_info'})}" for idx, action in enumerate(filtered_sequence, 1)])
    
    def __len__(self):
        return len(self.sequence)
    
    def __getitem__(self, index):
        return self.sequence[index]

class Binding(BaseModel):
    template_hash: int
    # right exclusive
    ranges: list[tuple[int, int]]

class ReplayInfo(BaseModel):
    fisrt_group_replayable: bool
    replay_groups: list[list[MobiAgentAction]]
    periods: list[tuple[int, int]] = []


class Subtask(BaseModel):
    description: str
    actions: Optional[list[MobiAgentAction]] = None

def append_subtask(subtasks: list[Subtask], new_subtask: Subtask) -> list[Subtask]:
    subtasks = subtasks + [new_subtask]
    # merge two consecutive non-replayable subtasks
    # if len(subtasks) >= 2 and subtasks[-2].actions is None and subtasks[-1].actions is None:
    #     merged_subtask = Subtask(
    #         description=subtasks[-2].description.rstrip("。") + "，然后" + subtasks[-1].description,
    #         actions=None,
    #     )
    #     return subtasks[:-2] + [merged_subtask]
    return subtasks

def convert_subtasks(subtasks: list[Subtask]) -> list[Union[str, list[MobiAgentAction]]]:
    ret: list[Union[str, list[MobiAgentAction]]] = []
    for subtask in subtasks:
        if subtask.actions is None:
            ret.append(subtask.description.rstrip("。") + "，然后结束任务，不要进行其他操作。")
        else:
            ret.extend(subtask.actions)
    return ret

class ExperienceRR:
    def __init__(self, planner_client: OpenAI, planner_model: str) -> None:
        if not planner_client:
            raise ValueError("planner_client is required")
        
        self.planner_client = planner_client
        self.planner_model = planner_model

        self.mid_level_table: dict[int, list[MidLevelSequence]] = defaultdict(list)
        self.low_level_table: dict[int, list[LowLevelSequence]] = defaultdict(list)
        self.bindings: dict[int, list[Binding]] = defaultdict(list)

        self.subtask_table: dict[int, dict[str, Subtask]] = defaultdict(dict)
        self.global_subtask_table: dict[str, Subtask] = {}

    def _update_subtasks(self, template_hash: int, idx: int) -> None:
        low_level_seq = self.low_level_table.get(template_hash)[idx]
        mid_level_seq = self.mid_level_table.get(template_hash)[idx]
        binding = self.bindings.get(template_hash)[idx]

        for i, subtask_desc in enumerate(mid_level_seq.sequence):
            range_start, range_end = binding.ranges[i]
            actions = low_level_seq.sequence[range_start:range_end]
            subtask = Subtask(description=subtask_desc, actions=actions)

            self.global_subtask_table[subtask_desc] = subtask
            self.subtask_table[template_hash][subtask_desc] = subtask

    def _bind(self, template_hash: int, idx: int) -> None:
        assert idx < len(self.bindings[template_hash]), f"Index out of range in binding table for {template_hash}"
        low_level_seq = self.low_level_table.get(template_hash)[idx]
        mid_level_seq = self.mid_level_table.get(template_hash)[idx]
        actions_str = str(low_level_seq)
        subtasks_str = str(mid_level_seq)
        prompt = BINDING_PROMPT.format(subtasks=subtasks_str, actions=actions_str)
        # logger.info(f"Binding prompt: {prompt}")
        response = self.planner_client.chat.completions.create(
            model=self.planner_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        ).choices[0].message.content
        if response.startswith("```json"):
            response = response[len("```json"):].strip()
        if response.endswith("```"):
            response = response[:-len("```")].strip()

        logger.info(f"Binding response: {response}")
        
        bindings_json = json.loads(response)
        # skip first open_app subtask
        # ranges = [(0, 0)]
        # cur_subtask_index = 1
        ranges = []
        cur_subtask_index = 0
        for item in bindings_json:
            # response_subtask_index = item["subtask_index"] - 1 + 1 # 0-based index, skip first
            response_subtask_index = item["subtask_index"] - 1
            if response_subtask_index != cur_subtask_index:
                raise ValueError(f"Subtask index mismatch: expected {cur_subtask_index}, got {response_subtask_index}")
            start_action_index = item["start_action_index"] - 1
            end_action_index = item["end_action_index"] - 1 + 1 # 0-based index, change to exclusive
            if start_action_index < 0 or end_action_index > len(low_level_seq):
                raise ValueError(f"Action index out of range: start {start_action_index}, end {end_action_index}, total {len(low_level_seq)}")
            if start_action_index >= end_action_index:
                raise ValueError(f"Invalid action index range: start {start_action_index}, end {end_action_index}")
            ranges.append((start_action_index, end_action_index))
            cur_subtask_index += 1
        
        # further validate ranges
        # the ranges should cover all actions continuously
        if ranges[0][0] != 0:
            raise ValueError(f"First subtask should start from action index 0, got {ranges[0][0]}")
        if ranges[-1][1] != len(low_level_seq):
            raise ValueError(f"Last subtask should end at action index {len(low_level_seq)}, got {ranges[-1][1]}")
        for i in range(len(ranges) - 1):
            if ranges[i][1] != ranges[i + 1][0]:
                raise ValueError(f"Action index ranges are not continuous between subtask {i} and {i+1}: end {ranges[i][1]}, start {ranges[i+1][0]}")
            
        binding = Binding(template_hash=template_hash, ranges=ranges)
        self.bindings[template_hash][idx] = binding
        self._update_subtasks(template_hash, idx)
        
    def record(self, final_desc: str, template: str, history: list[str], extra_info: list[Optional[dict[str, Any]]]) -> None:
        mid_level_seq = MidLevelSequence.from_experience(final_desc, template)
        # if any(existing == mid_level_seq for existing in self.mid_level_table[mid_level_seq.template_hash]):
        #     return
        self.mid_level_table[mid_level_seq.template_hash].append(mid_level_seq)
        low_level_seq = LowLevelSequence.from_history(mid_level_seq.template_hash, history, extra_info)
        self.low_level_table[low_level_seq.template_hash].append(low_level_seq)
        self.bindings[mid_level_seq.template_hash].append(None) # placeholder

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                self._bind(mid_level_seq.template_hash, len(self.mid_level_table[mid_level_seq.template_hash]) - 1)
                break
            except Exception as e:
                logger.error(f"Error processing bindings: {e.__class__.__name__}: {e}")
            if attempt == max_attempts - 1:
                logger.error(f"Failed to process bindings after {max_attempts} attempts. Recording skipped.")
                self.mid_level_table[mid_level_seq.template_hash].pop()
                self.low_level_table[low_level_seq.template_hash].pop()
                self.bindings[mid_level_seq.template_hash].pop()

    def _query(self, mid_level_seq: MidLevelSequence) -> list[Subtask]:
        result: tuple[MidLevelSequence, LowLevelSequence, Binding] = None
        max_match_len = 0
        existing_mid_level_seqs = self.mid_level_table.get(mid_level_seq.template_hash, [])
        for i, existing_mid_level_seq in enumerate(existing_mid_level_seqs):
            if len(existing_mid_level_seq) != len(mid_level_seq):
                continue
            match_len = 0
            for subtask1, subtask2 in zip(existing_mid_level_seq.sequence, mid_level_seq.sequence):
                if subtask1 == subtask2:
                    match_len += 1
            if match_len > max_match_len:
                max_match_len = match_len
                result = (
                    existing_mid_level_seq,
                    self.low_level_table[mid_level_seq.template_hash][i],
                    self.bindings[mid_level_seq.template_hash][i],
                )
        
        if result is None:
            return []
        
        existing_mid_level_seq, low_level_seq, binding = result

        ret_subtasks: list[Subtask] = []
        for i, (subtask_desc1, subtask_desc2) in enumerate(zip(existing_mid_level_seq.sequence, mid_level_seq.sequence)):
            range_start, range_end = binding.ranges[i]
            if subtask_desc1 == subtask_desc2:
                subtask = Subtask(description=subtask_desc1, actions=low_level_seq[range_start:range_end])
            else:
                subtask = Subtask(description=subtask_desc2, actions=None)
            ret_subtasks = append_subtask(ret_subtasks, subtask)
        return ret_subtasks

    def _query_cross_task(self, mid_level_seq: MidLevelSequence, enable_cross_template: bool = False) -> list[Subtask]:
        ret_subtasks: list[Subtask] = []
        available_subtasks = self.global_subtask_table if enable_cross_template else self.subtask_table.get(mid_level_seq.template_hash, {})
        for subtask_desc in mid_level_seq.sequence:
            if subtask_desc in available_subtasks:
                ret_subtasks = append_subtask(ret_subtasks, available_subtasks[subtask_desc])
            else:
                ret_subtasks = append_subtask(ret_subtasks, Subtask(description=subtask_desc, actions=None))
        if all(subtask.actions is None for subtask in ret_subtasks):
            return []
        return ret_subtasks

    def query(self, final_desc: str, template: str, enable_cross_task: bool = False, enable_cross_template: bool = False) -> list[Union[str, list[MobiAgentAction]]]:
        mid_level_seq = MidLevelSequence.from_experience(final_desc, template)
        subtasks: list[Subtask] = []
        if enable_cross_task:
            subtasks = self._query_cross_task(mid_level_seq, enable_cross_template)
        else:
            subtasks = self._query(mid_level_seq)
        return convert_subtasks(subtasks)

    # def query(self, final_desc: str, template: str) -> Optional[ReplayInfo]:
    #     mid_level_seq = MidLevelSequence.from_experience(final_desc, template)
    #     result: tuple[MidLevelSequence, LowLevelSequence, Binding] = None
    #     max_match_len = 0
    #     for i, existing_mid_level_seq in enumerate(self.mid_level_table.get(mid_level_seq.template_hash, [])):
    #         if len(existing_mid_level_seq) != len(mid_level_seq):
    #             continue
    #         match_len = 0
    #         for subtask1, subtask2 in zip(existing_mid_level_seq.sequence, mid_level_seq.sequence):
    #             if subtask1 == subtask2:
    #                 match_len += 1
    #         if match_len > max_match_len:
    #             max_match_len = match_len
    #             result = (
    #                 existing_mid_level_seq,
    #                 self.low_level_table[mid_level_seq.template_hash][i],
    #                 self.bindings[mid_level_seq.template_hash][i],
    #             )
        
    #     if result is None:
    #         return None
    #     existing_mid_level_seq, low_level_seq, binding = result
    #     replay_groups: list[list[MobiAgentAction]] = []
    #     periods: list[tuple[int, int]] = []
    #     last_subtask_replayable = False
    #     for i, (subtask1, subtask2) in enumerate(zip(existing_mid_level_seq.sequence, mid_level_seq.sequence)):
    #         range_start, range_end = binding.ranges[i]
    #         if subtask1 == subtask2:
    #             if i == 0:
    #                 first_group_replayable = True
    #             if last_subtask_replayable:
    #                 # merge with previous
    #                 replay_groups[-1].extend(low_level_seq.sequence[range_start:range_end])
    #             else:
    #                 group = low_level_seq[range_start:range_end]
    #                 replay_groups.append(group)
    #             last_subtask_replayable = True
    #         else:
    #             if i == 0:
    #                 first_group_replayable = False
    #             orig_steps = range_end - range_start
    #             min_steps = max(1, int(orig_steps * 0.75))
    #             max_steps = int(orig_steps * 1.5)
    #             periods.append((min_steps, max_steps))
    #     return ReplayInfo(fisrt_group_replayable=first_group_replayable, replay_groups=replay_groups, periods=periods)
        
if __name__ == "__main__":
    client = OpenAI(
        api_key = os.environ.get("OPENAI_API_KEY", "0"),
        base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8080/v1"),
    )
    experience_rr = ExperienceRR(planner_client=client)
    # test record
    final_desc = "请你使用携程App，帮我完成\"查询上海的汉庭酒店价格\"。主要操作流程为：\n1. 在首页点击“酒店”功能入口。\n2. 选择城市为\"上海\"。\n3. 输入并选定酒店名称为\"汉庭酒店\"。\n4. 执行酒店查询。\n5. 看到酒店搜索结果列表出现后，任务结束。"
    template = "请你使用{{APP名称}}App，帮我完成\"{{用户原始任务描述}}\"。主要操作流程为：\n1. 在首页点击“酒店”功能入口。\n2. 选择城市为\"{{城市名}}\"。\n3. 输入并选定酒店名称为\"{{酒店名}}\"。\n4. 执行酒店查询。\n5. 看到酒店搜索结果列表出现后，任务结束。"
    history = [
        '{"reasoning": "任务是查询全季酒店的价格。首先，我需要进入酒店预订功能。因此，我将点击主页上的“酒店”图标。", "action": "click", "parameters": {"target_element": "酒店图标"}}',
        '{"reasoning": "进入酒店预订页面后，我需要指定城市。当前城市显示的是“北京”，我需要点击它来修改目标城市。", "action": "click", "parameters": {"target_element": "城市选择框，当前显示北京"}}',
        '{"reasoning": "我需要点击热门城市中的上海来选择城市", "action": "click", "parameters": {"target_element": "热门城市中的上海"}}',
        '{"reasoning": "我已经完成了城市选择，接下来我需要指定要查询的酒店，为此，我需要点击位置/品牌/酒店输入框", "action": "click", "parameters": {"target_element": "位置/品牌/酒店输入框"}}',
        '{"reasoning": "我需要在搜索框中输入“全季酒店”以查找相关酒店。", "action": "input", "parameters": {"text": "汉庭酒店"}}',
        '{"reasoning": "输入“全季酒店”后，我需要点击键盘上的搜索按钮来执行搜索。", "action": "click", "parameters": {"target_element": "键盘上的搜索按钮"}}',
        '{"reasoning": "我已经完成了城市和酒店的选择，接下来我需要点击查询按钮来执行这次查询", "action": "click", "parameters": {"target_element": "查询按钮"}}',
        '{"reasoning": "屏幕上出现了酒店搜索结果，任务完成", "action": "done", "parameters": {"status": "success"}}',
    ]
    experience_rr.record(final_desc, template, history)
    # test query
    final_desc2 = "请你使用携程App，帮我完成\"查询北京的汉庭酒店价格\"。主要操作流程为：\n1. 在首页点击“酒店”功能入口。\n2. 选择城市为\"北京\"。\n3. 输入并选定酒店名称为\"汉庭酒店\"。\n4. 执行酒店查询。\n5. 看到酒店搜索结果列表出现后，任务结束。"
    queried_actions = experience_rr.query(final_desc2, template)
    for action in queried_actions:
        print(action)
    final_desc3 = "请你使用携程App，帮我完成\"查询上海的全季酒店价格\"。主要操作流程为：\n1. 在首页点击“酒店”功能入口。\n2. 选择城市为\"上海\"。\n3. 输入并选定酒店名称为\"全季酒店\"。\n4. 执行酒店查询。\n5. 看到酒店搜索结果列表出现后，任务结束。"
    queried_actions = experience_rr.query(final_desc3, template)
    for action in queried_actions:
        print(action)