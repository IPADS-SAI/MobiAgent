from collections import defaultdict
import itertools
from pathlib import Path
import random
from unittest import result
from pydantic import BaseModel
from typing import Any, Optional, Sequence, Union
from openai import OpenAI
import json, logging, os
import hashlib
import cv2
import numpy as np
from PIL import Image
from utils.load_md_prompt import load_prompt
from utils.local_experience import PromptTemplateSearch
import pandas as pd
from skimage.metrics import structural_similarity

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

GENERATE_TEXT_PROMPT = """根据以下描述生成一个随机文本，只返回生成的文本内容，不要包含任何解释、说明或其他额外信息：

{description}"""

GENERATE_TEXT_WITH_HISTORY_PROMPT = """根据以下描述生成一个随机文本，只返回生成的文本内容，不要包含任何解释、说明或其他额外信息：

描述：{description}

重要要求：生成的值**必须**与以下值**不重复**：
{history_text}

请确保生成一个全新的、与上述值都不相同的文本。"""


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
    template_hash: str
    sequence: list[str]

    @classmethod
    def from_experience(cls, final_desc: str, template: str) -> "MidLevelSequence":
        template_hash = hashlib.md5(template.encode('utf-8')).hexdigest()
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
    template_hash: str
    sequence: list[MobiAgentAction]

    @classmethod
    def from_history(cls, template_hash: str, history: list[str], extra_info: list[Optional[dict[str, Any]]]) -> "LowLevelSequence":
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
    template_hash: str
    # right exclusive
    ranges: list[tuple[int, int]]

class ReplayInfo(BaseModel):
    fisrt_group_replayable: bool
    replay_groups: list[list[MobiAgentAction]]
    periods: list[tuple[int, int]] = []


class Subtask(BaseModel):
    description: str
    actions: Optional[list[MobiAgentAction]] = None


class BBoxChangeResult(BaseModel):
    """
    表示一次UI元素变化检测的结果。
    """

    changed: bool
    change_ratio: float
    mean_intensity_delta: float
    ssim_score: Optional[float] = None


def _load_image(image: Union[str, Path, "np.ndarray", Image.Image]) -> "np.ndarray":
    """
    支持从路径或已有的numpy数组加载OpenCV图像。
    """

    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
        return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    image_path = Path(image)
    if not image_path.exists():
        raise FileNotFoundError(f"Screenshot not found: {image_path}")
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return img


def _sanitize_bbox(bbox: Sequence[int], width: int, height: int) -> tuple[int, int, int, int]:
    if len(bbox) != 4:
        raise ValueError(f"bbox must contain four values, got {bbox}")
    x1, y1, x2, y2 = map(int, bbox)
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    width = max(1, int(width))
    height = max(1, int(height))
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def detect_bbox_change(
    prev_screenshot: Union[str, Path, "np.ndarray", Image.Image],
    curr_screenshot: Union[str, Path, "np.ndarray", Image.Image],
    bbox: Sequence[int],
    *,
    ssim_threshold: float = 0.9,
    diff_activation: float = 0.15,
    blur_kernel_size: int = 3,
) -> BBoxChangeResult:
    """
    使用 SSIM (结构相似度) 判断 bbox 区域是否发生显著变化。

    Args:
        ssim_threshold: SSIM 分数低于该值则判定变化
        diff_activation: SSIM 差分图中视为变化的阈值，范围 0-1
    """

    prev_img = _load_image(prev_screenshot)
    curr_img = _load_image(curr_screenshot)

    if prev_img.shape[:2] != curr_img.shape[:2]:
        raise ValueError("Screenshots must share the same resolution for comparison")

    x1, y1, x2, y2 = _sanitize_bbox(bbox, prev_img.shape[1], prev_img.shape[0])
    prev_roi = prev_img[y1:y2, x1:x2]
    curr_roi = curr_img[y1:y2, x1:x2]

    if blur_kernel_size and blur_kernel_size > 1:
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        prev_roi = cv2.GaussianBlur(prev_roi, (blur_kernel_size, blur_kernel_size), 0)
        curr_roi = cv2.GaussianBlur(curr_roi, (blur_kernel_size, blur_kernel_size), 0)

    prev_gray = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY)

    ssim_score, diff_map = structural_similarity(prev_gray, curr_gray, full=True)
    diff_map = 1.0 - diff_map  # 差异越大数值越大
    change_mask = diff_map >= diff_activation
    change_ratio = float(np.mean(change_mask))
    mean_intensity_delta = float(diff_map.mean())
    changed = ssim_score <= ssim_threshold

    return BBoxChangeResult(
        changed=changed,
        change_ratio=change_ratio,
        mean_intensity_delta=mean_intensity_delta,
        ssim_score=ssim_score,
    )

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
    if all(subtask.actions is None for subtask in subtasks):
        return []
    ret: list[Union[str, list[MobiAgentAction]]] = []
    for subtask in subtasks:
        if subtask.actions is None:
            ret.append(subtask.description)
        else:
            ret.extend(subtask.actions)
    return ret

class ExperienceRR:
    def __init__(self, planner_client: OpenAI, planner_model: str) -> None:
        if not planner_client:
            raise ValueError("planner_client is required")
        
        self.planner_client = planner_client
        self.planner_model = planner_model

        self.mid_level_table: dict[str, list[MidLevelSequence]] = defaultdict(list)
        self.low_level_table: dict[str, list[LowLevelSequence]] = defaultdict(list)
        self.bindings: dict[str, list[Binding]] = defaultdict(list)

        self.subtask_table: dict[str, dict[str, Subtask]] = defaultdict(dict)
        self.global_subtask_table: dict[str, Subtask] = {}

        self.query_result_cache: dict[str, list[Subtask]] = {}

    def _update_subtasks(self, template_hash: str, idx: int) -> None:
        low_level_seq = self.low_level_table.get(template_hash)[idx]
        mid_level_seq = self.mid_level_table.get(template_hash)[idx]
        binding = self.bindings.get(template_hash)[idx]

        for i, subtask_desc in enumerate(mid_level_seq.sequence):
            range_start, range_end = binding.ranges[i]
            actions = low_level_seq.sequence[range_start:range_end]
            subtask = Subtask(description=subtask_desc, actions=actions)

            if subtask_desc not in self.subtask_table[template_hash]:
                self.subtask_table[template_hash][subtask_desc] = subtask
            if subtask_desc not in self.global_subtask_table:
                self.global_subtask_table[subtask_desc] = subtask

    def _bind(self, template_hash: str, idx: int) -> None:
        assert idx < len(self.bindings[template_hash]), f"Index out of range in binding table for {template_hash}"
        low_level_seq = self.low_level_table.get(template_hash)[idx]
        mid_level_seq = self.mid_level_table.get(template_hash)[idx]
        actions_str = str(low_level_seq)
        subtasks_str = str(mid_level_seq)
        prompt = BINDING_PROMPT.format(subtasks=subtasks_str, actions=actions_str)
        logger.info(f"Binding prompt: {prompt}")
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

        query_key = f"{final_desc}<SEP>{template}"
        cached_query_result = self.query_result_cache.get(query_key, None)
        if cached_query_result:
            logger.info("Using cached query result to update subtasks")
            cur_start_idx = 0
            ranges = []
            for subtask in cached_query_result:
                if subtask.actions is not None:
                    range_start = cur_start_idx
                    range_end = cur_start_idx + len(subtask.actions)
                    ranges.append((range_start, range_end))
                    cur_start_idx = range_end
                else:
                    desc = subtask.description
                    # find range in extra_info
                    filtered_info = [(i, info) for i, info in enumerate(extra_info) if info and info.get("subtask_desc", None) == desc]
                    range_start = filtered_info[0][0]
                    range_end = filtered_info[-1][0] + 1
                    ranges.append((range_start, range_end))
                    cur_start_idx = range_end
            binding = Binding(template_hash=mid_level_seq.template_hash, ranges=ranges)
            self.bindings[mid_level_seq.template_hash][-1] = binding
            self._update_subtasks(mid_level_seq.template_hash, len(self.mid_level_table[mid_level_seq.template_hash]) - 1)
        else:
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
        
        ret_subtasks: list[Subtask] = []
        if result is None:
            for subtask_desc in mid_level_seq.sequence:
                ret_subtasks = append_subtask(ret_subtasks, Subtask(description=subtask_desc, actions=None))
            return ret_subtasks
        
        existing_mid_level_seq, low_level_seq, binding = result

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
        return ret_subtasks

    def query(self, 
        final_desc: str, 
        template: str, 
        enable_cross_task: bool = False, 
        enable_cross_template: bool = False, 
        convert: bool = True
    ) -> list[MobiAgentAction] | list[Union[str, list[MobiAgentAction]]]:
        mid_level_seq = MidLevelSequence.from_experience(final_desc, template)
        subtasks: list[Subtask] = []
        if enable_cross_task:
            subtasks = self._query_cross_task(mid_level_seq, enable_cross_template)
        else:
            subtasks = self._query(mid_level_seq)
        # cache query result
        if any(subtask.actions is not None for subtask in subtasks):
            cache_key = f"{final_desc}<SEP>{template}"
            self.query_result_cache[cache_key] = subtasks
        if convert:
            subtasks = convert_subtasks(subtasks)
        return subtasks

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

class OracleAgent:
    def __init__(self) -> None:
        root_dir = Path(__file__).resolve().parent.parent.parent
        file_path = root_dir / "utils" / "experience" / "rr-oracle.json"
        with open(file_path, "r", encoding="utf-8") as f:
            oracle_data = json.load(f)["subtasks"]
            # oracle_data = json.load(f)["subtasks-opt"]
        self.mock_actions_table: dict[str, list[tuple[str]]] = {}
        for item in oracle_data:
            subtask = item["subtask"]
            actions = [tuple(action) for action in item["actions"]]
            self.mock_actions_table[subtask] = actions
        # self.mock_actions_table: dict[str, list[tuple[str]]] = {
        #     "在首页点击\"酒店\"功能入口": [
        #         ("点击首页的酒店按钮，进入酒店功能", "click", "酒店按钮"),
        #     ],
        #     "选择城市为\"{{城市名}}\"": [
        #         ("点击城市选择器准备选择城市", "click", "城市选择器"),
        #         ("输入用户指定的城市", "input", "{{城市名}}"),
        #         ("点击搜索按钮确认选择城市", "click", "搜索")
        #     ],
        #     "输入并选定酒店名称为\"{{酒店名}}酒店\"": [
        #         ("点击酒店名称输入框准备选择酒店", "click", "酒店名称输入框"),
        #         ("输入用户指定的酒店名称", "input", "{{酒店名}}"),
        #         ("点击搜索按钮确认选择酒店", "click", "搜索")
        #     ],
        #     "执行查询操作，看到酒店搜索结果列表出现后，结束任务": [
        #         ("点击查询按钮，执行查询操作", "click", "查询按钮"),
        #     ],
        # }

    def execute_subtask(self, subtask_desc: str, variables: dict[str, str]) -> list[MobiAgentAction]:
        subtask_desc = subtask_desc.replace("“", "\"").replace("”", "\"").rstrip("。")
        matched_mock_actions: list[tuple[str, str]] = []
        for subtask_template, mock_actions in self.mock_actions_table.items():
            for key, value in variables.items():
                subtask_template = subtask_template.replace(f"{{{{{key}}}}}", value)
                if subtask_template == subtask_desc:
                    matched_mock_actions = mock_actions
                    break
            if matched_mock_actions:
                break
        if not matched_mock_actions:
            raise ValueError(f"No mock actions found for subtask: {subtask_desc}")
        
        actions: list[MobiAgentAction] = []
        for reasoning, action_type, param_template in matched_mock_actions:
            param_str = param_template
            for key, value in variables.items():
                param_str = param_str.replace(f"{{{{{key}}}}}", value)
            parameters = {"param": param_str}
            action = MobiAgentAction(
                reasoning=reasoning,
                action=action_type,
                parameters=parameters,
            )
            actions.append(action)
        actions.append(MobiAgentAction(
            reasoning="任务成功完成",
            action="done",
            parameters={"param": ""},
        ))
        return actions

class ExperienceRRTest:
    def __init__(self, planner_client: OpenAI, planner_model: str, template_path: str) -> None:
        self.experience_rr = ExperienceRR(planner_client=planner_client, planner_model=planner_model)
        self.oracle_agent = OracleAgent()
        self.reset_metrics()
        self.search_engine = PromptTemplateSearch(template_path)
        # 存储每个description的历史生成值（最近20条）
        self.generated_value_history: dict[str, list[str]] = defaultdict(list)

    def reset_metrics(self):
        self.num_tasks = 0
        self.num_success = 0
        self.num_replayed_actions = 0
        self.num_fallback_actions = 0
        self.num_overhead_actions = 0

    # (theoretical) total action = replayed + fallback
    @property
    def num_total_actions(self) -> int:
        return self.num_replayed_actions + self.num_fallback_actions
    
    def get_metrics(self) -> dict[str, int]:
        return {
            "num_tasks": self.num_tasks,
            "num_success": self.num_success,
            "num_replayed_actions": self.num_replayed_actions,
            "num_fallback_actions": self.num_fallback_actions,
            "num_overhead_actions": self.num_overhead_actions,
            "num_total_actions": self.num_total_actions,
        }
    
    def retrive_template(self, task_description: str) -> str:
        template = self.search_engine.get_experience(task_description, 1)
        return template
    
    def fill_template(self, template: str, variables: dict[str, str]) -> str:
        filled_template = json.loads(template)["experience1"]
        for key, value in variables.items():
            filled_template = filled_template.replace(f"{{{{{key}}}}}", value)
        return filled_template

    def fill_template_planner(self, template: str, task_description: str) -> tuple[str, dict[str, str]]:
        planner_prompt_template = load_prompt("planner_oneshot.md")
        prompt = planner_prompt_template.format(
            task_description=task_description,
            experience_content=template,
        )
        response_str = self.experience_rr.planner_client.chat.completions.create(
            model = self.experience_rr.planner_model,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
        ).choices[0].message.content
        if response_str.startswith("```json"):
            response_str = response_str.replace("```json", "").replace("```", "").strip()
        response_json = json.loads(response_str)
        final_desc = response_json.get("final_task_description", task_description)
        return final_desc, {}

    def execute_task_acttree(self, task_description: str, variables: Optional[dict[str, str]]) -> None:
        self.num_tasks += 1
        template = self.retrive_template(task_description)
        if variables:
            final_desc = self.fill_template(template, variables)
        else:
            final_desc, variables = self.fill_template_planner(template, task_description)
        
        # hack experience_rr, no query/record

        # get all low-level seq from experience_rr.low_level_table
        mid_level_seq = MidLevelSequence.from_experience(final_desc, template)
        all_low_level_seqs = self.experience_rr.low_level_table.get(mid_level_seq.template_hash, [])
        
        # get ground truth
        gt_actions: list[MobiAgentAction] = []
        for i, subtask_desc in enumerate(mid_level_seq.sequence):
            subtask_gt_actions = self.oracle_agent.execute_subtask(subtask_desc, variables)
            if i != len(mid_level_seq) - 1:
                # exclude done action
                subtask_gt_actions = subtask_gt_actions[:-1]
            gt_actions.extend(subtask_gt_actions)
        
        # compare with all low-level seqs, find longest match
        max_match_len = 0
        for low_level_seq in all_low_level_seqs:
            existing_actions = low_level_seq.sequence
            match_len = 0
            for i, gt_action in enumerate(gt_actions):
                if i >= len(existing_actions):
                    break
                if gt_action.model_dump(exclude={'extra_info'}) == existing_actions[i].model_dump(exclude={'extra_info'}):
                    match_len += 1
                else:
                    break
            if match_len > max_match_len:
                max_match_len = match_len

        total_actions = len(gt_actions)
        self.num_replayed_actions += max_match_len
        self.num_fallback_actions += (total_actions - max_match_len)
        self.num_success += 1

        self.experience_rr.low_level_table[mid_level_seq.template_hash].append(
            LowLevelSequence(
                template_hash=mid_level_seq.template_hash,
                sequence=gt_actions,
            )
        )

    def execute_task(self, task_description: str, variables: Optional[dict[str, str]]) -> None:
        self.num_tasks += 1
        template = self.retrive_template(task_description)
        if variables:
            final_desc = self.fill_template(template, variables)
        else:
            final_desc, variables = self.fill_template_planner(template, task_description)
        subtasks: list[Subtask] = self.experience_rr.query(final_desc, template, enable_cross_task=True, convert=False)
        logger.info(f"Queried subtasks: {subtasks}")
        is_first_execute = all(subtask.actions is None for subtask in subtasks)
        actions: list[MobiAgentAction] = []
        history: list[str] = []
        extra_info: list[Optional[dict[str, Any]]] = []
        for i, subtask in enumerate(subtasks):
            is_replayed = subtask.actions is not None
            subtask_actions = self.execute_subtask(subtask, variables)
            if (i != len(subtasks) - 1) and (not is_replayed):
                if not is_first_execute:
                    # account done action in overheads
                    self.num_overhead_actions += 1
                # exclude done action
                self.num_fallback_actions -= 1
                subtask_actions = subtask_actions[:-1]
            if not is_replayed:
                extra_info.extend([{"subtask_desc": subtask.description}] * len(subtask_actions))
            else:
                extra_info.extend([None] * len(subtask_actions))
            actions.extend(subtask_actions)
            history.extend([action.model_dump_json(exclude={'extra_info'}) for action in subtask_actions])

        if is_first_execute:
            # must be correct
            self.num_success += 1
            self.experience_rr.record(final_desc, template, history, extra_info)
            return
        
        # get ground truth
        gt_actions: list[MobiAgentAction] = []
        for i, subtask in enumerate(subtasks):
            subtask_gt_actions = self.oracle_agent.execute_subtask(subtask.description, variables)
            if i != len(subtasks) - 1:
                # exclude done action
                subtask_gt_actions = subtask_gt_actions[:-1]
            gt_actions.extend(subtask_gt_actions)

        logger.debug("Executed actions: \n" + "\n".join(a.model_dump_json(exclude={"extra_info"}) for a in actions))
        logger.debug("Ground truth actions: \n" + "\n".join(a.model_dump_json(exclude={"extra_info"}) for a in gt_actions))

        # compare actions with gt_actions
        if len(actions) != len(gt_actions):
            logger.warning(f"Action length mismatch: got {len(actions)}, expected {len(gt_actions)}")
        elif all(a.model_dump(exclude={'extra_info'}) == b.model_dump(exclude={'extra_info'}) for a, b in zip(actions, gt_actions)):
            logger.info("Actions match ground truth")
            self.num_success += 1
            self.experience_rr.record(final_desc, template, history, extra_info)
        else:
            logger.warning("Actions do not match ground truth")
        

    def execute_subtask(self, subtask: Subtask, variables: dict[str, str]) -> list[MobiAgentAction]:
        if subtask.actions is not None:
            self.num_replayed_actions += len(subtask.actions)
            return subtask.actions
        else:
            actions = self.oracle_agent.execute_subtask(subtask.description, variables)
            self.num_fallback_actions += len(actions)
            return actions

    def generate_random_value(self, description: str) -> str:
        """根据描述生成随机文本，只返回文本内容，确保不与历史生成的值重复"""
        # 获取该description的历史生成值（最近20条）
        history_values = self.generated_value_history.get(description, [])
        
        # 构建prompt，包含历史值信息
        if history_values:
            history_text = "\n".join([f"- {value}" for value in history_values])
            prompt = GENERATE_TEXT_WITH_HISTORY_PROMPT.format(description=description, history_text=history_text)
        else:
            prompt = GENERATE_TEXT_PROMPT.format(description=description)
        
        response = self.experience_rr.planner_client.chat.completions.create(
            model=self.experience_rr.planner_model,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
            # temperature=0.4,
        ).choices[0].message.content
        # 清理可能的额外格式
        response = response.strip()
        if response.startswith("```"):
            # 移除可能的代码块标记
            lines = response.split("\n")
            if len(lines) > 1:
                response = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else response
        generated_value = response.strip()
        
        # 更新历史记录（只保留最近20条）
        history_list = self.generated_value_history[description]
        history_list.append(generated_value)
        if len(history_list) > 20:
            history_list.pop(0)
        
        logger.info(f"Generated value: {generated_value} for description: {description}")
        return generated_value

    def run_test_suite(self, use_acttree: bool = False) -> pd.DataFrame:
        root_dir = Path(__file__).resolve().parent.parent.parent
        file_path = root_dir / "utils" / "experience" / "rr-oracle.json"
        cache_file_path = root_dir / "utils" / "experience" / "task_dict_cache.json"
        
        # 检查缓存文件是否存在
        if cache_file_path.exists():
            logger.info(f"Loading task_dict from cache: {cache_file_path}")
            with open(cache_file_path, "r", encoding="utf-8") as f:
                task_dict = json.load(f)
            # 将普通字典转换回 defaultdict(list) 格式
            task_dict = defaultdict(list, task_dict)
        else:
            logger.info("Building task_dict from scratch...")
            with open(file_path, "r", encoding="utf-8") as f:
                task_data = json.load(f)["tasks"]

            task_dict: dict[str, list[dict[str, str]]] = defaultdict(list)
            min_tasks_per_template = 20
            max_tasks_per_template = 100
            for item in task_data:
                task_fmt = item["task"]
                if "外卖" not in task_fmt:
                    continue
                fixed_vars = item["variables"]["fixed"]
                dynamic_vars = item["variables"]["dynamic"]
                if fixed_vars == []:
                    for _ in range(min_tasks_per_template):
                        var_dict = {}
                        for dynamic_var in dynamic_vars:
                            key = dynamic_var["key"]
                            description = dynamic_var["description"]
                            value = self.generate_random_value(description)
                            var_dict[key] = value
                        task_dict[task_fmt].append(var_dict)
                elif dynamic_vars == []:
                    keys = [var["key"] for var in fixed_vars]
                    values = [var["values"] for var in fixed_vars]
                    var_dicts = []
                    for value_combination in itertools.product(*values):
                        var_dict = {k: v for k, v in zip(keys, value_combination)}
                        var_dicts.append(var_dict)
                    var_dicts = random.sample(var_dicts, min(len(var_dicts), max_tasks_per_template))
                    task_dict[task_fmt] = var_dicts
                else:
                    # total task num: len(fixed_vars[0]["values"]) ** (len(dynamic_vars) + len(fixed_vars))
                    value_num = len(fixed_vars[0]["values"])
                    num_repeat_fixed_vars = value_num ** len(dynamic_vars)

                    # num_repeat_fixed_vars * (value_num ** len(fixed_vars)) <= max_tasks_per_template
                    num_repeat_fixed_vars = min(max_tasks_per_template // (value_num ** len(fixed_vars)), num_repeat_fixed_vars)
                    
                    fixed_keys = [var["key"] for var in fixed_vars]
                    fixed_values = [var["values"] for var in fixed_vars]
                    for fixed_value_combination in itertools.product(*fixed_values):
                        for _ in range(num_repeat_fixed_vars):
                            var_dict = {k: v for k, v in zip(fixed_keys, fixed_value_combination)}
                            for dynamic_var in dynamic_vars:
                                key = dynamic_var["key"]
                                description = dynamic_var["description"]
                                value = self.generate_random_value(description)
                                var_dict[key] = value
                            task_dict[task_fmt].append(var_dict)
            
            # 保存 task_dict 到缓存文件
            logger.info(f"Saving task_dict to cache: {cache_file_path}")
            # 将 defaultdict 转换为普通字典以便序列化
            task_dict_serializable = dict(task_dict)
            with open(cache_file_path, "w", encoding="utf-8") as f:
                json.dump(task_dict_serializable, f, ensure_ascii=False)

        records = []
        for task_fmt, var_dicts in task_dict.items():
            for var_dict in var_dicts:
                task_description = task_fmt
                for k, v in var_dict.items():
                    task_description = task_description.replace(f"{{{{{k}}}}}", v)
                logger.info(f"Executing test task: {task_description} with variables {var_dict}")
                if use_acttree:
                    self.execute_task_acttree(task_description, var_dict)
                else:
                    self.execute_task(task_description, var_dict)
            metrics = self.get_metrics()
            records.append(metrics | {"task": task_fmt})
            self.reset_metrics()
        return pd.DataFrame(records)
        # for item in task_data:
        #     task_fmt = item["task"]
        #     variables = item["variables"]
        #     keys = [var["key"] for var in variables]
        #     values = [var["values"] for var in variables]
        #     for var_values in itertools.product(*values):
        #         var_dict = {k: v for k, v in zip(keys, var_values)}
        #         task_description = task_fmt
        #         for k, v in var_dict.items():
        #             task_description = task_description.replace(f"{{{{{k}}}}}", v)
        #         logger.info(f"Executing test task: {task_description} with variables {var_dict}")
        #         if use_acttree:
        #             self.execute_task_acttree(task_description, var_dict)
        #         else:
        #             self.execute_task(task_description, var_dict)
        #     metrics = self.get_metrics()
        #     records.append(metrics | {"task": task_fmt})
        #     self.reset_metrics()

        # tasks: list[tuple[str, dict[str, str]]] = []
        # cities = ["上海", "北京", "广州", "深圳", "杭州"]
        # hotels = ["汉庭", "全季", "如家", "7天", "锦江之星"]
        # for city, hotel in itertools.product(cities, hotels):
        #     task_description = f"查询{city}的{hotel}酒店价格"
        #     tasks.append((task_description, {"城市名": city, "酒店名": hotel}))
        # for task_description, variables in tasks:
        #     self.execute_task(task_description, variables)
        
if __name__ == "__main__":
    client = OpenAI()
    planner_model = "gemini-2.5-flash"
    # planner_model = ""
    experience_dir = Path(__file__).resolve().parent.parent.parent / "utils" / "experience"
    
    template_path = str(experience_dir / "templates-rr.json")
    experience_rr_test = ExperienceRRTest(planner_client=client, planner_model=planner_model, template_path=template_path)
    df = experience_rr_test.run_test_suite(use_acttree=True)
    
    logger.info("Test results:")
    logger.info(df)
    result_path = str(experience_dir / "acttree_oracle_2_1.csv")
    df.to_csv(result_path, index=False)
    logger.info(f"Results saved to {result_path}")
    # experience_rr = ExperienceRR(planner_client=client)
    # # test record
    # final_desc = "请你使用携程App，帮我完成\"查询上海的汉庭酒店价格\"。主要操作流程为：\n1. 在首页点击“酒店”功能入口。\n2. 选择城市为\"上海\"。\n3. 输入并选定酒店名称为\"汉庭酒店\"。\n4. 执行酒店查询。\n5. 看到酒店搜索结果列表出现后，任务结束。"
    # template = "请你使用{{APP名称}}App，帮我完成\"{{用户原始任务描述}}\"。主要操作流程为：\n1. 在首页点击“酒店”功能入口。\n2. 选择城市为\"{{城市名}}\"。\n3. 输入并选定酒店名称为\"{{酒店名}}\"。\n4. 执行酒店查询。\n5. 看到酒店搜索结果列表出现后，任务结束。"
    # history = [
    #     '{"reasoning": "任务是查询全季酒店的价格。首先，我需要进入酒店预订功能。因此，我将点击主页上的“酒店”图标。", "action": "click", "parameters": {"target_element": "酒店图标"}}',
    #     '{"reasoning": "进入酒店预订页面后，我需要指定城市。当前城市显示的是“北京”，我需要点击它来修改目标城市。", "action": "click", "parameters": {"target_element": "城市选择框，当前显示北京"}}',
    #     '{"reasoning": "我需要点击热门城市中的上海来选择城市", "action": "click", "parameters": {"target_element": "热门城市中的上海"}}',
    #     '{"reasoning": "我已经完成了城市选择，接下来我需要指定要查询的酒店，为此，我需要点击位置/品牌/酒店输入框", "action": "click", "parameters": {"target_element": "位置/品牌/酒店输入框"}}',
    #     '{"reasoning": "我需要在搜索框中输入“全季酒店”以查找相关酒店。", "action": "input", "parameters": {"text": "汉庭酒店"}}',
    #     '{"reasoning": "输入“全季酒店”后，我需要点击键盘上的搜索按钮来执行搜索。", "action": "click", "parameters": {"target_element": "键盘上的搜索按钮"}}',
    #     '{"reasoning": "我已经完成了城市和酒店的选择，接下来我需要点击查询按钮来执行这次查询", "action": "click", "parameters": {"target_element": "查询按钮"}}',
    #     '{"reasoning": "屏幕上出现了酒店搜索结果，任务完成", "action": "done", "parameters": {"status": "success"}}',
    # ]
    # experience_rr.record(final_desc, template, history)
    # # test query
    # final_desc2 = "请你使用携程App，帮我完成\"查询北京的汉庭酒店价格\"。主要操作流程为：\n1. 在首页点击“酒店”功能入口。\n2. 选择城市为\"北京\"。\n3. 输入并选定酒店名称为\"汉庭酒店\"。\n4. 执行酒店查询。\n5. 看到酒店搜索结果列表出现后，任务结束。"
    # queried_actions = experience_rr.query(final_desc2, template)
    # for action in queried_actions:
    #     print(action)
    # final_desc3 = "请你使用携程App，帮我完成\"查询上海的全季酒店价格\"。主要操作流程为：\n1. 在首页点击“酒店”功能入口。\n2. 选择城市为\"上海\"。\n3. 输入并选定酒店名称为\"全季酒店\"。\n4. 执行酒店查询。\n5. 看到酒店搜索结果列表出现后，任务结束。"
    # queried_actions = experience_rr.query(final_desc3, template)
    # for action in queried_actions:
    #     print(action)