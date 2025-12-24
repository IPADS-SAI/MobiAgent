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