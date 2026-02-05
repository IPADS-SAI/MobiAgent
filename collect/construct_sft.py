import os, json
from dataclasses import dataclass, asdict
from typing import List
from PIL import Image
import random
import argparse
from tqdm import tqdm

import re
from functools import reduce

from utils.load_md_prompt import load_prompt

def load_augmentation_rules(config_path="augment_config.json"):
    """读取数据扩充配置文件，返回规则列表"""
    if not os.path.exists(config_path):
        print(f"警告：配置文件 '{config_path}' 不存在，使用默认规则（无扩充）。")
        return []
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            rules = json.load(f)
        for rule in rules:
            if not isinstance(rule.get("dir"), list):
                raise ValueError(f"无效规则：{rule}，dir 必须是列表")
            if not isinstance(rule.get("pattern"), str):
                raise ValueError(f"无效规则：{rule}，pattern 必须是字符串")
            if not isinstance(rule.get("multiplier"), dict):
                raise ValueError(f"无效规则：{rule}，multiplier 必须是字典")
            rule["compiled_pattern"] = re.compile(rule["pattern"])
        return rules
    except Exception as e:
        print(f"读取配置文件失败：{e}，使用默认规则（无扩充）。")
        return []

def augment_data(action, rules):
    # 检查每个规则
    for rule in rules:
        try:
            field_value = reduce(lambda d, k: d[k], rule["dir"], action)
        except (KeyError, TypeError):
            continue
        if not isinstance(field_value, str):
            continue
        if rule["compiled_pattern"].search(field_value):
            return rule["multiplier"]
    return {"default": 1}

@dataclass
class AlpacaImageEntry:
    instruction: str
    output: str
    images: List[str]
    input: str = ""

grounder_prompt = load_prompt("grounder_coordinates.md")
grounder_prompt_bbox = load_prompt("grounder_bbox.md")
grounder_prompt_qwen3_coordinates = load_prompt("grounder_qwen3_coordinates.md")
grounder_prompt_qwen3_bbox = load_prompt("grounder_qwen3_bbox.md")

# decider_prompt = load_prompt("decider.md")
# decider_prompt_no_history = load_prompt("decider_nohistory.md")
decider_prompt = load_prompt("decider_v2.md")
decider_prompt_no_history = load_prompt("decider_nohistory_v2.md")
# decider_prompt_qwen3 = load_prompt("decider_qwen3.md")
# decider_prompt_qwen3_no_history = load_prompt("decider_qwen3_nohistory.md")
decider_prompt_qwen3 = decider_prompt
decider_prompt_qwen3_no_history = decider_prompt_no_history

e2e_prompt = load_prompt("e2e_qwen3.md")
e2e_prompt_no_history = load_prompt("e2e_nohistory_qwen3.md")

def dump_json_with_jsonl(out_path, data):
    # Write standard JSON and, when possible, a line-delimited JSONL for easier downstream loading
    with open(out_path, "w", encoding="UTF-8") as f:
        json.dump(data, f, ensure_ascii=False)

    if isinstance(data, list):
        jsonl_path = os.path.splitext(out_path)[0] + ".jsonl"
        with open(jsonl_path, "w", encoding="UTF-8") as jf:
            for row in data:
                jf.write(json.dumps(row, ensure_ascii=False) + "\n")

def history_str(history):
    if len(history) == 0:
        return "(No history)"
    else:
        return "\n".join(f"{idx}. {h}" for idx, h in enumerate(history, 1))


def position_num_repeat(index, total_length):
    if index == total_length - 1 or index / total_length <= 0.5:
        return 1
    else:
        return 2
    
def augment_num_repeat(part, augment_rule, is_train):
    return augment_rule.get(part, augment_rule.get("default", 1)) if is_train else 1

def create_entries_for_one_step(num_repeat, instruction, output, image_path):
    entry = AlpacaImageEntry(
        instruction=instruction,
        output=output,
        images=[image_path]
    )
    return [entry] * num_repeat

def resize_and_copy_image(part, img_path, data_path, out_path, factor, do_copy=False):
    relative_path = os.path.relpath(img_path, data_path)
    safe_filename = relative_path.replace(os.sep, "_").replace(":", "_")
    safe_filename = f"{part}_{safe_filename}"
    out_relpath = os.path.join(out_path, safe_filename)

    # Resize image并保存在同一目录下
    pil_img = Image.open(img_path)
    width, height = pil_img.size
    new_width = int(width * factor)
    new_height = int(height * factor)
    if do_copy:
        # 避免重复保存处理
        if os.path.exists(out_relpath):
            return os.path.abspath(out_relpath), new_width, new_height
        resized_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
        resized_img.save(out_relpath)

    out_abspath = os.path.abspath(out_relpath)
    return out_abspath, new_width, new_height

def validate_action(action_type, param):
    if action_type == "done":
        # 处理 done 类型的 status 参数, status(str) 可选 success / suspended / failed
        if isinstance(param, dict) and "status" in param:
            status = param["status"]
            if status in ["success", "suspended", "failed"]:
                return "done", {"status": status}
            
        return "done", {"status": "success"}
    # 一部分旧数据中存在 stop 和 terminate 两种类型，做兼容处理
    elif action_type == "stop":
        return "done", {"status": "suspended"}
    elif action_type == "terminate":
        return "done", {"status": "failed"}
    param_name_mapping = {
        "click": ["target_element","bbox"],
        "input": ["text"],
        "swipe": ["direction", "start_coords", "end_coords"],
        "wait": [],
        "done": ["status"]
    }
    if action_type not in param_name_mapping:
        raise ValueError(f"Unknown action type: {action_type}")
    
    valid_param_names = param_name_mapping[action_type]

    if not isinstance(param, dict):
        param = {}
    
    validated_param = {k: v for k, v in param.items() if k in valid_param_names}
    return action_type, validated_param

def format_qwen3_grounder_output(output_dict):
    return f'```json\n[\n    {json.dumps(output_dict, ensure_ascii=False)}\n]\n```'

def format_qwen3_decider_output(output_dict):
    # use the same format as qwen2.5 for now
    output_json = json.dumps(output_dict, ensure_ascii=False)
    return output_json, output_json

def relative_point(point, width, height):
    x, y = point
    rel_x = x / width * 1000
    rel_y = y / height * 1000
    return [int(rel_x), int(rel_y)]

def relative_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox
    rel_x1 = x1 / width * 1000
    rel_y1 = y1 / height * 1000
    rel_x2 = x2 / width * 1000
    rel_y2 = y2 / height * 1000
    return [int(rel_x1), int(rel_y1), int(rel_x2), int(rel_y2)]

def construct_ss_data(single_step_data_path, out_path, factor=0.5, train_ratio=0.9, do_copy=True, use_qwen3=False, e2e=False):
    if not os.path.exists(single_step_data_path):
        return [], [], [], []

    augment_config_path = os.path.join(os.path.dirname(__file__), 'augment_config.json')
    rules = load_augmentation_rules(augment_config_path)

    # 初始化所有返回变量
    decider_ss_entry_train = []
    decider_ss_entry_val = []
    grounder_ss_entry_train = []
    grounder_ss_entry_val = []

    decider_ss_path = os.path.join(single_step_data_path, "decider")
    if os.path.exists(decider_ss_path):
        for root, dirs, files in tqdm(os.walk(decider_ss_path), desc="constructing single step decider dataset"):
            if len(files) == 0:
                continue
            if "react.json" not in files:
                continue
            if "tasks.json" not in files:
                continue

            react_path = os.path.join(root, "react.json")
            with open(react_path, "r", encoding="UTF-8") as f:
                react_data = json.load(f)

            actions_path = os.path.join(root, "actions.json")
            actions = []
            if os.path.exists(actions_path):
                with open(actions_path, "r", encoding="UTF-8") as f:
                    try:
                        actions_data = json.load(f)
                        actions = actions_data.get("actions", [])
                    except:
                        pass

            tasks_path = os.path.join(root, "tasks.json")
            with open(tasks_path, "r", encoding="UTF-8") as f:
                tasks = json.load(f)

            for i, react in enumerate(react_data, 1):
                is_train = random.random() < train_ratio

                augment_rule = augment_data(react, rules)

                img_path = os.path.join(root, f"{i}.jpg")
                out_abspath, width, height = resize_and_copy_image("ss", img_path, single_step_data_path, out_path, factor, do_copy=do_copy)

                reasoning = react["reasoning"]
                action_type = react["function"]["name"]
                param = react["function"]["parameters"]
                
                action_type, param = validate_action(action_type, param)

                if e2e and action_type == "click":
                    if i - 1 < len(actions):
                        action = actions[i - 1]
                        bbox = action.get("bounds", None)
                        # 根据factor、width和height调整bbox,调整为1000*1000相对坐标
                        bbox = [int(bbox[0] * factor/width * 1000), int(bbox[1] * factor/height * 1000), int(bbox[2] * factor/width * 1000), int(bbox[3] * factor/height * 1000)] if bbox else None
                        if bbox:
                            param.update(dict(bbox=bbox))
                
                if e2e and action_type == "swipe":
                    if "direction" in param:
                        if i - 1 < len(actions):
                            action = actions[i - 1]
                            start_coords = [int(action["press_position_x"] * factor/width * 1000), int(action["press_position_y"] * factor/height * 1000)] if "press_position_x" in action and "press_position_y" in action else None
                            end_coords = [int(action["release_position_x"] * factor/width * 1000), int(action["release_position_y"] * factor/height * 1000)] if "release_position_x" in action and "release_position_y" in action else None
                            if start_coords and end_coords:
                                param.update(dict(start_coords=start_coords, end_coords=end_coords))
                            else:
                                print(f"[e2e]error: action {i} has no swipe coords in {root}")

                # 随机选择一个任务进行训练
                # random_tasks = random.sample(tasks, 1)
                # 按照顺序选择任务进行训练
                print_flag=False
                if len(tasks) == len(react_data):
                    random_tasks = [tasks[i - 1]]
                    print_flag=False
                else:
                    random_tasks = random.sample(tasks, 1)

                for task in random_tasks:
                    if print_flag:
                        print(f"Processing SS Decider - Train: {is_train}, Task: {task}")
                    output_dict = dict(reasoning=reasoning, action=action_type, parameters=param)
                    if use_qwen3:
                        output, _ = format_qwen3_decider_output(output_dict)
                        if e2e:
                            instruction = e2e_prompt_no_history.format(task=task)
                        else:
                            instruction = decider_prompt_qwen3_no_history.format(task=task)
                    else:
                        output = json.dumps(output_dict, ensure_ascii=False)
                        if e2e:
                            instruction = e2e_prompt_no_history.format(task=task)
                        else:
                            instruction = decider_prompt_no_history.format(task=task)

                    aug_num_repeat = augment_num_repeat("decider_no_history", augment_rule, is_train)
                    entries = create_entries_for_one_step(
                        num_repeat=aug_num_repeat,
                        instruction=instruction,
                        output=output,
                        image_path=out_abspath
                    )
                    if print_flag:
                        print(entries)
                    if is_train:
                        decider_ss_entry_train.extend(entries)
                    else:
                        decider_ss_entry_val.extend(entries)

    grounder_ss_path = os.path.join(single_step_data_path, "grounder")
    if os.path.exists(grounder_ss_path):
        for root, dirs, files in tqdm(os.walk(grounder_ss_path), desc="constructing single step grounder dataset"):
            if len(files) == 0:
                continue
            if "react.json" not in files:
                continue

            react_path = os.path.join(root, "react.json")
            with open(react_path, "r", encoding="UTF-8") as f:
                react_data = json.load(f)

            actions_path = os.path.join(root, "actions.json")
            actions = []
            if os.path.exists(actions_path):
                with open(actions_path, "r", encoding="UTF-8") as f:
                    try:
                        actions_data = json.load(f)
                        actions = actions_data.get("actions", [])
                    except:
                        pass

            for i, react in enumerate(react_data, 1):
                is_train = random.random() < train_ratio

                augment_rule = augment_data(react, rules)

                img_path = os.path.join(root, f"{i}.jpg")
                out_abspath, width, height = resize_and_copy_image("ss", img_path, single_step_data_path, out_path, factor, do_copy=do_copy)

                reasoning = react["reasoning"]
                action_type = react["function"]["name"]
                param = react["function"]["parameters"]

                # grounder训练集
                if action_type == "click":
                    bbox = react["bbox"]
                    bbox = [int(x * factor) for x in bbox]
                    aug_num_repeat = augment_num_repeat("grounder", augment_rule, is_train)
                    target_element = param["target_element"]
                    if use_qwen3:
                        instruction = grounder_prompt_qwen3_bbox.format(reasoning=reasoning, description=target_element)
                        rel_bbox = relative_bbox(bbox, width, height)
                        output = format_qwen3_grounder_output(dict(bbox=rel_bbox, label=target_element))
                    else:
                        instruction = grounder_prompt_bbox.format(reasoning=reasoning, description=target_element)
                        output = json.dumps(dict(bbox=bbox))
                    entries = create_entries_for_one_step(
                        num_repeat=aug_num_repeat,
                        instruction=instruction,
                        output=output,
                        image_path=out_abspath
                    )
                    if is_train:
                        grounder_ss_entry_train.extend(entries)
                    else:
                        grounder_ss_entry_val.extend(entries)

    return decider_ss_entry_train, decider_ss_entry_val, grounder_ss_entry_train, grounder_ss_entry_val

def create_grounder_entries_for_one_trace(react_data, actions, root, data_path, out_path, factor, rules, is_train, do_copy=False, use_qwen3=False):
    grounder_entries = []

    for i, react in enumerate(react_data, 1):
        augment_rule = augment_data(react, rules)
        grounder_aug_num_repeat = augment_num_repeat("grounder", augment_rule, is_train)

        img_path = os.path.join(root, f"{i}.jpg")
        out_abspath, width, height = resize_and_copy_image("main", img_path, data_path, out_path, factor, do_copy)

        reasoning = react["reasoning"]
        action_type = react["function"]["name"]
        param = react["function"]["parameters"]

        if action_type == "click":
            if "target_element" not in param:
                print(f"Warning: 'target_element' missing in click action at {root}, step {i}. Skipping.")
                continue

            if i - 1 >= len(actions):
                print(f"Warning: Action index {i-1} out of range for actions list (len={len(actions)}) at {root}. Skipping.")
                continue

            action = actions[i - 1]
            if "position_x" in action and "position_y" in action:
                coords = [int(action["position_x"]* factor), int(action["position_y"]* factor)]
                target_element = param["target_element"]
                if use_qwen3:
                    instruction = grounder_prompt_qwen3_coordinates.format(reasoning=reasoning, description=target_element)
                    rel_point = relative_point(coords, width, height)
                    output = format_qwen3_grounder_output(dict(point_2d=rel_point, label=target_element))
                else:
                    instruction = grounder_prompt.format(reasoning=reasoning, description=target_element)
                    output = json.dumps(dict(coordinates=coords))
                grounder_entries.extend(create_entries_for_one_step(
                    num_repeat=grounder_aug_num_repeat,
                    instruction=instruction,
                    output=output,
                    image_path=out_abspath
                ))
            else:
                print(f"warning: action {i} has no position_x / y in {root}")

            if "bounds" in action and isinstance(action["bounds"], list) and len(action["bounds"]) == 4:
                bbox = action["bounds"]
                bbox = [int(x * factor) for x in bbox]
                target_element = param["target_element"]
                if use_qwen3:
                    instruction = grounder_prompt_qwen3_bbox.format(reasoning=reasoning, description=target_element)
                    rel_bbox = relative_bbox(bbox, width, height)
                    output = format_qwen3_grounder_output(dict(bbox=rel_bbox, label=target_element))
                else:
                    instruction = grounder_prompt_bbox.format(reasoning=reasoning, description=target_element)
                    output = json.dumps(dict(bbox=bbox))
                grounder_entries.extend(create_entries_for_one_step(
                    num_repeat=grounder_aug_num_repeat,
                    instruction=instruction,
                    output=output,
                    image_path=out_abspath
                ))
            else:
                print(f"warning: action {i} has no valid bounds in {root}")
    return grounder_entries

def create_decider_entries_for_one_task(task, react_data, actions, root, data_path, out_path, factor, rules, unexpected_img_safe_abspaths, is_train, do_copy=False, e2e=False, use_qwen3=False):
    # decider
    normal_entries = []
    no_history_entries = []
    terminate_entries = []

    history = []

    # if e2e and use_qwen3:
    #     raise ValueError("qwen3 e2e is not supported")

    if e2e:
        prompt_template = e2e_prompt
        no_history_prompt_template = e2e_prompt_no_history
    elif use_qwen3:
        prompt_template = decider_prompt_qwen3
        no_history_prompt_template = decider_prompt_qwen3_no_history
    else:
        prompt_template = decider_prompt
        no_history_prompt_template = decider_prompt_no_history

    for i, react in enumerate(react_data, 1):
        augment_rule = augment_data(react, rules)
        pos_num_repeat = position_num_repeat(i, len(react_data)) #根据步骤位置设置重复次数
        reason_aug_num_repeat = augment_num_repeat("decider", augment_rule, is_train)  # 在训练时才进行数据扩充
        reason_no_history_aug_num_repeat = augment_num_repeat("decider_no_history", augment_rule, is_train)

        img_path = os.path.join(root, f"{i}.jpg")
        if e2e: 
            out_abspath, width, height = resize_and_copy_image("main", img_path, data_path, out_path, factor, do_copy)
        else:
            out_abspath, width, height = resize_and_copy_image("main", img_path, data_path, out_path, factor, do_copy=False)

        # 获取相关参数
        reasoning = react["reasoning"]
        action_type = react["function"]["name"]
        param = react["function"]["parameters"]

        action_type, param = validate_action(action_type, param)
        
        if e2e and action_type == "click":
            if i - 1 < len(actions):
                action = actions[i - 1]
                bbox = action.get("bounds", None)
                # 根据factor、width和height调整bbox,调整为1000*1000相对坐标
                bbox = [int(bbox[0] * factor/width * 1000), int(bbox[1] * factor/height * 1000), int(bbox[2] * factor/width * 1000), int(bbox[3] * factor/height * 1000)] if bbox else None
            else:
                print(f"[e2e]Error: Action index {i-1} out of range for actions list (len={len(actions)}) at {root}. Skipping bbox.")
                bbox = None
                return [], [], []

            if bbox:
                param.update(dict(bbox=bbox))
            else:
                print(f"[e2e]error: action {i} has no bbox in {root}")
                return [], [], []
        
        if e2e and action_type == "swipe":
            if "direction" in param:
                direction = param["direction"]
                # 从actions中获取start_coords和end_coords
                # start_coords :[press_position_x,press_position_y]
                # end_coords :[release_position_x,release_position_y]
                if i - 1 < len(actions):
                    action = actions[i - 1]
                    start_coords = [int(action["press_position_x"] * factor/width * 1000), int(action["press_position_y"] * factor/height * 1000)] if "press_position_x" in action and "press_position_y" in action else None
                    end_coords = [int(action["release_position_x"] * factor/width * 1000), int(action["release_position_y"] * factor/height * 1000)] if "release_position_x" in action and "release_position_y" in action else None
                    param.update(dict(start_coords=start_coords, end_coords=end_coords))
                else:
                    print(f"[e2e]Error: Action index {i-1} out of range for actions list (len={len(actions)}) at {root}. Skipping swipe coords.")
                    return [], [], []
            else:
                print(f"[e2e]Error: action {i} has no direction in {root}")
                return [], [], []

        output_dict = dict(reasoning=reasoning, action=action_type, parameters=param)
        if use_qwen3:
            output, brief_action = format_qwen3_decider_output(output_dict)
        else:
            output = json.dumps(output_dict, ensure_ascii=False)

        # partial_histories是当前action的前几个action
        # 对input类和done类型特殊处理
        if action_type in ["input"]:
            min_history_length = min(4, len(history))
            partial_histories = [history[i:] for i in range(len(history) + 1 - min_history_length)]
        else:
            partial_histories = [history[i:] for i in range(len(history) + 1)]

        partial_histories = [partial_histories[0]] + random.sample(partial_histories[1:], min(2, len(partial_histories) - 1))

        for partial_history in partial_histories:
            normal_entries.extend(create_entries_for_one_step(
                num_repeat=pos_num_repeat * reason_aug_num_repeat, 
                instruction=prompt_template.format(task=task, history=history_str(partial_history)), 
                output=output, 
                image_path=out_abspath
            ))

        if use_qwen3:
            history.append(brief_action)
        else:
            history.append(output)

        synthesize_terminate = action_type == "click" and len(unexpected_img_safe_abspaths) > 0
        # synthesize terminate samples
        if synthesize_terminate:
            terminate_reasoning_part1 = [
                "当前页面未按预期加载",
                "进入了错误的页面",
                "打开了不合预期的页面",
                "当前打开了错误页面",
                "当前页面不合预期",
                "错误进入了其他应用的页面"
            ]
            terminate_reasoning_part2 = [
                "需要用户介入",
                "需要用户接管",
                "任务无法继续执行"
            ]
            terminate_reasoning_part3 = [
                "任务提前结束",
                "中止任务执行"
            ]

            terminate_reasoning = "，".join(map(random.choice, [terminate_reasoning_part1, terminate_reasoning_part2, terminate_reasoning_part3]))
            terminate_output_dict = dict(reasoning=terminate_reasoning, action="done", parameters={"status": "failed"})
            if use_qwen3:
                terminate_output, _ = format_qwen3_decider_output(terminate_output_dict)
            else:
                terminate_output = json.dumps(terminate_output_dict, ensure_ascii=False)

            terminate_entries.extend(create_entries_for_one_step(
                num_repeat=1, # 终止样本不需要重复
                instruction=prompt_template.format(task=task, history=history_str(history)),
                output=terminate_output,
                image_path=random.choice(unexpected_img_safe_abspaths)
            ))

        
        # 无历史action训练集 (input类型不生成no history数据)
        if action_type not in ["input", "done"]:
            no_history_entries.extend(create_entries_for_one_step(
                num_repeat=pos_num_repeat * reason_no_history_aug_num_repeat,
                instruction=no_history_prompt_template.format(task=task),
                output=output,
                image_path=out_abspath
            ))

    return normal_entries, no_history_entries, terminate_entries

def construct_ds(data_path, single_step_data_path, unexpected_img_path, out_path, factor=0.5, train_ratio=0.9, e2e=False, do_copy=True, use_qwen3=False, json_dir=""):
    os.makedirs(out_path, exist_ok=True)
    
    e2e_entries_train = []
    e2e_terminate_entries_train = []
    e2e_no_history_entries_train = []
    
    e2e_entries_val = []
    e2e_terminate_entries_val = []
    e2e_no_history_entries_val = []

    # 训练集
    decider_entries_train = []
    terminate_entries_train = []
    decider_no_history_entries_train = []
    grounder_entries_train = []
    
    # 验证集
    decider_entries_val = []
    terminate_entries_val = []
    decider_no_history_entries_val = []
    grounder_entries_val = []

    augment_config_path = os.path.join(os.path.dirname(__file__), 'augment_config.json')
    rules = load_augmentation_rules(augment_config_path)

    if os.path.exists(unexpected_img_path):
        unexpected_img_dir = os.path.abspath(unexpected_img_path)
        unexpected_img_paths = os.listdir(unexpected_img_dir)
        unexpected_img_paths = [os.path.join(unexpected_img_dir, img) for img in unexpected_img_paths]

        unexpected_img_safe_abspaths = []
        for unexpected_img_path in unexpected_img_paths:
            out_abspath, width, height = resize_and_copy_image("unexpected", unexpected_img_path, unexpected_img_dir, out_path, factor, do_copy=do_copy)
            unexpected_img_safe_abspaths.append(out_abspath)
    else:
        unexpected_img_safe_abspaths = []
    # 1. 预统计目录数量
    total_dirs = sum(1 for _ in os.walk(data_path))

    # 2. 带 total 的 tqdm
    for root, dirs, files in tqdm(
            os.walk(data_path),
            total=total_dirs,
            desc="constructing dataset"
    ):
    # for root, dirs, files in tqdm(os.walk(data_path), desc="constructing dataset"):
        if len(files) == 0:
            continue
        if "actions.json" not in files or "react.json" not in files or "parse.error" in files:
            continue

        actions_json = os.path.join(root, "actions.json")
        with open(actions_json, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {root}.")
                raise e
        task_description = data.get("task_description")
        actions = data.get("actions")
        react_json = os.path.join(root, "react.json")
        with open(react_json, "r", encoding="UTF-8") as f:
            try:
                react_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {root}.")
                raise e

        # 多模式适配 将没有done的react补充done，目前全部修正带done
        index = 1
        while f"{index}.jpg" in files:
            index += 1
        num_img = index - 1
        if num_img == len(react_data) + 1:
            if isinstance(task_description, list):
                done_reasoning = f"我已经完成了目标任务\"{task_description[0]}\"，任务已结束。"
            else:
                done_reasoning = f"我已经完成了目标任务\"{task_description}\"，任务已结束。"
            react_data.append(
                {
                    "reasoning": done_reasoning,
                    "function": {
                        "name": "done",
                        "parameters": {
                            "status": "success"
                        }
                    }
                }
            )
        elif num_img != len(react_data):
            print(f"Warning: Number of images ({num_img}) does not match number of ReAct entries ({len(react_data)}) in {root}. Skipping this directory.")
            continue

        if not isinstance(task_description, list):
            task_description = [task_description]
        
        # 第一个任务：原始描述
        # 后三个任务：去除标点
        # 中间：泛化任务
        tasks = [task_description[0]]

        has_instruction_following = False
        # 长度为11或2,则最后一个任务为指令遵循，有90%概率加入训练
        if len(task_description) == 11 or len(task_description) == 2:
            has_instruction_following = True
            if random.random() < 0.9:
                tasks.append(task_description[-1])
            task_description = task_description[:-1]
        # 当任务描述长度>=4时，从最后三个泛化任务（无标点任务）中随机选取一个加入训练
        if len(task_description) >= 4:
            tasks += random.sample(task_description[-3:], 1)
        # 当任务描述长度>4时，从中间泛化任务中随机选取一个加入训练（如果没有指令遵循任务，或50%概率加入）
        if len(task_description) > 4:
            if (not has_instruction_following) or (random.random() < 0.5):
                tasks += random.sample(task_description[1:-3], 1)

        is_train = random.random() < train_ratio
        try:
            for i, task in enumerate(tasks):
                normal_entries, no_history_entries, terminate_entries = create_decider_entries_for_one_task(
                    task, react_data, actions, root, data_path, out_path, factor, rules, unexpected_img_safe_abspaths, is_train, do_copy=((i == 0) and do_copy), e2e=False, use_qwen3=use_qwen3
                )
                if normal_entries == [] and no_history_entries == [] and terminate_entries == []:
                    continue
                if i != 0:
                    normal_entries = random.sample(normal_entries, len(normal_entries) // 2)
                    no_history_entries = random.sample(no_history_entries, len(no_history_entries) // 2)
                    terminate_entries = random.sample(terminate_entries, len(terminate_entries) // 2)
                if is_train:
                    decider_entries_train.extend(normal_entries)
                    decider_no_history_entries_train.extend(no_history_entries)
                    terminate_entries_train.extend(terminate_entries)
                else:
                    decider_entries_val.extend(normal_entries)
                    decider_no_history_entries_val.extend(no_history_entries)
                    terminate_entries_val.extend(terminate_entries)
                if e2e:
                    e2e_normal_entries, e2e_history_entries, e2e_terminate_entries = create_decider_entries_for_one_task(
                        task, react_data, actions, root, data_path, out_path, factor, rules, unexpected_img_safe_abspaths, is_train, do_copy=((i == 0) and do_copy), e2e=True, use_qwen3=False
                    )
                    if e2e_normal_entries == [] and e2e_history_entries == [] and e2e_terminate_entries == []:
                        continue

                    if i !=0:
                        e2e_normal_entries = random.sample(e2e_normal_entries, len(e2e_normal_entries) // 2)
                        e2e_history_entries = random.sample(e2e_history_entries, len(e2e_history_entries) // 2)
                        e2e_terminate_entries = random.sample(e2e_terminate_entries, len(e2e_terminate_entries) // 2)
                        
                    if is_train:
                        e2e_entries_train.extend(e2e_normal_entries)
                        e2e_no_history_entries_train.extend(e2e_history_entries)
                        e2e_terminate_entries_train.extend(e2e_terminate_entries)
                    else:
                        e2e_entries_val.extend(e2e_normal_entries)
                        e2e_no_history_entries_val.extend(e2e_history_entries)
                        e2e_terminate_entries_val.extend(e2e_terminate_entries)

        except Exception as e:
            print(f"Error generating decider entries in {root}: {e}")
        if e2e is not True:
            try:
                grounder_entries = create_grounder_entries_for_one_trace(react_data, actions, root, data_path, out_path, factor, rules, is_train, do_copy=False, use_qwen3=use_qwen3)
                if is_train:
                    grounder_entries_train.extend(grounder_entries)
                else:
                    grounder_entries_val.extend(grounder_entries)
            except Exception as e:
                print(f"Error generating grounder entries in {root}: {e}")

    decider_ss_entry_train, decider_ss_entry_val, grounder_ss_entry_train, grounder_ss_entry_val = construct_ss_data(single_step_data_path, out_path, factor, train_ratio, do_copy=do_copy, use_qwen3=use_qwen3)

    # 合并训练集数据
    terminate_entries_train = random.sample(terminate_entries_train, min(len(decider_entries_train) // 75, len(terminate_entries_train)))
    terminate_entries_val = random.sample(terminate_entries_val, min(len(decider_entries_val) // 75, len(terminate_entries_val)))

    print(f"decider_entries_train: {len(decider_entries_train)}")
    print(f"decider_no_history_entries_train: {len(decider_no_history_entries_train)}")
    print(f"terminate_entries_train: {len(terminate_entries_train)}")
    print(f"grounder_entries_train: {len(grounder_entries_train)}")
    print(f"decider_ss_entry_train: {len(decider_ss_entry_train)}")
    print(f"grounder_ss_entry_train: {len(grounder_ss_entry_train)}")
    print()

    data = {
        "decider_entries_train": len(decider_entries_train),
        "decider_no_history_entries_train": len(decider_no_history_entries_train),
        "terminate_entries_train": len(terminate_entries_train),
        "grounder_entries_train": len(grounder_entries_train),
        "decider_ss_entry_train": len(decider_ss_entry_train),
        "grounder_ss_entry_train": len(grounder_ss_entry_train)
    }

    decider_entries_train = [asdict(entry) for entry in decider_entries_train]
    decider_entries_train.extend([asdict(entry) for entry in decider_no_history_entries_train])
    decider_entries_train.extend([asdict(entry) for entry in terminate_entries_train])
    decider_entries_train.extend([asdict(entry) for entry in decider_ss_entry_train])
    # random.shuffle(decider_entries_train)
    
    grounder_entries_train = [asdict(entry) for entry in grounder_entries_train]
    grounder_entries_train.extend([asdict(entry) for entry in grounder_ss_entry_train])
    # random.shuffle(grounder_entries_train)
    
    # 合并验证集数据
    print(f"decider_entries_val: {len(decider_entries_val)}")
    print(f"decider_no_history_entries_val: {len(decider_no_history_entries_val)}")
    print(f"terminate_entries_val: {len(terminate_entries_val)}")
    print(f"grounder_entries_val: {len(grounder_entries_val)}")
    print(f"decider_ss_entry_val: {len(decider_ss_entry_val)}")
    print(f"grounder_ss_entry_val: {len(grounder_ss_entry_val)}")

    # 添加验证集统计信息到data字典
    data.update({
        "decider_entries_val": len(decider_entries_val),
        "decider_no_history_entries_val": len(decider_no_history_entries_val),
        "terminate_entries_val": len(terminate_entries_val),
        "grounder_entries_val": len(grounder_entries_val),
        "decider_ss_entry_val": len(decider_ss_entry_val),
        "grounder_ss_entry_val": len(grounder_ss_entry_val)
    })

    decider_entries_val = [asdict(entry) for entry in decider_entries_val]
    decider_entries_val.extend([asdict(entry) for entry in decider_no_history_entries_val])
    decider_entries_val.extend([asdict(entry) for entry in terminate_entries_val])
    decider_entries_val.extend([asdict(entry) for entry in decider_ss_entry_val])
    # random.shuffle(decider_entries_val)
    
    grounder_entries_val_dict = [asdict(entry) for entry in grounder_entries_val]
    grounder_entries_val_dict.extend([asdict(entry) for entry in grounder_ss_entry_val])
    # random.shuffle(grounder_entries_val_dict)

    if e2e:
        e2e_entries_train = [asdict(entry) for entry in e2e_entries_train]
        e2e_entries_train.extend([asdict(entry) for entry in e2e_no_history_entries_train])
        e2e_entries_train.extend([asdict(entry) for entry in e2e_terminate_entries_train])
        e2e_entries_val = [asdict(entry) for entry in e2e_entries_val]
        e2e_entries_val.extend([asdict(entry) for entry in e2e_no_history_entries_val])
        e2e_entries_val.extend([asdict(entry) for entry in e2e_terminate_entries_val])
        data.update({
            "e2e_entries_train": len(e2e_entries_train),
            "e2e_no_history_entries_train": len(e2e_no_history_entries_train),
            "e2e_terminate_entries_train": len(e2e_terminate_entries_train),
            "e2e_entries_val": len(e2e_entries_val),
            "e2e_no_history_entries_val": len(e2e_no_history_entries_val),
            "e2e_terminate_entries_val": len(e2e_terminate_entries_val)
        })
        print(f"e2e_entries_train: {len(e2e_entries_train)}")
        print(f"e2e_no_history_entries_train: {len(e2e_no_history_entries_train)}")
        print(f"e2e_terminate_entries_train: {len(e2e_terminate_entries_train)}")
        print(f"e2e_entries_val: {len(e2e_entries_val)}")
        print(f"e2e_no_history_entries_val: {len(e2e_no_history_entries_val)}")
        print(f"e2e_terminate_entries_val: {len(e2e_terminate_entries_val)}")

        dump_json_with_jsonl(os.path.join(out_path, json_dir, "mobimind_e2e_train.json"), e2e_entries_train)
        dump_json_with_jsonl(os.path.join(out_path, json_dir, "mobimind_e2e_val.json"), e2e_entries_val)

    # 保存训练集
    dump_json_with_jsonl(os.path.join(out_path, json_dir, "mobimind_decider_train.json"), decider_entries_train)
    dump_json_with_jsonl(os.path.join(out_path, json_dir, "mobimind_grounder_train.json"), grounder_entries_train)
    
    # 保存验证集
    dump_json_with_jsonl(os.path.join(out_path, json_dir, "mobimind_decider_val.json"), decider_entries_val)
    dump_json_with_jsonl(os.path.join(out_path, json_dir, "mobimind_grounder_val.json"), grounder_entries_val_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training dataset construction with Alpaca format")
    parser.add_argument("--data_path", type=str, default="data", help="root path of raw data (default: data)")
    parser.add_argument("--ss_data_path", type=str, default="ss_data", help="root path of single-step data (default: ss_data)")
    parser.add_argument("--unexpected_img_path", type=str, default="unexpected_img", help="root path of unexpected image data (default: unexpected_data)")
    parser.add_argument("--out_path", type=str, default="output", help="output path of train dataset (default: output)")
    parser.add_argument("--factor", type=float, default=0.5, help="resize factor for images (default: 0.5)")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="ratio of training data (default: 0.9)")
    parser.add_argument('--e2e',action='store_true',help='construct e2e dataset')
    parser.add_argument('--no_copy', action='store_true', help='do not copy images to the output path')
    parser.add_argument('--use_qwen3', action='store_true', help='use qwen3-vl mobile agent format')
    parser.add_argument('--json_dir',type=str, default="", help="output json path of train dataset (default: null)")
    args = parser.parse_args()
    construct_ds(
        data_path=args.data_path,
        single_step_data_path=args.ss_data_path,
        unexpected_img_path=args.unexpected_img_path,
        out_path=args.out_path,
        factor=args.factor,
        train_ratio=args.train_ratio,
        e2e=args.e2e,
        do_copy=(not args.no_copy),
        use_qwen3=args.use_qwen3,
        json_dir=args.json_dir
    )
