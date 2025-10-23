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

e2e_prompt = load_prompt("e2e.md")
e2e_prompt_no_history = load_prompt("e2e_nohistory.md")

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
        resized_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
        resized_img.save(out_relpath)

    out_abspath = os.path.abspath(out_relpath)
    return out_abspath, new_width, new_height

def validate_action(action_type, param):
    if action_type == "done":
        return "done", {"status": "success"}
    elif action_type == "stop":
        return "done", {"status": "suspended"}
    elif action_type == "terminate":
        return "done", {"status": "failed"}
    param_name_mapping = {
        "click": ["target_element"],
        "input": ["text"],
        "swipe": ["direction"],
        "wait": [],
        "done": []
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
    output_json = json.dumps(output_dict, ensure_ascii=False)
    return output_json, output_json
    fmt_str = 'Thought: {reasoning}\nAction: {brief_action}\n<tool_call>{tool_call}</tool_call>'
    action_type = output_dict["action"]
    tool_call_dict = {
        "name": "mobile_use", 
        "arguments": {
            "action": action_type,
        }
    }

    if action_type == "click":
        target_element = output_dict["parameters"]["target_element"]
        brief_action = f'点击{target_element}'
        tool_call_dict["arguments"]["target_element"] = target_element
    elif action_type == "input":
        text = output_dict["parameters"]["text"]
        brief_action = f'在文本框中输入"{text}"'
        tool_call_dict["arguments"]["text"] = text
    elif action_type == "swipe":
        direction = output_dict["parameters"]["direction"]
        direction_mapping = {
            "UP": "上",
            "DOWN": "下",
            "LEFT": "左",
            "RIGHT": "右"
        }
        brief_action = f'向{direction_mapping[direction]}滑动屏幕'
        tool_call_dict["arguments"]["direction"] = direction
    elif action_type == "wait":
        brief_action = '等待页面加载'
    elif action_type == "done":
        brief_action = '任务已完成，结束操作'
    
    return fmt_str.format(
        reasoning=output_dict["reasoning"],
        brief_action=brief_action,
        tool_call=json.dumps(tool_call_dict, ensure_ascii=False)
    ), brief_action

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

def construct_ss_data(single_step_data_path, out_path, factor=0.5, train_ratio=0.9, do_copy=True, use_qwen3=False):
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

                random_tasks = random.sample(tasks, 1)

                for task in random_tasks:
                    output_dict = dict(reasoning=reasoning, action=action_type, parameters=param)
                    if use_qwen3:
                        output, _ = format_qwen3_decider_output(output_dict)
                        instruction = decider_prompt_qwen3_no_history.format(task=task)
                    else:
                        output = json.dumps(output_dict, ensure_ascii=False)
                        instruction = decider_prompt_no_history.format(task=task)

                    aug_num_repeat = augment_num_repeat("decider_no_history", augment_rule, is_train)
                    entries = create_entries_for_one_step(
                        num_repeat=aug_num_repeat,
                        instruction=instruction,
                        output=output,
                        image_path=out_abspath
                    )
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
                        output = format_qwen3_grounder_output(dict(bbox_2d=rel_bbox, label=target_element))
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
                    output = format_qwen3_grounder_output(dict(bbox_2d=rel_bbox, label=target_element))
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

    if e2e and use_qwen3:
        raise ValueError("qwen3 e2e is not supported")

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
        pos_num_repeat = position_num_repeat(i, len(react_data))
        reason_aug_num_repeat = augment_num_repeat("decider", augment_rule, is_train)
        reason_no_history_aug_num_repeat = augment_num_repeat("decider_no_history", augment_rule, is_train)

        img_path = os.path.join(root, f"{i}.jpg")
        out_abspath, width, height = resize_and_copy_image("main", img_path, data_path, out_path, factor, do_copy)

        # 获取相关参数
        reasoning = react["reasoning"]
        action_type = react["function"]["name"]
        param = react["function"]["parameters"]

        action_type, param = validate_action(action_type, param)
        
        if e2e and action_type == "click":
            action = actions[i - 1]
            bbox = action.get("bounds", None)

            if bbox:
                param.update(dict(bbox=bbox))
            else:
                print(f"warning: action {i} has no bbox in {root}")

        output_dict = dict(reasoning=reasoning, action=action_type, parameters=param)
        if use_qwen3:
            output, brief_action = format_qwen3_decider_output(output_dict)
        else:
            output = json.dumps(output_dict, ensure_ascii=False)

        # partial_histories是当前action的前几个action
        # 对input类和done类型特殊处理
        if action_type in ["input", "done"]:
            min_history_length = min(3, len(history))
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
                "当前页面不合预期"
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

def construct_ds(data_path, single_step_data_path, unexpected_img_path, out_path, factor=0.5, train_ratio=0.9, e2e=False, do_copy=True, use_qwen3=False):
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
            out_abspath, width, height = resize_and_copy_image("unexpected", unexpected_img_path, unexpected_img_dir, out_path, factor, do_copy=True)
            unexpected_img_safe_abspaths.append(out_abspath)
    else:
        unexpected_img_safe_abspaths = []

    for root, dirs, files in tqdm(os.walk(data_path), desc="constructing dataset"):
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
            done_reasoning = "我已经完成了目标任务，任务已结束。"
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
        # 长度为11,则最后一个任务为指令遵循
        if len(task_description) == 11:
            has_instruction_following = True
            if random.random() < 0.5:
                tasks.append(task_description[-1])
            task_description = task_description[:-1]
            
        if len(task_description) >= 4:
            tasks += random.sample(task_description[-3:], 1)
        if len(task_description) > 4:
            if (not has_instruction_following) or (random.random() < 0.5):
                tasks += random.sample(task_description[1:-3], 1)

        is_train = random.random() < train_ratio
        for i, task in enumerate(tasks):
            normal_entries, no_history_entries, terminate_entries = create_decider_entries_for_one_task(
                task, react_data, actions, root, data_path, out_path, factor, rules, unexpected_img_safe_abspaths, is_train, do_copy=((i == 0) and do_copy), e2e=False, use_qwen3=use_qwen3
            )
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
                    task, react_data, actions, root, data_path, out_path, factor, rules, unexpected_img_safe_abspaths, is_train, do_copy=False, e2e=True, use_qwen3=use_qwen3
                )
                if is_train:
                    e2e_entries_train.extend(e2e_normal_entries)
                    e2e_no_history_entries_train.extend(e2e_history_entries)
                    e2e_terminate_entries_train.extend(e2e_terminate_entries)
                else:
                    e2e_entries_val.extend(e2e_normal_entries)
                    e2e_no_history_entries_val.extend(e2e_history_entries)
                    e2e_terminate_entries_val.extend(e2e_terminate_entries)

        grounder_entries = create_grounder_entries_for_one_trace(react_data, actions, root, data_path, out_path, factor, rules, is_train, do_copy=False, use_qwen3=use_qwen3)
        if is_train:
            grounder_entries_train.extend(grounder_entries)
        else:
            grounder_entries_val.extend(grounder_entries)

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

        with open(os.path.join(out_path, f"mobimind_e2e_train.json"), "w", encoding="UTF-8") as f:
            json.dump(e2e_entries_train, f, ensure_ascii=False)
        with open(os.path.join(out_path, f"mobimind_e2e_val.json"), "w", encoding="UTF-8") as f:
            json.dump(e2e_entries_val, f, ensure_ascii=False)

    # 保存训练集
    with open(os.path.join(out_path, f"mobimind_decider_train.json"), "w", encoding="UTF-8") as f:
        json.dump(decider_entries_train, f, ensure_ascii=False)
    with open(os.path.join(out_path, f"mobimind_grounder_train.json"), "w", encoding="UTF-8") as f:
        json.dump(grounder_entries_train, f, ensure_ascii=False)
    
    # 保存验证集
    with open(os.path.join(out_path, f"mobimind_decider_val.json"), "w", encoding="UTF-8") as f:
        json.dump(decider_entries_val, f, ensure_ascii=False)
    with open(os.path.join(out_path, f"mobimind_grounder_val.json"), "w", encoding="UTF-8") as f:
        json.dump(grounder_entries_val_dict, f, ensure_ascii=False)

    with open(os.path.join(out_path, f"metadata.json"), "w", encoding="UTF-8") as f:
        json.dump(data, f, ensure_ascii=False)


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
        use_qwen3=args.use_qwen3
    )
