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

decider_prompt = load_prompt("decider.md")
decider_prompt_no_history = load_prompt("decider_nohistory.md")

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
    if do_copy:
        pil_img = Image.open(img_path)
        width, height = pil_img.size
        new_width = int(width * factor)
        new_height = int(height * factor)
        resized_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
        resized_img.save(out_relpath)

    out_abspath = os.path.abspath(out_relpath)
    return out_abspath

def construct_ss_data(single_step_data_path, out_path, factor=0.5, train_ratio=0.9):
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
                out_abspath = resize_and_copy_image("ss", img_path, single_step_data_path, out_path, factor, do_copy=True)

                reasoning = react["reasoning"]
                action = react["function"]["name"]
                param = react["function"]["parameters"]

                random_tasks = random.sample(tasks, 1)

                for task in random_tasks:
                    output_dict = dict(reasoning=reasoning, action=action, parameters=param)
                    output = json.dumps(output_dict, ensure_ascii=False)
                    aug_num_repeat = augment_num_repeat("decider_no_history", augment_rule, is_train)
                    entries = create_entries_for_one_step(
                        num_repeat=aug_num_repeat,
                        instruction=decider_prompt_no_history.format(task=task),
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
                out_abspath = resize_and_copy_image("ss", img_path, single_step_data_path, out_path, factor, do_copy=True)

                reasoning = react["reasoning"]
                action = react["function"]["name"]
                param = react["function"]["parameters"]

                # grounder训练集
                if action == "click":
                    bbox = react["bbox"]
                    bbox = [int(x * factor) for x in bbox]
                    aug_num_repeat = augment_num_repeat("grounder", augment_rule, is_train)
                    entries = create_entries_for_one_step(
                        num_repeat=aug_num_repeat,
                        instruction=grounder_prompt_bbox.format(reasoning=reasoning, description=param["target_element"]),
                        output=json.dumps(dict(bbox=bbox)),
                        image_path=out_abspath
                    )
                    if is_train:
                        grounder_ss_entry_train.extend(entries)
                    else:
                        grounder_ss_entry_val.extend(entries)

    return decider_ss_entry_train, decider_ss_entry_val, grounder_ss_entry_train, grounder_ss_entry_val

def create_grounder_entries_for_one_trace(react_data, actions, root, data_path, out_path, factor, rules, is_train, do_copy=False):
    grounder_entries = []

    for i, react in enumerate(react_data, 1):
        augment_rule = augment_data(react, rules)
        grounder_aug_num_repeat = augment_num_repeat("grounder", augment_rule, is_train)

        img_path = os.path.join(root, f"{i}.jpg")
        out_abspath = resize_and_copy_image("main", img_path, data_path, out_path, factor, do_copy)

        reasoning = react["reasoning"]
        action_type = react["function"]["name"]
        param = react["function"]["parameters"]

        if action_type == "click":
            action = actions[i - 1]
            coords = [int(action["position_x"]* factor), int(action["position_y"]* factor)]
            bbox = action.get("bounds", None)

            grounder_entries.extend(create_entries_for_one_step(
                num_repeat=grounder_aug_num_repeat,
                instruction=grounder_prompt.format(reasoning=reasoning, description=param["target_element"]),
                output=json.dumps(dict(coordinates=coords)),
                image_path=out_abspath
            ))

            if bbox:
                bbox = [int(x * factor) for x in bbox]
                grounder_entries.extend(create_entries_for_one_step(
                    num_repeat=grounder_aug_num_repeat,
                    instruction=grounder_prompt_bbox.format(reasoning=reasoning, description=param["target_element"]),
                    output=json.dumps(dict(bbox=bbox)),
                    image_path=out_abspath
                ))
    return grounder_entries

def create_decider_entries_for_one_task(task, react_data, root, data_path, out_path, factor, rules, unexpected_img_safe_abspaths, is_train, do_copy=False):
    # decider
    normal_entries = []
    no_history_entries = []
    terminate_entries = []

    history = []
    for i, react in enumerate(react_data, 1):
        augment_rule = augment_data(react, rules)
        pos_num_repeat = position_num_repeat(i, len(react_data))
        reason_aug_num_repeat = augment_num_repeat("decider", augment_rule, is_train)
        reason_no_history_aug_num_repeat = augment_num_repeat("decider_no_history", augment_rule, is_train)

        img_path = os.path.join(root, f"{i}.jpg")
        out_abspath = resize_and_copy_image("main", img_path, data_path, out_path, factor, do_copy)

        # 获取相关参数
        reasoning = react["reasoning"]
        action_type = react["function"]["name"]
        param = react["function"]["parameters"]
        
        output_dict = dict(reasoning=reasoning, action=action_type, parameters=param)
        output = json.dumps(output_dict, ensure_ascii=False)

        # partial_histories是当前action的前几个action
        # 对input类和done类型特殊处理
        if action_type == "input" or action_type == "done":
            min_history_length = min(3, len(history))
            partial_histories = [history[i:] for i in range(len(history) + 1 - min_history_length)]
        else:
            partial_histories = [history[i:] for i in range(len(history) + 1)]

        partial_histories = [partial_histories[0]] + random.sample(partial_histories[1:], min(2, len(partial_histories) - 1))

        for partial_history in partial_histories:
            normal_entries.extend(create_entries_for_one_step(
                num_repeat=pos_num_repeat * reason_aug_num_repeat, 
                instruction=decider_prompt.format(task=task, history=history_str(partial_history)), 
                output=output, 
                image_path=out_abspath
            ))

        history.append(output)

        synthesize_terminate = action_type != "wait" and action_type != "done" and action_type != "swipe"
        synthesize_terminate = synthesize_terminate and len(unexpected_img_safe_abspaths) > 0
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
            terminate_output_dict = dict(reasoning=terminate_reasoning, action="done", parameters={})
            terminate_output = json.dumps(terminate_output_dict, ensure_ascii=False)

            terminate_entries.extend(create_entries_for_one_step(
                num_repeat=pos_num_repeat * reason_aug_num_repeat,
                instruction=decider_prompt.format(task=task, history=history_str(history)),
                output=terminate_output,
                image_path=random.choice(unexpected_img_safe_abspaths)
            ))

        
        # 无历史action训练集 (input类型不生成no history数据)
        if action_type != "done" and action_type != "input":
            no_history_entries.extend(create_entries_for_one_step(
                num_repeat=pos_num_repeat * reason_no_history_aug_num_repeat,
                instruction=decider_prompt_no_history.format(task=task),
                output=output,
                image_path=out_abspath
            ))

    return normal_entries, no_history_entries, terminate_entries


def construct_ds(data_path, single_step_data_path, unexpected_img_path, out_path, factor=0.5, train_ratio=0.9):
    os.makedirs(out_path, exist_ok=True)
    
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

    #TODO: unexpected_img_path 不存在情况
    if os.path.exists(unexpected_img_path):
        unexpected_img_dir = os.path.abspath(unexpected_img_path)
        unexpected_img_paths = os.listdir(unexpected_img_dir)
        unexpected_img_paths = [os.path.join(unexpected_img_dir, img) for img in unexpected_img_paths]

        unexpected_img_safe_abspaths = []
        for unexpected_img_path in unexpected_img_paths:
            out_abspath = resize_and_copy_image("unexpected", unexpected_img_path, unexpected_img_dir, out_path, factor, do_copy=True)
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
            data = json.load(file)
        task_description = data.get("task_description")
        actions = data.get("actions")
        react_json = os.path.join(root, "react.json")
        with open(react_json, "r", encoding="UTF-8") as f:
            react_data = json.load(f)

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
                        "parameters": {}
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
        if len(task_description) >= 4:
            tasks += random.sample(task_description[-3:], 1)
        if len(task_description) > 4:
            tasks += random.sample(task_description[1:-3], 1)

        is_train = random.random() < train_ratio
        for i, task in enumerate(tasks):
            normal_entries, no_history_entries, terminate_entries = create_decider_entries_for_one_task(
                task, react_data, root, data_path, out_path, factor, rules, unexpected_img_safe_abspaths, is_train, do_copy=(i == 0)
            )
            if i != 0:
                normal_entries = random.sample(normal_entries, len(normal_entries) * 3 // 4 )
                no_history_entries = random.sample(no_history_entries, len(no_history_entries) * 3 // 4)
                terminate_entries = random.sample(terminate_entries, len(terminate_entries) * 3 // 4)
            if is_train:
                decider_entries_train.extend(normal_entries)
                decider_no_history_entries_train.extend(no_history_entries)
                terminate_entries_train.extend(terminate_entries)
            else:
                decider_entries_val.extend(normal_entries)
                decider_no_history_entries_val.extend(no_history_entries)
                terminate_entries_val.extend(terminate_entries)

        grounder_entries = create_grounder_entries_for_one_trace(react_data, actions, root, data_path, out_path, factor, rules, is_train, do_copy=False)
        if is_train:
            grounder_entries_train.extend(grounder_entries)
        else:
            grounder_entries_val.extend(grounder_entries)

    decider_ss_entry_train, decider_ss_entry_val, grounder_ss_entry_train, grounder_ss_entry_val = construct_ss_data(single_step_data_path, out_path, factor, train_ratio)

    # 合并训练集数据
    terminate_entries_train = random.sample(terminate_entries_train, len(terminate_entries_train) // 10)
    terminate_entries_val = random.sample(terminate_entries_val, len(terminate_entries_val) // 10)

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
    args = parser.parse_args()
    construct_ds(
        data_path=args.data_path,
        single_step_data_path=args.ss_data_path,
        unexpected_img_path=args.unexpected_img_path,
        out_path=args.out_path,
        factor=args.factor,
        train_ratio=args.train_ratio,
    )