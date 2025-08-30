import json, os, itertools
import random
from sklearn.model_selection import train_test_split
import re

from .task_template import get_app_task_trajectories

NUM_NEGATIVE = 10
MAX_REPEAT_TIMES = 10
EMBEDDING_QUERY_FORMAT = "Instruct: Given a phone-use task, retrieve similar tasks that shares at least **{n}** steps with the given task\nQuery:{query}"
EMBEDDING_INSTRUCT_FORMAT = "Instruct: Represent this phone-use task for level **{n}**\nQuery:{query}"

def get_lshare(trajectory1, trajectory2):
    k = 0
    l_share = 0
    while k < len(trajectory1) and k < len(trajectory2):
        if trajectory1[k] == trajectory2[k]:
            l_share += 1
        else:
            break
        k += 1
    return l_share

def single_split_embedding(path):
    task_trajectories = []
    app_tasks = {}
    for root, _, files in os.walk(path):
        if 'templates.json' in files:
            app_task_trajectories = get_app_task_trajectories(root)
            app_tasks.update({app: list(set(task for task, _ in app_task_trajectories[app])) for app in app_task_trajectories})
            task_trajectories.extend(list(set((task, tuple(trajectory)) for app in app_task_trajectories for task, trajectory in app_task_trajectories[app])))
    entries = []
    task_positives = {}
    task_negatives = {}
    task_app = {}
    for i, j in itertools.combinations(range(len(task_trajectories)), 2):
        task1, trajectory1 = task_trajectories[i]
        task2, trajectory2 = task_trajectories[j]
        app1 = trajectory1[0].split(' ')[1].replace("<", "").replace(">", "")
        app2 = trajectory2[0].split(' ')[1].replace("<", "").replace(">", "")
        task_app[task1] = app1
        task_app[task2] = app2
        is_same_app = app1 == app2
        l_share = get_lshare(trajectory1, trajectory2)
        if task1 not in task_positives:
            task_positives[task1] = {}
            task_negatives[task1] = {}
        if task2 not in task_positives:
            task_positives[task2] = {}
            task_negatives[task2] = {}
        for n in range(1, len(trajectory1) + 1):
            if n not in task_positives[task1]:
                task_positives[task1][n] = set([task1])
            if n not in task_negatives[task1]:
                task_negatives[task1][n] = set()
            if l_share >= n:
                task_positives[task1][n].add(task2)
            elif n == 1 or is_same_app:
                task_negatives[task1][n].add(task2)
        for n in range(1, len(trajectory2) + 1):
            if n not in task_positives[task2]:
                task_positives[task2][n] = set([task2])
            if n not in task_negatives[task2]:
                task_negatives[task2][n] = set()
            if l_share >= n:
                task_positives[task2][n].add(task1)
            elif n == 1 or is_same_app:
                task_negatives[task2][n].add(task1)
    for task in task_positives.keys():
        positive = task_positives[task]
        negative = task_negatives[task]
        for n in positive.keys():
            if len(positive[n]) == 1:
                positive[n] |= set([f"请{task}", f"请你{task}", f"请帮我{task}", f"帮我{task}", f"请你帮我{task}"])
            if len(negative[n]) == 0 and n > 1:
                # sample some tasks from other apps
                other_apps = [app for app in app_tasks if app != task_app[task]]
                for app in other_apps:
                    sample_num = (10 * NUM_NEGATIVE + len(negative[n]) - 1) // len(other_apps)
                    sample_num = min(sample_num, len(app_tasks[app]))
                    sampled_tasks = random.sample(app_tasks[app], sample_num)
                    negative[n].update(set(sampled_tasks))
            # print(f"Task: {task}, n: {n}, positives: {len(positive[n])}, negatives: {len(negative[n])}")
            repeat_times = (len(negative[n]) + len(positive[n]) - 1) // len(positive[n])
            repeat_times = (repeat_times + NUM_NEGATIVE - 1) // NUM_NEGATIVE
            repeat_times = max(1, repeat_times)
            start = 0
            rejected = list(negative[n])
            random.shuffle(rejected)
            old_num_entries = len(entries)
            for positive_task in positive[n]:
                # query = EMBEDDING_QUERY_FORMAT.format(n=n, query=task)
                query = EMBEDDING_INSTRUCT_FORMAT.format(n=n, query=task)
                # response = positive_task
                response = EMBEDDING_INSTRUCT_FORMAT.format(n=n, query=positive_task)
                for _ in range(repeat_times):
                    if start >= len(rejected):
                        start = 0
                    end = start + NUM_NEGATIVE
                    end = min(end, len(rejected))
                    entries.append({
                        "query": query,
                        "response": response,
                        # "rejected_response": [rejected[start:end]]
                        "rejected_response": [EMBEDDING_INSTRUCT_FORMAT.format(n=n, query=t) for t in rejected[start:end]],
                    })
                    start = end
            # print(f"Increment: {len(entries) - old_num_entries}")
    return entries


def balance_embedding(entries):
    level_entries = {}
    for entry in entries:
        match = re.search(r'\*\*(\d+)\*\*', entry['query'])
        if match:
            n = int(match.group(1))
            if n not in level_entries:
                level_entries[n] = []
            level_entries[n].append(entry)
    max_len = max(len(level_entries[n]) for n in level_entries)
    for n in level_entries:
        if len(level_entries[n]) < max_len // 2:
            multiplier = max_len // 2 // len(level_entries[n])
            level_entries[n] = level_entries[n] * multiplier
    ret = []
    for n in level_entries:
        ret.extend(level_entries[n])
    return ret

def embedding_main(train_path, test_path):
    entries_train = single_split_embedding(train_path)
    if test_path is None:
        entries_train, entries_test = train_test_split(entries_train, test_size=0.1, random_state=42)
    else:
        entries_test = single_split_embedding(test_path)
    entries_train = balance_embedding(entries_train)
    # random.shuffle(entries_train)
    # random.shuffle(entries_test)
    with open('embedding_mybench_data.jsonl', 'w', encoding='utf-8') as f:
        for entry in entries_train:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    with open('embedding_mybench_data_test.jsonl', 'w', encoding='utf-8') as f:
        for entry in entries_test:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

RERANKER_SYSTEM = "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
POSITIVE_TOKEN = "yes"
NEGATIVE_TOKEN = "no"
RERANKER_INPUT_FORMAT = "<Instruct>: Given a phone-use task, retrieve similar tasks that shares at least **{n}** steps with the given task\n<Query>: {query} \n<Document>: {document}"
RERANKER_OUTPUT_FORMAT = "<think>\n\n</think>\n\n{token}"

def single_app_reranker(task_trajectory_pairs):
    entries = []
    max_len = max(len(trajectory) for _, trajectory in task_trajectory_pairs)
    for i, j in itertools.combinations(range(len(task_trajectory_pairs)), 2):
        task1, trajectory1 = task_trajectory_pairs[i]
        task2, trajectory2 = task_trajectory_pairs[j]
        l_share = get_lshare(trajectory1, trajectory2)
        for n in range(1, max_len + 1):
            token = POSITIVE_TOKEN if l_share >= n else NEGATIVE_TOKEN
            input_text1 = RERANKER_INPUT_FORMAT.format(n=n, query=task1, document=task2)
            input_text2 = RERANKER_INPUT_FORMAT.format(n=n, query=task2, document=task1)
            output_text = RERANKER_OUTPUT_FORMAT.format(token=token)
            entries.extend([
                {
                    "system": RERANKER_SYSTEM,
                    "input": input_text,
                    "output": output_text
                }
                for input_text in [input_text1, input_text2]
            ])
    return entries

def single_domain_reranker(domain_dir):
    app_task_trajectories = get_app_task_trajectories(domain_dir)
    entries = []
    for app in app_task_trajectories:
        entries.extend(single_app_reranker(app_task_trajectories[app]))
    return entries

def cross_app_step1_reranker(path):
    app_tasks = {}
    task_app = {}
    for root, _, files in os.walk(path):
        if 'templates.json' in files:
            app_task_trajectories = get_app_task_trajectories(root)
            app_tasks.update({app: list(set(task for task, _ in app_task_trajectories[app])) for app in app_task_trajectories})
            task_app.update({task: app for app in app_task_trajectories for task, _ in app_task_trajectories[app]})
    entries = []
    for task in task_app:
        app = task_app[task]
        other_apps = [a for a in app_tasks if a != app]
        for other_app in other_apps:
            sample_num = (30 + len(app_tasks[app]) - 1) // len(other_apps)
            sample_num = min(sample_num, len(app_tasks[other_app]))
            sampled_tasks = random.sample(app_tasks[other_app], sample_num)
            for sampled_task in sampled_tasks:
                input_text = RERANKER_INPUT_FORMAT.format(n=1, query=task, document=sampled_task)
                output_text = RERANKER_OUTPUT_FORMAT.format(token=NEGATIVE_TOKEN)
                entries.append({
                    "system": RERANKER_SYSTEM,
                    "input": input_text,
                    "output": output_text
                })
    return entries

def reranker_main(train_path, test_path):
    entries_train = []
    entries_test = []
    for root, _, files in os.walk(train_path):
        if 'templates.json' in files:
            entries_train.extend(single_domain_reranker(root))
    entries_train.extend(cross_app_step1_reranker(train_path))
    if test_path is None:
        entries_train, entries_test = train_test_split(entries_train, test_size=0.1, random_state=42)
    else:
        for root, _, files in os.walk(test_path):
            if 'templates.json' in files:
                entries_test.extend(single_domain_reranker(root))
        entries_test.extend(cross_app_step1_reranker(test_path))

    with open('reranker_mybench_data.jsonl', 'w', encoding='utf-8') as f:
        for entry in entries_train:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    with open('reranker_mybench_data_test.jsonl', 'w', encoding='utf-8') as f:
        for entry in entries_test:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['embedding', 'reranker', 'both'], required=True,
                        help="Specify the task to generate data for.")
    parser.add_argument('--train_path', type=str, default='train/train_data',
                        help="Path to the training data directory.")
    parser.add_argument('--test_path', type=str, default='train/test_data',
                        help="Path to the test data directory.")
    args = parser.parse_args()
    if args.task == 'embedding':
        embedding_main(args.train_path, args.test_path)
    elif args.task == 'reranker':
        reranker_main(args.train_path, args.test_path)
    elif args.task == 'both':
        embedding_main(args.train_path, args.test_path)
        reranker_main(args.train_path, args.test_path)
    else:
        print("Invalid task specified. Use 'embedding' or 'reranker' or 'both'.")
