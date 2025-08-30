import os, itertools, json

def get_task_templates(raw_template):
    results = []
    left_pos = 0
    while True:
        left = raw_template.find('(', left_pos)
        if left == -1:
            break
        right = raw_template.find(')', left + 1)
        if right == -1:
            break

        content = raw_template[left + 1: right]
        contents = content.replace("NULL", "").split('|')

        results.append({
            'left': left,
            'right': right,
            'contents': contents
        })

        left_pos = right + 1
    combinations = itertools.product(*[result['contents'] for result in results])
    task_templates = []
    for combination in combinations:
        segments = []
        last_left = 0
        for i, content in enumerate(combination):
            left = results[i]['left']
            right = results[i]['right']
            segments.append(raw_template[last_left:left])
            segments.append(content)
            last_left = right + 1
        segments.append(raw_template[last_left:])
        task_templates.append(''.join(segments))
    return task_templates

def get_trajectory(trajectory_template, fmt):
    trajectory = []
    for act in trajectory_template:
        for k in fmt.keys():
            if f"{{{k}}}" in act:
                act = act.replace(f"{{{k}}}", fmt[k])
            if f"<{k}>" in act:
                act = act.replace(f"<{k}>", f"<{fmt[k]}>")
        trajectory.append(act)
    return trajectory

def get_app_task_trajectories(domain_dir):
    with open(os.path.join(domain_dir, "templates.json"), encoding='utf-8') as f:
        templates = json.load(f)
    print(f"Domain: {domain_dir}")
    task_trajectories = {}
    for template in templates:
        raw_task_template = template["task"]
        trajectory_template = template["trajectory"]
        task_templates = get_task_templates(raw_task_template)
        candidates = template["candidates"]
        dependency = template.get("dependency", "no")
        keys = list(candidates.keys())

        if dependency == "one-to-one":
            combinations = zip(*[candidates[k] for k in keys])
        elif dependency == "no":
            combinations = itertools.product(*[candidates[k] for k in keys])
        else:
            print(f"Unknonw dependency type: {dependency}")
            continue

        for combination in combinations:
            fmt = {}
            for i, k in enumerate(keys):
                fmt[k] = combination[i]
            trajectory = get_trajectory(trajectory_template, fmt)
            for task_template in task_templates:
                task = task_template.format(**fmt)
                task_trajectories[task] = trajectory
    # print(task_trajectories)
    app_task_trajectories = {}
    for task, trajectory in task_trajectories.items():
        app = trajectory[0].split(' ')[1]
        app = app.replace("<", "").replace(">", "")
        if app not in app_task_trajectories:
            app_task_trajectories[app] = []
        app_task_trajectories[app].append((task, trajectory))

    return app_task_trajectories