import logging, os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())

from train.task_template import get_app_task_trajectories
from agent.agent import Agent
from agent.env import Environment
import os, random, math
from action_cache.action import Action
from action_cache.tree import ActionTree, Task, MatchMode
import pandas as pd

class MybenchTasks:
    def __init__(self, data_path):
        self.app_task_trajectories = {}
        for root, _, files in os.walk(data_path):
            if 'templates.json' in files:
                domain_app_task_trajectories = get_app_task_trajectories(root)
                for app, tasks in domain_app_task_trajectories.items():
                    if app not in self.app_task_trajectories:
                        self.app_task_trajectories[app] = []
                    self.app_task_trajectories[app].extend(tasks)
        for app in self.app_task_trajectories.keys():
            old_task_trajectories = self.app_task_trajectories[app]
            new_task_trajectories = []
            for task, trajectory in old_task_trajectories:
                new_trajectory = []
                for action in trajectory:
                    fields = action.split(" ")
                    new_trajectory.append(Action(name=fields[0], param={str(i) : field for i, field in enumerate(fields[1:])}, extra={}))
                new_task_trajectories.append((task, new_trajectory))
            self.app_task_trajectories[app] = new_task_trajectories

    def get_app_task_trajectories(self):
        return self.app_task_trajectories


class MybenchAgent(Agent):
    def __init__(self, tasks: MybenchTasks):
        super().__init__()
        self.reset_cnt()
        self.reset_cur_task()
        self.tasks = tasks
        self.task_trajectory = {}
        self.task_step = {}
        app_task_trajectories = self.tasks.get_app_task_trajectories()
        for app, task_trajectories in app_task_trajectories.items():
            for task, trajectory in task_trajectories:
                self.task_trajectory[task] = trajectory
                self.task_step[task] = -1

    def reset_cnt(self):
        self.generate_cnt = 0

    def reset_cur_task(self, account=False):
        if account:
            self.generate_cnt += self.cur_generate_cnt
        self.cur_generate_cnt = 0

    def print_cnt(self):
        logger.info(f"generate_cnt: {self.generate_cnt}")

    def generate(self, agent_input):
        self.cur_generate_cnt += 1
        task = agent_input["task"]
        trajectory = self.task_trajectory[task]
        cur_step = self.task_step[task]
        if cur_step >= len(trajectory):
            return {"name":"done", "param":{}, "extra":{}}
        action = trajectory[cur_step]
        return {"name": action.name, "param": action.param, "extra": action.extra}

class MybenchEnvironment(Environment):
    def __init__(self, agent: MybenchAgent):
        super().__init__()
        self.agent = agent
        self.cur_task = ""
        self.reset_cnt()
        self.reset_cur_task()

    def reset_cnt(self):
        self.execute_cnt = 0
        self.total_task_cnt = 0
        self.correct_task_cnt = 0

    def reset_cur_task(self):
        self.cur_execute_cnt = 0
        self.cur_success = True

    def print_cnt(self):
        logger.info(f"execute_cnt: {self.execute_cnt}, total_task_cnt: {self.total_task_cnt}, correct_task_cnt: {self.correct_task_cnt}")

    def get_agent_input(self, history, task_description):
        self.agent.task_step[task_description] += 1
        self.cur_task = task_description
        return {"task": task_description, "history": history}

    def execute(self, action):
        logger.debug(f"env executing: {action}")
        self.cur_execute_cnt += 1
        step = self.agent.task_step[self.cur_task]
        if step >= len(self.agent.task_trajectory[self.cur_task]):
            logger.debug(f"incorrect: expected done action, actual action is {action}")
            self.cur_success = False
            return
        ground_truth = self.agent.task_trajectory[self.cur_task][step]
        if action != ground_truth:
            logger.debug(f"incorrect: {action} != {ground_truth} in step {step}")
            self.cur_success = False

    def check_done(self):
        step = self.agent.task_step[self.cur_task]
        self.cur_execute_cnt += 1
        if not self.cur_success:
            self.agent.reset_cur_task(account=False)
            return
        if step == len(self.agent.task_trajectory[self.cur_task]):
            self.execute_cnt += self.cur_execute_cnt
            self.correct_task_cnt += 1
        else:
            self.cur_success = False
            logger.debug("incorrect: done mismatch")
        self.agent.reset_cur_task(account=self.cur_success)

def main(args):
    agent = MybenchAgent(MybenchTasks(args.data_path))
    env = MybenchEnvironment(agent)
    tree = ActionTree(env, agent, Action, done=lambda a: a.name == 'done',
                      mode=MatchMode.FUZZY,
                      embedder_config={
                          "path": args.embedder_path
                      },
                      reranker_config={
                          "path": args.reranker_path
                      })

    app_task_trajectories = agent.tasks.get_app_task_trajectories()
    records = []
    for app, task_trajectories in app_task_trajectories.items():
        tree.clear()
        tasks = [t for t, _ in task_trajectories]
        random.shuffle(tasks)
        redistributed_tasks = []
        if args.distribution == 'uniform':
            redistributed_tasks = tasks
        elif args.distribution == 'power_law':
            num_task20 = math.ceil(0.2 * len(tasks))
            task20 = tasks[:num_task20]
            task80 = tasks[num_task20:]
            redistributed_tasks = task20 * 16 + task80
            random.shuffle(redistributed_tasks)
            redistributed_tasks = random.sample(redistributed_tasks, len(tasks))
        else:
            raise ValueError(f"Unknown distribution: {args.distribution}")
        for task in redistributed_tasks:
            logger.info(f"Current task: {task}")
            tree.execute(task)
            env.check_done()
            if not env.cur_success:
                tree.root.remove_task_trace(Task(task))
            agent.task_step[task] = -1
            env.reset_cur_task()
            env.total_task_cnt += 1
        logger.info(f"Current app: {app}")
        env.print_cnt()
        agent.print_cnt()
        logger.info(f"embedding time: {tree.embedding_counter}")
        if args.csv_path:
            records.append({
                "app": app,
                "total_task": env.total_task_cnt,
                "correct_task": env.correct_task_cnt,
                "total_actions": env.execute_cnt,
                "replayed_actions": env.execute_cnt - agent.generate_cnt,
                "total_embedding_time": tree.embedding_counter,
                "embedding_time_per_step": tree.embedding_counter / env.execute_cnt if env.execute_cnt > 0 else 0.0,
                "avg_steps": round(env.execute_cnt / env.total_task_cnt, 2) if env.total_task_cnt > 0 else 0.0,
                "correct_rate": round(env.correct_task_cnt / env.total_task_cnt, 2) if env.total_task_cnt > 0 else 0.0,
                "replay_rate": round(1 - agent.generate_cnt / env.execute_cnt, 2) if env.execute_cnt > 0 else 0.0
            })
        env.reset_cnt()
        agent.reset_cnt()
        tree.reset_counter()
    if args.csv_path:
        df = pd.DataFrame.from_records(records)
        df.to_csv(args.csv_path, index=False)
        logger.info(f"Results exported to {args.csv_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedder_path', type=str, required=True)
    parser.add_argument('--reranker_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--distribution', choices=['uniform', 'power_law'], default='uniform')
    parser.add_argument('--csv_path', type=str, required=False, default=None)
    args = parser.parse_args()
    main(args)
