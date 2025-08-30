from train.task_template import get_app_task_trajectories
from agent.agent import Agent
from agent.env import Environment
import os
from action_cache.action import Action
from action_cache.tree import ActionTree, Task, MatchMode

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
        print(f"generate_cnt: {self.generate_cnt}")

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
        print(f"execute_cnt: {self.execute_cnt}, total_task_cnt: {self.total_task_cnt}, correct_task_cnt: {self.correct_task_cnt}")

    def get_agent_input(self, history, task_description):
        self.agent.task_step[task_description] += 1
        if self.cur_task != task_description:
            self.reset_cur_task()
            self.total_task_cnt += 1
        self.cur_task = task_description
        return {"task": task_description, "history": history}

    def execute(self, action):
        print(f"env executing: {action}")
        self.cur_execute_cnt += 1
        step = self.agent.task_step[self.cur_task]
        if step >= len(self.agent.task_trajectory[self.cur_task]):
            print(f"incorrect: expected done action, actual action is {action}")
            self.cur_success = False
            return
        ground_truth = self.agent.task_trajectory[self.cur_task][step]
        if action != ground_truth:
            print(f"incorrect: {action} != {ground_truth} in step {step}")
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
            print("incorrect: done mismatch")
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
    for app, task_trajectories in app_task_trajectories.items():
        print(f"Current app: {app}")
        tree.clear()
        for task, _ in task_trajectories:
            print(f"Current task: {task}")
            tree.execute(task)
            env.check_done()
            if not env.cur_success:
                tree.root.remove_task_trace(Task(task))
        env.print_cnt()
        agent.print_cnt()
        input("Press enter to continue")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedder_path', type=str, required=True)
    parser.add_argument('--reranker_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()
    main(args)
