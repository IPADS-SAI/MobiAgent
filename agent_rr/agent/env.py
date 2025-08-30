from PIL import Image
import base64
import io
import requests
import time

from .agent import ReplayLevel

class Environment:
    def __init__(self):
        pass

    def get_agent_input(self, history, task_description):
        pass

    def get_agent_input_speculative(self, history, task_description, draft_action):
        pass

    def execute(self, action):
        pass


def request_screenshot(url):
    body = {"action": "screenshot", "param": {}}
    response = requests.post(url, json=body)
    if response.status_code == 200:
        encoded_image = response.json()['data']['image']
        image_data = base64.b64decode(encoded_image)
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    else:
        return None

class MultiLevelGeneralEnvironment(Environment):
    def __init__(self, agent, replay_level=ReplayLevel.ALL):
        super().__init__()
        self.agent = agent
        self.replay_level = replay_level
        self.decider_prompt_fmt = '''
You are a phone-use AI agent. Now your task is "{task}".
Your action history is:
{history}
Please provide the next action based on the screenshot and your action history. You should do careful reasoning before providing the action.
Your action space includes:
- Name: click, Parameters: target_element (a high-level description of the UI element to click).
- Name: swipe, Parameters: direction (one of UP, DOWN, LEFT, RIGHT).
- Name: input, Parameters: text (the text to input).
- Name: wait, Parameters: (no parameters, will wait for 1 second).
- Name: done, Parameters: (no parameters).
Your output should be a JSON object with the following format:
{{"reasoning": "Your reasoning here", "action": "The next action (one of click, input, swipe, wait, done)", "parameters": {{"param1": "value1", ...}}}}'''
        if replay_level == ReplayLevel.REASONING:
            self.grounder_prompt_fmt = '''
Based on the screenshot, user's intent and the description of the target UI element, provide the bounding box of the element using **absolute coordinates**.
User's intent: {reasoning}
Target element's description: {description}
Your output should be a JSON object with the following format:
{{"bbox": [x1, y1, x2, y2]}}'''

    def get_screenshot(self):
        pass

    def get_agent_input(self, history, task_description):
        image = self.get_screenshot()
        if len(history) == 0:
            history_str = "(No history)"
        else:
            history_str = "\n".join(f"{idx}. {action.extra['decider_raw_output']}" for idx, action in enumerate(history, 1))
        query = self.decider_prompt_fmt.format(task=task_description, history=history_str)
        return {"image": image, "query": query}
    
    def execute(self, action):
        pass

class RemoteMultiLevelGeneralEnvironment(MultiLevelGeneralEnvironment):
    def __init__(self, agent, replay_level=ReplayLevel.ALL, url="http://localhost:8766/adb"):
        super().__init__(agent, replay_level)
        self.url = url
        self.last_screenshot = None
        self.factor = 0.5

    def get_screenshot(self):
        image = request_screenshot(self.url)

        if image is not None:
            width, height = image.size
            new_width = int(width * self.factor)
            new_height = int(height * self.factor)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        self.last_screenshot = image
        return image
    
    def execute(self, action):
        name = action.name
        if name in ["click", "longclick"] and self.replay_level == ReplayLevel.REASONING:
            query = self.grounder_prompt_fmt.format(
                reasoning=action.extra['reasoning'],
                description=action.param['target_element']
            )
            agent_input = {
                "image": self.last_screenshot,
                "query": query,
                "replay_level": self.replay_level
            }
            agent_output = self.agent.generate(agent_input)
            action = action.__class__(**agent_output)
        if name in ["click", "longclick"]:
            x1, y1, x2, y2 = action.param["bbox"]
            x = int((x1 + x2) / 2 / self.factor)
            y = int((y1 + y2) / 2 / self.factor)
            param = {"x": x, "y": y}
        elif name == "input":
            param = {"text": action.param["text"]}
        elif name == "scroll":
            direction = action.param["direction"]
            if direction == "DOWN":
                y1, y2 = 300, 700
            elif direction == "UP":
                y1, y2 = 700, 300
            x1, x2 = 500, 500
            x1, x2 = map(lambda x: x / 1000 * self.screenshot_width, [x1, x2])
            y1, y2 = map(lambda y: y / 1000 * self.screenshot_height, [y1, y2])
            param = {"x1": x1, "x2": x2, "y1": y1, "y2": y2}
            name = "scroll"
        else:
            name = ""

        if name:
            body = {"action": name, "param": param}
            requests.post(self.url, json=body)
        time.sleep(1)
