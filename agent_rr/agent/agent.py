from enum import Enum
import json
from openai import OpenAI
import io, base64

class Agent:
    def __init__(self):
        pass

    def generate(self, agent_input):
        pass


class ReplayLevel(Enum):
    ALL = 1
    REASONING = 2

class RemoteMultiLevelGeneralAgent(Agent):
    def __init__(self, decider_url, grounder_url):
        super().__init__()
        self.decider_client = OpenAI(api_key="0", base_url=decider_url)
        self.grounder_client = OpenAI(api_key="0", base_url=grounder_url)

    def generate(self, agent_input):
        if "replay_level" in agent_input:
            replay_level = agent_input["replay_level"]
        else:
            replay_level = ReplayLevel.ALL
        
        image = agent_input["image"]
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        query = agent_input["query"]

        action_dict = {}
        if replay_level == ReplayLevel.ALL:
            response = self.decider_client.chat.completions.create(
                model="",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                            {"type": "text", "text": query},
                        ],
                    }
                ],
                temperature=0
            )
            decider_response = response.choices[0].message.content
            decider_json = json.loads(decider_response)
            reasoning = decider_json["reasoning"]
            action = decider_json["action"]
            param = decider_json["parameters"]
            action_dict["name"] = action
            action_dict["parameters"] = param
            action_dict["extra"] = {"reasoning": reasoning, "decider_raw_output": decider_response}
            if action in ["click", "longclick"]:
                target_element = param["target_element"]
                query = '''
Based on the screenshot, user's intent and the description of the target UI element, provide the bounding box of the element using **absolute coordinates**.
User's intent: {reasoning}
Target element's description: {description}
Your output should be a JSON object with the following format:
{{"bbox": [x1, y1, x2, y2]}}'''.format(reasoning=reasoning, description=target_element)
            else:
                return action_dict
        
        # do grounding
        # case 1: a cache miss happened
        # case 2: replaying cached reasoning
        grounder_response = self.grounder_client.chat.completions.create(
            model="",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        {"type": "text", "text": query},
                    ],
                }
            ],
            temperature=0
        )
        grounder_response = grounder_response.choices[0].message.content
        grounder_json = json.loads(grounder_response)
        action_dict["parameters"]["bbox"] = grounder_json["bbox"]
        return action_dict
