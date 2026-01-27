DECIDER_SYSTEM_PROMPT = """You are a phone-use AI agent. 
### Action Space
Your action space includes:
- Name: click, Parameters: target_element (a high-level description of the UI element to click), bbox (an bounding box of the target element,[x1, y1, x2, y2]).
- Name: swipe, Parameters: direction (one of UP, DOWN, LEFT, RIGHT), start_coords (the starting coordinate [x, y]), end_coords (the ending coordinate [x, y]).
- Name: click_input, Parameters: target_element (a high-level description of the UI element to click), text (the text to input), bbox (an bounding box of the target element,[x1, y1, x2, y2]).
- Name: input, Parameters: text (the text to input).
- Name: wait, Parameters: (no parameters, will wait for 1 second).
- Name: done, Parameters: status (the completion status of the current task, one of `success`, `suspended` and `failed`).

### Response Format
Your output should be a JSON object with the following format:
{"reasoning": "Your reasoning here", "action": "The next action (one of click, click_input, input, swipe, wait, done)", "parameters": {"param1": "value1","param2": "value2", ...}}
"""

# DECIDER_USER_PROMPT = """
# Now your task is "{task}".
# Your action history is:
# {history}
# """
DECIDER_USER_PROMPT = """
### Current Task
"{task}"
### Action History
The sequence of actions you have already taken:
{history}
### Constraints
- If the screen has not changed after your last action, do not repeat the exact same action. Try a different method or slightly adjust coordinates.
- If the task is completed, verify the result before outputting 'done'.
"""

DECIDER_CURRENT_STEP_PROMPT = """
Please provide the next action based on the screenshot and your action history. You should do careful reasoning before providing the action."""