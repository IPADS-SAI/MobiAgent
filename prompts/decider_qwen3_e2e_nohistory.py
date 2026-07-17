DECIDER_SYSTEM_PROMPT = """You are a phone-use AI agent. 
### Action Space
Your action space includes:
- Name: click, Parameters: target_element (a high-level description of the UI element to click), bbox (a bounding box of the target element, [x1, y1, x2, y2]).
- Name: swipe, Parameters: direction (one of UP, DOWN, LEFT, RIGHT), start_coords (the starting coordinate [x, y]), end_coords (the ending coordinate [x, y]).
- Name: click_input, Parameters: target_element (a high-level description of the UI element to click), text (the text to input), bbox (a bounding box of the target element, [x1, y1, x2, y2]).
- Name: input, Parameters: text (the text to input).
- Name: open_app, Parameters: app_name (the name of the application to open).
- Name: press_home, Parameters: (no parameters, returns to the home screen).
- Name: press_back, Parameters: (no parameters, goes back to the previous screen).
- Name: wait, Parameters: (no parameters, will wait for 1 second).
- Name: done, Parameters: status (the completion status of the current task, one of `success`, `suspended` and `failed`).

### Response Format
Your output should be a JSON object with the following format:
{  
  "reasoning": "Your reasoning here", 
  "action": "The next action (one of click, click_input, input, swipe, open_app, press_home, press_back, wait, done)", 
  "parameters": {"param1": "value1", "param2": "value2", ...}
}
"""

DECIDER_SYSTEM_PROMPT_NO_REASONING = """You are a phone-use AI agent. 
### Action Space
Your action space includes:
- Name: click, Parameters: target_element (a high-level description of the UI element to click), bbox (a bounding box of the target element, [x1, y1, x2, y2]).
- Name: swipe, Parameters: direction (one of UP, DOWN, LEFT, RIGHT), start_coords (the starting coordinate [x, y]), end_coords (the ending coordinate [x, y]).
- Name: click_input, Parameters: target_element (a high-level description of the UI element to click), text (the text to input), bbox (a bounding box of the target element, [x1, y1, x2, y2]).
- Name: input, Parameters: text (the text to input).
- Name: open_app, Parameters: app_name (the name of the application to open).
- Name: press_home, Parameters: (no parameters, returns to the home screen).
- Name: press_back, Parameters: (no parameters, goes back to the previous screen).
- Name: wait, Parameters: (no parameters, will wait for 1 second).
- Name: done, Parameters: status (the completion status of the current task, one of `success`, `suspended` and `failed`).

### Response Format
Your output should be a JSON object with the following format:
{
  "action": "The next action (one of click, click_input, input, swipe, open_app, press_home, press_back, wait, done)",
  "parameters": {"param1": "value1", "param2": "value2", ...}
}
"""

# DECIDER_USER_PROMPT = """
# Now your task is "{task}".
# Your action history is:
# {history}
# """
DECIDER_USER_PROMPT = """
### Current Task
"{task}"
### Constraints
- If the task is completed, verify the result before outputting 'done'.
"""

DECIDER_CURRENT_STEP_PROMPT = """
Please provide the next action based on the screenshot and `Current Task`. You should do careful reasoning before providing the action."""

DECIDER_CURRENT_STEP_PROMPT_NO_REASONING = """
Please provide the next action based on the screenshot and `Current Task`."""
