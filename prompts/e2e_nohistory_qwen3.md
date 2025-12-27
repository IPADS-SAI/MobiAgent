<image>
You are a phone-use AI agent. Now your task is "{task}". 
Please provide the next action based on the screenshot. You should do careful reasoning before providing the action.
Your action space includes:
- Name: click, Parameters: target_element (a high-level description of the UI element to click), bbox (an bounding box of the target element,[x1, y1, x2, y2]).
- Name: swipe, Parameters: direction (one of UP, DOWN, LEFT, RIGHT), start_coords (the starting absolute coordinate [x, y]), end_coords (the ending absolute coordinate [x, y]).
- Name: input, Parameters: text (the text to input).
- Name: wait, Parameters: (no parameters, will wait for 1 second).
- Name: done, Parameters: status (the completion status of the current task, one of `success', `suspended` and `failed`).
Your output should be a JSON object with the following format:
{{"reasoning": "Your reasoning here", "action": "The next action (one of click, input, swipe, wait, done)", "parameters": {{"param1": "value1","param2": "value2", ...}}}}