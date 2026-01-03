"""Action parser for AutoGLM model responses."""

import ast
import re
from typing import Any, Dict, Tuple


def parse_action(response: str) -> Dict[str, Any]:
    """
    Parse action from model response.

    Args:
        response: Raw response string from the model.

    Returns:
        Parsed action dictionary with format:
        {
            "_metadata": "do" or "finish",
            "action": "Tap" | "Type" | ...,
            "element": [x, y],  # for Tap/Long Press/Double Tap
            "text": "...",      # for Type
            "start": [x, y],    # for Swipe
            "end": [x, y],      # for Swipe
            "app": "...",       # for Launch
            "message": "...",   # for finish
        }

    Raises:
        ValueError: If the response cannot be parsed.
    """
    try:
        response = response.strip()
        
        # ==================== 处理简写形式 ====================
        # [Back]、[Home] 等简写形式
        shorthand_map = {
            '[back]': {"_metadata": "do", "action": "Back"},
            '[home]': {"_metadata": "do", "action": "Home"},
        }
        # '[wait]': {"_metadata": "do", "action": "Wait", "duration": "2 seconds"},
        response_lower = response.lower()
        if response_lower in shorthand_map:
            return shorthand_map[response_lower]
        
        # 处理包含简写的响应：如 "[Back]do(action="Back")"
        for shorthand, action_dict in shorthand_map.items():
            if response_lower.startswith(shorthand):
                remaining = response[len(shorthand):].strip()
                if not remaining:  # 只有简写
                    return action_dict
                response = remaining  # 继续解析剩余部分
                break
        
        # ==================== 处理 Take_over 的各种格式 ====================
        # 格式1: do(action="Take_over", message="...")
        # 格式2: [{'type': 'Take_over", message="..."}
        # 格式3: Take_over(message="...")
        # if 'take_over' in response_lower or "type': 'take_over" in response_lower:
        #     # 尝试提取 message
        #     message = "User intervention required"
        #     if 'message=' in response:
        #         try:
        #             msg_start = response.find('message=') + len('message=')
        #             if msg_start < len(response) and response[msg_start] in ['"', "'"]:
        #                 quote = response[msg_start]
        #                 msg_end = response.find(quote, msg_start + 1)
        #                 if msg_end > msg_start:
        #                     message = response[msg_start + 1:msg_end]
        #         except Exception:
        #             pass
        #     return {"_metadata": "do", "action": "Take_over", "message": message}
        
        # ==================== 标准格式解析 ====================
        # Handle Type action with special text parsing
        if response.startswith('do(action="Type"') or response.startswith(
            'do(action="Type_Name"'
        ):
            text = response.split("text=", 1)[1][1:-2]
            action = {"_metadata": "do", "action": "Type", "text": text}
            return action
            
        elif response.startswith("do"):
            # Use AST parsing for safety
            try:
                # Escape special characters
                response = response.replace('\n', '\\n')
                response = response.replace('\r', '\\r')
                response = response.replace('\t', '\\t')

                tree = ast.parse(response, mode="eval")
                if not isinstance(tree.body, ast.Call):
                    raise ValueError("Expected a function call")

                call = tree.body
                action = {"_metadata": "do"}
                for keyword in call.keywords:
                    key = keyword.arg
                    value = ast.literal_eval(keyword.value)
                    action[key] = value

                return action
            except (SyntaxError, ValueError) as e:
                raise ValueError(f"Failed to parse do() action: {e}")

        elif response.startswith("finish"):
            action = {
                "_metadata": "finish",
                "message": response.replace("finish(message=", "")[1:-2],
            }
            return action
        
        else:
            raise ValueError(f"Failed to parse action: {response}")
            
    except Exception as e:
        raise ValueError(f"Failed to parse action: {e}")


def parse_response(content: str) -> Tuple[str, str]:
    """
    Parse the model response into thinking and action parts.

    Parsing rules:
    1. If content contains 'finish(message=', everything before is thinking,
       everything from 'finish(message=' onwards is action.
    2. If rule 1 doesn't apply but content contains 'do(action=',
       everything before is thinking, everything from 'do(action=' onwards is action.
    3. Fallback: If content contains '<answer>', use legacy parsing with XML tags.
    4. Otherwise, return empty thinking and full content as action.

    Args:
        content: Raw response content.

    Returns:
        Tuple of (thinking, action).
    """
    # Rule 1: Check for finish(message=
    if "finish(message=" in content:
        parts = content.split("finish(message=", 1)
        thinking = parts[0].strip()
        action = "finish(message=" + parts[1]
        return thinking, action

    # Rule 2: Check for do(action=
    if "do(action=" in content:
        parts = content.split("do(action=", 1)
        thinking = parts[0].strip()
        action = "do(action=" + parts[1]
        return thinking, action

    # Rule 3: Fallback to legacy XML tag parsing
    if "<answer>" in content:
        parts = content.split("<answer>", 1)
        thinking = parts[0].replace("<think>", "").replace("</think>", "").strip()
        action = parts[1].replace("</answer>", "").strip()
        return thinking, action

    # Rule 4: No markers found, return content as action
    return "", content
