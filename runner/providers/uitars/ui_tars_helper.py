import re
import logging
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 内置提示词模板
PROMPT_TEMPLATE = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(point='<point>x1 y1</point>')
long_press(point='<point>x1 y1</point>')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
press_home()
press_back()
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.

## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.
- To open an app, use click() to tap on the app icon you can see in the screenshot, don't use open_app().
- Always look for app icons, buttons, or UI elements in the current screenshot and click on them.

## User Instruction
{instruction}"""

class UITarsHelper:
    @staticmethod
    def parse_response(response: str, image_width: int, image_height: int, model_name: str = "UI-TARS-1.5-7B") -> Tuple[str, str, List[Dict]]:
        """
        解析模型响应
        Returns:
            (thought, raw_action, parsed_actions_list)
        """
        try:
            # 由ui_tars添加到路径中
            import ui_tars.action_parser as ui_tars_parser
        except ImportError:
            logger.error("未能导入utils(UI-TARS)。请确保sys.path正确。")
            raise
        
        # 计算smart resize尺寸
        smart_height, smart_width = ui_tars_parser.smart_resize(
            image_height, image_width, 
            factor=ui_tars_parser.IMAGE_FACTOR
        )

        # 使用官方解析
        actions = ui_tars_parser.parse_action_to_structure_output(
            response,
            factor=ui_tars_parser.IMAGE_FACTOR,
            origin_resized_height=smart_height,
            origin_resized_width=smart_width,
            model_type="qwen25vl"
        )
        
        if not actions:
            logger.warning("未解析到结构化动作。")
            return "", response, []

        action = actions[0]
        thought = action.get('thought', '')
        raw_action = response.split("Action:")[-1].strip().split('\n')[0] if "Action:" in response else response

        # 转换为PyAutoGUI代码（用于日志）
        pyautogui_code = ui_tars_parser.parsing_response_to_pyautogui_code(
            action, image_height, image_width
        )
        logger.debug(f"PyAutoGUI代码: {pyautogui_code}")

        # 转换为内部格式
        internal_action = UITarsHelper._convert_action_to_internal(action, pyautogui_code, image_width, image_height, model_name)
        
        return thought, raw_action, [internal_action]

    @staticmethod
    def _convert_action_to_internal(action: Dict, pyautogui_code: str, width: int, height: int, model_name: str) -> Dict:
        """
        转换为框架的标准动作格式 {type: ..., params: ...}
        """
        action_type = action.get('action_type', '')
        action_inputs = action.get('action_inputs', {})
        raw_text = action.get('text', '')
        
        # 从原始文本解析(x,y)的辅助函数，支持多种格式
        def extract_coords(text, *keys):
            for key in keys:
                # 模式1: key='(x,y)'
                match = re.search(f"{key}=['\"]\((\d+),(\d+)\)['\"]", text)
                if match: return int(match.group(1)), int(match.group(2))
                match = re.search(f"{key}=\((\d+),(\d+)\)", text)
                if match: return int(match.group(1)), int(match.group(2))
                
                # 模式2: key='<point>x y</point>'
                match = re.search(f"{key}=['\"]<point>(\d+)\s+(\d+)</point>['\"]", text)
                if match: return int(match.group(1)), int(match.group(2))
            return None

        # UI-TARS-1.5-7B特殊处理
        # 使用返回的绝对坐标
        if model_name == "UI-TARS-1.5-7B":
            # 点击/长按
            if action_type in ["click", "left_single", "left_double", "right_single", "hover", "long_press"]:
                coords = extract_coords(raw_text, "start_box", "point")
                if coords:
                    logger.info(f"解析的绝对坐标: {coords}")
                    return {'type': 'click', 'params': {'points': [coords[0], coords[1]]}}

            # 拖拽/滑动
            elif action_type == "drag":
                start = extract_coords(raw_text, "start_box", "start_point")
                end = extract_coords(raw_text, "end_box", "end_point")
                if start and end:
                    return {
                        'type': 'swipe', 
                        'params': {'points': [start[0], start[1], end[0], end[1]]}
                    }

            # 滚动
            elif action_type == "scroll":
                 direction = action_inputs.get('direction', 'down')
                 coords = extract_coords(raw_text, "start_box", "point")
                 params = {'direction': direction}
                 if coords:
                     pass
                 return {'type': 'scroll', 'params': params}

            # 输入/类型
            elif action_type == "type":
                content = action_inputs.get('content', '')
                return {'type': 'input', 'params': {'text': content}}

            # 完成
            elif action_type == "finished":
                content = action_inputs.get('content', 'success')
                return {'type': 'done', 'params': {'status': 'success', 'message': content}}

            # 首页/返回
            elif action_type == "press_home":
                return {'type': 'home', 'params': {}}
            elif action_type == "press_back":
                return {'type': 'back', 'params': {}}

        # 其他模型或7B解析未命中的备选方案
        
        # 内部映射结果
        result_type = None
        result_params = {}

        if pyautogui_code.strip() == "DONE":
             content = action_inputs.get('content', 'success')
             return {'type': 'done', 'params': {'status': 'success', 'message': content}}

        if action_type in ["click", "long_press"]:
            click_match = re.search(r'pyautogui\.click\((\d+(?:\.\d+)?), (\d+(?:\.\d+)?)', pyautogui_code)
            if click_match:
                x = float(click_match.group(1))
                y = float(click_match.group(2))
                
                if model_name == "UI-TARS-1.5":
                     if 0 <= x <= 1 and 0 <= y <= 1:
                        x = int(x * width)
                        y = int(y * height)
                     else:
                        x, y = int(x), int(y)
                elif 0 <= x <= 1 and 0 <= y <= 1:
                     x = int(x * width)
                     y = int(y * height)
                else: 
                     x, y = int(x), int(y)
                
                result_type = 'click'
                result_params = {'points': [int(x), int(y)]}

        elif action_type == "type":
             content = action_inputs.get('content', '')
             result_type = 'input'
             result_params = {'text': content}
             
        elif action_type == "press_home":
             result_type = 'home'
        elif action_type == "press_back":
             result_type = 'back'
        elif action_type == "finished":
             result_type = 'done'
             result_params = {'status': 'success'}

        elif action_type == "scroll":
            scroll_match = re.search(r'pyautogui\.scroll\((-?\d+)', pyautogui_code)
            direction = 'up'
            if scroll_match:
                val = int(scroll_match.group(1))
                direction = 'up' if val < 0 else 'down'
            result_type = 'swipe'
            result_params = {'direction': direction}
        elif action_type == "drag":
            move_match = re.search(r'pyautogui\.moveTo\((\d+(?:\.\d+)?), (\d+(?:\.\d+)?)', pyautogui_code)
            drag_match = re.search(r'pyautogui\.dragTo\((\d+(?:\.\d+)?), (\d+(?:\.\d+)?)', pyautogui_code)
            if move_match and drag_match:
                sx = float(move_match.group(1))
                sy = float(move_match.group(2))
                ex = float(drag_match.group(1))
                ey = float(drag_match.group(2))
                result_type = 'swipe'
                result_params = {'points': [int(sx), int(sy), int(ex), int(ey)]}
        if result_type:
            return {'type': result_type, 'params': result_params}
            
        return {'type': 'wait', 'params': {}}
