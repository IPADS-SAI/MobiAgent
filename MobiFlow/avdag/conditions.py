from __future__ import annotations
from typing import Dict, Callable, Any, List, Optional
import re
import numpy as np

from .types import Frame, VerifierOptions
from .logger import get_condition_logger

class ConditionChecker:
    def check(self, frame: Frame, params: Dict[str, Any], options: Optional[VerifierOptions] = None) -> bool:
        raise NotImplementedError

_REGISTRY: Dict[str, ConditionChecker] = {}

def register_condition(name: str):
    def _wrap(cls):
        _REGISTRY[name] = cls()
        return cls
    return _wrap


def get_checker(name: str) -> ConditionChecker:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown condition type: {name}")
    return _REGISTRY[name]


@register_condition("text_match")
class TextMatch(ConditionChecker):
    def check(self, frame: Frame, params: Dict[str, Any], options: Optional[VerifierOptions] = None) -> bool:
        text = str(frame.get("text", ""))
        any_words: List[str] = params.get("any", [])
        all_words: List[str] = params.get("all", [])
        if any_words and not any(w in text for w in any_words):
            return False
        if all_words and not all(w in text for w in all_words):
            return False
        return bool(any_words or all_words)


@register_condition("regex_match")
class RegexMatch(ConditionChecker):
    def check(self, frame: Frame, params: Dict[str, Any], options: Optional[VerifierOptions] = None) -> bool:
        text = str(frame.get("text", ""))
        pattern = params.get("pattern")
        if not pattern:
            return False
        flags = 0
        if params.get("ignore_case"):
            flags |= re.IGNORECASE
        return re.search(pattern, text, flags) is not None


@register_condition("ui_flag")
class UIFlag(ConditionChecker):
    def check(self, frame: Frame, params: Dict[str, Any], options: Optional[VerifierOptions] = None) -> bool:
        ui = frame.get("ui", {}) or {}
        key = params.get("key")
        if key is None:
            return False
        value = ui.get(key)
        if "equals" in params:
            return value == params["equals"]
        if "in" in params and isinstance(params["in"], list):
            return value in params["in"]
        return value is not None


@register_condition("xml_text_match")
class XmlTextMatch(ConditionChecker):
    def check(self, frame: Frame, params: Dict[str, Any], options: Optional[VerifierOptions] = None) -> bool:
        xml_text = str((frame.get("xml_text") or frame.get("xml") or ""))
        any_words: List[str] = params.get("any", [])
        all_words: List[str] = params.get("all", [])
        if any_words and not any(w in xml_text for w in any_words):
            return False
        if all_words and not all(w in xml_text for w in all_words):
            return False
        return bool(any_words or all_words)


@register_condition("escalate")
class EscalateChecker(ConditionChecker):
    """按策略升级顺序尝试检查器，当任意一个检查器返回True时立即结束。

    严格按照 escalation_order 顺序执行：["text", "regex", "ui", "action", "dynamic_match", "ocr", "llm"]
    
    params 可包含各种子条件配置：
    - text: 文本匹配参数
    - regex: 正则表达式匹配参数
    - ui: UI状态检查参数
    - action: 动作匹配参数
    - dynamic_match: 动态匹配参数
    - ocr: OCR识别参数
    - llm: LLM验证参数
    
    按照 escalation_order 顺序依次尝试，任意一个检查器返回 True 则立即返回 True。
    """

    def __init__(self):
        self._logger = get_condition_logger()

    def check(self, frame: Frame, params: Dict[str, Any], options: Optional[VerifierOptions] = None) -> bool:
        options = options or VerifierOptions()
        order = options.escalation_order
        
        # 如果强制LLM验证且配置了LLM和LLM条件，则只检查LLM
        if (options.force_llm_verification and 
            options.llm is not None and 
            params.get("llm") is not None):
            self._logger.debug(f"强制LLM验证模式，frame索引: {frame.get('_index', '?')}")
            ctx = {
                "frame": frame,
                "params": params["llm"],
                "options": options,  # 传递options给LLM函数
            }
            res = options.llm(ctx)
            self._logger.debug(f"LLM验证结果: {res}")
            return res is True

        # 严格按照 escalation_order 顺序执行检查器
        # self._logger.debug(f"升级顺序: {order}")
        self._logger.debug(f"配置的检查器: {list(params.keys())}")
        
        for checker_name in order:
            # 检查当前检查器是否在params中配置
            if params.get(checker_name) is not None:
                self._logger.debug(f"尝试检查器: {checker_name}")
                
                try:
                    result = False
                    
                    if checker_name == "text":
                        result = get_checker("text_match").check(frame, params["text"], options)
                        
                    elif checker_name == "regex":
                        result = get_checker("regex_match").check(frame, params["regex"], options)
                        
                    elif checker_name == "ui":
                        result = get_checker("ui_flag").check(frame, params["ui"], options)
                        
                    elif checker_name == "action":
                        # 处理 action 的两种配置方式
                        action_params = params["action"]
                        if isinstance(action_params, dict) and action_params.get("type") == "action_match":
                            result = get_checker("action_match").check(frame, action_params.get("params") or {}, options)
                        else:
                            result = get_checker("action_match").check(frame, action_params, options)
                            
                    # elif checker_name == "xml":
                    #     result = get_checker("xml_text_match").check(frame, params["xml"], options)
                        
                    elif checker_name == "dynamic_match":
                        result = get_checker("dynamic_match").check(frame, params["dynamic_match"], options)
                        
                    elif checker_name == "icons":
                        # 使用专门的图标检查器
                        self._logger.debug(f"调用图标检查器，frame索引: {frame.get('_index', '未知')}")
                        result = get_checker("icons_match").check(frame, params["icons"], options)
                        self._logger.debug(f"图标检查结果: {result}")
                        # 如果图标检查失败且未配置LLM，则直接返回结果
                        if not result and options.llm is None:
                            self._logger.debug(f"图标检查失败，未配置LLM，frame索引: {frame.get('_index', '未知')}")
                            return False
                        return result
                    elif checker_name == "ocr" and options.ocr is not None:
                        # 使用专门的OCR检查器
                        self._logger.debug(f"调用OCR检查器，frame索引: {frame.get('_index', '未知')}")
                        result = get_checker("ocr_match").check(frame, params["ocr"], options)
                        self._logger.debug(f"OCR检查结果: {result}")
                        # TODO: 当前暂时避免ocr检测任务不满足时，总是调用llm检测
                        # 若注释，则不管OCR一旦检测为不满足，都会继续尝试LLM验证
                        # return result
                    
                    elif checker_name == "llm" and options.llm is not None:
                        ctx = {
                            "frame": frame,
                            "params": params["llm"],
                            "options": options,  # 传递options给LLM函数
                        }
                        llm_result = options.llm(ctx)
                        result = llm_result is True
                    
                    self._logger.debug(f"{checker_name} 检查结果: {result}")
                    
                    # 如果当前检查器返回True，立即返回True（escalate的核心逻辑）
                    if result:
                        self._logger.debug(f"{checker_name} 检查成功，立即返回True")
                        return True
                        
                except Exception as e:
                    self._logger.warning(f"{checker_name} 检查器执行失败: {e}")
                    continue
            else:
                if checker_name in ["ocr", "icons"]:
                    self._logger.debug(f"跳过未配置的检查器: {checker_name}")

        # 所有配置的检查器都失败
        self._logger.debug("所有检查器都失败，返回False")
        return False


@register_condition("juxtaposition")
class JuxtapositionChecker(ConditionChecker):
    """并列检查器：要求所有配置的检查器都必须通过且结果一致。

    params 可包含多个子条件配置：
    - text / regex / ui / xml: 与对应基础检查器兼容的参数
    - action: 动作匹配参数
    - dynamic_match: 动态匹配参数
    - ocr: OCR识别参数
    - llm: LLM验证参数
    
    所有配置的检查器都必须返回 True，才认为该节点验证成功。
    """

    def __init__(self):
        self._logger = get_condition_logger()

    def check(self, frame: Frame, params: Dict[str, Any], options: Optional[VerifierOptions] = None) -> bool:
        options = options or VerifierOptions()
        
        # 收集所有配置的检查器及其结果
        configured_checkers = []
        results = []
        
        # 1) text 检查
        if params.get("text") is not None:
            configured_checkers.append("text_match")
            result = get_checker("text_match").check(frame, params["text"], options)
            results.append(result)
            self._logger.debug(f"text_match 结果: {result}")
            if not result:
                self._logger.debug("text_match 检查失败，跳过后续检查")
                return False
            
        # 2) regex 检查
        if params.get("regex") is not None:
            configured_checkers.append("regex_match")
            result = get_checker("regex_match").check(frame, params["regex"], options)
            results.append(result)
            self._logger.debug(f"regex_match 结果: {result}")
            if not result:
                self._logger.debug("regex_match 检查失败，跳过后续检查")
                return False
            
        # 3) ui 检查
        if params.get("ui") is not None:
            configured_checkers.append("ui_flag")
            result = get_checker("ui_flag").check(frame, params["ui"], options)
            results.append(result)
            self._logger.debug(f"ui_flag 结果: {result}")
            if not result:
                self._logger.debug("ui_flag 检查失败，跳过后续检查")
                return False
            
        # 4) action 检查
        if params.get("action") is not None:
            configured_checkers.append("action_match")
            # 处理嵌套的action配置
            action_params = params["action"]
            if isinstance(action_params, dict) and action_params.get("type") == "action_match":
                result = get_checker("action_match").check(frame, action_params.get("params") or {}, options)
            else:
                result = get_checker("action_match").check(frame, action_params, options)
            results.append(result)
            self._logger.debug(f"action_match 结果: {result}")
            if not result:
                self._logger.debug("action_match 检查失败，跳过后续检查")
                return False
            
        # 5) xml 检查
        if params.get("xml") is not None:
            configured_checkers.append("xml_text_match")
            result = get_checker("xml_text_match").check(frame, params["xml"], options)
            results.append(result)
            self._logger.debug(f"xml_text_match 结果: {result}")
            if not result:
                self._logger.debug("xml_text_match 检查失败，跳过后续检查")
                return False
            
        # 6) dynamic_match 检查
        if params.get("dynamic_match") is not None:
            configured_checkers.append("dynamic_match")
            result = get_checker("dynamic_match").check(frame, params["dynamic_match"], options)
            results.append(result)
            self._logger.debug(f"dynamic_match 结果: {result}")
            if not result:
                self._logger.debug("dynamic_match 检查失败，跳过后续检查")
                return False
            
        # 7) icons 检查
        if params.get("icons") is not None:
            configured_checkers.append("icons")
            # 使用专门的图标检查器
            icons_result = get_checker("icons_match").check(frame, params["icons"], options)
            results.append(icons_result)
            self._logger.debug(f"图标检测最终结果: {icons_result}")
            if not icons_result:
                self._logger.debug("图标检测失败，跳过后续检查")
                return False

        # 8) ocr 检查（需要 options.ocr 支持）
        if params.get("ocr") is not None and options.ocr is not None:
            configured_checkers.append("ocr")
            # 使用专门的OCR检查器
            ocr_result = get_checker("ocr_match").check(frame, params["ocr"], options)
            results.append(ocr_result)
            self._logger.debug(f"OCR最终结果: {ocr_result}")
            if not ocr_result:
                self._logger.debug("OCR 检查失败，跳过后续检查")
                return False

        # 9) llm 检查（需要 options.llm 支持）
        if params.get("llm") is not None and options.llm is not None:
            configured_checkers.append("llm")
            ctx = {
                "frame": frame,
                "params": params["llm"],
                "options": options,  # 传递options给LLM函数
            }
            llm_result = options.llm(ctx)
            results.append(llm_result is True)
            self._logger.debug(f"llm 结果: {llm_result is True}")

        # 检查是否至少配置了一个检查器
        if not configured_checkers:
            self._logger.warning("没有配置任何检查器")
            return False

        # 所有配置的检查器都必须返回 True
        final_result = all(results)
        self._logger.debug(f"配置的检查器: {configured_checkers}")
        self._logger.debug(f"各检查器结果: {results}")
        self._logger.debug(f"最终结果: {final_result}")
        
        return final_result


@register_condition("ocr_match")
class OCRMatch(ConditionChecker):
    """OCR匹配检查器，使用增强的文本处理功能"""
    
    def __init__(self):
        self._logger = get_condition_logger()
    
    def check(self, frame: Frame, params: Dict[str, Any], options: Optional[VerifierOptions] = None) -> bool:
        self._logger.debug("=====开始OCR匹配检查=====")
        self._logger.debug(f"frame索引: {frame.get('_index', '未知')}")
        self._logger.debug(f"检查params: {params}")
        self._logger.debug(f"options存在: {options is not None}")
        self._logger.debug(f"options.ocr存在: {options.ocr is not None if options else False}")
        
        # 初始化结果记录
        matched_keywords = []
        unmatched_keywords = []
        checker_result = ""
        
        if not options or not options.ocr:
            self._logger.warning("OCR选项不可用，返回False")
            checker_result = "OCR选项不可用"
            # 在frame中记录检查结果
            frame['_last_ocr_result'] = {
                'success': False,
                'reason': checker_result,
                'matched_keywords': matched_keywords,
                'unmatched_keywords': unmatched_keywords
            }
            return False
        
        # 获取OCR文本
        ocr_text = options.ocr(frame) or ""
        if not ocr_text.strip():
            checker_result = "OCR识别文本为空"
            frame['_last_ocr_result'] = {
                'success': False,
                'reason': checker_result,
                'matched_keywords': matched_keywords,
                'unmatched_keywords': unmatched_keywords
            }
            return False
        
        # self._logger.debug(f"OCR原始返回: {ocr_text[:100]}...")
        self._logger.debug(f"OCR原始返回: {ocr_text}")

        # 检查是否有缓存的处理结果
        processed_result = frame.get('_ocr_processed') or frame.get('_xml_processed')
        
        # 文本包含匹配 - any条件
        if "any" in params:
            any_keywords = params["any"]
            matched_any = []
            
            # 方式1：原始OCR文本检查
            for w in any_keywords:
                if w in ocr_text:
                    matched_any.append(w)
            
            if matched_any:
                self._logger.debug(f"any匹配(原始): {matched_any}")
                matched_keywords.extend(matched_any)
                checker_result = f"OCR识别成功，匹配关键词: {matched_any}"
                frame['_last_ocr_result'] = {
                    'success': True,
                    'reason': checker_result,
                    'matched_keywords': matched_keywords,
                    'unmatched_keywords': unmatched_keywords
                }
                return True
            
            # 方式2：智能匹配
            if processed_result:
                try:
                    from .ocr_processor import get_ocr_processor
                    processor = get_ocr_processor()
                    for keyword in any_keywords:
                        if processor.smart_text_contains(processed_result, keyword):
                            self._logger.debug(f"any匹配(智能): {keyword}")
                            matched_keywords.append(keyword)
                            checker_result = f"OCR智能匹配成功，匹配关键词: {keyword}"
                            frame['_last_ocr_result'] = {
                                'success': True,
                                'reason': checker_result,
                                'matched_keywords': matched_keywords,
                                'unmatched_keywords': unmatched_keywords
                            }
                            return True
                except ImportError:
                    self._logger.warning("OCRProcessor不可用，使用基础匹配")
            
            # 记录所有未匹配的any关键词
            unmatched_keywords.extend([w for w in any_keywords if w not in matched_any])
        
        # 文本包含匹配 - all条件
        if "all" in params:
            all_keywords = params["all"]
            matched_all = []
            
            # 方式1：原始OCR文本检查
            for w in all_keywords:
                if w in ocr_text:
                    matched_all.append(w)
            
            if len(matched_all) == len(all_keywords):
                self._logger.debug(f"all匹配(原始): {all_keywords}")
                matched_keywords.extend(matched_all)
                checker_result = f"OCR识别成功，匹配所有关键词: {all_keywords}"
                frame['_last_ocr_result'] = {
                    'success': True,
                    'reason': checker_result,
                    'matched_keywords': matched_keywords,
                    'unmatched_keywords': unmatched_keywords
                }
                return True
            # 方式2：智能匹配
            elif processed_result:
                try:
                    from .ocr_processor import get_ocr_processor
                    processor = get_ocr_processor()
                    smart_matched = []
                    for keyword in all_keywords:
                        if processor.smart_text_contains(processed_result, keyword):
                            smart_matched.append(keyword)
                    
                    if len(smart_matched) == len(all_keywords):
                        self._logger.debug(f"all匹配(智能): {all_keywords}")
                        matched_keywords.extend(smart_matched)
                        checker_result = f"OCR智能匹配成功，匹配所有关键词: {all_keywords}"
                        frame['_last_ocr_result'] = {
                            'success': True,
                            'reason': checker_result,
                            'matched_keywords': matched_keywords,
                            'unmatched_keywords': unmatched_keywords
                        }
                        return True
                    else:
                        # 记录智能匹配下未匹配的关键词
                        unmatched_keywords.extend([w for w in all_keywords if w not in smart_matched])
                        matched_keywords.extend(smart_matched)
                except ImportError:
                    self._logger.warning("OCRProcessor不可用，使用基础匹配")
                    # 记录原始匹配下未匹配的关键词
                    unmatched_keywords.extend([w for w in all_keywords if w not in matched_all])
                    matched_keywords.extend(matched_all)
            else:
                # 记录原始匹配下未匹配的关键词
                unmatched_keywords.extend([w for w in all_keywords if w not in matched_all])
                matched_keywords.extend(matched_all)
        
        # 正则匹配
        if "pattern" in params:
            pattern = params["pattern"]
            flags = re.IGNORECASE if params.get("ignore_case") else 0
            
            # 方式1：对原始文本应用正则
            if re.search(pattern, ocr_text, flags):
                self._logger.debug(f"正则匹配(原始): {pattern}")
                checker_result = f"OCR正则匹配成功，模式: {pattern}"
                frame['_last_ocr_result'] = {
                    'success': True,
                    'reason': checker_result,
                    'matched_keywords': matched_keywords,
                    'unmatched_keywords': unmatched_keywords
                }
                return True
            # 方式2：对处理后的文本格式应用正则
            elif processed_result:
                for text_format in [processed_result.cleaned, processed_result.no_spaces, ' '.join(processed_result.words)]:
                    if text_format and re.search(pattern, text_format, flags):
                        self._logger.debug(f"正则匹配(处理): {pattern} -> {text_format[:50]}...")
                        checker_result = f"OCR智能正则匹配成功，模式: {pattern}"
                        frame['_last_ocr_result'] = {
                            'success': True,
                            'reason': checker_result,
                            'matched_keywords': matched_keywords,
                            'unmatched_keywords': unmatched_keywords
                        }
                        return True
            
            # 正则匹配失败，记录模式
            unmatched_keywords.append(f"pattern: {pattern}")
        
        # 构建失败原因
        if unmatched_keywords:
            checker_result = f"OCR识别失败，未匹配关键词: {unmatched_keywords}"
            self._logger.debug(f"未匹配的关键词: {unmatched_keywords}")
        else:
            checker_result = "OCR识别失败，无匹配条件"
        
        if processed_result:
            self._logger.debug(f"check keywords: any: {params.get('any', [])} / all: {params.get('all', [])}")
            # self._logger.debug(f"处理文本格式 - 清理: {processed_result.cleaned[:50]}...")
            # self._logger.debug(f"处理文本格式 - 无空格: {processed_result.no_spaces[:50]}...")
            self._logger.debug(f"处理文本格式 - 清理: {processed_result.cleaned}")
            self._logger.debug(f"处理文本格式 - 无空格: {processed_result.no_spaces}")
            self._logger.debug(f"处理文本格式 - 词语数: {len(processed_result.words)}")
        
        # 记录失败结果
        frame['_last_ocr_result'] = {
            'success': False,
            'reason': checker_result,
            'matched_keywords': matched_keywords,
            'unmatched_keywords': unmatched_keywords
        }
        
        return False

__all__ = ["ConditionChecker", "register_condition", "get_checker"]


@register_condition("action_match")
class ActionMatch(ConditionChecker):
    def check(self, frame: Frame, params: Dict[str, Any], options: Optional[VerifierOptions] = None) -> bool:
        act = frame.get("action") or {}
        if not isinstance(act, dict):
            return False
        t = params.get("type")
        if t and act.get("type") != t:
            return False
        contains: Dict[str, Any] = params.get("contains") or {}
        for k, v in contains.items():
            if act.get(k) != v:
                return False
        return True if (t or contains) else False


@register_condition("dynamic_match")
class DynamicMatchChecker(ConditionChecker):
    """基于动态配置的通用匹配检查器，支持从任务描述中提取关键信息进行匹配"""
    
    def check(self, frame: Frame, params: Dict[str, Any], options: Optional[VerifierOptions] = None) -> bool:
        """
        动态匹配检查器，支持多种匹配策略：
        
        params 支持的配置：
        - extract_from: 指定从哪个字段提取信息 (task_description, reasoning等)
        - condition_patterns: 条件模式映射，每个模式包含匹配关键词和对应的验证关键词
        - verification_fields: 验证字段列表，指定在哪些字段中查找验证关键词
        - fallback_llm: 当基础匹配不确定时是否使用LLM验证
        """
        extract_from = params.get("extract_from", "task_description")
        condition_patterns = params.get("condition_patterns", {})
        verification_fields = params.get("verification_fields", ["reasoning", "text"])
        
        # 提取源文本
        source_text = frame.get(extract_from, "").lower()
        if not source_text:
            return False
        
        # 找到匹配的条件模式
        matched_condition = None
        for condition_name, pattern_config in condition_patterns.items():
            trigger_keywords = pattern_config.get("trigger_keywords", [])
            if any(keyword.lower() in source_text for keyword in trigger_keywords):
                matched_condition = condition_name
                break
        
        if not matched_condition:
            return False
        
        # 获取对应的验证关键词
        pattern_config = condition_patterns[matched_condition]
        verify_keywords = pattern_config.get("verify_keywords", [])
        
        # 在验证字段中查找关键词
        for field in verification_fields:
            field_text = frame.get(field, "").lower()
            if any(keyword.lower() in field_text for keyword in verify_keywords):
                return True
        
        # 如果基础匹配失败，使用LLM作为后备验证
        if params.get("fallback_llm") and options and options.llm:
            llm_prompt = pattern_config.get("llm_prompt") or f"该步骤是否执行了与'{matched_condition}'相关的操作？"
            ctx = {
                "frame": frame,
                "params": {
                    "prompt": llm_prompt,
                    "expected_true": True
                },
                "options": options,  # 传递options给LLM函数
            }
            return options.llm(ctx) is True
        
        return False


@register_condition("icons_match")
class IconsMatch(ConditionChecker):
    """图标匹配检查器，使用图像模板匹配检测图标是否存在"""
    
    def __init__(self):
        self._logger = get_condition_logger()
        self._detection_service = None
    
    def _get_detection_service(self):
        """延迟导入图标检测服务"""
        if self._detection_service is None:
            try:
                # 延迟导入避免循环依赖
                import sys
                from pathlib import Path
                project_root = Path(__file__).parent.parent
                sys.path.insert(0, str(project_root))
                
                from tools.Icon_detection import get_icon_detection_service
                self._detection_service = get_icon_detection_service()
                self._logger.debug("图标检测服务初始化成功")
            except Exception as e:
                self._logger.error(f"初始化图标检测服务失败: {e}")
                self._detection_service = None
        return self._detection_service
    
    def _extract_image_from_frame(self, frame: Frame) -> Optional[np.ndarray]:
        """从frame中提取图像数据"""
        # 检查frame中可能的图像字段,当前是使用frame(字典)中的img存储图像文件的完整路径
        image_fields = ['img', 'screenshot', 'image', 'frame_image', 'screen']
        
        for field in image_fields:
            if field in frame and frame[field] is not None:
                image_data = frame[field]
                
                # 如果是文件路径
                if isinstance(image_data, str):
                    try:
                        import cv2
                        img = cv2.imread(image_data)
                        if img is not None:
                            self._logger.debug(f"从路径加载图像: {image_data}")
                            return img
                    except Exception as e:
                        self._logger.warning(f"从路径加载图像失败 {image_data}: {e}")
                        continue
                
                # 如果是numpy数组
                elif isinstance(image_data, np.ndarray):
                    self._logger.debug(f"从字段 {field} 获取图像数据，形状: {image_data.shape}")
                    return image_data
                
                # 如果是字节数据
                elif isinstance(image_data, (bytes, bytearray)):
                    try:
                        import cv2
                        nparr = np.frombuffer(image_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if img is not None:
                            self._logger.debug(f"从字节数据解码图像成功")
                            return img
                    except Exception as e:
                        self._logger.warning(f"从字节数据解码图像失败: {e}")
                        continue
        
        self._logger.warning("未在frame中找到有效的图像数据")
        return None
    
    def check(self, frame: Frame, params: Dict[str, Any], options: Optional[VerifierOptions] = None) -> bool:
        self._logger.debug("=====开始图标匹配检查=====")
        self._logger.debug(f"frame索引: {frame.get('_index', '未知')}")
        self._logger.debug(f"检查params: {params}")
        
        # 初始化结果记录
        matched_icons = []
        unmatched_icons = []
        checker_result = ""
        
        # 获取图标检测服务
        detection_service = self._get_detection_service()
        if detection_service is None:
            checker_result = "图标检测服务不可用"
            frame['_last_icons_result'] = {
                'success': False,
                'reason': checker_result,
                'matched_icons': matched_icons,
                'unmatched_icons': unmatched_icons
            }
            self._logger.error(checker_result)
            return False
        
        # 提取图像数据
        image = self._extract_image_from_frame(frame)
        if image is None:
            checker_result = "无法从frame中提取图像数据"
            frame['_last_icons_result'] = {
                'success': False,
                'reason': checker_result,
                'matched_icons': matched_icons,
                'unmatched_icons': unmatched_icons
            }
            self._logger.error(checker_result)
            return False
        
        # 获取应用ID（用于确定图标搜索路径）
        app_id = frame.get('app_id') or frame.get('package_name') or frame.get('app_name')
        
        # 获取相似度阈值
        threshold = params.get('threshold')
        
        # 处理any条件
        if "any" in params:
            any_icons = params["any"]
            if not isinstance(any_icons, list):
                any_icons = [any_icons]
            
            self._logger.debug(f"检查any图标: {any_icons}")
            
            result = detection_service.detect_icons(
                image, 
                any_icons, 
                app_id, 
                threshold, 
                match_mode='any'
            )
            
            if result['success']:
                matched_icons.extend(result['matched_icons'])
                self._logger.debug(f"any图标匹配成功: {result['matched_icons']}")
                # 记录成功结果
                frame['_last_icons_result'] = {
                    'success': True,
                    'reason': f"成功匹配图标: {result['matched_icons']}",
                    'matched_icons': result['matched_icons'],
                    'unmatched_icons': result['unmatched_icons'],
                    'details': result['details']
                }
                return True
            else:
                unmatched_icons.extend(result['unmatched_icons'])
                self._logger.debug(f"any图标匹配失败: {result['unmatched_icons']}")
        
        # 处理all条件
        if "all" in params:
            all_icons = params["all"]
            if not isinstance(all_icons, list):
                all_icons = [all_icons]
                
            self._logger.debug(f"检查all图标: {all_icons}")
            
            result = detection_service.detect_icons(
                image,
                all_icons,
                app_id,
                threshold,
                match_mode='all'
            )
            
            if result['success']:
                matched_icons.extend(result['matched_icons'])
                self._logger.debug(f"all图标匹配成功: {result['matched_icons']}")
                # 记录成功结果
                frame['_last_icons_result'] = {
                    'success': True,
                    'reason': f"成功匹配所有图标: {result['matched_icons']}",
                    'matched_icons': result['matched_icons'],
                    'unmatched_icons': result['unmatched_icons'],
                    'details': result['details']
                }
                return True
            else:
                unmatched_icons.extend(result['unmatched_icons'])
                self._logger.debug(f"all图标匹配失败: {result['unmatched_icons']}")
        
        # 构建失败原因
        if unmatched_icons:
            checker_result = f"图标检测失败，未匹配图标: {unmatched_icons}"
        else:
            checker_result = "图标检测失败，无匹配条件"
        
        # 记录失败结果
        frame['_last_icons_result'] = {
            'success': False,
            'reason': checker_result,
            'matched_icons': matched_icons,
            'unmatched_icons': unmatched_icons
        }
        
        self._logger.debug(checker_result)
        return False
