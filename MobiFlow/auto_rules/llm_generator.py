import openai
import yaml
import logging
from typing import Dict, Any, Optional

from prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class LLMConfigGenerator:
    """基于LLM的配置生成器"""
    
    def __init__(self, 
                 api_key: str = None,
                 base_url: str = "https://api.openai.com/v1",
                 model: str = "gpt-4"):
        """
        初始化LLM配置生成器
        
        Args:
            api_key: OpenAI API密钥
            base_url: API基础URL
            model: 使用的模型名称
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        
        # 创建OpenAI客户端
        if api_key:
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url
            )
        else:
            self.client = None
    
    def generate_config_from_prompt(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        根据完整的提示词生成配置
        
        Args:
            prompt: 完整的LLM提示词
            
        Returns:
            生成的配置字典或原始文本
        """
        if not self.client:
            logger.warning("未提供API密钥，无法生成配置")
            return None
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=4000,
                timeout=40,  # 40秒超时
            )
            
            response_text = response.choices[0].message.content
            
            # 尝试解析为YAML
            config = self._parse_llm_response(response_text)
            
            if config:
                logger.info("LLM成功生成了配置")
                return config
            else:
                # 如果解析失败，返回原始文本
                logger.warning("LLM生成的内容无法解析为YAML，返回原始文本")
                return response_text
                
        except Exception as e:
            logger.error(f"LLM生成配置失败: {e}")
            return None
    
    def _parse_llm_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """解析LLM响应"""
        try:
            # 提取YAML部分
            if "```yaml" in response_text:
                yaml_start = response_text.find("```yaml") + 7
                yaml_end = response_text.find("```", yaml_start)
                yaml_text = response_text[yaml_start:yaml_end].strip()
            elif "```" in response_text:
                yaml_start = response_text.find("```") + 3
                yaml_end = response_text.find("```", yaml_start)
                yaml_text = response_text[yaml_start:yaml_end].strip()
            else:
                yaml_text = response_text.strip()
            
            # 解析YAML
            config = yaml.safe_load(yaml_text)
            
            # 简单验证配置格式
            if isinstance(config, dict) and 'task_id' in config and 'nodes' in config:
                # 格式化配置以确保数组字段使用方括号格式
                formatted_config = self._format_config_arrays(config)
                return formatted_config
            else:
                logger.warning("LLM生成的配置格式无效")
                return None
                
        except Exception as e:
            logger.error(f"解析LLM响应失败: {e}")
            return None
    
    def _format_config_arrays(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化配置中的数组字段，确保使用方括号格式
        
        Args:
            config: 原始配置字典
            
        Returns:
            格式化后的配置字典
        """
        def format_recursive(obj):
            if isinstance(obj, dict):
                formatted = {}
                for key, value in obj.items():
                    # 特定字段需要格式化为数组
                    if key in ['deps', 'next', 'all', 'any', 'any_of'] and isinstance(value, list):
                        formatted[key] = value
                    else:
                        formatted[key] = format_recursive(value)
                return formatted
            elif isinstance(obj, list):
                return [format_recursive(item) for item in obj]
            else:
                return obj
        
        return format_recursive(config)
