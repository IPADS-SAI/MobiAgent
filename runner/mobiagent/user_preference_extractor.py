"""
用户偏好提取器
基于任务执行记录异步提取用户偏好，存储到Mem0中
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    from mem0 import MemoryClient, Memory
except ImportError:
    logging.warning("mem0 not installed, user preference extraction will be disabled")
    MemoryClient = None
    Memory = None

# 全局配置
USER_ID = "default_user"

class PreferenceExtractor:
    """用户偏好提取器"""
    
    def __init__(self, planner_client=None, planner_model="", use_graphrag=False):
        """
        初始化偏好提取器
        
        Args:
            planner_client: OpenAI客户端，用于调用LLM模型
            planner_model: 模型名称
            use_graphrag: 是否使用GraphRAG
        """
        self.planner_client = planner_client
        self.planner_model = planner_model
        self.use_graphrag = use_graphrag
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        # 初始化Mem0客户端
        if MemoryClient and Memory:
            try:
                self.config = self._get_mem0_config()
                self.mem = Memory.from_config(config_dict=self.config)
                logging.info("Mem0 client initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize Mem0 client: {e}")
                self.mem = None
        else:
            self.mem = None
            logging.warning("Mem0 not available, preference extraction disabled")
    
    def __del__(self):
        """析构函数，确保线程池正确关闭"""
        if hasattr(self, 'executor') and not self.executor._shutdown:
            logging.info("Waiting for preference extraction tasks to complete...")
            self.executor.shutdown(wait=True)  # 等待所有任务完成
            logging.info("All preference extraction tasks completed")
    
    def _get_mem0_config(self):
        """
        获取GraphRAG配置
        
        Returns:
            GraphRAG配置字典
        """
        # 从配置文件读取GraphRAG设置
        mem0_config = {
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": os.getenv('EMBEDDING_MODEL'),
                }
            },
            "vector_store": {
                "provider": "milvus",
                "config": {
                    "collection_name": "mobiagent",
                    "embedding_model_dims": os.getenv('EMBEDDING_MODEL_DIMS'),
                    "url": os.getenv('MILVUS_URL'),
                    "db_name": "default",
                    "token": "",
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": "gpt-4o-mini",
                    "api_key": os.getenv('OPENAI_API_KEY'),
                    "openai_base_url": os.getenv('OPENAI_BASE_URL'),
                }
            },
        }
        
        if self.use_graphrag:
            mem0_config["graph_store"] = {
                "provider": "neo4j",
                "config": {
                    "url": os.getenv('NEO4J_URL'),
                    "username": os.getenv('NEO4J_USERNAME'),
                    "password": os.getenv('NEO4J_PASSWORD'),
                },
            }
        
        return mem0_config
    
    def extract_async(self, task_data: Dict[str, Any]):
        """
        异步提取用户偏好
        
        Args:
            task_data: 任务数据，包含task_description, actions, reacts等
        """
        if not self.mem or not self.planner_client:
            return
            
        try:
            self.executor.submit(self._extract, task_data)
        except Exception as e:
            logging.error(f"Failed to submit preference extraction task: {e}")
    
    def _extract(self, task_data: Dict[str, Any]):
        """
        实际的偏好提取逻辑
        
        Args:
            task_data: 任务数据
        """
        try:
            # 使用LLM分析执行记录
            result = self._analyze_with_llm(task_data)
            
            if result:
                task_type = result.get("task_type", "general")
                preferences = result.get("preferences", {})
                
                # 存储所有偏好
                self._store_preferences(preferences, task_type)
                logging.info(f"Extracted preferences: {preferences}")
                
        except Exception as e:
            logging.error(f"Preference extraction failed: {e}")
    
    def _analyze_with_llm(self, task_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        使用LLM分析执行记录，提取任务类型和用户偏好
        
        Args:
            task_data: 任务数据
            
        Returns:
            包含task_type和preferences的字典
        """
        try:
            # 导入偏好模板
            from .preference_templates import identify_task_type, generate_preference_extraction_prompt
            
            # 识别任务类型
            task_type = identify_task_type(task_data['task_description'])
            
            # 生成针对性的prompt
            analysis_prompt = generate_preference_extraction_prompt(task_data, task_type)
            
            # 使用现有的planner模型
            response = self.planner_client.chat.completions.create(
                model=self.planner_model,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0
            )
            
            response_text = response.choices[0].message.content
            
            # 提取JSON部分
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text.strip()
            
            return json.loads(json_str)
            
        except Exception as e:
            logging.error(f"Failed to analyze with Qwen: {e}")
            return None
    
    def _store_preferences(self, preferences: Dict[str, str], task_type: str):
        """
        存储偏好到Mem0
        
        Args:
            preferences: 偏好字典
            task_type: 任务类型
        """
        if not self.mem:
            logging.warning("Mem0 client not available, cannot store preferences")
            return
            
        try:
            logging.info(f"Storing {len(preferences)} preferences for task_type: {task_type}")
            for key, value in preferences.items():
                if value and value.strip():  # 只要不是空值就存储
                    preference_text = f"用户偏好{key}：{value}"
                    
                    if self.use_graphrag:
                        # 使用GraphRAG存储，支持更丰富的元数据
                        logging.info(f"Storing with GraphRAG: {preference_text}")
                        result = self.mem.add(
                            preference_text, 
                            user_id=USER_ID,
                            metadata={
                                "type": "preference",
                                "task_type": task_type,
                                "preference_key": key,
                                "preference_value": value,
                                "confidence": 1.0,
                                "timestamp": time.time()
                            }
                        )
                        logging.info(f"GraphRAG storage result: {result}")
                    else:
                        # 使用普通Mem0存储
                        logging.info(f"Storing with vector search: {preference_text}")
                        result = self.mem.add(preference_text, metadata={
                            "type": "preference",
                            "task_type": task_type,
                            "user_id": USER_ID,
                            "timestamp": time.time()
                        })
                        logging.info(f"Vector storage result: {result}")
                    
                    logging.info(f"Stored preference: {preference_text}")
                else:
                    logging.info(f"Skipping empty preference: {key} = {value}")
                    
        except Exception as e:
            logging.error(f"Failed to store preferences: {e}")

    def clear_all_memories(self) -> int:
        """
        尝试清空当前 user 的所有记忆。返回成功删除的条数（尽力而为）。
        若底层不支持删除，将记录日志并返回0。
        """
        if not getattr(self, 'mem', None):
            logging.warning("Mem0 client not available, cannot clear memories")
            return 0
        deleted = 0
        try:
            # 列出所有记忆（尽量用通配/空查询）
            try:
                results = self.mem.search("", user_id=USER_ID)
            except Exception:
                results = []
            if isinstance(results, dict) and "results" in results:
                results_list = results["results"]
            else:
                results_list = results if isinstance(results, list) else []
            for r in results_list:
                mem_id = r.get("id")
                try:
                    if hasattr(self.mem, 'delete') and mem_id:
                        self.mem.delete(id=mem_id, user_id=USER_ID)
                        deleted += 1
                    elif hasattr(self.mem, 'remove') and mem_id:
                        self.mem.remove(id=mem_id, user_id=USER_ID)
                        deleted += 1
                except Exception as e:
                    logging.warning(f"Failed to delete memory {mem_id}: {e}")
            logging.info(f"Cleared {deleted} memories for user_id={USER_ID}")
            return deleted
        except Exception as e:
            logging.error(f"Failed to clear memories: {e}")
            return deleted


def retrieve_user_preferences(task_description: str, mem_client=None, use_graphrag: bool = None) -> list:
    """
    检索用户偏好，直接以检索到的所有原文list形式返回
    Returns: List[str]
    """
    if not mem_client:
        return []
    try:
        if use_graphrag:
            return _retrieve_with_graphrag(task_description, mem_client)
        else:
            return _retrieve_with_vector_search(task_description, mem_client)
    except Exception as e:
        logging.error(f"Failed to retrieve user preferences: {e}")
        return []


def _retrieve_with_vector_search(task_description: str, mem_client) -> list:
    """使用内存向量/检索原始偏好文本list"""
    try:
        results = mem_client.search(task_description, user_id=USER_ID)
        memory_texts = []
        # 返回格式直接看是否list或dict
        if isinstance(results, dict) and "results" in results:
            results_list = results["results"]
        else:
            results_list = results if isinstance(results, list) else []
        for r in results_list:
            memory_text = r.get("memory", "")
            if memory_text:
                memory_texts.append(memory_text)
        logging.info(f"Final preferences from vector search (raw text): {memory_texts}")
        return memory_texts
    except Exception as e:
        logging.error(f"Failed to retrieve with vector search: {e}")
        return []


def _retrieve_with_graphrag(task_description: str, mem_client) -> list:
    """使用GraphRAG检索用户偏好，直接返回检索到的原始偏好文本组成的list"""
    try:
        results = mem_client.search(task_description, user_id=USER_ID)
        memory_texts = []
        # GraphRAG返回格式可能有dict或list
        if isinstance(results, dict) and "results" in results:
            results_list = results["results"]
        else:
            results_list = results if isinstance(results, list) else []
        logging.info(f"GraphRAG search results: {results_list}")
        for i, result in enumerate(results_list):
            memory_text = result.get("memory", "")
            if memory_text:
                memory_texts.append(memory_text)
        logging.info(f"Final preferences from GraphRAG (raw text): {memory_texts}")
        return memory_texts
    except Exception as e:
        logging.error(f"Failed to retrieve with GraphRAG: {e}")
        return []


def should_extract_preferences(task_result: Dict[str, Any]) -> bool:
    """
    判断是否应该提取偏好
    
    Args:
        task_result: 任务结果
        
    Returns:
        是否应该提取偏好
    """
    try:
        actions = task_result.get("actions", [])
        if not actions:
            return False
            
        # 检查任务是否成功完成
        last_action = actions[-1]
        return last_action.get("type") == "done"
        
    except Exception:
        return False


def combine_context(experience_content: str, user_preferences: List[str]) -> str:
    """
    结合experience和用户偏好
    
    Args:
        experience_content: 经验内容
        user_preferences: 用户偏好
        
    Returns:
        增强的上下文
    """
    if not user_preferences:
        return experience_content
    
    preferences_text = "用户偏好原文：\n" + "\n".join(f"- {p}" for p in user_preferences)
    
    return f"""
{experience_content}

{preferences_text}
"""
