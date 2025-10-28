import json
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Document, 
    Settings,
    StorageContext,  # 导入 StorageContext
    load_index_from_storage  # 导入加载函数
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pathlib import Path
import os
import re

# 获取当前文件的绝对路径（Path 对象）
current_file_path = Path(__file__).resolve()

# 获取当前文件所在目录
current_dir = current_file_path.parent

# 默认模板路径
default_template_path = current_dir / "experience" / "templates-new.json"
# 默认持久化存储路径
DEFAULT_STORAGE_DIR = current_dir / "experience" / "vector_storage"

# Disable default OpenAI LLM globally
Settings.llm = None

class PromptTemplateSearch:
    def __init__(self, 
                 template_path: str = default_template_path, 
                 storage_dir: str = DEFAULT_STORAGE_DIR):
        
        self.template_path = Path(template_path)
        self.storage_dir = Path(storage_dir)
        self.templates = []
        self.index = None

        # 1. 初始化嵌入模型 (加载和构建都需要)
        model_path = current_dir / "experience" / "BAAI" / "bge-small-zh"
        
        # 使用您在原始代码中提供的绝对路径
        embed_model_path_str = "/home/zhaoxi/ipads/llm-agent/MobiAgent/utils/experience/BAAI/bge-small-zh"
        
        # 检查路径是否存在，如果不存在，尝试使用相对路径
        if not Path(embed_model_path_str).exists():
            print(f"警告: 绝对路径 {embed_model_path_str} 未找到。")
            embed_model_path_str = str(model_path)
            print(f"回退到相对路径: {embed_model_path_str}")
            if not Path(embed_model_path_str).exists():
                 raise FileNotFoundError(f"嵌入模型未在 {embed_model_path_str} 或原始绝对路径找到。")

        print(f"使用嵌入模型路径: {embed_model_path_str}")
        self.embed_model = HuggingFaceEmbedding(model_name=embed_model_path_str)

        # 2. 加载或构建索引
        self._load_or_build_index()

    def _load_templates(self):
        """从 JSON 文件加载模板。"""
        with open(self.template_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 处理列表和字典格式
            if isinstance(data, dict) and "templates" in data:
                self.templates = data["templates"]
            elif isinstance(data, list):
                self.templates = data
            else:
                self.templates = []

    def _load_or_build_index(self):
        """如果存储中存在索引，则加载它，否则构建它。"""
        # 检查一个关键文件（如 docstore.json）是否存在，以判断索引是否已持久化
        if (self.storage_dir / "docstore.json").exists():
            try:
                print(f"从 {self.storage_dir} 加载现有索引...")
                storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
                self.index = load_index_from_storage(
                    storage_context, 
                    embed_model=self.embed_model
                )
                print("索引加载成功。")
            except Exception as e:
                print(f"从存储加载索引失败: {e}。 正在重建索引...")
                # 如果加载失败（例如，版本不兼容），则重建
                self._build_index()
        else:
            print(f"在 {self.storage_dir} 未找到现有索引。正在构建新索引...")
            self._build_index()

    def _build_index(self):
        """从加载的模板构建 llama_index 并将其保存到存储中。"""
        # 1. 加载模板（仅在构建时需要）
        self._load_templates()
        if not self.templates:
            print("未加载任何模板。无法构建索引。")
            return

        # 2. 创建文档
        print("正在从模板创建文档...")
        documents = [
            Document(
                text=json.dumps({
                    "name": template['name'],
                    "description": template['description'],
                    "Full Description": template['full_description']
                }, ensure_ascii=False),
                metadata={
                    "keywords": template.get("keywords", []),
                    "description": template.get("description", "")
                }
            )
            for template in self.templates
        ]

        # 3. 创建新的存储上下文
        storage_context = StorageContext.from_defaults()

        # 4. 构建索引
        print("正在从文档构建索引...")
        self.index = VectorStoreIndex.from_documents(
            documents, 
            embed_model=self.embed_model,
            storage_context=storage_context  # 传递上下文
        )
        
        # 5. 持久化到磁盘
        print(f"正在将索引保存到 {self.storage_dir}...")
        self.index.storage_context.persist(persist_dir=self.storage_dir)
        print("索引构建并保存成功。")

    def query(self, task_description, top_k=1):
        """查询索引以查找最相关的模板。"""
        if not self.index:
            return "索引未初始化。"
        query_engine = self.index.as_query_engine(llm=None, similarity_top_k=top_k)
        response = query_engine.query(task_description)
        return response
    
    def extract_full_description(self, result):
        """从结果中的多个 JSON 对象中提取 Full Description 内容。"""
        experiences = {}
        
        # 方法 1: 使用正则表达式查找 JSON 对象
        json_pattern = r'\{"name":\s*"[^"]*",\s*"description":\s*"[^"]*",\s*"Full Description":\s*"(?:[^"\\]|\\.)*"\}'
        
        # 确保 result 是字符串
        result_str = str(result)
        
        json_matches = re.findall(json_pattern, result_str)
        
        for i, json_match in enumerate(json_matches, 1):
            try:
                parsed_json = json.loads(json_match)
                full_description = parsed_json.get("Full Description", None)
                if full_description:
                    experiences[f"experience{i}"] = full_description
            except json.JSONDecodeError:
                continue
        
        # 方法 2: 如果 regex 失败，使用更稳健的逐字符解析
        if not experiences:
            i = 0
            while i < len(result_str):
                start_pos = result_str.find('{"name":', i)
                if start_pos == -1:
                    break
                
                brace_count = 0
                in_string = False
                escape_next = False
                end_pos = start_pos
                
                for j in range(start_pos, len(result_str)):
                    char = result_str[j]
                    
                    if escape_next:
                        escape_next = False
                        continue
                    if char == '\\':
                        escape_next = True
                        continue
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    
                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = j + 1
                                break
                
                json_str = result_str[start_pos:end_pos]
                if '"Full Description"' in json_str:
                    try:
                        parsed_json = json.loads(json_str)
                        full_description = parsed_json.get("Full Description", None)
                        if full_description:
                            experiences[f"experience{len(experiences) + 1}"] = full_description
                    except json.JSONDecodeError:
                        pass
                
                i = end_pos
        
        return json.dumps(experiences, ensure_ascii=False, indent=2) if experiences else None
    
    def get_experience(self, query_content, template_path:str = default_template_path, top_k=1):
        """通过查询模板并提取 Full Description 字段来获取经验。"""
        
        new_template_path = Path(template_path)
        # 检查 template_path 是否已更改
        if self.template_path != new_template_path:
            print(f"模板路径已更改。正在为 {new_template_path} 重新加载/构建索引...")
            self.template_path = new_template_path
            
            # 基于新的模板文件名创建一个新的、唯一的存储目录
            # 例如: "templates-new.json" -> "vector_storage_templates-new"
            new_storage_name = f"vector_storage_{self.template_path.stem}"
            self.storage_dir = self.template_path.parent / new_storage_name
            
            # 强制重新加载或构建新路径的索引
            self._load_or_build_index()
        
        # 查询索引
        result = self.query(query_content, top_k)
        
        # 提取 Full Description 字段
        if result and hasattr(result, 'response'):
            full_description = self.extract_full_description(result.response)
            return full_description if full_description else "未找到Full Description字段"
        elif isinstance(result, str): # 处理来自 query 的错误消息
             return result
        
        return "未找到Full Description字段"
    

if __name__ == "__main__":
    # 使用原始的 templates.json 文件进行测试
    template_file = Path(__file__).parent / "experience" / "templates.json"
    
    # === 第一次运行 ===
    # 这一次将构建索引并将其保存到 'experience/vector_storage_templates'
    print("--- 第一次运行 (使用 templates.json) ---")
    search_engine_1 = PromptTemplateSearch(template_file)

    user_query = "帮我收能量"
    result_1 = search_engine_1.query(user_query, top_k=2)

    if hasattr(result_1, 'response'):
        print("\n对应的模版内容:")
        print(result_1.response)
        print("\n对应的模版内容的Full Description字段:")
        full_description_1 = search_engine_1.extract_full_description(result_1.response)
        print(full_description_1 if full_description_1 else "未找到Full Description字段")
    else:
        print(f"查询失败: {result_1}")

    # === 第二次运行 ===
    # 这一次将从 'experience/vector_storage_templates' 加载索引，速度会快得多
    print("\n--- 第二次运行 (使用 templates.json, 从存储加载) ---")
    search_engine_2 = PromptTemplateSearch(template_file)
    full_description_2 = search_engine_2.get_experience(user_query, template_file, top_k=2)
    print("\n通过get_experience方法获取的Full Description字段:")
    print(full_description_2)

    # === 第三次运行 (使用不同的模板文件) ===
    # 这一次将构建一个 *新* 索引并将其保存到 'experience/vector_storage_templates-new'
    print("\n--- 第三次运行 (使用 templates-new.json, 构建新索引) ---")
    template_file_new = Path(__file__).parent / "experience" / "templates-new.json"
    search_engine_3 = PromptTemplateSearch(template_file_new)
    full_description_3 = search_engine_3.get_experience(user_query, template_file_new, top_k=2)
    print("\n通过get_experience方法获取的Full Description字段:")
    print(full_description_3)
