import json
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pathlib import Path

# 获取当前文件的绝对路径（Path 对象）
current_file_path = Path(__file__).resolve()

# 获取当前文件所在目录
current_dir = current_file_path.parent

# default_template_path = current_dir / "experience" / "templates.json"
default_template_path = current_dir / "experience" / "templates-new.json"
# Disable default OpenAI LLM globally
Settings.llm = None

class PromptTemplateSearch:
    def __init__(self, template_path: str = default_template_path):
        self.template_path = template_path
        self.index = None
        self.templates = []
        self._load_templates()
        self._build_index()

    def _load_templates(self):
        """Load templates from the JSON file."""
        with open(self.template_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Handle both list and dict format for templates
            if isinstance(data, dict) and "templates" in data:
                self.templates = data["templates"]
            elif isinstance(data, list):
                self.templates = data
            else:
                self.templates = []

    def _build_index(self):
        """Build the llama_index from the loaded templates."""
        documents = [
            Document(
                text=json.dumps({
                    "name": template['name'],
                    "description": template['description'],
                    "Full Description": template['full_description']
                }, ensure_ascii=False),
                metadata={
                    "keywords": template.get("keywords", []),
                    "description": template.get("description", "")  # Include description in metadata
                }
            )
            for template in self.templates
        ]
        model_path = current_dir / "experience" / "BAAI" / "bge-small-zh"
        print("Using embedding model path:", model_path)
        embed_model = HuggingFaceEmbedding(model_name= "BAAI/bge-small-zh")
        self.index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

    def query(self, task_description, top_k=1):
        """Query the index to find the most relevant template."""
        query_engine = self.index.as_query_engine(llm=None, similarity_top_k=top_k)
        response = query_engine.query(task_description)
        return response
    
    def extract_full_description(self, result):
        """Extract the Full Description content from multiple JSON objects in the result."""
        import re
        import json
        
        experiences = {}
        
        # Method 1: Find JSON objects using regex with proper handling of nested quotes
        # This pattern matches JSON objects that contain both "name" and "Full Description" fields
        json_pattern = r'\{"name":\s*"[^"]*",\s*"description":\s*"[^"]*",\s*"Full Description":\s*"(?:[^"\\]|\\.)*"\}'
        json_matches = re.findall(json_pattern, result)
        
        for i, json_match in enumerate(json_matches, 1):
            try:
                parsed_json = json.loads(json_match)
                full_description = parsed_json.get("Full Description", None)
                if full_description:
                    experiences[f"experience{i}"] = full_description
            except json.JSONDecodeError:
                continue
        
        # Method 2: If regex fails, use a more robust character-by-character parsing
        if not experiences:
            i = 0
            while i < len(result):
                # Look for start of JSON object
                start_pos = result.find('{"name":', i)
                if start_pos == -1:
                    break
                
                # Find the matching closing brace
                brace_count = 0
                in_string = False
                escape_next = False
                end_pos = start_pos
                
                for j in range(start_pos, len(result)):
                    char = result[j]
                    
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
                
                # Extract and parse the JSON object
                json_str = result[start_pos:end_pos]
                if '"Full Description"' in json_str:
                    try:
                        parsed_json = json.loads(json_str)
                        full_description = parsed_json.get("Full Description", None)
                        if full_description:
                            experiences[f"experience{len(experiences) + 1}"] = full_description
                    except json.JSONDecodeError:
                        pass
                
                i = end_pos
        
        # Return as JSON string if experiences found, otherwise return None
        return json.dumps(experiences, ensure_ascii=False, indent=2) if experiences else None
    
    def get_experience(self, query_content, template_path:str = default_template_path, top_k=1):
        """Get experience by querying templates and extracting Full Description fields."""
        # Initialize with the specified template path if different
        if self.template_path != template_path:
            self.template_path = template_path
            self.index = None
            self.templates = []
            self._load_templates()
            self._build_index()
        
        # Query the index
        result = self.query(query_content, top_k)
        
        # Extract Full Description fields
        if result and hasattr(result, 'response'):
            full_description = self.extract_full_description(result.response)
            return full_description if full_description else "未找到Full Description字段"
        
        return "未找到Full Description字段"
    

if __name__ == "__main__":
    # Path to the templates.json file
    template_file = Path(__file__).parent / "experience" / "templates.json"

    # Initialize the search class
    search_engine = PromptTemplateSearch(template_file)

    # Example query
    user_query = "帮我收能量"
    result = search_engine.query(user_query, top_k=2)

    print("\n对应的模版内容:")
    print(result.response)  # Assuming the response contains the full template details
    print("\n对应的模版内容的Full Description字段:")
    full_description = search_engine.extract_full_description(result.response)
    print(full_description if full_description else "未找到Full Description字段")

    search_engine = PromptTemplateSearch(template_file)
    full_description = search_engine.get_experience(user_query, template_file, top_k=2)
    print("\n通过get_experience方法获取的Full Description字段:")
    print(full_description)