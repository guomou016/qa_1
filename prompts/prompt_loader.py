import os
from typing import Dict

class PromptLoader:
    """提示词加载器，用于从prompt_all.txt文件中加载各种模型的提示词"""
    
    def __init__(self, prompt_file_path: str = None):
        if prompt_file_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompt_file_path = os.path.join(current_dir, "prompt_all.txt")
        
        self.prompt_file_path = prompt_file_path
        self._prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, str]:
        """从文件中加载所有提示词"""
        prompts = {}
        
        try:
            with open(self.prompt_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析提示词
            import re
            pattern = r'\[([A-Z_]+)\]\n(.*?)\n\[/\1\]'
            matches = re.findall(pattern, content, re.DOTALL)
            
            for prompt_name, prompt_content in matches:
                prompts[prompt_name] = prompt_content.strip()
            
        except FileNotFoundError:
            print(f"警告：提示词文件 {self.prompt_file_path} 未找到")
        except Exception as e:
            print(f"加载提示词时出错：{e}")
        
        return prompts
    
    def get_prompt(self, prompt_name: str) -> str:
        """获取指定名称的提示词"""
        return self._prompts.get(prompt_name, "")
    
    def get_intent_recognition_prompt(self) -> str:
        """获取意图识别提示词"""
        return self.get_prompt("INTENT_RECOGNITION_PROMPT")
    
    def get_answer_generation_prompt(self) -> str:
        """获取答案生成提示词"""
        return self.get_prompt("ANSWER_GENERATION_PROMPT")
    
    def get_table_summary_prompt(self) -> str:
        """获取表格总结提示词"""
        return self.get_prompt("TABLE_SUMMARY_PROMPT")
    
    def get_archive_summary_prompt(self) -> str:
        """获取档案总结提示词"""
        return self.get_prompt("ARCHIVE_SUMMARY_PROMPT")
    
    def get_json_extraction_prompt(self) -> str:
        """获取JSON提取提示词"""
        return self.get_prompt("JSON_EXTRACTION_PROMPT")