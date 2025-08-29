import os
import sys
from openai import OpenAI
from typing import List, Dict, Union

# 添加prompts目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
prompts_dir = os.path.join(project_root, "prompts")
if prompts_dir not in sys.path:
    sys.path.insert(0, prompts_dir)

from prompt_loader import PromptLoader

class TableSummarizer:
    """
    一个用于总结表格数据的类。
    它会逐行读取表格数据，并调用AI模型为每一行生成一个总结。
    """

    def __init__(self, api_key: str = None, model: str = "qwen-plus"):
        """
        初始化 TableSummarizer。

        Args:
            api_key (str, optional): 您的阿里云百炼 API Key。如果未提供，将尝试从环境变量 DASHSCOPE_API_KEY 中读取。
            model (str, optional): 要使用的模型名称。默认为 "qwen-plus"。
        """
        # 如果未直接提供 api_key，则从环境变量中获取
        resolved_api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not resolved_api_key:
            raise ValueError("API Key not found. Please provide it as an argument or set the DASHSCOPE_API_KEY environment variable.")

        self.client = OpenAI(
            api_key=resolved_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = model
        
        # 从prompt_all.txt加载提示词
        prompt_loader = PromptLoader()
        self.table_summary_prompt = prompt_loader.get_table_summary_prompt()
        self.archive_summary_prompt = prompt_loader.get_archive_summary_prompt()

    def _get_summary_for_row(self, headers: List[str], row_data: List[str]) -> str:
        """
        为单行数据生成总结。这是一个内部辅助方法。

        Args:
            headers (List[str]): 表格的表头列表。
            row_data (List[str]): 单行数据的列表。

        Returns:
            str: AI模型生成的该行数据的总结文本。
        """
        # 将表头和行数据组合成更易于模型理解的格式
        row_description = ", ".join([f"{header}: {data}" for header, data in zip(headers, row_data)])

        messages = [
            {
                "role": "system",
                "content": self.table_summary_prompt
            },
            {
                "role": "user",
                "content": f"请为以下业务信息生成总结: {row_description}"
            }
        ]
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"An error occurred while processing a row: {e}")
            return "无法生成总结"

    def summarize_table(self, headers: List[str], data: List[List[str]]) -> List[str]:
        """
        对整个表格的每一行进行总结。

        Args:
            headers (List[str]): 表格的表头列表。
            data (List[List[str]]): 包含多行数据的列表，其中每一行也是一个列表。

        Returns:
            List[str]: 一个包含每一行数据总结的列表。
        """
        summaries = []
        print(f"开始处理 {len(data)} 行数据...")
        for i, row in enumerate(data):
            if len(headers) != len(row):
                print(f"警告: 第 {i+1} 行的数据列数与表头列数不匹配，已跳过。")
                summaries.append("数据格式错误，已跳过")
                continue
            
            summary = self._get_summary_for_row(headers, row)
            summaries.append(summary)
            print(f"已完成第 {i+1} 行的总结。")
        
        print("所有数据处理完毕。")
        return summaries

# --- 使用示例 ---
if __name__ == '__main__':
    # 1. 准备你的表格数据
    # 表头
    table_headers = ["产品名称", "处理器", "屏幕尺寸", "价格(元)", "库存状态"]
    # 数据行
    table_data = [
        ["旗舰手机Pro", "骁龙 8 Gen 3", "6.7英寸", "4999", "有货"],
        ["商务笔记本X1", "酷睿 i7-13700H", "14英寸", "8999", "库存紧张"],
        ["游戏平板G9", "天玑 9300", "11英寸", "2999", "有货"],
        ["超薄上网本Air", "酷睿 i5-1340P", "13.3英寸", "5299", "已售罄"],
    ]

    # 2. 创建 TableSummarizer 实例
    # 确保你已经设置了环境变量 DASHSCOPE_API_KEY，或者在这里直接传入 api_key="sk-xxx"
    try:
        summarizer = TableSummarizer()

        # 3. 调用方法进行总结
        all_summaries = summarizer.summarize_table(headers=table_headers, data=table_data)

        # 4. 打印结果
        print("\n--- 表格数据总结报告 ---")
        for i, summary in enumerate(all_summaries):
            original_row = ", ".join(table_data[i])
            print(f"原始数据: {original_row}")
            print(f"AI 总结: {summary}\n")
            
    except ValueError as e:
        print(e)

