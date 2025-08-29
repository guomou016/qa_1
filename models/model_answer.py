import os
import sys
from openai import OpenAI

# 添加prompts目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
prompts_dir = os.path.join(project_root, "prompts")
if prompts_dir not in sys.path:
    sys.path.insert(0, prompts_dir)

from prompt_loader import PromptLoader

class DashScopeChatBot:
    """通义千问（Qwen）大模型聊天封装"""
    def __init__(self, model_name="qwen-plus", temperature=0.1, api_key=None, base_url=None, timeout=20.0, proxy=None):
        """
        初始化聊天机器人
        :param model_name: 使用的模型名称
        :param temperature: 温度参数
        :param api_key: API密钥，默认从环境变量获取
        :param base_url: API基础URL
        :param timeout: 超时时间
        :param proxy: 代理设置
        """
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = model_name
        self.temperature = temperature
        
        # 从prompt_all.txt加载系统提示词模板
        prompt_loader = PromptLoader()
        self.system_template = prompt_loader.get_answer_generation_prompt()

    def chat(self, user_question: str, db_info: str, history: list = None, stream: bool = True, stream_options=None, on_chunk=None):
        """
        向AI提问并获取回答
        :param user_question: 用户问题
        :param db_info: 数据库信息（字符串或字典）
        :param history: 历史对话（当前未使用）
        :param stream: 是否启用流式输出
        :param stream_options: 流式输出选项，例如 {"include_usage": True}
        :param on_chunk: 当开启流式时，每个chunk的回调函数 on_chunk(chunk)
        :return: AI生成的回答（非流式模式）或完整回答文本（流式模式）
        """
        # 将数据库信息转换为字符串格式
        formatted_db_info = self._format_db_info(db_info)
        
        # 构建动态系统提示
        system_prompt = self.system_template.format(db_info=formatted_db_info)
        
        # 准备API调用参数
        params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question},
            ]
        }
        
        # 如果启用流式，加入相关参数
        if stream:
            params["stream"] = True
            # 若未提供自定义 stream_options，则默认包含 usage
            params["stream_options"] = stream_options or {"include_usage": True}
        
        try:
            # 调用API
            completion = self.client.chat.completions.create(**params)
            
            # 非流式模式：直接返回完整回答
            if not stream:
                return completion.choices[0].message.content
            
            # 流式模式：收集所有内容片段并返回完整文本
            pieces = []
            for chunk in completion:
                try:
                    choice0 = chunk.choices[0] if getattr(chunk, "choices", None) else None
                    delta = getattr(choice0, "delta", None) if choice0 else None
                    content_piece = (
                        (getattr(delta, "content", None) if delta else None)
                        or (getattr(getattr(choice0, "message", None), "content", None) if choice0 else None)
                        or ""
                    )
                    if content_piece:
                        pieces.append(content_piece)
                        # 如果提供了回调函数，调用它
                        if on_chunk:
                            on_chunk(content_piece)
                except Exception:
                    # 忽略无法识别的结构
                    pass
            
            # 返回完整的回答文本
            return "".join(pieces)
            
        except Exception as e:
            return f"Error: {str(e)}"

    def chat_stream(self, user_question: str, db_info: str):
        """流式聊天方法，返回生成器"""
        # 将数据库信息转换为字符串格式
        formatted_db_info = self._format_db_info(db_info)
        
        # 构建动态系统提示
        system_prompt = self.system_template.format(db_info=formatted_db_info)
        
        # 准备API调用参数
        params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question},
            ],
            "stream": True,
            "stream_options": {"include_usage": True}
        }
        
        try:
            # 调用API
            completion = self.client.chat.completions.create(**params)
            
            # 流式处理：逐个yield内容片段
            for chunk in completion:
                try:
                    choice0 = chunk.choices[0] if getattr(chunk, "choices", None) else None
                    delta = getattr(choice0, "delta", None) if choice0 else None
                    content_piece = (
                        (getattr(delta, "content", None) if delta else None)
                        or (getattr(getattr(choice0, "message", None), "content", None) if choice0 else None)
                        or ""
                    )
                    if content_piece:
                        yield content_piece
                except Exception:
                    # 忽略无法识别的结构
                    pass
                    
        except Exception as e:
            yield f"Error: {str(e)}"

    def _format_db_info(self, db_info):
        """格式化数据库信息为可读字符串"""
        if isinstance(db_info, str):
            return db_info
        elif isinstance(db_info, dict):
            return "\n".join([f"- {key}: {value}" for key, value in db_info.items()])
        elif isinstance(db_info, list):
            return "\n".join([f"- {item}" for item in db_info])
        else:
            return str(db_info)

# 使用示例
# if __name__ == "__main__":
#     # 初始化聊天机器人
#     bot = DashScopeChatBot(model="qwen-plus")
    
#     # 定义数据库信息（可以是字符串、字典或列表）
#     database_info = {
#         "产品库存": "手机: 120台, 平板: 75台",
#         "用户数据": "注册用户: 15,000人, 活跃用户: 8,500人",
#         "销售数据": "本月销售额: ¥1,250,000, 同比增长: 15%"
#     }
    
#     # 用户问题
#     user_query = "我们还有多少手机库存？"
    
#     # 获取回答
#     response = bot.ask_question(user_query, database_info)
#     print("AI回复:", response)