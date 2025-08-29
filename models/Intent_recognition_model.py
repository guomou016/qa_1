# -*- coding: utf-8 -*-
import os
import sys
from openai import OpenAI
from typing import Optional, List, Dict, Any

# 添加prompts目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
prompts_dir = os.path.join(project_root, "prompts")
if prompts_dir not in sys.path:
    sys.path.insert(0, prompts_dir)

from prompt_loader import PromptLoader

class QwenChat:
    """
    通义千问大模型（Qwen）的封装类，通过 OpenAI 兼容接口进行调用。

    调用方式:
    ----------
    1. 确保已设置环境变量 `DASHSCOPE_API_KEY`，或在初始化时传入 `api_key`。

    2. 实例化 (可在此处自定义系统提示词):
       chat_client = QwenChat(
           api_key="sk-your_key_here",
           system_prompt="你是一个精通中国古诗词的AI助手。"
       )

    3. 准备消息并发起请求:
       # 普通调用
       response = chat_client.chat("李白是谁？", model="qwen-max")
       print(response.choices[0].message.content)

       # 流式调用
       stream_response = chat_client.chat("介绍一下杜甫", model="qwen-max", stream=True)
       for chunk in stream_response:
           print(chunk.choices[0].delta.content or "", end="")

    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        system_prompt: str = None,
    ) -> None:
        """
        初始化 OpenAI 兼容客户端。

        :param api_key: DashScope 的 API Key。如果为 None，则会从环境变量 `DASHSCOPE_API_KEY` 中读取。
        :param base_url: API 的 base URL。
        :param system_prompt: 默认的系统提示词。如果为None，则从prompt_all.txt加载。
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY 未设置，请在环境变量中配置或在初始化时传入 api_key 参数。"
            )

        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        
        # 从prompt_all.txt加载系统提示词
        if system_prompt is None:
            prompt_loader = PromptLoader()
            self.system_prompt = prompt_loader.get_intent_recognition_prompt()
        else:
            self.system_prompt = system_prompt

    def chat(
        self,
        user_prompt: str,
        model: str = "qwen-max",
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        使用指定模型发起聊天补全请求。

        :param user_prompt: 用户的提问。
        :param model: 要使用的模型名称。
                      模型列表参考: https://help.aliyun.com/zh/model-studio/getting-started/models
        :param history: 对话历史（可选）。
        :param kwargs: 其他要传递给 `client.chat.completions.create` 的参数,
                       例如 `stream=True`, `stream_options={"include_usage": True}` 等。
        :return: 返回 OpenAI API 的原始响应对象。
        """
        if not user_prompt:
            raise ValueError("user_prompt 不能为空。")

        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        return self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )