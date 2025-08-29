import os
import json
from pathlib import Path  # 导入 pathlib
from typing import List, Union, Dict
from openai import OpenAI

class DashScopeEmbeddingGenerator:
    def __init__(self, api_key=None, model="text-embedding-v4", dimensions=1024, encoding_format="float"):
        """
        初始化向量生成器
        :param api_key: API密钥，默认从环境变量获取
        :param model: 使用的嵌入模型名称
        :param dimensions: 向量维度
        :param encoding_format: 向量编码格式
        """
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = model
        self.dimensions = dimensions
        self.encoding_format = encoding_format
        self.next_id = 0  # 自增ID计数器
    
    def generate_embeddings(self, inputs: Union[str, List[str]]) -> List[Dict]:
        """
        生成文本向量
        :param inputs: 单个文本字符串或文本列表
        :return: 结构化向量结果列表，每个元素包含id、text和vector
        """
        # 确保输入是列表格式
        if isinstance(inputs, str):
            inputs = [inputs]
        
        # 调用API生成向量
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=inputs,
                dimensions=self.dimensions,
                encoding_format=self.encoding_format
            )
            embeddings = response.data
        except Exception as e:
            raise RuntimeError(f"向量生成失败: {str(e)}")
        
        # 处理并格式化结果
        results = []
        # 使用 enumerate 获取原始文本和索引
        for i, emb in enumerate(embeddings):
            results.append({
                "id": self._get_next_id(),  # 使用自增ID
                "text": inputs[i],         # 保存原始文本
                "vector": emb.embedding    # 保留向量数据
            })
        
        return results
    
    def _get_next_id(self) -> int:
        """获取下一个自增ID"""
        current_id = self.next_id
        self.next_id += 1
        return current_id
    
    def generate_embeddings_json(self, inputs: Union[str, List[str]], indent=None) -> str:
        """
        生成JSON格式的向量结果
        :param inputs: 单个文本字符串或文本列表
        :param indent: JSON缩进格式
        :return: JSON字符串
        """
        embeddings = self.generate_embeddings(inputs)
        return json.dumps(embeddings, ensure_ascii=False, indent=indent)
    
    def reset_id_counter(self, start_id=0):
        """重置ID计数器"""
        self.next_id = start_id

# 使用示例
if __name__ == "__main__":
    # 初始化向量生成器
    embedder = DashScopeEmbeddingGenerator(model="text-embedding-v4", dimensions=1024)
    
    # 示例文本 (修改为更细粒度的文本块，并为每个块增加核心主题)
    texts = [
        "项目名称：居民户报装",
        "居民户报装所需材料：安全用气协议（必）、居民用户供用气合同（必）、不动产权证（必）、燃气安全告知书（必）、用户的居民身份证（必）",
        "居民户报装办理地址：新建区长麦南路6号C区公共事务服务区 C2窗口",
        "居民户报装咨询热线：0791-83753710"
    ]
    
    # 重置ID计数器，确保每次运行都从0开始
    embedder.reset_id_counter()
    
    # 生成向量
    embeddings = embedder.generate_embeddings(texts)
    
    # 打印结果
    print("结构化向量结果:")
    for emb in embeddings:
        print(f"\nID: {emb['id']}")
        print(f"向量长度: {len(emb['vector'])}")
        print(f"前5维向量值: {emb['vector'][:5]}")
    
    # 获取JSON格式结果
    json_result = embedder.generate_embeddings_json(texts, indent=2)
    print("\nJSON格式输出:")
    print(json_result)

    # --- 新增：保存到文件 ---
    try:
        # 1. 定义保存目录 (使用你指定的绝对路径)
        output_dir = Path(r"C:\Users\pc123\Desktop\新事心办-1\db")
        
        # 2. 确保目录存在，如果不存在则创建
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. 定义完整的文件路径
        output_file_path = output_dir / "embeddings.json"
        
        # 4. 将JSON结果写入文件
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(json_result)
            
        print(f"\n✅ 结果已成功保存到: {output_file_path}")
        
    except Exception as e:
        print(f"\n❌ 保存文件时出错: {e}")