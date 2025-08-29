import os
import dashscope
from openai import OpenAI
from http import HTTPStatus
from typing import List, Dict, Any
import json
import sys
import numpy as np # 引入Numpy用于向量计算

class SemanticSearchReranker:
    """
    一个封装了文本嵌入和重排序功能的工具类。
    它模拟了信息检索中的“召回”和“精排”两个阶段。
    """

    def __init__(self, api_key: str = None, 
                 embedding_model: str = "text-embedding-v4", 
                 rerank_model: str = "gte-rerank-v2"):
        """
        初始化客户端。

        Args:
            api_key (str, optional): 您的阿里云百炼 API Key。如果未提供，
                                     将尝试从环境变量 DASHSCOPE_API_KEY 中读取。
            embedding_model (str, optional): 用于文本嵌入的模型。
            rerank_model (str, optional): 用于文本重排序的模型。
        
        Raises:
            ValueError: 如果没有提供 API Key 或在环境变量中找不到。
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("API Key not found. Please provide it as an argument or set the DASHSCOPE_API_KEY environment variable.")
        
        # 为 dashscope SDK 设置 API Key
        dashscope.api_key = self.api_key

        # 初始化用于 embedding 的 OpenAI 兼容客户端
        self.embedding_client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model

    def embed_documents(self, documents: List[str], dimensions: int = 1024) -> List[List[float]]:
        """
        将文档列表转换为向量表示。
        在实际应用中，您会先将整个数据库的文档进行向量化并存储。

        Args:
            documents (List[str]): 需要向量化的文档字符串列表。
            dimensions (int, optional): 向量维度。默认为 1024。

        Returns:
            List[List[float]]: 包含每个文档向量的列表。
        """
        print(f"开始使用模型 '{self.embedding_model}' 对 {len(documents)} 个文档进行向量化...")
        embeddings = []
        for doc in documents:
            try:
                response = self.embedding_client.embeddings.create(
                    model=self.embedding_model,
                    input=doc,
                    dimensions=dimensions,
                    encoding_format="float"
                )
                embeddings.append(response.data[0].embedding)
            except Exception as e:
                print(f"文档 '{doc[:20]}...' 向量化失败: {e}")
                embeddings.append([]) # 添加一个空列表作为占位符
        print("所有文档向量化完成。")
        return embeddings

    def search_and_rerank(self, query: str, documents: List[str], top_n: int = 3) -> List[Dict[str, Any]]:
        """
        接收一个查询和一组文档，返回与查询最相关的前 N 个文档。
        这个方法直接使用 rerank 模型，它在内部处理语义相关性，无需预先对查询进行向量化。

        Args:
            query (str): 用户的查询问题。
            documents (List[str]): 候选文档的列表（模拟从向量数据库中召回的结果）。
            top_n (int, optional): 需要返回的最相关文档的数量。默认为 3。

        Returns:
            List[Dict[str, Any]]: 一个排序后的字典列表，每个字典包含文档内容和相关性分数。
        """
        print(f"\n正在使用模型 '{self.rerank_model}' 对查询进行重排序...")
        try:
            response = dashscope.TextReRank.call(
                model=self.rerank_model,
                query=query,
                documents=documents,
                top_n=top_n,
                return_documents=True
            )

            if response.status_code == HTTPStatus.OK:
                print("重排序成功。")
                # response.output.results 是已经排好序的列表
                return response.output.results
            else:
                print(f"重排序失败: {response.message}")
                return []

        except Exception as e:
            print(f"调用重排序API时发生错误: {e}")
            return []

# --- 使用示例 (已修改为向量匹配逻辑) ---
if __name__ == '__main__':
    
    # --- 1. 定义和加载包含向量的文档 ---
    documents_path = r"C:\Users\pc123\Desktop\新事心办-1\db\embeddings.json"
    try:
        with open(documents_path, 'r', encoding='utf-8') as f:
            # 加载包含向量和文本的完整数据
            documents_with_vectors = json.load(f)
        
        if not documents_with_vectors or not isinstance(documents_with_vectors, list):
            print(f"❌ 错误: 文档文件 '{documents_path}' 为空或格式不正确。应为一个JSON列表。")
            sys.exit(1)
        
        print(f"✅ 成功从 {documents_path} 加载了 {len(documents_with_vectors)} 个文档向量。")

    except FileNotFoundError:
        print(f"❌ 错误: 找不到文档文件 '{documents_path}'。请先运行 rag/embedding.py 来生成它。")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ 错误: 解析文档文件时出错: {e}")
        sys.exit(1)

    # --- 2. 初始化工具并向量化查询 ---
    user_query = "居民户报装需要我提供什么材料"
    try:
        # 初始化工具类
        search_tool = SemanticSearchReranker()

        # 将用户查询向量化
        print(f"\n正在向量化查询: '{user_query}'...")
        query_vector = search_tool.embed_documents([user_query])[0]

        if not query_vector:
            print("❌ 错误: 查询向量化失败，无法继续。")
            sys.exit(1)
        
        # 新增：打印向量化后的部分内容
        print(f"查询向量 (前 5 维): {query_vector[:5]}...")
        
        print("查询向量化完成。")

    except ValueError as e:
        print(e)
        sys.exit(1)
    except Exception as e:
        print(f"❌ 初始化或查询向量化时发生未知错误: {e}")
        sys.exit(1)

    # --- 3. 计算余弦相似度并排序 ---
    def cosine_similarity(vec1, vec2):
        """计算两个向量之间的余弦相似度"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        return dot_product / (norm_vec1 * norm_vec2)

    print("\n正在计算查询与各文档的相似度...")
    similarities = []
    for doc in documents_with_vectors:
        # 确保文档包含 'vector' 和 'text' 键，并且向量不为空
        if 'vector' in doc and 'text' in doc and doc['vector']:
            score = cosine_similarity(query_vector, doc['vector'])
            similarities.append({
                "score": score,
                "text": doc['text']
            })

    # 按相似度得分从高到低排序
    sorted_results = sorted(similarities, key=lambda x: x['score'], reverse=True)
    
    # --- 4. 新增：根据阈值过滤结果 ---
    similarity_threshold = 0.9
    # 创建一个新的列表，只包含分数高于或等于阈值的结果
    filtered_results = [result for result in sorted_results if result['score'] >= similarity_threshold]

    # --- 5. 打印过滤后的结果 ---
    top_n = 3
    print(f"\n--- 查询结果（相似度 > {similarity_threshold}，Top {top_n}） ---")
    if filtered_results:
        for i, result in enumerate(filtered_results[:top_n]):
            print(f"Top {i+1} - 相似度分数: {result['score']:.4f}")
            print(f"文档内容: {result['text']}\n")
    else:
        # 如果过滤后列表为空，则告知用户没有找到足够相似的内容
        print(f"未能找到相似度高于 {similarity_threshold} 的匹配内容。")

