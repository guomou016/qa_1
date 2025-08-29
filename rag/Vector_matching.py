import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量，主要为了获取API密钥
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', 'main', '.env'))

class DashScopeEmbeddingGenerator:
    """通义千问向量生成器"""
    def __init__(self, model="text-embedding-v4", dimensions=1024, api_key=None, base_url=None):
        """
        初始化向量生成器。
        :param model: 使用的模型名称。
        :param dimensions: 向量维度。
        :param api_key: API密钥，如果未提供，则从环境变量 DASHSCOPE_API_KEY 获取。
        :param base_url: API的基础URL。
        """
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model = model
        self.dimensions = dimensions

    def get_embedding(self, text: str) -> list[float]:
        """
        获取单个文本的向量。
        :param text: 输入的文本。
        :return: 文本的向量表示。
        """
        # --- 关键修改：确保将 dimensions 参数传递给 API ---
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
            dimensions=self.dimensions
        )
        return response.data[0].embedding

class VectorSearcher:
    """向量检索器，用于在知识库中进行相似度搜索"""
    def __init__(self, db_path: str, embedding_model: str = "text-embedding-v4", dimensions: int = 1024):
        """
        初始化向量检索器。
        :param db_path: 知识库文件路径 (embeddings.json)。
        :param embedding_model: 用于生成查询向量的模型名称。
        :param dimensions: 向量维度。
        """
        self.db_path = db_path
        self.knowledge_base = self._load_knowledge_base()
        # --- 确保将 dimensions 传递给生成器 ---
        self.embedding_generator = DashScopeEmbeddingGenerator(model=embedding_model, dimensions=dimensions)

    def _load_knowledge_base(self) -> list:
        """从JSON文件加载知识库。"""
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"错误：知识库文件未找到，路径：{self.db_path}")
            return []
        except json.JSONDecodeError:
            print(f"错误：无法解析知识库文件，请检查JSON格式：{self.db_path}")
            return []

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量的余弦相似度。"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        return dot_product / (norm_vec1 * norm_vec2)

    def search(self, user_query, top_k=3, similarity_threshold=0.8, debug=False):
        """
        在知识库中搜索与用户查询最相关的文本块。
        新增debug模式，用于打印所有潜在匹配项的相似度。
        """
        query_vector = self.embedding_generator.get_embedding(user_query)
        
        all_matches = []
        for item in self.knowledge_base:
            # 确保向量存在且非空
            if 'vector' not in item or not item['vector']:
                continue
            
            similarity = self._cosine_similarity(query_vector, item['vector'])
            all_matches.append({
                'id': item.get('id', 'N/A'),
                'text': item.get('text', 'N/A'),
                'similarity': similarity
            })

        # 按相似度对所有候选项进行排序
        sorted_matches = sorted(all_matches, key=lambda x: x['similarity'], reverse=True)

        # 如果开启调试模式，打印Top5的原始匹配信息
        if debug:
            print("\n--- [DEBUG] RAG Search Internals ---")
            print(f"Query: '{user_query}'")
            print(f"Threshold: {similarity_threshold}")
            print("Top 5 potential matches (regardless of threshold):")
            for i, match in enumerate(sorted_matches[:5]):
                print(f"  {i+1}. Sim: {match['similarity']:.4f}, ID: {match['id']}, Text: '{match['text'][:80]}...'")
            print("-------------------------------------\n")

        # 根据阈值过滤结果
        final_results = [match for match in sorted_matches if match['similarity'] >= similarity_threshold]
        
        return final_results[:top_k]

# 主程序入口示例代码保持不变
if __name__ == "__main__":
    # 知识库文件路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_file_path = os.path.join(project_root, 'db', 'embeddings.json')

    # 检查知识库文件是否存在
    if not os.path.exists(db_file_path):
        print(f"错误：知识库文件 'embeddings.json' 在路径 '{db_file_path}' 中不存在。")
        print("请先确保知识库文件已正确生成并放置在指定目录。")
    else:
        # --- 确保在初始化时传入正确的模型和维度 ---
        searcher = VectorSearcher(db_path=db_file_path, embedding_model="text-embedding-v4", dimensions=1024)
        
        # 定义用户问题
        user_query = "水表登记需要什么材料？"


        
        # 执行搜索
        results = searcher.search(user_query, top_k=3, similarity_threshold=5)
        
        # 打印结果
        print(f"\n🔍 对于问题: '{user_query}'")
        if results:
            print("✅ 找到以下相关内容:")
            for result in results:
                # 从结果中获取 'text' 字段
                print(f"   - [相似度: {result['similarity']:.4f}] 匹配内容: {result['text']}")
        else:
            print("❌ 未找到相关内容。")