import pymysql
import pymysql.cursors
import json
import os
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

# 导入您项目中的另外两个核心模块
from qwen_summary import TableSummarizer
from embedding import DashScopeEmbeddingGenerator

# 修改环境变量加载，指定正确的.env文件路径
env_path = Path(__file__).parent.parent / 'main' / '.env'
load_dotenv(dotenv_path=env_path)

# --- 从环境变量读取数据库连接配置 ---
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT')),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_DATABASE'),
    'charset': os.getenv('DB_CHARSET'),
    'cursorclass': pymysql.cursors.DictCursor
}

# --- 从环境变量读取 SQL 查询配置 ---
SQL_QUERIES = {
    'test_table': os.getenv('CS_SQL_QUERY_TEST_TABLE'),
    'items_doc': os.getenv('CS_SQL_QUERY_ITEMS_DOC')
}

# --- 从环境变量读取表名配置 ---
TABLE_NAMES = {
    'test_table': os.getenv('CS_TABLE_NAME_TEST'),
    'items_doc': os.getenv('CS_TABLE_NAME_ITEMS_DOC')
}

# --- 从环境变量读取输出路径配置 ---
OUTPUT_CONFIG = {
    'dir': os.getenv('CS_OUTPUT_DIR'),
    'filename': os.getenv('CS_OUTPUT_FILENAME')
}

def process_data_pipeline_final():
    """
    执行最终的数据处理流水线：
    1. 分别从两个表获取所有数据。
    2. 在Python中按ID对数据进行分组和拼接（已增加类型转换，更健壮）。
    3. 对拼接后的完整信息进行AI总结。
    4. 将总结文本转换为向量并保存。
    """
    connection = None
    try:
        # 验证必要的环境变量
        if not os.getenv('DASHSCOPE_API_KEY'):
            raise ValueError("请在 .env 文件中设置 DASHSCOPE_API_KEY")
        
        # 验证数据库配置
        required_db_vars = ['DB_HOST', 'DB_PORT', 'DB_USER', 'DB_PASSWORD', 'DB_DATABASE', 'DB_CHARSET']
        for var in required_db_vars:
            if not os.getenv(var):
                raise ValueError(f"请在 .env 文件中设置 {var}")
        
        # 验证SQL查询配置
        required_sql_vars = ['CS_SQL_QUERY_TEST_TABLE', 'CS_SQL_QUERY_ITEMS_DOC']
        for var in required_sql_vars:
            if not os.getenv(var):
                raise ValueError(f"请在 .env 文件中设置 {var}")
        
        # 验证表名配置
        required_table_vars = ['CS_TABLE_NAME_TEST', 'CS_TABLE_NAME_ITEMS_DOC']
        for var in required_table_vars:
            if not os.getenv(var):
                raise ValueError(f"请在 .env 文件中设置 {var}")
        
        # 验证输出配置
        required_output_vars = ['CS_OUTPUT_DIR', 'CS_OUTPUT_FILENAME']
        for var in required_output_vars:
            if not os.getenv(var):
                raise ValueError(f"请在 .env 文件中设置 {var}")
        
        # --- 步骤 0: 初始化模型 ---
        print("--- 正在初始化AI模型 ---")
        summarizer = TableSummarizer()
        embedder = DashScopeEmbeddingGenerator(model="text-embedding-v4", dimensions=1024)
        print("--- AI模型初始化完成 ---")

        # --- 步骤 1: 一次性获取所有数据 ---
        print(f"\n--- 正在连接到数据库 {DB_CONFIG['host']}:{DB_CONFIG['port']} ---")
        connection = pymysql.connect(**DB_CONFIG)
        print("--- 成功连接到数据库 ---")

        with connection.cursor() as cursor:
            print(f"正在从 '{TABLE_NAMES['test_table']}' 表获取数据...")
            print(f"执行 SQL: {SQL_QUERIES['test_table']}")
            cursor.execute(SQL_QUERIES['test_table'])
            test_table_results = cursor.fetchall()

            print(f"正在从 '{TABLE_NAMES['items_doc']}' 表获取数据...")
            print(f"执行 SQL: {SQL_QUERIES['items_doc']}")
            cursor.execute(SQL_QUERIES['items_doc'])
            items_doc_results = cursor.fetchall()

        # --- 步骤 2: 在代码中对数据进行分组和拼接 ---
        print("\n--- 正在内存中对数据进行分组处理 ---")
        
        # 将 test_table 存入字典，并将id强制转为int，确保类型一致
        test_table_map = {int(item['id']): item['business_name'] for item in test_table_results}
        
        # 将 items_doc 按 uid 分组，并将uid强制转为int
        docs_by_uid = defaultdict(list)
        for doc in items_doc_results:
            docs_by_uid[int(doc['uid'])].append(doc)

        print(f"已找到 {len(test_table_map)} 个独立业务项，其中 {len(docs_by_uid)} 项有关联文档。")

        # --- 步骤 3: 循环处理每个ID，生成总结和向量 ---
        processed_knowledge_base = []
        for item_id, business_name in test_table_map.items():
            print(f"\n--- [正在处理 ID: {item_id}] ---")
            
            # 检查是否有关联文档
            if item_id in docs_by_uid:
                related_docs = docs_by_uid[item_id]
                
                # 按照用户要求的新格式拼接内容
                doc_strings = []
                for doc in related_docs:
                    doc_strings.append(f"doc_name:{doc['doc_name']}")
                    doc_strings.append(f"doc_text:{doc['doc_text']}")

                summary_input_text = f"[business_name：{business_name}，{'，'.join(doc_strings)}]"
                
                print(f"  1. 正在为 ID {item_id} 的完整内容生成总结...")
                
                completion = summarizer.client.chat.completions.create(
                    model=summarizer.model,
                    messages=[
                        {"role": "system", "content": "你是一名专业的档案管理员，请根据提供的业务名称和所有相关的文档内容，生成一段简短、精确、包含核心要点的摘要，用于后续的向量检索。"},
                        {"role": "user", "content": f"请为以下信息生成摘要: {summary_input_text}"}
                    ]
                )
                summary_text = completion.choices[0].message.content
                print(f"  2. 已生成总结: \"{summary_text[:50]}...\"")

                print(f"  3. 正在为该总结生成向量...")
                embedding_result = embedder.generate_embeddings(summary_text)
                vector = embedding_result[0]['vector']
                print(f"  4. 已生成向量 (维度: {len(vector)})")

                processed_knowledge_base.append({
                    "id": item_id,
                    "text": summary_text,
                    "vector": vector,
                    "source_data": {
                        "business_name": business_name,
                        "documents": [{"name": doc['doc_name'], "text": doc['doc_text']} for doc in related_docs]
                    }
                })
            else:
                # 没有关联文档的情况，只使用业务名称生成总结
                print(f"  注意: ID {item_id} 在 '{TABLE_NAMES['items_doc']}' 表中未找到关联文档，仅使用业务名称生成总结。")
                
                summary_input_text = f"[business_name：{business_name}]"
                
                print(f"  1. 正在为 ID {item_id} 的业务名称生成总结...")
                
                completion = summarizer.client.chat.completions.create(
                    model=summarizer.model,
                    messages=[
                        {"role": "system", "content": "你是一名专业的档案管理员，请根据提供的业务名称生成一段简短、精确、包含核心要点的摘要，用于后续的向量检索。"},
                        {"role": "user", "content": f"请为以下业务生成摘要: {summary_input_text}"}
                    ]
                )
                summary_text = completion.choices[0].message.content
                print(f"  2. 已生成总结: \"{summary_text[:50]}...\"")

                print(f"  3. 正在为该总结生成向量...")
                embedding_result = embedder.generate_embeddings(summary_text)
                vector = embedding_result[0]['vector']
                print(f"  4. 已生成向量 (维度: {len(vector)})")

                processed_knowledge_base.append({
                    "id": item_id,
                    "text": summary_text,
                    "vector": vector,
                    "source_data": {
                        "business_name": business_name,
                        "documents": []  # 空文档列表
                    }
                })
        
        # --- 步骤 4: 保存到文件 ---
        if not processed_knowledge_base:
            print(f"\n--- 没有可保存的数据，请检查数据库中 '{TABLE_NAMES['test_table']}.id' 与 '{TABLE_NAMES['items_doc']}.uid' 是否能匹配上。 ---")
            return

        output_dir = Path(OUTPUT_CONFIG['dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = output_dir / OUTPUT_CONFIG['filename']

        print(f"\n--- 正在将 {len(processed_knowledge_base)} 条处理结果保存到文件 ---")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(processed_knowledge_base, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 数据处理流水线完成！结果已成功保存到: {output_file_path}")

    except (pymysql.MySQLError, ValueError, Exception) as e:
        print(f"\n--- 发生错误 ---")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        if isinstance(e, ValueError) and "API Key" in str(e):
            print("提醒: 请检查您的 DASHSCOPE_API_KEY 环境变量是否已正确设置。")
        elif isinstance(e, ValueError):
            print("提醒: 请检查 .env 文件中的配置是否完整。")
        elif isinstance(e, pymysql.MySQLError):
            print("提醒: 请检查数据库连接配置是否正确。")

    finally:
        if connection:
            connection.close()
            print("\n--- 数据库连接已关闭 ---")

if __name__ == '__main__':
    process_data_pipeline_final()