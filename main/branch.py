# -*- coding: utf-8 -*-
"""
branch_1: 直连数据库按ID聚合两表数据并交给模型回答
用法:
    python branch_1.py 1 "居民户报装电话是多少？"
或者直接运行后按提示输入
"""

import os
import sys
import json
import pymysql
import pymysql.cursors
from dotenv import load_dotenv

# 设置项目根路径，保证可以导入 models 包
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.model_answer import DashScopeChatBot

# 加载 .env 文件
dotenv_path = os.path.join(current_script_path, '.env')
load_dotenv(dotenv_path=dotenv_path)

# 从环境变量中读取数据库配置
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_DATABASE'),
    'charset': os.getenv('DB_CHARSET'),
    'cursorclass': pymysql.cursors.DictCursor
}

# 从环境变量加载表名
TABLE_NAME_ITEMS_INFO = os.getenv('QA_TABLE_NAME_ITEMS_INFO')
TABLE_NAME_ITEMS_DOC = os.getenv('QA_TABLE_NAME_ITEMS_DOC')


def fetch_data_by_id(item_id: int):
    """根据 item_id 同时查询 items_info(id) 和 business_document(uid)"""
    # 检查表名是否已配置
    if not TABLE_NAME_ITEMS_INFO or not TABLE_NAME_ITEMS_DOC:
        print("ERROR: 数据库表名未在.env文件中配置。请检查 QA_TABLE_NAME_ITEMS_INFO 和 QA_TABLE_NAME_ITEMS_DOC。")
        return None, []
        
    connection = None
    try:
        connection = pymysql.connect(**DB_CONFIG)
        with connection.cursor() as cursor:
            # 基于环境变量中的表名动态构建SQL查询
            sql_item = f"SELECT * FROM {TABLE_NAME_ITEMS_INFO} WHERE id = %s;"
            cursor.execute(sql_item, (item_id,))
            item_row = cursor.fetchone()  # 期望唯一

            # 基于环境变量中的表名动态构建SQL查询
            sql_docs = f"SELECT * FROM {TABLE_NAME_ITEMS_DOC} WHERE uid = %s;"
            cursor.execute(sql_docs, (item_id,))
            docs_rows = cursor.fetchall() or []

        return item_row, docs_rows

    except Exception as e:
        print(f"ERROR: 数据库查询失败: {e}")
        return None, []
    finally:
        if connection:
            connection.close()


def build_context_payload(item_row, docs_rows):
    """
    将两表的原始结构化数据组织为一个字典，直接传给模型。
    不做字段名映射，保持数据库中的原始键值。
    """
    payload = {
        "items_info": item_row if item_row is not None else {},
        "items_doc": docs_rows

    }
    return payload


def main():
    # 解析命令行参数
    if len(sys.argv) >= 3:
        try:
            item_id = int(sys.argv[1])
        except ValueError:
            print("请提供正确的数值类型ID，例如: python branch_1.py 1 \"您的问题\"")
            return
        user_question = " ".join(sys.argv[2:])
    else:
        # 交互式输入
        try:
            item_id = int(input("请输入ID: ").strip())
        except ValueError:
            print("ID 需要是整数。")
            return
        user_question = input("请输入您的问题: ").strip()

    print(f"SYSTEM: 正在根据ID={item_id}查询数据库...")

    # 查询数据库
    item_row, docs_rows = fetch_data_by_id(item_id)

    # 组织上下文
    context_payload = build_context_payload(item_row, docs_rows)

    # 可选：打印调试信息，确认传入模型的数据内容
    print("\n--- [DEBUG] Context Payload (原始结构化数据) ---")
    try:
        print(json.dumps(context_payload, ensure_ascii=False, indent=2))
    except Exception:
        # 若存在非JSON可序列化内容，退化为字符串显示
        print(str(context_payload))
    print("---------------------------------------------\n")

    # 在模型回答前输出 JSON 格式的 ID 信息
    print(f'''json{{"id":"{item_id}"}}''')

    # 交给模型生成答案
    try:
        bot = DashScopeChatBot()
        print("AI: ", end="", flush=True)  # 开始输出标识
        answer_parts = []
        
        # 使用流式输出
        for chunk in bot.chat_stream(user_question=user_question, db_info=context_payload):
            print(chunk, end="", flush=True)  # 实时显示每个chunk
            answer_parts.append(chunk)  # 收集完整回答
        
        print()  # 换行
        answer = "".join(answer_parts)  # 组装完整回答（如果需要后续处理）
        
    except Exception as e:
        answer = f"模型调用失败: {e}"
        print(f"AI: {answer}")

    # print(f"AI: {answer}")


if __name__ == "__main__":
    main()