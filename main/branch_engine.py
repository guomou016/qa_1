# -*- coding: utf-8 -*-
"""
branch_1_engine: 封装 branch_1 功能，支持流式调用
"""

import os
import sys
import json
import pymysql
import pymysql.cursors
from typing import Generator, Tuple, Dict, List, Any, Optional
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


class Branch1Engine:
    """封装 branch_1 功能的引擎类"""
    
    def __init__(self):
        # 从环境变量加载数据库配置
        self.db_config = {
            'host': os.getenv('DB_HOST'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'database': os.getenv('DB_DATABASE'),
            'charset': os.getenv('DB_CHARSET'),
            'cursorclass': pymysql.cursors.DictCursor
        }
        # 从环境变量加载表名
        self.table_items_info = os.getenv('QA_TABLE_NAME_ITEMS_INFO')
        self.table_items_doc = os.getenv('QA_TABLE_NAME_ITEMS_DOC')
        
        self.bot = DashScopeChatBot()
    
    def fetch_data_by_id(self, item_id: int) -> Tuple[Optional[Dict], List[Dict]]:
        """根据 item_id 同时查询 items_info(id) 和 items_doc(uid)"""
        if not self.table_items_info or not self.table_items_doc:
            raise ValueError("数据库表名未在.env文件中配置。请检查 QA_TABLE_NAME_ITEMS_INFO 和 QA_TABLE_NAME_ITEMS_DOC。")

        connection = None
        try:
            connection = pymysql.connect(**self.db_config)
            with connection.cursor() as cursor:
                # 动态构建SQL查询
                sql_item = f"SELECT * FROM {self.table_items_info} WHERE id = %s;"
                cursor.execute(sql_item, (item_id,))
                item_row = cursor.fetchone()  # 期望唯一

                # 动态构建SQL查询
                sql_docs = f"SELECT * FROM {self.table_items_doc} WHERE uid = %s;"
                cursor.execute(sql_docs, (item_id,))
                docs_rows = cursor.fetchall() or []

            return item_row, docs_rows

        except Exception as e:
            raise Exception(f"数据库查询失败: {e}")
        finally:
            if connection:
                connection.close()
    
    def build_context_payload(self, item_row: Optional[Dict], docs_rows: List[Dict]) -> Dict[str, Any]:
        """
        将两表的原始结构化数据组织为一个字典，直接传给模型。
        不做字段名映射，保持数据库中的原始键值。
        """
        payload = {
            "items_info": item_row if item_row is not None else {},
            "items_doc": docs_rows
        }
        return payload
    
    def handle_query_stream(self, item_id: int, user_question: str) -> Generator[str, None, None]:
        """
        处理查询并返回流式响应
        
        Args:
            item_id: 数据库中的项目ID
            user_question: 用户问题
            
        Yields:
            str: 流式响应的文本块
        """
        try:
            # 查询数据库
            item_row, docs_rows = self.fetch_data_by_id(item_id)
            
            # 组织上下文
            context_payload = self.build_context_payload(item_row, docs_rows)
            
            # 输出 JSON 格式的 ID 信息
            yield f'json{{"id":"{item_id}"}}\n'
            
            # 使用流式输出生成答案
            for chunk in self.bot.chat_stream(user_question=user_question, db_info=context_payload):
                if chunk:
                    yield chunk
                    
        except Exception as e:
            yield f"处理失败: {str(e)}"
    
    def handle_query(self, item_id: int, user_question: str) -> Dict[str, Any]:
        """
        处理查询并返回完整响应（非流式）
        
        Args:
            item_id: 数据库中的项目ID
            user_question: 用户问题
            
        Returns:
            Dict: 包含答案和相关信息的字典
        """
        try:
            # 查询数据库
            item_row, docs_rows = self.fetch_data_by_id(item_id)
            
            # 组织上下文
            context_payload = self.build_context_payload(item_row, docs_rows)
            
            # 收集流式输出为完整答案
            answer_parts = []
            for chunk in self.bot.chat_stream(user_question=user_question, db_info=context_payload):
                if chunk:
                    answer_parts.append(chunk)
            
            answer = "".join(answer_parts)
            
            return {
                "item_id": item_id,
                "answer": answer,
                "context_info": {
                    "items_info_found": item_row is not None,
                    "docs_count": len(docs_rows)
                }
            }
            
        except Exception as e:
            return {
                "item_id": item_id,
                "answer": f"处理失败: {str(e)}",
                "error": str(e)
            }