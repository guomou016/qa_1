# -*- coding: utf-8 -*-
import sys
import os
import json
import time
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

import pymysql
import pymysql.cursors
from dotenv import load_dotenv

# 路径设置，确保可以导入 models/ 和 rag/
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 加载环境变量
dotenv_path = os.path.join(current_script_path, '.env')
load_dotenv(dotenv_path=dotenv_path)

# 导入模型
from models.Intent_recognition_model import QwenChat
from rag.Vector_matching import VectorSearcher
from models.model_answer import DashScopeChatBot


def setup_simple_logging():
    """设置简单的用户日志系统"""
    logs_dir = os.path.join(project_root, "logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # 固定的用户日志文件
    log_file = os.path.join(logs_dir, "qa_system_user_logs.log")
    
    # 创建专用的用户日志记录器
    user_logger = logging.getLogger("QASystem")
    user_logger.setLevel(logging.INFO)
    
    # 避免重复添加handler
    if not user_logger.handlers:
        handler = logging.FileHandler(log_file, encoding='utf-8')
        user_logger.addHandler(handler)
    
    return user_logger


DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_DATABASE'),
    'charset': os.getenv('DB_CHARSET'),
    'cursorclass': pymysql.cursors.DictCursor
}

# QA Engine 专用 SQL 查询配置
QA_SQL_QUERIES = {
    'item_details': os.getenv('QA_SQL_QUERY_ITEM_DETAILS'),
    'business_name': os.getenv('QA_SQL_QUERY_BUSINESS_NAME')
}

# QA Engine 表名配置
QA_TABLE_NAMES = {
    'items_info': os.getenv('QA_TABLE_NAME_ITEMS_INFO'),
    'items_doc': os.getenv('QA_TABLE_NAME_ITEMS_DOC')
}


class QASessionState:
    """会话状态：用于维持多轮对话上下文"""
    def __init__(self, session_id=None):
        self.intent_history: List[Dict[str, str]] = []
        self.int_id: int = 0  # 追问计数器
        # 生成唯一的会话ID
        if session_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_id = f"session_{timestamp}_{str(uuid.uuid4())[:8]}"
        else:
            self.session_id = session_id


class QAEngine:
    """问答引擎：封装原 main_2 的核心逻辑，便于在API/脚本中复用"""
    def __init__(self, db_path: Optional[str] = None, similarity_threshold: float = 0.5, session_id: Optional[str] = None):
        # 初始化简单的用户日志系统
        self.user_logger = setup_simple_logging()
        
        # 检查关键配置
        if not all(DB_CONFIG.values()) or not all(QA_SQL_QUERIES.values()):
            error_msg = "数据库配置或SQL查询未在.env文件中完全配置。"
            raise ValueError(error_msg)

        # 模型初始化
        self.intent_recognizer = QwenChat()
        self.answer_generator = DashScopeChatBot()

        # 检索器初始化
        if db_path is None:
            db_path = os.path.join(project_root, "db", "knowledge_base_final.json")
        self.searcher = VectorSearcher(db_path=db_path)

        self.similarity_threshold = similarity_threshold
    
    def _log_user_interaction(self, user_id: str, session_count: int, user_query: str, 
                            intent_result: str, int_id: int, vector_matches: list, 
                            sql_data: list, assistant_response: str):
        """记录用户交互日志"""
        log_data = {
            "user_id": user_id,
            "session_count": session_count,
            "user_query": user_query,
            "intent_result": intent_result,
            "int_id": int_id,
            "vector_matches": vector_matches,
            "sql_data": sql_data,
            "assistant_response": assistant_response
        }
        
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": user_id,
            "level": "INFO",
            "logger": "QASystem",
            "message": json.dumps(log_data, ensure_ascii=False),
            "module": "main",
            "function": "main",
            "line": 459
        }
        
        self.user_logger.info(json.dumps(log_entry, ensure_ascii=False))

    def handle_query(self, user_query: str, session: QASessionState, user_id: str = "default_user") -> Dict[str, Any]:
        """处理一次用户问题，返回结构化结果"""
        session_start = time.time()
        resp: Dict[str, Any] = {"type": "", "data": {}, "meta": {}}

        # 步骤1：意图识别
        try:
            intent_response = self.intent_recognizer.chat(user_query, history=session.intent_history)
            intent_result = intent_response.choices[0].message.content.strip().lower()
        except Exception as e:
            intent_result = "t2"  # 默认当作新话题

        # 更新追问计数器
        if intent_result == "t1":
            session.int_id += 1
        else:
            session.int_id = 1

        # 步骤2：向量检索
        try:
            search_results = self.searcher.search(
                user_query, top_k=3, similarity_threshold=self.similarity_threshold, debug=True
            )
            num_results = len(search_results)
        except Exception as e:
            search_results = []
            num_results = 0

        # 准备日志数据
        vector_matches = []
        sql_data = []
        
        # 步骤3：分支处理
        if num_results > 0 and session.int_id < 3:
            if num_results == 1:
                # 单一结果：从DB取详情 -> 生成答案
                result = search_results[0]
                item_id = result.get("id")
                
                # 记录向量匹配结果
                vector_matches.append({
                    "id": item_id,
                    "similarity": result.get("similarity", 0.0),
                    "text": result.get("text", "")
                })

                context_for_model = ""
                connection = None
                try:
                    connection = pymysql.connect(**DB_CONFIG)
                    with connection.cursor() as cursor:
                        cursor.execute(QA_SQL_QUERIES['item_details'], (item_id,))
                        db_results = cursor.fetchall()
                        
                        # 记录SQL数据
                        sql_data = db_results

                    if db_results:
                        # 组装上下文
                        item_info = {k: v for k, v in db_results[0].items() if k not in ['doc_name', 'doc_text']}
                        ctx_lines = ["--- 业务详情 ---"]
                        for k, v in item_info.items():
                            if v is not None:
                                ctx_lines.append(f"{k}: {v}")
                        doc_lines = []
                        for row in db_results:
                            name = row.get('doc_name')
                            text = row.get('doc_text')
                            if name:
                                doc_lines.append(f"- {name}: {text or '无'}")
                        if doc_lines:
                            ctx_lines.append("\n--- 相关文档 ---")
                            ctx_lines.extend(doc_lines)
                        context_for_model = "\n".join(ctx_lines)
                    else:
                        context_for_model = result.get("text", "无可用信息")
                except Exception as e:
                    context_for_model = result.get("text", "无可用信息")
                finally:
                    if connection:
                        connection.close()

                # 生成答案
                try:
                    answer = self.answer_generator.chat(user_query, context_for_model)
                except Exception as e:
                    answer = "抱歉，生成回答时出现了技术问题，请稍后重试。"

                resp["type"] = "answer"
                resp["data"] = {"answer": answer}
            else:
                # 多结果：返回选项列表
                business_name_list = []
                connection = None
                try:
                    connection = pymysql.connect(**DB_CONFIG)
                    with connection.cursor() as cursor:
                        for r in search_results:
                            item_id = r.get("id")
                            cursor.execute(QA_SQL_QUERIES['business_name'], (item_id,))
                            row = cursor.fetchone()
                            if row and row.get("business_name"):
                                business_name_list.append({
                                    "id": row["id"],
                                    "business_name": row["business_name"]
                                })
                            else:
                                business_name_list.append({
                                    "id": item_id,
                                    "business_name": r.get("text", "未知业务")
                                })
                except Exception as e:
                    business_name_list = [
                        {"id": r.get("id"), "business_name": r.get("text", "未知业务")}
                        for r in search_results
                    ]
                finally:
                    if connection:
                        connection.close()

                resp["type"] = "options"
                resp["data"] = {
                    "old_query": user_query,
                    "options": business_name_list
                }
        else:
            # 无结果或追问超限
            resp["type"] = "answer"
            resp["data"] = {
                "answer": "抱歉，我在知识库中没有找到与您问题直接相关的信息。"
            }

        # 更新会话历史
        answer_for_history = resp["data"]["answer"] if resp["type"] == "answer" else json.dumps(resp["data"], ensure_ascii=False)
        session.intent_history.append({"role": "user", "content": user_query})
        session.intent_history.append({"role": "assistant", "content": answer_for_history})

        # 记录用户交互日志
        self._log_user_interaction(
            user_id=user_id,
            session_count=len(session.intent_history) // 2,
            user_query=user_query,
            intent_result=intent_result,
            int_id=session.int_id,
            vector_matches=vector_matches,
            sql_data=sql_data,
            assistant_response=answer_for_history
        )

        # 元数据
        resp["meta"] = {
            "intent": intent_result,
            "int_id": session.int_id,
            "num_results": num_results,
            "duration_sec": round(time.time() - session_start, 3)
        }
        return resp

    def handle_query_stream(self, query: str, session: QASessionState, user_id: str = "default_user"):
        """
        处理用户查询（流式版本），并流式返回结果。
        """
        try:
            # 步骤1：意图识别
            try:
                intent_response = self.intent_recognizer.chat(query, history=session.intent_history)
                intent_result = intent_response.choices[0].message.content.strip().lower()
            except Exception:
                intent_result = "t2"  # 发生异常时，默认为新话题

            # 更新追问计数器
            if intent_result == "t1":
                session.int_id += 1
            else:
                session.int_id = 1

            # 步骤2：向量检索
            try:
                search_results = self.searcher.search(
                    query, top_k=3, similarity_threshold=self.similarity_threshold, debug=True
                )
                num_results = len(search_results)
            except Exception:
                search_results = []
                num_results = 0

            # 准备日志数据
            vector_matches = []
            sql_data = []
            assistant_response = ""

            # 步骤3：生成流式答案
            if num_results > 0 and session.int_id < 3:
                if num_results == 1:
                    # 单一结果：获取上下文并流式生成答案
                    result = search_results[0]
                    item_id = result.get("id")
                    
                    vector_matches.append({
                        "id": item_id,
                        "similarity": result.get("similarity", 0.0),
                        "text": result.get("text", "")
                    })
                    
                    id_json = f"```json\\n{{\\\"id\\\": \\\"{item_id}\\\"}}\\n```\\n\\n"
                    yield id_json
                    
                    context_for_model = self._get_context_for_item(item_id, result)
                    
                    connection = None
                    try:
                        connection = pymysql.connect(**DB_CONFIG)
                        with connection.cursor() as cursor:
                            cursor.execute(QA_SQL_QUERIES['item_details'], (item_id,))
                            sql_data = cursor.fetchall()
                    finally:
                        if connection:
                            connection.close()
                    
                    answer_chunks = []
                    for chunk in self.answer_generator.chat_stream(query, context_for_model):
                        answer_chunks.append(chunk)
                        yield chunk
                    assistant_response = "".join(answer_chunks)
                else:
                    # 多结果：返回结构化选项数据
                    business_name_list = self._get_structured_options(search_results)
                    options_data = {
                        "type": "options",
                        "data": {
                            "old_query": query,
                            "options": business_name_list
                        }
                    }
                    assistant_response = json.dumps(options_data, ensure_ascii=False)
                    yield assistant_response
            else:
                # 无结果
                assistant_response = "抱歉，我在知识库中没有找到与您问题直接相关的信息。"
                yield assistant_response
            
            # 更新会话历史
            session.intent_history.append({"role": "user", "content": query})
            session.intent_history.append({"role": "assistant", "content": assistant_response})

            # 在流式处理的最后记录用户交互
            self._log_user_interaction(
                user_id=user_id,
                session_count=len(session.intent_history) // 2,
                user_query=query,
                intent_result=intent_result,
                int_id=session.int_id,
                vector_matches=vector_matches,
                sql_data=sql_data,
                assistant_response=assistant_response
            )

        except Exception as e:
            self.logger.error(f"Error in handle_query_stream: {e}", exc_info=True)
            # 在流中返回错误信息
            yield json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)

    def _get_structured_options(self, search_results):
        """获取结构化的选项数据"""
        business_name_list = []
        connection = None
        try:
            connection = pymysql.connect(**DB_CONFIG)
            with connection.cursor() as cursor:
                for r in search_results:
                    item_id = r.get("id")
                    cursor.execute(QA_SQL_QUERIES['business_name'], (item_id,))
                    row = cursor.fetchone()
                    if row and row.get("business_name"):
                        business_name_list.append({
                            "id": row["id"],
                            "business_name": row["business_name"]
                        })
                    else:
                        business_name_list.append({
                            "id": item_id,
                            "business_name": r.get("text", "未知业务")
                        })
        except Exception as e:
            business_name_list = [
                {"id": r.get("id"), "business_name": r.get("text", "未知业务")}
                for r in search_results
            ]
        finally:
            if connection:
                connection.close()
        
        return business_name_list
    
    def _get_context_for_item(self, item_id, result):
        """获取单个项目的上下文信息"""
        context_for_model = ""
        connection = None
        try:
            connection = pymysql.connect(**DB_CONFIG)
            with connection.cursor() as cursor:
                cursor.execute(QA_SQL_QUERIES['item_details'], (item_id,))
                db_results = cursor.fetchall()

            if db_results:
                # 组装上下文
                item_info = {k: v for k, v in db_results[0].items() if k not in ['doc_name', 'doc_text']}
                ctx_lines = ["--- 业务详情 ---"]
                for k, v in item_info.items():
                    if v is not None:
                        ctx_lines.append(f"{k}: {v}")
                doc_lines = []
                for row in db_results:
                    name = row.get('doc_name')
                    text = row.get('doc_text')
                    if name:
                        doc_lines.append(f"- {name}: {text or '无'}")
                if doc_lines:
                    ctx_lines.append("\n--- 相关文档 ---")
                    ctx_lines.extend(doc_lines)
                context_for_model = "\n".join(ctx_lines)
            else:
                context_for_model = result.get("text", "无可用信息")
        except Exception as e:
            context_for_model = result.get("text", "无可用信息")
        finally:
            if connection:
                connection.close()
        
        return context_for_model
    
    def _format_options(self, search_results):
        """格式化多个选项"""
        options_text = "我找到了多个相关业务，请选择您需要了解的具体业务：\n\n"
        for i, result in enumerate(search_results, 1):
            options_text += f"{i}. {result.get('text', '未知业务')[:100]}...\n"
        return options_text