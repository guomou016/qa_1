# -*- coding: utf-8 -*-
"""
主控制脚本 (Main Controller Script)

本脚本是整个问答系统的核心调度中心。它整合了多个AI模型的功能，
实现一个完整且具备多轮对话能力的问答流程：

1.  **路径管理**: 动态设置Python模块的搜索路径，确保可以跨文件夹导入自定义模块。
2.  **模型加载**: 初始化意图识别、向量检索和答案生成三个核心模型。
3.  **主循环**:
    - 接收用户输入。
    - 调用【意图识别模型】判断用户本次提问是否与上一轮相关。
    - 根据意图识别结果，维护一个【追问计数器】。
    - 调用【向量检索模块】在本地知识库中查找与用户问题最相关的信息。
    - 根据【追问计数器】和【检索结果数量】执行复杂的判断逻辑，决定下一步行动：
        a. 直接回答。
        b. 向用户澄清问题。
        c. 告知无法处理。
    - 将最终答复返回给用户。
    - 更新【意图识别模型】的对话历史，为其提供上下文。
"""

# --- 核心库导入 ---
import sys
import os
import json
import numpy as np
from dotenv import load_dotenv
import pymysql
import pymysql.cursors
import logging
import time
from datetime import datetime

# --- 1. 设置项目根目录，确保可以正确导入模块 ---
# 获取当前脚本 (main_1.py) 所在的目录的绝对路径
# 例如: C:\Users\pc123\Desktop\新事心办-1\main
current_script_path = os.path.dirname(os.path.abspath(__file__))
# 从当前目录向上回退一级，获取项目的根目录
# 例如: C:\Users\pc123\Desktop\新事心办-1
project_root = os.path.dirname(current_script_path)
# 将项目根目录添加到 Python 的模块搜索路径列表的最前面
# 这样做可以确保 `from models.xxx import ...` 这样的语句能够成功找到模块
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 2. 自定义JSON格式化器 ---
class JsonFormatter(logging.Formatter):
    """自定义JSON格式化器，将日志记录格式化为JSON字符串"""
    
    def format(self, record):
        # 创建日志字典
        log_dict = {
            'timestamp': datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': getattr(record, 'user_id', 'unknown'),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # 如果有异常信息，添加到字典中
        if record.exc_info:
            log_dict['exception'] = self.formatException(record.exc_info)
        
        # 返回JSON字符串
        return json.dumps(log_dict, ensure_ascii=False)

# --- 3. 设置日志配置 ---
def setup_logging():
    """设置日志系统"""
    # 创建logs目录（如果不存在）
    logs_dir = os.path.join(project_root, "logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # 统一的日志文件名（按用户ID区分的日志都存储在这个文件中）
    log_file = os.path.join(logs_dir, "qa_system_user_logs.log")
    
    # 创建根logger
    logger = logging.getLogger('QASystem')
    logger.setLevel(logging.INFO)
    
    # 清除已有的处理器（避免重复添加）
    logger.handlers.clear()
    
    # 创建文件处理器，使用JSON格式化器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(JsonFormatter())
    
    # 创建控制台处理器，使用普通格式（便于阅读）
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# --- 4. 导入你需要的自定义模块 ---
# 从 models 文件夹导入意图识别模型类
from models.Intent_recognition_model import QwenChat
# 从 rag 文件夹导入新的向量检索器
from rag.Vector_matching import VectorSearcher
# 从 models 文件夹导入最终答案生成的模型类
from models.model_answer import DashScopeChatBot

# --- 5. 加载环境变量 & 数据库配置 ---
# 构建 .env 文件的完整路径 (位于 main 文件夹下)
dotenv_path = os.path.join(current_script_path, '.env')
# 从指定的 .env 文件中加载环境变量 (主要是 DASHSCOPE_API_KEY)
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

# 从环境变量加载SQL查询语句
QA_SQL_QUERY_ITEM_DETAILS = os.getenv('QA_SQL_QUERY_ITEM_DETAILS')
QA_SQL_QUERY_BUSINESS_NAME = os.getenv('QA_SQL_QUERY_BUSINESS_NAME')


def main():
    """主函数，运行整个问答流程的无限循环。"""
    # 初始化日志系统
    logger = setup_logging()
    
    # 获取用户ID
    user_id = input("请输入您的用户ID: ").strip()
    if not user_id:
        user_id = "anonymous"
    
    # 创建带用户ID的LoggerAdapter
    logger_adapter = logging.LoggerAdapter(logger, {'user_id': user_id})
    
    # 移除系统启动日志
    # logger_adapter.info("=== 智能问答系统启动 ===")
    print("--- 智能问答系统已启动 ---")
    
    try:
        # 初始化意图识别模型（移除日志）
        start_time = time.time()
        intent_recognizer = QwenChat()
        init_time = time.time() - start_time
        
        # 初始化最终答案生成模型（移除日志）
        start_time = time.time()
        answer_generator = DashScopeChatBot()
        init_time = time.time() - start_time
        
    except (ValueError, Exception) as e:
        # 保留错误日志
        logger_adapter.error(f"模型初始化失败: {e}")
        print(f"初始化模型失败: {e}")
        print("请确保 .env 文件中已正确配置 DASHSCOPE_API_KEY。")
        return

    # 构建知识库文件的绝对路径
    db_path = os.path.join(project_root, "db", "knowledge_base_final.json")
    
    # 初始化新的向量检索器（移除日志）
    try:
        start_time = time.time()
        searcher = VectorSearcher(db_path=db_path)
        init_time = time.time() - start_time
    except Exception as e:
        logger_adapter.error(f"向量检索器初始化失败: {e}")
        print(f"初始化向量检索器失败: {e}")
        return

    # 用于存储意图识别模型上下文的列表
    intent_history = []
    # 初始化意图追问计数器
    int_id = 0
    # 会话计数器
    session_count = 0

    # 检查SQL查询是否已配置
    if not QA_SQL_QUERY_ITEM_DETAILS or not QA_SQL_QUERY_BUSINESS_NAME:
        logger_adapter.error("SQL查询语句未在.env文件中配置。请检查 QA_SQL_QUERY_ITEM_DETAILS 和 QA_SQL_QUERY_BUSINESS_NAME。")
        print("错误：SQL查询语句未配置，请检查.env文件。")
        return

    # 移除组件初始化完成日志
    # logger_adapter.info("所有组件初始化完成，开始主交互循环")

    # 启动主交互循环
    while True:
        user_query = input("\n你: ")
        if user_query.lower() in ["exit", "退出", "quit"]:
            # 移除退出日志
            # logger_adapter.info("用户选择退出系统")
            print("AI: 再见！")
            break

        session_count += 1
        session_start_time = time.time()
        
        # 初始化本轮对话的日志数据
        session_log_data = {
            "user_id": user_id,
            "session_count": session_count,
            "user_query": user_query,
            "intent_result": None,
            "int_id": None,
            "vector_matches": [],
            "sql_data": None,
            "assistant_response": None
        }

        # --- 步骤 1: 意图识别 ---
        try:
            intent_start_time = time.time()
            # 调用意图模型，传入当前问题和历史对话，判断意图
            intent_response = intent_recognizer.chat(user_query, history=intent_history)
            intent_result = intent_response.choices[0].message.content.strip().lower()
            intent_time = time.time() - intent_start_time
            
            session_log_data["intent_result"] = intent_result
            print(f"[意图识别模型分析: {intent_result}]") # 打印内部状态，供调试
            
        except Exception as e:
            logger_adapter.error(f"意图识别过程中发生错误: {e}")
            intent_result = 't2'  # 默认为新话题
            session_log_data["intent_result"] = intent_result

        # --- 步骤 1.1: 更新意图追问计数器 ---
        old_int_id = int_id
        
        if intent_result == 't1':
            int_id += 1  # 如果意图相同 (t1)，计数器加1
        else:
            int_id = 1   # 如果意图改变 (t2)，重置计数器为1

        session_log_data["int_id"] = int_id
        print(f"DEBUG: 意图识别后, int_id = {int_id}")

        # --- 步骤 2: 向量检索 ---
        try:
            search_start_time = time.time()
            # 启用DEBUG模式进行检索，并使用您设置的0.5阈值
            search_results = searcher.search(user_query, top_k=3, similarity_threshold=0.5, debug=True)
            search_time = time.time() - search_start_time
            num_results = len(search_results)
            
            # 记录向量检索结果
            vector_matches = []
            for result in search_results:
                vector_matches.append({
                    "id": result.get('id'),
                    "similarity": result.get('similarity', 0),
                    "text": result.get('text', '')
                })
            session_log_data["vector_matches"] = vector_matches
                
        except Exception as e:
            logger_adapter.error(f"向量检索过程中发生错误: {e}")
            search_results = []
            num_results = 0
            session_log_data["vector_matches"] = []
        
        assistant_response = ""

        # --- 步骤 3: 核心判断逻辑 (简化版) ---
        if num_results > 0 and int_id < 3:
            
            if num_results == 1:
                result = search_results[0]
                item_id = result.get('id')
                similarity_score = result['similarity']
                print(f"SYSTEM: 已找到1条相关信息 (ID: {item_id}, 相似度: {similarity_score:.4f})，正在从数据库实时获取详细信息...")

                # 先输出获取到的数据库items_info表的id
                print(f"\n获取到的数据库items_info表ID: {item_id}")

                context_for_model = ""
                connection = None
                try:
                    # 步骤 A: 实时连接数据库
                    db_start_time = time.time()
                    connection = pymysql.connect(**DB_CONFIG)
                    
                    with connection.cursor() as cursor:
                        # 步骤 B: 查询与ID相关的所有原始信息 (从环境变量获取SQL)
                        cursor.execute(QA_SQL_QUERY_ITEM_DETAILS, (item_id,))
                        db_results = cursor.fetchall()
                        db_time = time.time() - db_start_time
                        
                        # 记录SQL查询结果
                        session_log_data["sql_data"] = db_results

                    if db_results:
                        # 步骤 C: 格式化从数据库获取的完整数据作为上下文
                        
                        # 1. 提取唯一的业务信息 (所有行都一样，取第一行即可)
                        item_info_data = {k: v for k, v in db_results[0].items() if k not in ['doc_name', 'doc_text']}
                        
                        context_lines = ["--- 业务详情 ---"]
                        for key, value in item_info_data.items():
                            if value is not None:
                                context_lines.append(f"{key}: {value}")

                        # 2. 提取所有关联的文档信息
                        document_lines = []
                        for row in db_results:
                            doc_name = row.get('doc_name')
                            doc_text = row.get('doc_text')
                            if doc_name: # 确保文档名存在
                                document_lines.append(f"- {doc_name}: {doc_text or '无'}")
                        
                        if document_lines:
                            context_lines.append("\n--- 相关文档 ---")
                            context_lines.extend(document_lines)
                        
                        context_for_model = "\n".join(context_lines)
                        print("SYSTEM: 数据库信息获取成功，上下文准备完毕。")
                        
                        # --- 打印从数据库获取并格式化后的详细内容 ---
                        print("\n--- [DEBUG] Context for Model ---")
                        print(context_for_model)
                        print("---------------------------------\n")
                        
                    else:
                        # 如果数据库中没有找到（理论上不应发生），则回退
                        context_for_model = result.get('text', '无可用信息')
                        print(f"WARNING: 在数据库中未找到 ID 为 {item_id} 的信息，将使用总结文本作为上下文。")

                except Exception as e:
                    print(f"ERROR: 数据库查询失败: {e}")
                    print("WARNING: 将使用总结文本作为上下文。")
                    context_for_model = result.get('text', '无可用信息')
                    session_log_data["sql_data"] = None
                finally:
                    if connection:
                        connection.close()

                # 生成答案 - 使用流式输出
                
                # 先输出JSON格式的ID信息
                id_json = json.dumps({"id": item_id}, ensure_ascii=False)
                print(f"\n```json\n{id_json}\n```")
                
                print("\nAI: ", end="", flush=True)  # 开始AI回答，不换行
                try:
                    answer_start_time = time.time()
                    answer_parts = []  # 收集完整回答
                    
                    # 使用流式输出
                    for chunk in answer_generator.chat_stream(user_query, context_for_model):
                        print(chunk, end="", flush=True)  # 实时显示每个chunk
                        answer_parts.append(chunk)  # 收集完整回答
                    
                    print()  # 换行
                    assistant_response = "".join(answer_parts)  # 组装完整回答
                    session_log_data["assistant_response"] = assistant_response
                    
                    answer_time = time.time() - answer_start_time
                except Exception as e:
                    logger_adapter.error(f"答案生成失败: {e}")
                    assistant_response = "抱歉，生成回答时出现了技术问题，请稍后重试。"
                    session_log_data["assistant_response"] = assistant_response
                    print(assistant_response)
            else: # num_results > 1
                
                # 当找到多个相关结果时，以JSON格式返回业务名称供用户选择
                business_name_list = []
                connection = None
                try:
                    db_start_time = time.time()
                    connection = pymysql.connect(**DB_CONFIG)
                    with connection.cursor() as cursor:
                        for i, result in enumerate(search_results):
                            item_id = result.get('id')
                            
                            # 查询数据库获取业务名称（同时取回id） (从环境变量获取SQL)
                            cursor.execute(QA_SQL_QUERY_BUSINESS_NAME, (item_id,))
                            db_result = cursor.fetchone()
                            
                            if db_result and db_result.get('business_name'):
                                business_name_list.append({
                                    'id': db_result['id'],
                                    'business_name': db_result['business_name']
                                })
                            else:
                                # 如果数据库中没找到，则用检索ID和摘要文本作为后备
                                fallback_name = result.get('text', '未知业务')
                                business_name_list.append({
                                    'id': item_id,
                                    'business_name': fallback_name
                                })
                                print(f"WARNING: 未能为 ID {item_id} 找到 business_name，使用后备信息。")
                    
                    db_time = time.time() - db_start_time
                    
                    # 记录SQL查询结果
                    session_log_data["sql_data"] = business_name_list

                    # 组装最终返回：包含 old_query 与 options
                    assistant_response = json.dumps(
                        {
                            "old_query": user_query,
                            "options": business_name_list
                        },
                        indent=4,
                        ensure_ascii=False
                    )
                    session_log_data["assistant_response"] = assistant_response

                except Exception as e:
                    logger_adapter.error(f"多结果分支数据库查询失败: {e}")
                    print(f"ERROR: 数据库查询失败 (多结果分支): {e}")
                    # 如果数据库查询失败，使用检索结果作为后备，但仍保持输出结构
                    fallback_options = []
                    for result in search_results:
                        fallback_options.append({
                            'id': result.get('id'),
                            'business_name': result.get('text', '未知业务')
                        })
                    
                    session_log_data["sql_data"] = None
                    assistant_response = json.dumps(
                        {
                            "old_query": user_query,
                            "options": fallback_options
                        },
                        indent=4,
                        ensure_ascii=False
                    )
                    session_log_data["assistant_response"] = assistant_response
                finally:
                    if connection:
                        connection.close()

                print("\nAI: ")
                print(assistant_response)
        else:
            # 无法回答的情况
            if num_results == 0:
                assistant_response = "抱歉，我在知识库中没有找到与您问题相关的信息。请尝试换个方式提问，或者联系相关部门获取帮助。"
            else:  # int_id >= 3
                assistant_response = "您已经连续追问了多次，我可能无法准确理解您的需求。建议您重新描述问题，或者联系相关部门获取更详细的帮助。"
            
            session_log_data["assistant_response"] = assistant_response
            print(f"\nAI: {assistant_response}")

        # 记录本轮对话的完整日志（一行一个字典）
        session_log_json = json.dumps(session_log_data, ensure_ascii=False)
        logger_adapter.info(session_log_json)

        # 更新意图识别模型的历史记录
        intent_history.append(user_query)
        intent_history.append(assistant_response)
        
        # 限制历史记录长度，避免过长
        if len(intent_history) > 10:
            intent_history = intent_history[-10:]
        
        # 移除历史对话更新日志
        # logger_adapter.info(f"历史对话已更新，当前历史长度: {len(intent_history)}")
        # 移除轮次结束日志
        # logger_adapter.info(f"=== 第 {session_count} 轮对话结束 ===\n")

        session_time = time.time() - session_start_time

        # 删除这行重复输出
        # print(f"AI: {assistant_response}")

        # 将本轮的用户问题和AI回答追加到历史记录中
        # 这使得下一轮的意图识别模型能"记住"这次对话
        intent_history.append({"role": "user", "content": user_query})
        intent_history.append({"role": "assistant", "content": assistant_response})
        
        # logger_adapter.info(f"历史对话已更新，当前历史长度: {len(intent_history)}")
        # logger_adapter.info(f"=== 第 {session_count} 轮对话结束 ===\n")


# 当该脚本作为主程序直接运行时，调用 main() 函数
if __name__ == "__main__":
    main()