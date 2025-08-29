# -*- coding: utf-8 -*-
import os
import sys
import json
import pymysql
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

import sqlite3

# 设置路径
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
main_dir = os.path.join(project_root, "main")

# 加载环境变量
dotenv_path = os.path.join(main_dir, '.env')
load_dotenv(dotenv_path=dotenv_path)

# 数据库配置
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'port': int(os.getenv('DB_PORT', 3306)),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_DATABASE'),
    'charset': os.getenv('DB_CHARSET', 'utf8mb4'),
    'cursorclass': pymysql.cursors.DictCursor
}

def init_db():
    conn = sqlite3.connect('db/knowledge_base_final.db')
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS items_info_ (
            id INTEGER PRIMARY KEY,
            items_type TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS items_doc_ (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uid INTEGER,
            doc_name TEXT,
            img_path TEXT,
            FOREIGN KEY (uid) REFERENCES items_info_(id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS items_annex_mark (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            uid INTEGER,
            annex_name TEXT,
            annex_path TEXT,
            FOREIGN KEY (uid) REFERENCES items_info_(id)
        )
    ''')

    # Insert dummy data for "不填表" case (id=1)
    cursor.execute("INSERT OR IGNORE INTO items_info_ (id, items_type) VALUES (1, '不填表')")
    cursor.execute("INSERT OR IGNORE INTO items_doc_ (uid, doc_name, img_path) VALUES (1, 'doc1_不填表', 'img1_path_不填表')")
    cursor.execute("INSERT OR IGNORE INTO items_doc_ (uid, doc_name, img_path) VALUES (1, 'doc2_不填表', 'img2_path_不填表')")

    # Insert dummy data for "填表" case (id=2)
    cursor.execute("INSERT OR IGNORE INTO items_info_ (id, items_type) VALUES (2, '填表')")
    cursor.execute("INSERT OR IGNORE INTO items_doc_ (uid, doc_name, img_path) VALUES (2, 'doc1_填表', 'img1_path_填表')")
    cursor.execute("INSERT OR IGNORE INTO items_doc_ (uid, doc_name, img_path) VALUES (2, 'doc2_填表', 'img2_path_填表')")
    cursor.execute("INSERT OR IGNORE INTO items_annex_mark (uid, annex_name, annex_path) VALUES (2, 'annex_table_A', 'annex_path_A')")
    cursor.execute("INSERT OR IGNORE INTO items_annex_mark (uid, annex_name, annex_path) VALUES (2, 'annex_table_B', 'annex_path_B')")

    # Create dynamic tables for "填表" case
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS annex_table_A (
            col1 TEXT,
            col2 TEXT
        )
    ''')
    cursor.execute("INSERT OR IGNORE INTO annex_table_A (col1, col2) VALUES ('valueA1', 'valueA2')")
    cursor.execute("INSERT OR IGNORE INTO annex_table_A (col1, col2) VALUES ('valueA3', 'valueA4')")

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS annex_table_B (
            fieldX TEXT,
            fieldY TEXT
        )
    ''')
    cursor.execute("INSERT OR IGNORE INTO annex_table_B (fieldX, fieldY) VALUES ('valueB1', 'valueB2')")
    cursor.execute("INSERT OR IGNORE INTO annex_table_B (fieldX, fieldY) VALUES ('valueB3', 'valueB4')")

    conn.commit()
    conn.close()


def get_data_by_id(item_id):
    conn = None
    try:
        conn = sqlite3.connect('db/knowledge_base_final.db')
        cursor = conn.cursor()

        # 1. Query items_info_ to get items_type
        cursor.execute("SELECT items_type FROM items_info_ WHERE id = ?", (item_id,))
        item_info = cursor.fetchone()

        if not item_info:
            return f"No item found for id: {item_id}"

        items_type = item_info[0]
        results = []

        # 2. Query items_doc_
        cursor.execute("SELECT doc_name, img_path FROM items_doc_ WHERE uid = ?", (item_id,))
        docs = cursor.fetchall()

        if items_type == "不填表":
            for doc in docs:
                results.append({"doc_name": doc[0], "img_path": doc[1]})
        elif items_type == "填表":
            # 3. Query items_annex_mark
            cursor.execute("SELECT annex_name, annex_path FROM items_annex_mark WHERE uid = ?", (item_id,))
            annexes = cursor.fetchall()

            # Combine docs and annexes, handling multiple docs and annexes correctly
            # Create a base structure for each doc, then add annex details
            for doc in docs:
                base_item = {"doc_name": doc[0], "img_path": doc[1]}
                if not annexes:
                    # If no annexes, just append the doc info
                    results.append(base_item)
                else:
                    # For each doc, iterate through annexes and add them
                    for annex in annexes:
                        annex_name = annex[0]
                        annex_path = annex[1]
                        
                        # 4. Query dynamic table named by annex_name
                        table_data = []
                        try:
                            # IMPORTANT: Direct string interpolation for table name, be cautious.
                            # This assumes annex_name values are trusted table names.
                            cursor.execute(f"PRAGMA table_info({annex_name})")
                            columns_info = cursor.fetchall()
                            column_names = [col[1] for col in columns_info]

                            cursor.execute(f"SELECT * FROM {annex_name}")
                            rows = cursor.fetchall()

                            for row in rows:
                                table_data.append(dict(zip(column_names, row)))
                        except sqlite3.OperationalError as e:
                            print(f"Warning: Could not query table {annex_name}. Error: {e}")
                            table_data = f"Error: Table {annex_name} not found or accessible."

                        combined_item = base_item.copy()
                        combined_item.update({
                            "annex_name": annex_name,
                            "annex_path": annex_path,
                            "table": table_data
                        })
                        results.append(combined_item)
        return results

    except sqlite3.Error as e:
        return f"Database error: {e}"
    finally:
        if conn:
            conn.close()


# Call init_db() once to set up the database and populate it with initial data.
# After the first run, you should comment this line out to prevent re-initialization.
# init_db()


def get_item_data(item_id: int) -> List[Dict[str, Any]]:
    """
    根据item_id获取相关数据
    
    Args:
        item_id: 要查询的项目ID
    
    Returns:
        包含所有相关数据的列表，每个元素是一个字典
    """
    result_list = []
    connection = None
    
    try:
        # 连接数据库
        connection = pymysql.connect(**DB_CONFIG)
        
        with connection.cursor() as cursor:
            # 1. 首先查询items_info_表获取items_type
            sql_get_type = "SELECT items_type FROM items_info_ WHERE id = %s"
            cursor.execute(sql_get_type, (item_id,))
            type_result = cursor.fetchone()
            
            if not type_result:
                return []  # 如果找不到对应的item_id，返回空列表
            
            items_type = type_result['items_type']
            
            # 如果是"不填表"，只从items_doc_表获取文档信息
            if items_type == '不填表':
                sql_get_docs = """
                    SELECT doc_name, img_path 
                    FROM items_doc_ 
                    WHERE uid = %s
                """
                cursor.execute(sql_get_docs, (item_id,))
                doc_results = cursor.fetchall()
                for doc in doc_results:
                    result_list.append({
                        'doc_name': doc['doc_name'],
                        'img_path': doc['img_path']
                    })
            
            # 如果是"填表"，只获取附件信息和表数据
            elif items_type == '填表':
                # 3. 获取附件信息
                sql_get_annex = """
                    SELECT annex_name, annex_path 
                    FROM items_annex_mark 
                    WHERE uid = %s
                """
                cursor.execute(sql_get_annex, (item_id,))
                annex_results = cursor.fetchall()
                
                # 4. 对每个附件进行处理
                for annex in annex_results:
                    annex_info = {
                        'annex_name': annex['annex_name'],
                        'annex_path': annex['annex_path']
                    }
                    
                    # 5. 获取附件表中的数据
                    if annex['annex_name']:
                        try:
                            # 注意：这里应该添加表名验证以防止SQL注入
                            table_name = annex['annex_name']
                            # 验证表名是否合法（只允许字母、数字和下划线）
                            if not table_name.replace('_', '').isalnum():
                                continue
                                
                            sql_get_table_data = f"SELECT * FROM {table_name}"
                            cursor.execute(sql_get_table_data)
                            table_data = cursor.fetchall()
                            annex_info['table'] = table_data
                        except Exception as e:
                            # 如果表不存在或查询出错，设置为空列表
                            annex_info['table'] = []
                    
                    result_list.append(annex_info)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return []
    
    finally:
        if connection:
            connection.close()
    
    return result_list

def test_get_item_data(item_id: int):
    """
    测试函数
    """
    result = get_item_data(item_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    # 测试代码
    test_id = 1  # 替换为要测试的ID
    test_get_item_data(test_id)