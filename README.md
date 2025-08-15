# 新事心办-智能问答系统

本项目是一个基于大型语言模型的智能问答系统，旨在为用户提供高效、准确的业务咨询服务。系统通过API接口对外提供服务，支持流式和非流式两种响应模式。

## 项目结构
├── api/                  # FastAPI 接口服务
│   └── api_server.py
├── db/                   # 数据库和知识库文件
│   └── knowledge_base_final.json
├── logs/                 # 日志文件
├── main/                 # 主要业务逻辑
│   ├── .env              # 环境变量配置
│   ├── branch_1.py       # 命令行测试脚本
│   ├── branch_1_engine.py # branch1 业务引擎
│   └── qa_engine.py      # 核心问答引擎
├── models/               # 模型相关代码
├── prompts/              # Prompt模板
└── rag/                  # RAG 相关模块


## 快速开始

### 1. 环境配置

在 `main/.env` 文件中配置必要的环境变量。

#### 通用配置

*   `DASHSCOPE_API_KEY`: 您的 DashScope API 密钥。

#### 数据库配置

*   `DB_HOST`: 数据库主机地址
*   `DB_PORT`: 数据库端口
*   `DB_USER`: 数据库用户名
*   `DB_PASSWORD`: 数据库密码
*   `DB_DATABASE`: 数据库名称
*   `DB_CHARSET`: 数据库字符集

#### `cs_123.py` 专用配置 (数据处理脚本)

*   `CS_SQL_QUERY_TEST_TABLE`: 用于从 `items_info` 表查询业务名称的SQL语句。
*   `CS_SQL_QUERY_ITEMS_DOC`: 用于从 `items_doc` 表查询文档内容的SQL语句。
*   `CS_TABLE_NAME_TEST`: `items_info` 表名。
*   `CS_TABLE_NAME_ITEMS_DOC`: `items_doc` 表名。
*   `CS_OUTPUT_DIR`: 生成的知识库JSON文件的输出目录。
*   `CS_OUTPUT_FILENAME`: 生成的知识库JSON文件名。

#### `qa_engine.py` 专用配置 (问答引擎)

*   `QA_SQL_QUERY_ITEM_DETAILS`: 根据ID查询项目详细信息的SQL语句。
*   `QA_SQL_QUERY_BUSINESS_NAME`: 根据ID查询业务名称的SQL语句。
*   `QA_TABLE_NAME_ITEMS_INFO`: `items_info` 表名。
*   `QA_TABLE_NAME_ITEMS_DOC`: `items_doc` 表名。

### 2. 安装依赖

```bash
pip install -r requirements.txt
```
*(注意: 项目中暂未提供 requirements.txt 文件, 请根据需要自行创建)*

### 3. 启动API服务

```bash
python api/api_server.py
```
服务将在 `http://127.0.0.1:8000` 启动。由于设置了 `reload=True`，任何代码更改都会自动重启服务。

## 如何使用

### API 端点

#### 1. 健康检查

*   **URL**: `/health`
*   **Method**: `GET`
*   **Description**: 检查服务是否正常运行。

#### 2. 标准问答

*   **URL**: `/chat`
*   **Method**: `POST`
*   **Description**: 通用的问答接口。
*   **Body**:
    ```json
    {
      "query": "你的问题",
      "session_id": "可选的会话ID",
      "stream": true
    }
    ```
    * `stream` 默认为 `true`，表示流式响应。如需一次性返回，请设置为 `false`。

#### 3. Branch1 专用问答

*   **URL**: `/branch1/chat`
*   **Method**: `POST`
*   **Description**: 针对特定业务场景的问答接口。
*   **Body**:
    ```json
    {
      "item_id": 3,
      "query": "这个业务的办理地址在哪里？",
      "stream": true
    }
    ```
    * `stream` 默认为 `true`。

### 使用 `curl` 测试

#### 测试流式响应

```bash
curl -X POST "http://127.0.0.1:8000/branch1/chat" \
-H "Content-Type: application/json" \
-d '{
  "item_id": 3,
  "query": "这个业务的办理地址在哪里？"
}'
```

#### 测试非流式响应

```bash
curl -X POST "http://127.0.0.1:8000/branch1/chat" \
-H "Content-Type: application/json" \
-d '{
  "item_id": 3,
  "query": "这个业务的办理地址在哪里？",
  "stream": false
}'
```

这个文档应该能帮助您和您的团队更好地理解和使用这个项目。如果您需要任何进一步的修改或补充，请随时告诉我。