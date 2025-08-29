# 新事心办-智能问答系统

本项目是一个基于大型语言模型的智能问答系统，旨在为用户提供高效、准确的业务咨询服务。系统通过API接口对外提供服务，支持流式和非流式两种响应模式，并可通过Docker进行容器化部署。

## 项目结构
.
├── api/
│   ├── api_server.py       # 主API服务 (FastAPI)
│   └── item_api.py         # 项目数据查询API (Flask)
├── data/
│   ├── fetch_data.py       # 数据获取逻辑
│   └── main_data.py        # 数据测试脚本
├── db/
│   └── knowledge_base_final.json # 知识库数据
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── logs/
│   └── qa_system_user_logs.log # 系统日志
├── main/
│   ├── .env                # 环境变量配置
│   ├── branch_engine.py    # 分支业务逻辑引擎
│   ├── main.py             # 主程序入口
│   └── qa_engine.py        # 核心问答引擎
├── models/
│   ├── Intent_recognition_model.py # 意图识别模型
│   ├── embedding_1.py      # 文本嵌入模型
│   └── model_answer.py     # 答案生成模型
├── prompts/
│   ├── prompt_all.txt      # Prompt模板
│   └── prompt_loader.py    # Prompt加载器
├── rag/
│   ├── Vector_matching.py  # 向量匹配
│   ├── emb_db.py           # 向量数据库
│   ├── embedding.py        # 文本嵌入
│   └── qwen_summary.py     # 文本摘要
└── requirements.txt        # Python依赖



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

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动服务

#### 方式一：直接运行（开发环境）

```bash
# 启动 FastAPI 主服务
python main/main.py

# 启动 Flask 项目数据查询服务
python api/item_api.py
```

#### 方式二：Docker 启动（生产环境）

确保已安装 Docker 和 Docker Compose。Docker 会自动从 `main/.env` 文件加载环境变量。

1.  **构建并启动服务**
    ```bash
    cd docker
    docker-compose up --build -d
    ```

2.  **查看服务状态**
    ```bash
    docker-compose ps
    ```

3.  **查看服务日志**
    ```bash
    docker-compose logs -f qa_api_server
    docker-compose logs -f item_api_server
    ```

4.  **停止服务**
    ```bash
    docker-compose down
    ```

**服务访问地址:**
*   **FastAPI 服务**: `http://localhost:8000`
*   **Flask 服务**: `http://localhost:5000`

## API 端点

#### 1. 健康检查
*   **URL**: `/health`
*   **Method**: `GET`
*   **Description**: 检查服务是否正常运行。

#### 2. 标准问答
*   **URL**: `/chat`
*   **Method**: `POST`
*   **Body**:
    ```json
    {
      "query": "你的问题",
      "session_id": "可选的会话ID",
      "stream": true,
      "user_id": "可选的用户ID"
    }
    ```
    *`stream` 默认为 `true`。如需一次性返回，请设置为 `false`。*

#### 3. Branch1 专用问答
*   **URL**: `/branch1/chat`
*   **Method**: `POST`
*   **Body**:
    ```json
    {
      "item_id": 3,
      "query": "这个业务的办理地址在哪里？",
      "stream": true
    }
    ```

#### 4. 查询项目数据
*   **URL**: `/item/<item_id>`
*   **Method**: `GET`
*   **Description**: 根据项目ID获取相关数据。

## 使用 `curl` 测试

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

#### 测试查询项目数据
```bash
curl -X GET "http://127.0.0.1:5000/item/1"
```