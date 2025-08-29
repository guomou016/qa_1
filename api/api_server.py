# -*- coding: utf-8 -*-
from typing import Optional, Dict
import os
import sys
import uuid
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 路径设置
current_script_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_script_path)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入引擎
from main.qa_engine import QAEngine, QASessionState
from main.branch_engine import Branch1Engine

app = FastAPI(title="Q&A API", version="1.0.0", description="基于现有问答系统的标准API服务")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)

# 全局引擎与会话池
engine = QAEngine()
branch1_engine = Branch1Engine()
SESSIONS: Dict[str, QASessionState] = {}

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    stream: Optional[bool] = True  # 添加流式参数
    user_id: Optional[str] = "default_user"  # 添加用户ID参数

class Branch1Request(BaseModel):
    item_id: int
    query: str
    stream: Optional[bool] = True

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    # 管理会话
    session_id = req.session_id or str(uuid.uuid4())
    if session_id not in SESSIONS:
        SESSIONS[session_id] = QASessionState()
    session = SESSIONS[session_id]

    try:
        if req.stream:
            # 流式响应
            def generate_stream():
                try:
                    result = engine.handle_query_stream(req.query, session, req.user_id)
                    # 发送初始元数据
                    yield f"data: {json.dumps({'type': 'metadata', 'session_id': session_id})}\n\n"
                    
                    # 流式发送内容
                    for chunk in result:
                        if chunk:
                            yield f"data: {json.dumps({'type': 'content', 'answer': chunk})}\n\n"
                    
                    # 发送结束标记
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            # 非流式响应（保持原有逻辑）
            result = engine.handle_query(req.query, session, req.user_id)
            return {
                "status": "ok",
                "session_id": session_id,
                "result": result
            }
    except Exception as e:
        # 捕获任何未处理异常，返回标准错误
        return {
            "status": "error",
            "session_id": session_id,
            "message": str(e)
        }

@app.post("/branch1/chat")
def branch1_chat(req: Branch1Request):
    """Branch1 专用的聊天接口"""
    try:
        if req.stream:
            # 流式响应
            def generate_stream():
                try:
                    # 发送初始元数据
                    yield f"data: {json.dumps({'type': 'metadata', 'item_id': req.item_id})}\n\n"
                    
                    # 流式发送内容
                    for chunk in branch1_engine.handle_query_stream(req.item_id, req.query):
                        if chunk:
                            yield f"data: {json.dumps({'type': 'content', 'answer': chunk})}\n\n"
                    
                    # 发送结束标记
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
            )
        else:
            # 非流式响应
            result = branch1_engine.handle_query(req.item_id, req.query)
            return {
                "status": "ok",
                "result": result
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/sessions/{session_id}")
def get_session_info(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="session not found")
    s = SESSIONS[session_id]
    return {
        "session_id": session_id,
        "int_id": s.int_id,
        "history_len": len(s.intent_history)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.api_server:app", host="0.0.0.0", port=8000, reload=True)