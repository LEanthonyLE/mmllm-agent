"""
FastAPI REST API — 多模态图片理解
=================================
POST /analyze  图片上传 + 文本查询 → AI 回复
GET  /health   健康检查

启动方式：uv run uvicorn api:app --host 0.0.0.0 --port 8000
"""

import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

load_dotenv()

# ============================================================
# LangGraph 图定义（自包含，不依赖 agent.py）
# ============================================================

import base64
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    image_path: str


def encode_image_to_base64(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
    }
    mime_type = mime_map.get(path.suffix.lower(), "image/jpeg")
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def prepare_message(state: AgentState) -> dict:
    image_data_uri = encode_image_to_base64(state["image_path"])
    return {"messages": [HumanMessage(content=[
        {"type": "text", "text": state["user_query"]},
        {"type": "image_url", "image_url": {"url": image_data_uri}},
    ])]}


def call_llm(state: AgentState) -> dict:
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        temperature=0.7,
        max_tokens=1024,
    )
    return {"messages": [llm.invoke(state["messages"])]}


_graph = StateGraph(AgentState)
_graph.add_node("prepare_message", prepare_message)
_graph.add_node("call_llm", call_llm)
_graph.add_edge(START, "prepare_message")
_graph.add_edge("prepare_message", "call_llm")
_graph.add_edge("call_llm", END)
graph_app = _graph.compile()


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(title="多模态图片理解 API")


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(..., description="图片文件"),
    query: str = Form(..., description="文本问题"),
):
    """上传图片并提问，返回 AI 回复"""
    # 保存上传图片到临时文件
    suffix = Path(image.filename).suffix if image.filename else ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(image.file, tmp)
        tmp.close()

        # 调用 LangGraph 图
        result = graph_app.invoke({
            "user_query": query,
            "image_path": tmp.name,
            "messages": [],
        })
        response_text = result["messages"][-1].content
        return {"response": response_text}

    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM 调用失败: {e}")
    finally:
        Path(tmp.name).unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
