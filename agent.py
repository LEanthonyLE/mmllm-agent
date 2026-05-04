"""
LangGraph 多模态智能体 — 核心模块
==================================
提供 run_agent(image_path, query) 对外接口，供 api.py / ui.py / app.py 调用。
"""

import base64
import os
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()


# ============================================================
# 1. 状态定义
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    image_path: str


# ============================================================
# 2. 图片编码
# ============================================================

def encode_image_to_base64(image_path: str) -> str:
    """读取本地图片，返回 base64 data URI。"""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(path.suffix.lower(), "image/jpeg")

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


# ============================================================
# 3. 图节点
# ============================================================

def prepare_message(state: AgentState) -> dict:
    """节点1：构造多模态 HumanMessage。"""
    image_data_uri = encode_image_to_base64(state["image_path"])

    multimodal_message = HumanMessage(
        content=[
            {"type": "text", "text": state["user_query"]},
            {"type": "image_url", "image_url": {"url": image_data_uri}},
        ]
    )
    return {"messages": [multimodal_message]}


def call_llm(state: AgentState) -> dict:
    """节点2：调用 LLM 获取回复。"""
    llm = ChatOpenAI(
        model=os.getenv("MODEL_NAME", "gpt-4o"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        temperature=0.7,
        max_tokens=1024,
    )
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# ============================================================
# 4. 图构建
# ============================================================

_graph = StateGraph(AgentState)
_graph.add_node("prepare_message", prepare_message)
_graph.add_node("call_llm", call_llm)
_graph.add_edge(START, "prepare_message")
_graph.add_edge("prepare_message", "call_llm")
_graph.add_edge("call_llm", END)
_app = _graph.compile()


# ============================================================
# 5. 对外接口
# ============================================================

def run_agent(image_path: str, query: str) -> str:
    """运行多模态智能体，返回 AI 回复文本。"""
    result = _app.invoke({
        "user_query": query,
        "image_path": image_path,
        "messages": [],
    })
    return result["messages"][-1].content
