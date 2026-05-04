"""
Streamlit Web UI — 多模态图片理解 Agent
========================================
启动方式：uv run streamlit run ui.py
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# LangGraph 图定义（自包含）
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
# Streamlit 界面
# ============================================================

st.set_page_config(page_title="多模态图片理解", page_icon="🖼️")
st.title("多模态图片理解 Agent")

# 侧边栏：配置信息
with st.sidebar:
    st.header("配置信息")
    st.text(f"模型: {os.getenv('MODEL_NAME', '未配置')}")
    st.text(f"API: {os.getenv('OPENAI_BASE_URL', '未配置')}")

# 主区域
uploaded_image = st.file_uploader(
    "上传图片",
    type=["jpg", "jpeg", "png", "webp"],
    help="支持 JPG / PNG / WebP 格式",
)

query = st.text_input(
    "输入问题",
    value="请描述这张图片的内容",
)

if uploaded_image:
    st.image(uploaded_image, caption="预览", width=400)

if st.button("分析", type="primary", disabled=not uploaded_image):
    if not uploaded_image:
        st.warning("请先上传图片")
    else:
        # 保存上传文件到临时目录
        suffix = Path(uploaded_image.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_image.getvalue())
            tmp_path = tmp.name

        try:
            with st.spinner("正在分析..."):
                result = graph_app.invoke({
                    "user_query": query,
                    "image_path": tmp_path,
                    "messages": [],
                })

            st.subheader("AI 回复")
            st.write(result["messages"][-1].content)

            # 消息历史（可展开）
            with st.expander("查看完整消息历史"):
                for i, msg in enumerate(result["messages"]):
                    role = type(msg).__name__
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    st.markdown(f"**[{i}] {role}**")
                    st.text(content[:500] + ("..." if len(content) > 500 else ""))

        except Exception as e:
            st.error(f"分析失败: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)
