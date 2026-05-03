"""
LangGraph 多模态入门示例
========================
学习目标：理解图片如何作为消息的一部分在 LangGraph 图中流动。

核心概念：
1. StateGraph 的状态定义（TypedDict + Annotated）
2. 多模态消息的构造（HumanMessage 含 image_url 内容块）
3. 图的构建与编译（StateGraph -> add_node -> add_edge -> compile）
4. 图的调用（app.invoke）

数据流：
  __start__ --> prepare_message --> call_llm --> __end__
                   ↑                   ↑
              构造多模态消息        调用大模型
"""

import base64
import os
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()


# ============================================================
# 1. 状态定义 (State Definition)
# ============================================================
# 这是 LangGraph 最核心的概念之一。
# State 决定了图中流动的数据结构。
#
# - messages: 消息列表，使用 add_messages 作为 reducer
#   reducer 的作用：当一个节点返回 {"messages": [新消息]} 时，
#   add_messages 会将新消息 *追加* 到现有列表，
#   而不是 *替换* 整个列表。
#
# - user_query: 用户的文本问题（纯字符串，无 reducer → 覆盖语义）
# - image_path: 图片文件路径

class AgentState(TypedDict):
    messages: Annotated[list, add_messages] # 通过add_messages方法实现append（添加信息content是list）
    user_query: str
    image_path: str


# ============================================================
# 2. 图片编码工具函数
# ============================================================
# OpenAI 兼容 API 接受两种图片格式：
#   a) URL: "https://example.com/image.jpg"
#   b) Base64 data URI: "data:image/jpeg;base64,/9j/4AAQ..."
#
# 本地文件需要先转为 base64 格式。

def encode_image_to_base64(image_path: str) -> str:
    """读取本地图片文件，返回 base64 编码的 data URI。"""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    suffix = path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(suffix, "image/jpeg")

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


# ============================================================
# 3. 图节点 (Graph Nodes)
# ============================================================
# 节点是图中的处理单元。每个节点：
#   - 接收当前 State
#   - 返回一个 dict，包含要更新的 State 字段

def prepare_message(state: AgentState) -> dict:
    """
    节点1：将文本问题和图片路径组合成一条多模态 HumanMessage。
    """
    query = state["user_query"]
    image_path = state["image_path"]

    image_data_uri = encode_image_to_base64(image_path)

    # 构造多模态消息 —— 这是核心知识点！
    # content 是一个列表，每个元素是一个"内容块"(content block)：
    #   - type: "text"      → 文本内容
    #   - type: "image_url" → 图片内容（URL 或 base64 data URI）
    multimodal_message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": query,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_data_uri,
                },
            },
        ]
    )

    return {"messages": [multimodal_message]}


def call_llm(state: AgentState) -> dict:
    """
    节点2：将消息历史发送给 LLM，获取回复。
    """
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
# 4. 图构建 (Graph Construction)
# ============================================================
# 构建过程：StateGraph → add_node → add_edge → compile
#
# 流程示意：
#   __start__ --> prepare_message --> call_llm --> __end__

graph = StateGraph(AgentState)

graph.add_node("prepare_message", prepare_message)
graph.add_node("call_llm", call_llm)

graph.add_edge(START, "prepare_message")
graph.add_edge("prepare_message", "call_llm")
graph.add_edge("call_llm", END)

app = graph.compile()


# ============================================================
# 5. 运行图 (Run the Graph)
# ============================================================

if __name__ == "__main__":
    image_path = "./image/img.jpg"                # ← 替换为你的图片路径
    user_query = "请描述这张图片的内容"       # ← 替换为你的问题

    print("=" * 50)
    print("LangGraph 多模态示例")
    print("=" * 50)
    print(f"图片路径: {image_path}")
    print(f"用户问题: {user_query}")
    print("-" * 50)

    # 调用图
    # invoke 接收初始状态的 dict，LangGraph 按边的定义依次执行节点
    result = app.invoke({
        "user_query": user_query,
        "image_path": image_path,
        "messages": [],
    })

    # 输出 AI 回复
    print("\nAI 回复：")
    print(result["messages"][-1].content)

    # 打印完整消息历史（教学用）
    print("\n" + "=" * 50)
    print("完整消息历史：")
    for i, msg in enumerate(result["messages"]):
        role = type(msg).__name__
        preview = msg.content[:100] if isinstance(msg.content, str) else str(msg.content)[:100]
        print(f"  [{i}] {role}: {preview}...")
