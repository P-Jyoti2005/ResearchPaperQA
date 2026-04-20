# ============================================================
# RESEARCH PAPER Q&A AGENT (FINAL VERSION 🔥)
# ============================================================

import chromadb
from typing import TypedDict, List
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime

# ============================================================
# 🔥 LLM (CHANGE MODEL HERE IF NEEDED)
# ============================================================
llm = ChatGroq(
    model="llama-3.1-8b-instant",   # ✅ optimized model
    temperature=0
)

# ============================================================
# EMBEDDING MODEL
# ============================================================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ============================================================
# CONFIG
# ============================================================
TOP_K = 3
SLIDING_WINDOW = 6

# ============================================================
# KNOWLEDGE BASE (RESEARCH PAPER DOMAIN)
# ============================================================
documents = [
    {
        "id": "doc1",
        "topic": "What is a Research Paper",
        "text": "A research paper is a structured academic document that presents original findings, analysis, or review of a topic."
    },
    {
        "id": "doc2",
        "topic": "Abstract Section",
        "text": "The abstract is a short summary of a research paper including the problem, methodology, results, and conclusion."
    },
    {
        "id": "doc3",
        "topic": "Methodology Section",
        "text": "The methodology section explains how the research was conducted including data collection, models, and experimental setup."
    },
    {
        "id": "doc4",
        "topic": "Results Section",
        "text": "The results section presents findings of the research often using tables, graphs, and performance metrics."
    },
    {
        "id": "doc5",
        "topic": "Evaluation Metrics",
        "text": "Common NLP evaluation metrics include BLEU, ROUGE, F1 score, and Perplexity."
    },
    {
        "id": "doc6",
        "topic": "Finding Research Papers",
        "text": "Research papers can be found on platforms like Google Scholar, arXiv, Semantic Scholar, IEEE Xplore, and ACM Digital Library."
    }
]

# ============================================================
# CHROMADB SETUP
# ============================================================
client = chromadb.Client()

try:
    client.delete_collection("research_kb")
except:
    pass

collection = client.create_collection("research_kb")

texts = [d["text"] for d in documents]

collection.add(
    ids=[d["id"] for d in documents],
    documents=texts,
    embeddings=embedder.encode(texts).tolist(),
    metadatas=[{"topic": d["topic"]} for d in documents]
)

# ============================================================
# STATE
# ============================================================
class AgentState(TypedDict):
    question: str
    messages: List[dict]
    route: str
    retrieved: str
    sources: List[str]
    tool_result: str
    answer: str
    user_name: str


# ============================================================
# RETRIEVAL
# ============================================================
def retrieve(query):
    qe = embedder.encode(query).tolist()
    res = collection.query(query_embeddings=[qe], n_results=TOP_K,
                           include=["documents", "metadatas"])

    context = []
    sources = []

    for i in range(len(res["documents"][0])):
        context.append(res["documents"][0][i])
        sources.append(res["metadatas"][0][i]["topic"])

    return {
        "context": "\n\n".join(context),
        "sources": sources
    }


# ============================================================
# MEMORY NODE
# ============================================================
def memory_node(state):
    msgs = state.get("messages", [])
    msgs.append({"role": "user", "content": state["question"]})

    if len(msgs) > SLIDING_WINDOW:
        msgs = msgs[-SLIDING_WINDOW:]

    return {"messages": msgs}


# ============================================================
# ROUTER NODE
# ============================================================
def router_node(state):
    prompt = f"""
Classify this question:

retrieve → research paper topics (abstract, methodology, results, metrics)
tool → date/time
memory_only → greeting

Question: {state["question"]}

Answer ONE word:
"""
    r = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()

    if "tool" in r:
        route = "tool"
    elif "memory" in r:
        route = "memory_only"
    else:
        route = "retrieve"

    return {"route": route}


# ============================================================
# RETRIEVAL NODE
# ============================================================
def retrieval_node(state):
    res = retrieve(state["question"])
    return {
        "retrieved": res["context"],
        "sources": res["sources"]
    }


# ============================================================
# TOOL NODE
# ============================================================
def tool_node(state):
    now = datetime.now()
    return {
        "tool_result": now.strftime("%A, %d %B %Y")
    }


# ============================================================
# ANSWER NODE
# ============================================================
def answer_node(state):
    context = state.get("retrieved", "")
    tool = state.get("tool_result", "")

    full_context = f"{context}\n{tool}"

    prompt = f"""
You are a Research Paper Assistant.

Use ONLY the context below to answer.

If answer not found, say:
"I don't have that information in my knowledge base."

Context:
{full_context}

Question:
{state["question"]}
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {"answer": response.content}


# ============================================================
# SAVE NODE
# ============================================================
def save_node(state):
    msgs = state.get("messages", [])
    msgs.append({"role": "assistant", "content": state["answer"]})
    return {"messages": msgs}


# ============================================================
# GRAPH BUILD
# ============================================================
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("memory", memory_node)
    g.add_node("router", router_node)
    g.add_node("retrieve", retrieval_node)
    g.add_node("tool", tool_node)
    g.add_node("answer", answer_node)
    g.add_node("save", save_node)

    g.set_entry_point("memory")

    g.add_edge("memory", "router")

    g.add_conditional_edges(
        "router",
        lambda s: s["route"],
        {
            "retrieve": "retrieve",
            "tool": "tool",
            "memory_only": "answer"
        }
    )

    g.add_edge("retrieve", "answer")
    g.add_edge("tool", "answer")
    g.add_edge("answer", "save")
    g.add_edge("save", END)

    return g.compile(checkpointer=MemorySaver())


# ============================================================
# PUBLIC FUNCTION
# ============================================================
graph = build_graph()

def ask(question, thread_id="default"):
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke(
        {
            "question": question,
            "messages": [],
            "route": "",
            "retrieved": "",
            "sources": [],
            "tool_result": "",
            "answer": "",
            "user_name": ""
        },
        config=config
    )

    return result