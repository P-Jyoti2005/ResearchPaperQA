import streamlit as st
import uuid
import os
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
from datetime import datetime

# ============================================================
# PAGE CONFIG — must be first!
# ============================================================
st.set_page_config(
    page_title="PaperIQ – Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PREMIUM CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@300;400;500;600;700&family=Geist+Mono:wght@400;500&display=swap');

/* ── Design Tokens ── */
:root {
    --bg:           #06060a;
    --surface:      #0d0d14;
    --surface2:     #13131c;
    --surface3:     #1a1a26;
    --surface4:     #21212f;
    --border:       rgba(255,255,255,0.055);
    --border2:      rgba(255,255,255,0.10);
    --border3:      rgba(255,255,255,0.16);
    --accent:       #7c6af7;
    --accent-soft:  rgba(124,106,247,0.12);
    --accent-glow:  rgba(124,106,247,0.25);
    --gold:         #e8b84b;
    --gold-soft:    rgba(232,184,75,0.10);
    --teal:         #2dd4bf;
    --teal-soft:    rgba(45,212,191,0.10);
    --rose:         #fb7185;
    --text:         #eeeef5;
    --text2:        #a0a0b8;
    --text3:        #636378;
    --font-sans:    'Geist', sans-serif;
    --font-serif:   'Instrument Serif', serif;
    --font-mono:    'Geist Mono', monospace;
    --radius:       14px;
    --radius-sm:    9px;
    --radius-lg:    20px;
}

/* ── Global ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: var(--bg) !important;
    font-family: var(--font-sans) !important;
    color: var(--text) !important;
}
[data-testid="stHeader"]       { background: transparent !important; }
#MainMenu, footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"]   { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar            { width: 3px; }
::-webkit-scrollbar-track      { background: transparent; }
::-webkit-scrollbar-thumb      { background: var(--surface4); border-radius: 99px; }

/* ════════════════════════════════════════
   SIDEBAR
════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* brand strip */
.brand-strip {
    padding: 1.6rem 1.2rem 1.4rem;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(160deg, rgba(124,106,247,0.08) 0%, transparent 70%);
}
.brand-mark {
    display: flex; align-items: center; gap: 11px;
}
.brand-icon {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #7c6af7, #a78bfa);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    box-shadow: 0 8px 24px rgba(124,106,247,0.35);
    flex-shrink: 0;
}
.brand-name {
    font-family: var(--font-serif);
    font-size: 22px;
    color: var(--text);
    letter-spacing: -0.01em;
    line-height: 1;
}
.brand-tagline {
    font-size: 11px;
    color: var(--text3);
    margin-top: 3px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-weight: 500;
}

/* section headers */
.sec-hdr {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text3);
    padding: 0 1.2rem;
    margin-bottom: 6px;
}

/* stat cards */
.stats-row {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 8px; padding: 0 1.2rem; margin-bottom: 4px;
}
.stat-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 10px 12px;
}
.stat-lbl {
    font-size: 10px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--text3); margin-bottom: 3px;
}
.stat-val {
    font-family: var(--font-mono);
    font-size: 22px; font-weight: 500;
    color: var(--accent);
    line-height: 1;
}
.stat-val.neutral { color: var(--text2); }

/* quick-q pills */
[data-testid="stSidebar"] .stButton > button {
    background: var(--surface2) !important;
    color: var(--text2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-sans) !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    text-align: left !important;
    padding: 9px 13px !important;
    width: 100% !important;
    transition: all 0.18s ease !important;
    margin-bottom: 3px !important;
    line-height: 1.4 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--surface3) !important;
    color: var(--text) !important;
    border-color: var(--border2) !important;
    transform: translateX(4px) !important;
}

/* new-chat button override */
.new-chat-wrap > div > button {
    background: linear-gradient(135deg, #7c6af7 0%, #a78bfa 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-sans) !important;
    font-size: 13px !important; font-weight: 600 !important;
    padding: 10px 14px !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 18px rgba(124,106,247,0.3) !important;
    letter-spacing: 0.02em !important;
}
.new-chat-wrap > div > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 7px 24px rgba(124,106,247,0.42) !important;
}

/* kb topic list */
.kb-topic {
    font-size: 12px;
    color: var(--text3);
    padding: 3px 1.2rem;
    display: flex; align-items: center; gap: 6px;
}
.kb-topic::before {
    content: '';
    width: 4px; height: 4px;
    border-radius: 50%;
    background: var(--accent);
    opacity: 0.5;
    flex-shrink: 0;
}

/* sidebar text overrides */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: var(--text2) !important; font-size: 13px !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--text) !important; font-family: var(--font-sans) !important; }

/* ════════════════════════════════════════
   MAIN LAYOUT
════════════════════════════════════════ */
[data-testid="stMainBlockContainer"],
.block-container {
    padding: 0 !important; max-width: 100% !important;
}

/* ── Hero ── */
.hero {
    padding: 3rem 3.5rem 2rem;
    border-bottom: 1px solid var(--border);
    position: relative; overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse 60% 80% at 20% -20%, rgba(124,106,247,0.08) 0%, transparent 70%),
        radial-gradient(ellipse 40% 60% at 90% 110%, rgba(45,212,191,0.05) 0%, transparent 60%);
    pointer-events: none;
}
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(124,106,247,0.08);
    border: 1px solid rgba(124,106,247,0.18);
    border-radius: 99px; padding: 5px 14px;
    font-size: 11px; font-weight: 500;
    color: #a78bfa; letter-spacing: 0.06em;
    text-transform: uppercase; margin-bottom: 16px;
}
.hero-title {
    font-family: var(--font-serif);
    font-size: clamp(2rem, 3.5vw, 3rem);
    font-weight: 400;
    line-height: 1.12;
    color: var(--text);
    margin: 0 0 10px;
    letter-spacing: -0.02em;
}
.hero-title em { color: #a78bfa; font-style: italic; }
.hero-sub {
    font-size: 15px; color: var(--text2); font-weight: 300;
    max-width: 560px; line-height: 1.65; margin: 0;
}
.hero-pills {
    display: flex; flex-wrap: wrap; gap: 8px; margin-top: 20px;
}
.hero-pill {
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 99px;
    padding: 4px 12px;
    font-size: 12px; color: var(--text3);
    font-weight: 500; letter-spacing: 0.02em;
}

/* ── Chat area ── */
.chat-area { padding: 2rem 3.5rem; min-height: 52vh; }

/* ── Empty state ── */
.empty-state {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    min-height: 42vh; text-align: center; gap: 10px;
}
.empty-orb {
    width: 80px; height: 80px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, #7c6af7, #0d0d14);
    box-shadow: 0 0 60px rgba(124,106,247,0.2), 0 0 120px rgba(124,106,247,0.08);
    display: flex; align-items: center; justify-content: center;
    font-size: 32px; margin-bottom: 8px;
}
.empty-title {
    font-family: var(--font-serif);
    font-size: 1.5rem; color: var(--text);
}
.empty-desc {
    font-size: 14px; color: var(--text3);
    max-width: 380px; line-height: 1.7;
}
.suggestion-grid {
    display: grid; grid-template-columns: repeat(2, 1fr);
    gap: 8px; margin-top: 16px; max-width: 480px;
}
.suggestion-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 10px 14px;
    font-size: 13px; color: var(--text2);
    text-align: left; cursor: pointer;
    transition: all 0.18s ease;
    line-height: 1.4;
}
.suggestion-card:hover {
    background: var(--surface3);
    border-color: var(--border2);
    color: var(--text);
}

/* ── Messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin-bottom: 1.8rem !important;
}
[data-testid="stChatMessageContent"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 15px 20px !important;
    font-size: 14.5px !important;
    line-height: 1.75 !important;
    color: var(--text) !important;
}

/* User bubble accent */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
    background: var(--surface3) !important;
    border-color: var(--border2) !important;
}

/* Avatars */
[data-testid="chatAvatarIcon-user"] > div {
    background: linear-gradient(135deg, #2dd4bf, #0ea5e9) !important;
    border-radius: 10px !important;
    width: 36px !important; height: 36px !important;
}
[data-testid="chatAvatarIcon-assistant"] > div {
    background: linear-gradient(135deg, #7c6af7, #a78bfa) !important;
    border-radius: 10px !important;
    width: 36px !important; height: 36px !important;
}

/* ── Meta badges ── */
.meta-row {
    display: flex; flex-wrap: wrap; gap: 5px;
    margin-top: 10px; padding-top: 10px;
    border-top: 1px solid var(--border);
}
.badge {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 10px; border-radius: 99px;
    font-size: 11px; font-weight: 500;
    font-family: var(--font-mono);
    letter-spacing: 0.01em;
}
.badge-route  { background: rgba(124,106,247,0.1); border: 1px solid rgba(124,106,247,0.2); color: #a78bfa; }
.badge-source { background: var(--teal-soft);  border: 1px solid rgba(45,212,191,0.18); color: var(--teal); }
.badge-score  { background: rgba(232,184,75,0.08); border: 1px solid rgba(232,184,75,0.2); color: var(--gold); }
.badge-low    { background: rgba(251,113,133,0.08); border: 1px solid rgba(251,113,133,0.2); color: var(--rose); }

/* ── Thinking ── */
.thinking {
    display: inline-flex; align-items: center; gap: 12px;
    padding: 12px 18px;
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: var(--radius);
    font-size: 13px; color: var(--text3);
}
.dot-trio { display: flex; gap: 4px; align-items: center; }
.dot-trio span {
    width: 6px; height: 6px;
    background: var(--accent); border-radius: 50%;
    animation: pulse 1.4s infinite ease-in-out;
}
.dot-trio span:nth-child(2) { animation-delay: 0.22s; }
.dot-trio span:nth-child(3) { animation-delay: 0.44s; }
@keyframes pulse {
    0%,80%,100% { transform: scale(0.55); opacity: 0.35; }
    40%          { transform: scale(1);    opacity: 1; }
}

/* ── Chat input bar ── */
[data-testid="stBottom"] {
    background: linear-gradient(0deg, var(--bg) 80%, transparent) !important;
    border-top: none !important;
    padding: 1.4rem 3.5rem 1.6rem !important;
}
[data-testid="stChatInput"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: var(--radius-lg) !important;
    color: var(--text) !important;
    font-family: var(--font-sans) !important;
    font-size: 14.5px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(124,106,247,0.45) !important;
    box-shadow: 0 0 0 3px rgba(124,106,247,0.12), 0 4px 24px rgba(0,0,0,0.35) !important;
}
[data-testid="stChatInput"] textarea { color: var(--text) !important; }
[data-testid="stChatInput"] textarea::placeholder { color: var(--text3) !important; }
[data-testid="stChatInput"] button {
    background: var(--accent) !important;
    border-radius: 10px !important;
    color: white !important;
}
[data-testid="stChatInput"] button:hover { background: #9580ff !important; }

/* ── Markdown refinements ── */
[data-testid="stChatMessageContent"] p  { margin-bottom: 0.55rem !important; }
[data-testid="stChatMessageContent"] ul,
[data-testid="stChatMessageContent"] ol { padding-left: 1.4rem !important; }
[data-testid="stChatMessageContent"] li { margin-bottom: 0.3rem !important; }
[data-testid="stChatMessageContent"] code {
    background: var(--surface4) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 5px !important;
    padding: 1px 6px !important;
    font-size: 12.5px !important;
    color: var(--teal) !important;
    font-family: var(--font-mono) !important;
}
[data-testid="stChatMessageContent"] strong { color: var(--text) !important; }

/* ── Alert ── */
[data-testid="stAlert"] {
    background: rgba(45,212,191,0.06) !important;
    border: 1px solid rgba(45,212,191,0.18) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--teal) !important;
}

/* ── Dividers ── */
hr { border-color: var(--border) !important; margin: 0.8rem 0 !important; }

/* ── Sidebar padding helper ── */
.sb-pad { padding: 0 1.2rem; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# AGENT SETUP (cached — loads only once)
# ============================================================
@st.cache_resource
def load_agent():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.environ.get("GROQ_API_KEY"),
        max_tokens=1024
    )
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="research_papers_kb")

    documents = [
        {"id": "doc_001", "topic": "What is a Research Paper",
         "text": "A research paper is a formal document that presents original research, analysis, or arguments on a specific topic. It is written by researchers, scientists, or academics and published in journals or conference proceedings. A typical research paper has several key sections: Abstract, Introduction, Literature Review, Methodology, Results, Discussion, Conclusion, and References. Research papers go through a peer review process where experts evaluate the work before publication. The goal of a research paper is to contribute new knowledge to a specific field. Papers are identified by a DOI (Digital Object Identifier), author names, and publication year."},
        {"id": "doc_002", "topic": "How to Read the Abstract",
         "text": "The abstract is a short summary of the entire research paper, usually 150 to 300 words long. It appears at the very beginning of the paper. The abstract tells you: what problem the paper is solving, what method was used, what the key results are, and what the main conclusion is. Reading the abstract first helps you decide if the paper is relevant to your research. A good abstract covers the motivation, the approach, the key finding, and the implication. You should never skip the abstract — it is the fastest way to understand what a paper is about. Keywords listed below the abstract help in searching for related papers."},
        {"id": "doc_003", "topic": "Understanding the Methodology Section",
         "text": "The methodology section explains HOW the research was conducted. It describes the experimental setup, the dataset used, the algorithms or techniques applied, and the evaluation metrics. For computer science papers, the methodology often includes the model architecture, training procedure, hyperparameters, and hardware used. For medical papers, it includes the study design, sample size, control groups, and statistical tests. The methodology should be detailed enough that another researcher can reproduce the experiment. Weaknesses in methodology are often the main target of peer review criticism."},
        {"id": "doc_004", "topic": "How to Find Key Results and Findings",
         "text": "The results section presents the outcome of the experiments or analysis. Key findings are usually shown in tables, graphs, or charts. Look for numbers that compare the proposed method against baseline methods — this is called a comparison table or ablation study. The most important result is usually highlighted in bold in the table. In deep learning papers, results are often measured using metrics like accuracy, F1 score, BLEU score, or ROUGE score depending on the task. Always check the dataset on which results are reported — a method that performs well on one dataset may not generalize to others."},
        {"id": "doc_005", "topic": "Understanding the Conclusion Section",
         "text": "The conclusion section summarizes what the paper achieved and what it means for the field. It restates the main contribution of the paper, discusses limitations of the work, and suggests directions for future research. A strong conclusion answers: What was proven or demonstrated? Why does it matter? What should future researchers explore? The conclusion is different from the abstract — the abstract tells you what will happen, the conclusion tells you what happened and what it means. Limitations mentioned in the conclusion are important for understanding the boundaries of the research."},
        {"id": "doc_006", "topic": "How to Identify Paper Authors and Affiliations",
         "text": "Authors of a research paper are listed on the title page, usually right below the paper title. The first author is typically the person who did most of the work. The last author is often the senior researcher or supervisor. Authors include their affiliations — the university, company, or research lab they belong to. Contact information (email) is provided for the corresponding author. In Google Scholar and Semantic Scholar, you can click on an author's name to see all their published papers. The h-index of an author indicates their research impact."},
        {"id": "doc_007", "topic": "What is a Literature Review",
         "text": "The literature review section surveys existing work related to the paper's topic. It explains what has already been done, what gaps exist in current knowledge, and how this paper fills those gaps. A good literature review groups related work into themes and critically compares approaches. References cited in the literature review are the most relevant prior papers to read if you want background on the topic. The literature review justifies why the new paper is needed. Reading the literature review helps you understand the history of the problem and the state of the art before the current paper was published."},
        {"id": "doc_008", "topic": "How to Use Citations and References",
         "text": "References are listed at the end of a research paper. Each in-text citation like [1] or (Smith, 2020) corresponds to a full reference in this list. References include: author names, paper title, journal or conference name, year, volume, pages, and DOI. You can use tools like Google Scholar, Semantic Scholar, or ResearchGate to find the full text of cited papers. Citation count tells you how many other papers have referenced this work — higher citation count usually means higher impact. A paper cited 1000+ times is considered a landmark or seminal paper in the field."},
        {"id": "doc_009", "topic": "Common Research Paper Evaluation Metrics",
         "text": "Different research domains use different metrics to evaluate results. In Natural Language Processing (NLP): BLEU score measures translation quality, ROUGE score measures summarization quality, Perplexity measures language model quality, F1 score measures classification balance. In Computer Vision: Accuracy measures overall correctness, mAP (mean Average Precision) measures object detection quality, IoU (Intersection over Union) measures segmentation quality. When reading a paper, always check which dataset and which metric is used. State-of-the-art (SOTA) means the best performing method on a benchmark at the time of publication."},
        {"id": "doc_010", "topic": "How to Search and Find Research Papers",
         "text": "The best sources to find research papers are: Google Scholar (scholar.google.com) for broad search across all fields, arXiv (arxiv.org) for free preprints in CS, Physics, and Math, Semantic Scholar for AI-powered search with citation graphs, IEEE Xplore for electronics and computer engineering papers, ACM Digital Library for computer science papers, and PubMed for medical and biological research. To search effectively, use specific keywords from the methodology or topic. Use quotation marks for exact phrases. Filter by year to find recent papers. Saving papers to Zotero or Mendeley helps organize your reading list."}
    ]

    texts      = [d["text"]  for d in documents]
    ids        = [d["id"]    for d in documents]
    metadatas  = [{"topic": d["topic"]} for d in documents]
    embeddings = embedder.encode(texts).tolist()
    collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)

    class CapstoneState(TypedDict):
        question: str; messages: list; route: str; retrieved: str
        sources: list; tool_result: str; answer: str
        faithfulness: float; eval_retries: int
        paper_topic: str; user_name: str

    MAX_RETRIES = 2
    FAITH_THRESHOLD = 0.7

    def memory_node(state):
        msgs = state.get("messages", [])[-6:]
        q    = state["question"]
        name = state.get("user_name", "")
        if "my name is" in q.lower():
            parts = q.lower().split("my name is")
            if len(parts) > 1:
                name = parts[1].strip().split()[0].capitalize()
        msgs.append({"role": "user", "content": q})
        return {"messages": msgs, "user_name": name,
                "tool_result": "", "retrieved": "", "sources": [],
                "eval_retries": state.get("eval_retries", 0)}

    def router_node(state):
        q    = state["question"]
        hist = "".join(f"{m['role']}: {m['content']}\n" for m in state.get("messages", [])[-4:])
        prompt = f"""You are a routing assistant for a Research Paper Q&A system.
Conversation:\n{hist}\nQuestion: {q}

Reply with ONE word only:
- retrieve   → about research papers, abstract, methodology, authors, citations, results, metrics
- tool       → needs current date/time or live web search
- memory_only → casual chat, greetings, thank you

ONE word:"""
        r = llm.invoke(prompt).content.strip().lower()
        if r not in ["retrieve", "tool", "memory_only"]:
            r = "retrieve"
        return {"route": r}

    def retrieval_node(state):
        q_emb = embedder.encode([state["question"]]).tolist()
        res   = collection.query(query_embeddings=q_emb, n_results=3)
        topics  = [m["topic"] for m in res["metadatas"][0]]
        context = "".join(f"[{t}]\n{c}\n\n" for t, c in zip(topics, res["documents"][0]))
        return {"retrieved": context, "sources": topics}

    def skip_retrieval_node(state):
        return {"retrieved": "", "sources": []}

    def tool_node(state):
        q = state["question"].lower()
        try:
            if any(w in q for w in ["date", "time", "today", "day", "year"]):
                res = f"Current date and time: {datetime.now().strftime('%A, %d %B %Y, %I:%M %p')}"
            elif any(w in q for w in ["citation", "journal", "published", "latest", "recent"]):
                res = "For latest citation counts and recent papers, visit: scholar.google.com or semanticscholar.org"
            else:
                res = "For latest research papers, visit arxiv.org or scholar.google.com"
        except Exception as e:
            res = f"Tool error: {str(e)}"
        return {"tool_result": res}

    def answer_node(state):
        name    = state.get("user_name", "")
        retries = state.get("eval_retries", 0)
        hist    = "".join(f"Assistant: {m['content']}\n"
                          for m in state.get("messages", [])[-4:] if m["role"] == "assistant")
        retry_note = "\nIMPORTANT: Previous answer was not faithful. Answer STRICTLY from context only." if retries > 0 else ""
        prompt = f"""You are a Research Paper Q&A assistant.{retry_note}

RULES:
1. Answer ONLY from the CONTEXT below.
2. If not in context say: "I don't have that info. Check Google Scholar or arXiv."
3. Never fabricate paper titles, authors, or statistics.
4. Be concise and helpful.
{"5. Address " + name + " by name." if name else ""}

HISTORY:\n{hist}
CONTEXT:\n{state.get('retrieved', '')}
TOOL RESULT:\n{state.get('tool_result', '')}
QUESTION: {state['question']}

Answer:"""
        return {"answer": llm.invoke(prompt).content.strip()}

    def eval_node(state):
        retrieved = state.get("retrieved", "")
        retries   = state.get("eval_retries", 0)
        if not retrieved.strip():
            return {"faithfulness": 1.0, "eval_retries": retries}
        prompt = f"""Rate faithfulness of the ANSWER to the CONTEXT. Reply with a decimal 0.0-1.0 only.
1.0 = only uses context | 0.5 = mixes context+outside | 0.0 = ignores context

CONTEXT:\n{retrieved[:800]}
ANSWER:\n{state.get('answer', '')}

Score:"""
        try:
            score = float(llm.invoke(prompt).content.strip())
            score = max(0.0, min(1.0, score))
        except:
            score = 0.5
        return {"faithfulness": score, "eval_retries": retries + 1}

    def save_node(state):
        msgs = state.get("messages", [])
        msgs.append({"role": "assistant", "content": state.get("answer", "")})
        return {"messages": msgs}

    def route_decision(state):
        r = state.get("route", "retrieve")
        return "tool" if r == "tool" else ("skip" if r == "memory_only" else "retrieve")

    def eval_decision(state):
        if state.get("faithfulness", 1.0) < FAITH_THRESHOLD and state.get("eval_retries", 0) < MAX_RETRIES:
            return "answer"
        return "save"

    g = StateGraph(CapstoneState)
    for node_name, fn in [("memory", memory_node), ("router", router_node),
                           ("retrieve", retrieval_node), ("skip", skip_retrieval_node),
                           ("tool", tool_node), ("answer", answer_node),
                           ("eval", eval_node), ("save", save_node)]:
        g.add_node(node_name, fn)

    g.set_entry_point("memory")
    g.add_edge("memory", "router")
    g.add_edge("retrieve", "answer")
    g.add_edge("skip", "answer")
    g.add_edge("tool", "answer")
    g.add_edge("answer", "eval")
    g.add_edge("save", END)
    g.add_conditional_edges("router", route_decision, {"retrieve": "retrieve", "skip": "skip", "tool": "tool"})
    g.add_conditional_edges("eval", eval_decision, {"answer": "answer", "save": "save"})

    mem = MemorySaver()
    app = g.compile(checkpointer=mem)
    return app, CapstoneState


app, CapstoneState = load_agent()


def ask(question: str, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    return app.invoke({
        "question": question, "messages": [], "route": "",
        "retrieved": "", "sources": [], "tool_result": "",
        "answer": "", "faithfulness": 0.0, "eval_retries": 0,
        "paper_topic": "", "user_name": ""
    }, config=config)


# ============================================================
# SESSION STATE
# ============================================================
if "messages"  not in st.session_state: st.session_state.messages  = []
if "thread_id" not in st.session_state: st.session_state.thread_id = str(uuid.uuid4())
if "q_count"   not in st.session_state: st.session_state.q_count   = 0
if "quick_q"   not in st.session_state: st.session_state.quick_q   = None

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="brand-strip">
        <div class="brand-mark">
            <div class="brand-icon">🔬</div>
            <div>
                <div class="brand-name">PaperIQ</div>
                <div class="brand-tagline">Research Intelligence</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Stats
    st.markdown('<div class="sec-hdr">Session</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-lbl">Queries</div>
            <div class="stat-val">{st.session_state.q_count}</div>
        </div>
        <div class="stat-card">
            <div class="stat-lbl">KB Docs</div>
            <div class="stat-val neutral">10</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">Quick Questions</div>', unsafe_allow_html=True)

    quick_qs = [
        "What is an abstract?",
        "Explain methodology section",
        "What are NLP evaluation metrics?",
        "How to find research papers?",
        "How do citations work?",
        "What is a literature review?",
    ]
    for q in quick_qs:
        if st.button(q, key=f"qq_{q}"):
            st.session_state.quick_q = q
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">Actions</div>', unsafe_allow_html=True)
    st.markdown('<div class="new-chat-wrap sb-pad">', unsafe_allow_html=True)
    if st.button("＋  New Conversation", key="new_chat"):
        st.session_state.messages  = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.q_count   = 0
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-hdr">Knowledge Base</div>', unsafe_allow_html=True)
    topics = ["Abstract & Keywords", "Methodology", "Results & Findings",
              "Conclusion", "Authors & Affiliations", "Citations & DOI",
              "Literature Review", "NLP / CV Metrics", "Paper Discovery", "Paper Structure"]
    for t in topics:
        st.markdown(f'<div class="kb-topic">{t}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="padding:0 1.2rem; font-size:11px; color:#3d3d55; line-height:1.6;">
        Powered by LangGraph · ChromaDB · Groq LLaMA
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">✦ RAG &nbsp;·&nbsp; LangGraph &nbsp;·&nbsp; ChromaDB</div>
    <h1 class="hero-title">Research Paper<br><em>Intelligence</em></h1>
    <p class="hero-sub">
        Ask anything about research papers — methodology, results,
        citations, evaluation metrics, or where to discover them online.
    </p>
    <div class="hero-pills">
        <span class="hero-pill">📄 10 KB Documents</span>
        <span class="hero-pill">⚡ Groq LLaMA 3.3</span>
        <span class="hero-pill">🔍 Semantic Search</span>
        <span class="hero-pill">✅ Faithfulness Eval</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# CHAT AREA
# ============================================================
st.markdown('<div class="chat-area">', unsafe_allow_html=True)

# Empty state
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-orb">🧠</div>
        <div class="empty-title">Ready to decode your research</div>
        <div class="empty-desc">
            Ask about any section of a research paper, or pick a quick question
            from the sidebar to get started.
        </div>
        <div class="suggestion-grid">
            <div class="suggestion-card">📝 What sections does a research paper have?</div>
            <div class="suggestion-card">📊 How do I read a results table?</div>
            <div class="suggestion-card">🔗 What is a DOI and why does it matter?</div>
            <div class="suggestion-card">🌐 Where can I find free research papers?</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Render message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            meta   = msg["meta"]
            badges = ""
            if meta.get("route"):
                badges += f'<span class="badge badge-route">⇢ {meta["route"]}</span>'
            for s in (meta.get("sources") or [])[:2]:
                badges += f'<span class="badge badge-source">◈ {s[:30]}</span>'
            if meta.get("faith") and meta["faith"] > 0:
                cls = "badge-score" if meta["faith"] >= 0.7 else "badge-low"
                badges += f'<span class="badge {cls}">✦ {meta["faith"]:.0%} faithful</span>'
            if badges:
                st.markdown(f'<div class="meta-row">{badges}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# INPUT HANDLING
# ============================================================
user_input = st.chat_input("Ask anything about research papers…")

if st.session_state.quick_q:
    user_input = st.session_state.quick_q
    st.session_state.quick_q = None

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.q_count += 1

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        st.markdown("""
        <div class="thinking">
            <div class="dot-trio"><span></span><span></span><span></span></div>
            Searching knowledge base…
        </div>
        """, unsafe_allow_html=True)

        result  = ask(user_input, thread_id=st.session_state.thread_id)
        answer  = result.get("answer", "Sorry, I couldn't generate a response.")
        sources = result.get("sources", [])
        route   = result.get("route", "")
        faith   = result.get("faithfulness", 0.0)

        st.markdown(answer)

        badges = ""
        if route:
            badges += f'<span class="badge badge-route">⇢ {route}</span>'
        for s in sources[:2]:
            badges += f'<span class="badge badge-source">◈ {s[:30]}</span>'
        if faith and faith > 0:
            cls = "badge-score" if faith >= 0.7 else "badge-low"
            badges += f'<span class="badge {cls}">✦ {faith:.0%} faithful</span>'
        if badges:
            st.markdown(f'<div class="meta-row">{badges}</div>', unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "meta": {"route": route, "sources": sources, "faith": faith}
    })
    st.rerun()
