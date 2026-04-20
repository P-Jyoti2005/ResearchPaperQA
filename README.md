# 📄 Research Paper Q&A Assistant (RAG + RAGAS)

An AI-powered Research Paper Assistant that helps users understand academic papers by answering questions about **abstracts, methodology, results, and evaluation metrics** using a **Retrieval-Augmented Generation (RAG)** pipeline.

---

## 🚀 Features

* 🔍 **Context-Aware Retrieval** using ChromaDB
* 🤖 **LLM-based Answer Generation** (Groq - LLaMA 3.1)
* 🧠 **Conversation Memory** using LangGraph
* 📊 **Evaluation with RAGAS Metrics**
* 💬 **Interactive UI using Streamlit**
* ⚡ **Optimized for low token usage**

---

## 🏗️ System Architecture

User Query → Retriever (ChromaDB) → Context
→ LLM (Groq) → Answer
→ Evaluation (RAGAS)

---

## 🧠 Tech Stack

* **LLM**: Groq (LLaMA 3.1 - 8B Instant)
* **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
* **Vector DB**: ChromaDB
* **Framework**: LangGraph + LangChain
* **Evaluation**: RAGAS
* **Frontend**: Streamlit

---
## 💡 Example Queries

* What is an abstract in a research paper?
* Explain methodology section
* What metrics are used in NLP papers?
* How to find research papers?

---

## 📊 Evaluation (RAGAS)

We evaluate the system using:

* **Faithfulness** → Is answer grounded in context?
* **Context Precision** → Are retrieved documents relevant?

Example Output:

```
Faithfulness       : 0.905
Context Precision  : 1.000
Average Score      : 0.952
```

---

## ⚠️ Limitations

* Small knowledge base (demo-level)
* Limited evaluation due to API rate limits
* No PDF upload (future work)

---

## 🔮 Future Improvements

* 📄 Upload and analyze research papers (PDF)
* 🔎 Advanced retrieval (reranking)
* 🎤 Voice-based queries
* 📊 Visualization of evaluation metrics

---

## 👨‍💻 Author

**P Jyoti**
B.Tech Student | AI/ML Enthusiast

---

## ⭐ Acknowledgements

* LangChain / LangGraph
* ChromaDB
* Groq API
* RAGAS Framework

---

## 📌 Conclusion

This project demonstrates how a **RAG-based system** can be used to build a reliable and grounded **Research Paper Assistant**, combining retrieval, generation, and evaluation into a single pipeline.

---
