#  Memory-Augmented Legal AI Chatbot

A production-oriented **Legal AI Chatbot for Indian Law** that answers questions related to the **Indian Penal Code (IPC)** and **Criminal Procedure Code (CrPC)** using a **memory-augmented Retrieval-Augmented Generation (RAG)** architecture.

The system combines **vector search, symbolic section matching, and conversational memory** to provide accurate, context-aware legal responses.

## Features

* **Hybrid Retrieval**

  * Exact IPC/CrPC section matching
  * Semantic vector search using Sentence Transformers
  * Keyword fallback for robustness

* **Memory Architecture**

  * **Short-term memory**: Recent conversation context
  * **Medium-term memory**: Session summaries stored in vector DB
  * **Long-term memory**: Persistent legal knowledge base (IPC)

* **Legal-Aware Reasoning**

  * Prioritizes section-based queries
  * Avoids hallucinated sections
  * Cites only retrieved legal provisions

* **Response Validation**

  * Semantic similarity validation
  * Citation consistency checks

* **Interactive UI**

  * Streamlit-based frontend
  * Real-time memory visualization

