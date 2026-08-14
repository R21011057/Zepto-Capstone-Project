# Module 3 — Support Assistant

This module implements a RAG-based local support assistant using LangGraph, ChromaDB, and FastAPI.

## Architecture

1. **Document Loading & Chunking**: The exact 8-document Zepto policy corpus is loaded from `corpus.py`. Each policy acts as its own discrete chunk.
2. **Embedding**: `sentence-transformers` (`all-MiniLM-L6-v2`) is used to convert the policy texts into dense vector embeddings.
3. **Storage**: Vectors and documents are stored in an in-memory `chromadb` collection.
4. **Agent Logic (LangGraph)**:
    - **classify_intent**: Determines if a query is a `policy_question` (using strict keyword heuristics) or a `general_question`.
    - **route_intent**: Directs flow.
    - **retrieve_and_answer**: Queries ChromaDB, retrieves the top policy chunk, and returns a deterministic mock response (since `MOCK_LLM=1` is default).
    - **direct_answer**: Bypasses retrieval entirely and returns a general canned mock response.
5. **FastAPI**: Exposes a `/ask` endpoint ensuring responses adhere to a strict Pydantic schema (`answer`, `sources`, `confidence`).

## Test Results

Since the graded path requires `MOCK_LLM=1` to be deterministic and avoid cloud dependencies, here are the actual recorded JSON outputs from testing the state graph logic:

### Policy Query (Triggers Retrieval)
**Query**: "What is your return policy?"
**Response**:
```json
{
  "answer": "MOCK POLICY ANSWER: Based on 'Zepto Return Policy: Perishable items cannot be returned once delivered. Non-perishable items can be returned within 3 days of delivery if they are unopened and in their original packaging.'",
  "sources": [
    "zepto_policy_db"
  ],
  "confidence": 0.4649718403816223
}
```

### General Query (Skips Retrieval)
**Query**: "Hello there"
**Response**:
```json
{
  "answer": "MOCK GENERAL ANSWER: I am a Zepto support bot. How can I help you with your order?",
  "sources": [],
  "confidence": 1.0
}
```

## Docker
A lightweight `Dockerfile` is provided that packages the application and exposes port 8000 for local containerized execution.
