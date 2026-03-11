# Contextual_RAG_Engine 

Contextual_RAG_Engine  is a production-oriented **Retrieval-Augmented Generation (RAG) system** designed to generate accurate, context-grounded responses using semantic retrieval, metadata filtering, and hybrid reranking.

The system retrieves relevant knowledge from a vector database and generates answers using an LLM while applying strict filtering and ranking strategies to reduce hallucinations and improve precision.

The project is implemented as a **FastAPI service**, allowing the RAG engine to be accessed through REST APIs.

---

# Key Features

### Retrieval Pipeline
- Semantic search using vector embeddings
- Vector storage with Pinecone
- Metadata-aware retrieval
- Category filtering to remove irrelevant documents

### Ranking Improvements
- Score threshold filtering
- Hybrid reranking (semantic similarity + keyword relevance)
- Context prioritization

### Generation Layer
- Context-grounded response generation
- Hallucination guard when context is insufficient
- Structured output

### API Layer
- FastAPI service exposing the RAG pipeline
- Request / response schemas with validation
- Interactive Swagger documentation

---

# System Architecture

User Query  
↓  
Query Embedding (HuggingFace)  
↓  
Pinecone Vector Search  
↓  
Metadata Filtering  
↓  
Hybrid Reranking  
↓  
Context Assembly  
↓  
LLM Generation (Groq)  
↓  
API Response  

---

# Tech Stack

### Backend
- Python
- FastAPI

### Vector Retrieval
- Pinecone

### Embeddings
- HuggingFace sentence-transformers

### LLM Generation
- Groq API

### Utilities
- python-dotenv
- Pydantic

---


# Hallucination Guard

If the retrieval pipeline cannot find relevant context above the score threshold, the system returns:

```
No relevant information found in the knowledge base.
```

This prevents the LLM from generating unsupported answers.

---
