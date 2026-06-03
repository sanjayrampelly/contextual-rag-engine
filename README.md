# Contextual RAG Engine

A modular Retrieval-Augmented Generation pipeline that combines dense vector
search, metadata filtering, and hybrid (vector + BM25) reranking on top of
Pinecone, with answer generation by a Groq-hosted LLM. The system is designed
to demonstrate the building blocks of a real RAG service, not to be a
plug-and-play API.

## What it does

1. **Ingest** plain-text documents from `data/`, chunk them with a recursive
   character splitter, attach metadata (`doc_id`, `chunk_id`, `source`,
   `chunk_index`, `category`, `text`), and upsert into Pinecone.
2. **Retrieve** the top-k most similar chunks for a query, with an optional
   Pinecone metadata filter (used here to disambiguate "Pinecone the vector
   database" from "pinecones the seed cones of a pine tree").
3. **Rerank** retrieved chunks using one of four strategies (see below).
4. **Generate** an answer with Groq using only the top-N reranked chunks as
   context. If no chunk clears the score threshold, the system declines to
   answer.

## Reranking strategies

| Strategy           | File                                | Notes                                                  |
| ------------------ | ----------------------------------- | ------------------------------------------------------ |
| `SimpleReranker`   | `retrieval/reranker.py`             | Score + length weighting.                              |
| `KeywordRanker`    | `retrieval/keyword_reranker.py`     | Token-overlap with the query.                          |
| `BM25Reranker`     | `retrieval/bm25_ranker.py`          | `rank-bm25` (BM25Okapi).                               |
| `HybridReranker`   | `retrieval/hybrid_reranker.py`      | Min-max normalises vector + BM25, then weighted sum.   |

The hybrid reranker is the default used by `retrieval/retrieval.py`.

## Hallucination guard

After reranking, chunks below `MIN_SCORE` (0.60) are dropped. If nothing
survives, the generator is fed an empty context and the prompt instructs it
to reply *"I don't know based on the provided documents."* rather than
fabricate an answer.

## Tech stack

- Python 3.10+
- Pinecone (vector store)
- HuggingFace `sentence-transformers` via `langchain-huggingface`
  (`BAAI/bge-small-en-v1.5`)
- `langchain-text-splitters` (recursive character splitter)
- `rank-bm25` (BM25Okapi)
- Groq (LLM, defaults to `llama3-8b-8192`)
- `python-dotenv`

## Project structure

```text
contextual-rag-engine/
├── chunking/         # text splitter wrapper
├── data/             # sample .txt documents
├── embeddings/       # HuggingFace embeddings wrapper
├── generation/       # Groq LLM with strict grounding prompt
├── ingestion/        # ingest.py: chunk -> embed -> upsert
├── retrieval/        # retrieve, rerankers, end-to-end query script
├── storage/          # filesystem storage abstraction
└── vectorstore/      # Pinecone wrapper (upsert, delete_by_source)
```

## Setup

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # fill in PINECONE_API_KEY, GROQ_API_KEY, ...
```

## Run

From the project root (so the imports resolve):

```bash
# 1. Ingest sample data into Pinecone
python -m ingestion.ingest

# 2. Run an end-to-end query
python -m retrieval.retrieval
```

## Notes and limitations

- This is a script-driven engine, not an HTTP service. There is no FastAPI
  app yet; wiring one on top of `retrieval.run_query()` is the natural next
  step.
- No automated tests at the moment. The hybrid reranker's normalisation logic
  is the obvious first thing to cover.
- `data/sample3.txt` is intentionally a document about *botanical* pine
  cones; it is used to validate that the metadata filter
  `{"category": {"$ne": "pinecone_nature"}}` correctly excludes off-topic
  matches.
