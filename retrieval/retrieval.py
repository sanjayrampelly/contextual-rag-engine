from embeddings.embeder import Embeder
from .pinecone_retrieval import PineconeRetriever
import os
from .reranker import SimpleReranker as Reranker
from .keyword_reranker import KeywordRanker as KeywordRanker
from .bm25_ranker import BM25Reranker
from generation.groq_llm import GroqGenerator
from retrieval.hybrid_reranker import HybridReranker
from dotenv import load_dotenv
load_dotenv()

def main():
    retriever = PineconeRetriever(index_name=os.getenv("PINECONE_INDEX"))

    embeder = Embeder()
    query = "Why use Pinecone in RAG systems?"  
    query_embedding = embeder.embed_query(query)
    matches = retriever.retrieve(query_embedding, top_k=5, filter={"category": {"$ne": "pinecone_nature"}})

    for m in matches:
        print("ID:", m["id"])
        print("Doc Id:", m["metadata"]["doc_id"])
        print("Score:", m["score"])
        print("Source:", m["metadata"]["source"])
        print("Chunk Index:", m["metadata"]["chunk_index"])
        print("Category:", m["metadata"].get("category", "N/A"))
        print("Text:", m["metadata"]["text"][:200])  # preview first 200 chars
        print("-" * 50)

    filtered = [
    m for m in matches
    if m["score"] >= 0.50
    ]

    context = "\n\n---\n\n".join(
    m["metadata"]["text"] for m in filtered
    )
    print("-" * 50)
    print("Context for RAG:")
    print(context)
    print("-" * 50)
    # reranker = Reranker()
    # reranked_matches = reranker.rerank(matches)
    reranker= HybridReranker(alpha=0.75)
    reranked_matches = reranker.rerank(query, matches)
    for rm in reranked_matches:
        print("Reranked ID:", rm["id"])
        print("DOC ID:", rm["metadata"]["doc_id"])
        print("Reranked Hybrid Score:", rm["hybrid_score"])
        print("category:", rm["metadata"].get("category", "N/A"))
        print("Reranked Score:", rm["score"])
        print("Reranked Source:", rm["metadata"]["source"])
        print("Reranked Chunk Index:", rm["metadata"]["chunk_index"])
        print("Reranked Text:", rm["metadata"]["text"][:200])
        print("-" * 50)

    MIN_SCORE = 0.60

    reraked_filtered = [
        m for m in reranked_matches
        if m["score"] >= MIN_SCORE
    ]
    final_results = reraked_filtered[:3]
    reranked_context = "\n\n---\n\n".join(
        m["metadata"]["text"] for m in final_results
    )

    # ranker=BM25Reranker()
    # reranked_matches = ranker.rerank(query, matches)
    # for rm in reranked_matches:
    #     print("Reranked ID:", rm["id"])
    #     print("Reranked Score:", rm["score"])
    #     print("BM25 Score:", rm["bm25_score"])
    #     print("Reranked Source:", rm["metadata"]["source"])
    #     print("Reranked Chunk Index:", rm["metadata"]["chunk_index"])
    #     print("Reranked Text:", rm["metadata"]["text"][:200])
    #     print("-" * 50)
    
    # FINAL_TOP_K = 3

    # final_results = reranked_matches[:FINAL_TOP_K]

    # reranked_context = "\n\n---\n\n".join(
    #     m["metadata"]["text"] for m in final_results
    # )

    print("\n========== FINAL CONTEXT FOR GENERATION ==========\n")
    print(reranked_context)

    generator = GroqGenerator()
    answer = generator.generate(query, reranked_context)

    print("\n========== GENERATED ANSWER ==========\n")
    print(answer)


if __name__ == "__main__":
    main()