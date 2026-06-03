import os
from dotenv import load_dotenv

from embeddings.embeder import Embeder
from generation.groq_llm import GroqGenerator
from retrieval.hybrid_reranker import HybridReranker
from retrieval.pinecone_retrieval import PineconeRetriever

load_dotenv()


def run_query(query: str, top_k: int = 5, min_score: float = 0.60, final_top_k: int = 3) -> str:
    """End-to-end retrieval + hybrid rerank + generation for a single query."""
    retriever = PineconeRetriever(index_name=os.getenv("PINECONE_INDEX"))
    embeder = Embeder()

    query_embedding = embeder.embed_query(query)

    # Exclude the "nature pine cones" documents to avoid the classic
    # ambiguity with Pinecone the vector database.
    matches = retriever.retrieve(
        query_embedding,
        top_k=top_k,
        filter={"category": {"$ne": "pinecone_nature"}},
    )

    reranker = HybridReranker(alpha=0.75)
    reranked = reranker.rerank(query, matches)

    final = [m for m in reranked if m["score"] >= min_score][:final_top_k]
    context = "\n\n---\n\n".join(m["metadata"]["text"] for m in final)

    generator = GroqGenerator()
    return generator.generate(query, context)


def main():
    query = "Why use Pinecone in RAG systems?"
    print("Query:", query)
    print("-" * 50)
    print(run_query(query))


if __name__ == "__main__":
    main()
