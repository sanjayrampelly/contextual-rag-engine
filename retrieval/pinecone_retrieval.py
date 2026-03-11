from pinecone import Pinecone
import os

class PineconeRetriever:
    def __init__(self, index_name: str):
        pc = Pinecone()
        self.index = pc.Index(index_name)

    def retrieve(self, query_embedding, top_k=6, filter=None):
        result = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter
        )
        return result["matches"]
