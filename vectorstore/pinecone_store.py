from pinecone import Pinecone
from uuid import uuid4

class PineconeStore:
    def __init__(self, index_name: str):
        pc=Pinecone()
        self.index = pc.Index(index_name)

    # def upsert(self, embeddings, metadatas):
    #     vectors = []
    #     for emb, meta in zip(embeddings, metadatas):
    #         vectors.append((
    #             str(uuid4()),
    #             emb,
    #             meta
    #         ))
    #     self.index.upsert(vectors=vectors)

    def upsert(self, vectors):
        self.index.upsert(vectors=vectors)

    def delete_by_source(self, source: str):
        self.index.delete(filter={"source": source})
