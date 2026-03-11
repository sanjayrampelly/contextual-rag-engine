# from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# class Embeder:
#     def __init__(self, model="text-embedding-3-small"):
#         self.client = OpenAI()
#         self.model = model

#     def embed_text(self, text: list[str]) -> list[list[float]]:
#         response = self.client.embeddings.create(
#             input=text,
#             model=self.model
#         )
#         return [item.embedding for item in response.data]


class Embeder:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.embedder = HuggingFaceEmbeddings(
            model_name=model_name
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.embed_query(text)
