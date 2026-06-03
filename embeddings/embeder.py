from langchain_huggingface import HuggingFaceEmbeddings


class Embeder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.embedder = HuggingFaceEmbeddings(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.embed_query(text)
