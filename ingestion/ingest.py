import os
from dotenv import load_dotenv

from chunking.splitter import TextSplitter
from embeddings.embeder import Embeder
from storage.local import LocalStorage
from vectorstore.pinecone_store import PineconeStore

load_dotenv()

# Manual mapping of source files to logical categories. Anything not in this
# map falls back to keyword-based detection.
CATEGORY_MAP = {
    "sample1.txt": "rag",
    "sample2.txt": "pinecone_db",
    "sample3.txt": "pinecone_nature",
}


def detect_category_from_text(text: str) -> str:
    t = text.lower()

    if "retrieval-augmented generation" in t or "rag" in t:
        return "rag"

    if "vector database" in t or "embeddings" in t or "pinecone" in t:
        return "pinecone_db"

    if "pine cones" in t or "seed dispersal" in t or "forestry" in t:
        return "pinecone_nature"

    return "general"


def main():
    storage = LocalStorage()
    splitter = TextSplitter()
    embeder = Embeder()
    vector_store = PineconeStore(index_name=os.getenv("PINECONE_INDEX"))

    files = storage.list_files()
    print("Files found:", files)

    for file in files:
        content = storage.read_file(file)
        chunks = splitter.chunk_text(content)

        file_category = CATEGORY_MAP.get(file)
        embeddings = embeder.embed_documents(chunks)

        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            category = file_category or detect_category_from_text(chunk)

            doc_id = file.replace(".txt", "")
            chunk_id = f"{doc_id}_chunk_{i}"

            vectors.append({
                "id": chunk_id,
                "values": embedding,
                "metadata": {
                    "doc_id": doc_id,
                    "source": file,
                    "chunk_index": i,
                    "chunk_id": chunk_id,
                    "category": category,
                    "text": chunk,
                },
            })

        vector_store.upsert(vectors)

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
