from openai import embeddings
from storage.local import LocalStorage
from chunking.splitter import TextSplitter
from vectorstore.pinecone_store import PineconeStore
from embeddings.embeder import Embeder
from dotenv import load_dotenv
import os
load_dotenv()

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
    all_chunks = []
    all_metadatas = []
    storage = LocalStorage()
    splitter = TextSplitter()
    embeder = Embeder()
    vector_store = PineconeStore(index_name=os.getenv("PINECONE_INDEX"))
    files = storage.list_files()
    print("Files found:", files)

    for file in files:
        content = storage.read_file(file)
        # print(f"\n--- {file} ---")
        # print(content[:200])  # preview
        chunks = splitter.chunk_text(content)
        # print(f"Chunks preview: {chunks[:2]}")  # preview first 2 chunks
        # print(f"Number of chunks: {len(chunks)}")
        # for i, c in enumerate(chunks[:2]):
        #     print(f"Chunk {i} length: {len(c)}")
        # for i, c in enumerate(chunks):
        #     all_chunks.append(c)
        #     all_metadatas.append({"source": file, "chunk_index": i,"text": c})
        # embeddings = embeder.embed_documents(all_chunks)
        # vector_store.upsert(embeddings, all_metadatas)
        # all_chunks.extend(chunks)
        vectors = []
        file_category = CATEGORY_MAP.get(file)
        embeddings = embeder.embed_documents(chunks)
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            
            category= file_category or detect_category_from_text(chunk)
            # vector_id = f"{file}_chunk_{i}"
            # vectors.append({
            #     "id": vector_id,          //for index in v1
            #     "values": embedding,
            #     "metadata": {
            #         "doc_id": file,
            #         "source": file,
            #         "chunk_index": i,
            #         "text": chunk
            #     }
            # })

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
                    "text": chunk
                }
            })

        vector_store.upsert(vectors)

    print("Ingestion complete.")
        
   

    


if __name__ == "__main__":
    main()
