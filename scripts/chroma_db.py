import chromadb
from chromadb.config import Settings
import numpy as np
import os
import time
from timer import timer
import psutil

#embedding model
MODEL_NAME = "MiniLM"  # "MiniLM", "MPNet", "InstructorXL"

#adjust dimensions based on embedding model
VECTOR_DIM = {
    "MiniLM": 384,
    "MPNet": 768,
    "InstructorXL": 768
}[MODEL_NAME]

CHROMA_COLLECTION_NAME = f"ds4300_{MODEL_NAME}_collection"

#set up chroma client
chroma_client = chromadb.PersistentClient(path="./chroma_store")

collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

def get_memory_usage(label=""):
    used_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    print(f"{label} Python memory usage: {used_mb:.2f} MB")
    return used_mb

@timer(f"chroma_indexing_{MODEL_NAME}")
def insert_embeddings(file_path):
    embeddings_dict = np.load(file_path, allow_pickle=True).item()
    ids, documents, embeddings = [], [], []

    for doc_id, embedding_array in embeddings_dict.items():
        txt_path = os.path.join("materials", doc_id)
        if not os.path.exists(txt_path):
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            raw_chunks = f.read().split("\n\n")

        for i, vec in enumerate(embedding_array):
            chunk_id = f"{MODEL_NAME}_{doc_id}_{i}"
            chunk_text = raw_chunks[i] if i < len(raw_chunks) else "No text available"
            ids.append(chunk_id)
            documents.append(chunk_text)
            embeddings.append(vec.tolist())

    collection.add(documents=documents, embeddings=embeddings, ids=ids)
    print(f"Inserted {len(ids)} embeddings from {file_path}")

@timer(f"chroma_query_{MODEL_NAME}")
def search_similar(query_embedding, k=5):
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=k)
    return list(zip(results['ids'][0], results['documents'][0], results['distances'][0]))

#indexing and memory usage
if __name__ == "__main__":
    before_mem = get_memory_usage("Before indexing")

    embedded_files_dir = "embedded_files"
    embedding_filename = f"{MODEL_NAME}_embedded.npy"
    file_path = os.path.join(embedded_files_dir, embedding_filename)

    if os.path.exists(file_path):
        insert_embeddings(file_path)
    else:
        print(f"Embedding file not found: {file_path}")

    after_mem = get_memory_usage("After indexing")
    print(f"\nMemory increase from indexing: {after_mem - before_mem:.2f} MB")