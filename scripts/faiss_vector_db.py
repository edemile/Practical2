import os
import time
import json
import numpy as np
import faiss
import psutil
from timer import timer

#models = "MiniLM", "MPNet", "InstructorXL"
MODEL_NAME = "MiniLM"

VECTOR_DIM = {
    "MiniLM": 384,
    "MPNet": 768,
    "InstructorXL": 768
}[MODEL_NAME]

INDEX_PATH = f"faiss_indexes/{MODEL_NAME}.index"
METADATA_PATH = f"faiss_indexes/{MODEL_NAME}_metadata.json"

INDEX = faiss.IndexFlatIP(VECTOR_DIM)
METADATA = []

#memory analytics
def get_memory_usage(label=""):
    used_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    print(f"{label} memory usage: {used_mb:.2f} MB")
    return used_mb


@timer(f"faiss_indexing_{MODEL_NAME}")
def insert_embeddings(file_path):
    embeddings_dict = np.load(file_path, allow_pickle=True).item()
    all_vecs = []
    metadata_out = []

    for doc_id, embedding_array in embeddings_dict.items():
        txt_path = os.path.join("materials", doc_id)
        if not os.path.exists(txt_path):
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            raw_chunks = f.read().split("\n\n")

        for i, vec in enumerate(embedding_array):
            vec = vec.astype(np.float32)
            vec = vec / np.linalg.norm(vec) if np.linalg.norm(vec) != 0 else vec
            all_vecs.append(vec)
            chunk_text = raw_chunks[i] if i < len(raw_chunks) else "No text available"
            metadata_out.append((f"{doc_id}_{i}", chunk_text))

    INDEX.add(np.array(all_vecs))
    print(f"Inserted {len(all_vecs)} vectors into FAISS for {MODEL_NAME}")

    #save index and metadata
    os.makedirs("faiss_indexes", exist_ok=True)
    faiss.write_index(INDEX, INDEX_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata_out, f)
    print(f"FAISS index and metadata saved")

@timer(f"faiss_query_{MODEL_NAME}")
def search_similar(query_embedding, k=5):
    vec = query_embedding.astype(np.float32)
    vec = vec / np.linalg.norm(vec) if np.linalg.norm(vec) != 0 else vec
    scores, indices = INDEX.search(np.array([vec]), k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(METADATA):
            doc_id, chunk_text = METADATA[idx]
            results.append((doc_id, chunk_text, float(scores[0][i])))
        else:
            print(f"Index {idx} out of bounds (metadata len {len(METADATA)})")
    return results

if __name__ == "__main__":
    INDEX.reset()
    METADATA.clear()

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
