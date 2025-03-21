import redis
import numpy as np
import struct
import os
import time
from redis.commands.search.field import TagField, VectorField, TextField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from timer import timer

#models = "MiniLM", "MPNet", "InstructorXL"
MODEL_NAME = "MiniLM"

#adjust vector dimension
VECTOR_DIM = {
    "MiniLM": 384,
    "MPNet": 768,
    "InstructorXL": 768
}[MODEL_NAME]

INDEX_NAME = "ds4300_vector_index"
VECTOR_FIELD = "embedding"

#connection to Redis
redis_client = redis.Redis(host="localhost", port=6379)

def get_memory_usage(label=""):
    memory_info = redis_client.info("memory")
    used_bytes = memory_info.get("used_memory", 0)
    used_mb = used_bytes / (1024 * 1024)
    print(f"{label} Redis memory usage: {used_mb:.2f} MB")
    return used_mb

#schema
schema = [
    TagField("id"),
    TextField("text"),
    VectorField(VECTOR_FIELD, "HNSW", {
        "TYPE": "FLOAT32",
        "DIM": VECTOR_DIM,
        "DISTANCE_METRIC": "COSINE"
    }),
]

#create index
def create_index():
    try:
        redis_client.ft(INDEX_NAME).dropindex(delete_documents=True)
    except:
        pass
    index_def = IndexDefinition(prefix=["doc:"], index_type=IndexType.HASH)
    redis_client.ft(INDEX_NAME).create_index(schema, definition=index_def)
    print(f"\nRedis index '{INDEX_NAME}' created for model: {MODEL_NAME} (dim={VECTOR_DIM})")

def timed_insertion(file_path):
    @timer(f"redis_indexing_{MODEL_NAME}")
    def inner():
        insert_embeddings(file_path)
    inner()

def insert_embeddings(file_path):
    embeddings_dict = np.load(file_path, allow_pickle=True).item()
    for doc_id, embedding_array in embeddings_dict.items():
        txt_path = os.path.join("materials", doc_id)
        if not os.path.exists(txt_path):
            continue
        with open(txt_path, "r", encoding="utf-8") as f:
            raw_chunks = f.read().split("\n\n")
        for i, vec in enumerate(embedding_array):
            key = f"doc:{MODEL_NAME}_{doc_id}_{i}"
            chunk_text = raw_chunks[i] if i < len(raw_chunks) else "No text available"
            redis_client.hset(key, mapping={
                "id": f"{doc_id}_{i}",
                "text": chunk_text,
                VECTOR_FIELD: struct.pack(f"{len(vec)}f", *vec)
            })
    print(f"Inserted embeddings from {file_path}")

@timer("redis_query")
def search_similar(query_embedding, k=5):
    query_vector = struct.pack(f"{len(query_embedding)}f", *query_embedding)
    query = (Query(f"*=>[KNN {k} @{VECTOR_FIELD} $vec as score]")
             .sort_by("score")
             .return_fields("id", "text", "score")
             .paging(0, k)
             .dialect(2))
    results = redis_client.ft(INDEX_NAME).search(query, query_params={"vec": query_vector})
    return [(doc.id, doc.text, float(doc.score)) for doc in results.docs]

if __name__ == "__main__":
    create_index()
    before_mem = get_memory_usage("Before indexing")

    embedded_files_dir = "embedded_files"
    embedding_filename = f"{MODEL_NAME}_embedded.npy"
    file_path = os.path.join(embedded_files_dir, embedding_filename)

    if os.path.exists(file_path):
        timed_insertion(file_path)
    else:
        print(f"Embedding file not found: {file_path}")

    after_mem = get_memory_usage("After indexing")
    print(f"\nTotal memory increase from indexing: {after_mem - before_mem:.2f} MB")
