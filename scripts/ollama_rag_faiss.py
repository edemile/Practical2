import ollama
import numpy as np
import json
import faiss
import time
from sentence_transformers import SentenceTransformer

#models = "MiniLM", "MPNet", "InstructorXL"
EMBEDDING_MODEL_NAME = "MiniLM"

EMBEDDING_MODELS = {
    "MiniLM": {
        "path": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384
    },
    "MPNet": {
        "path": "sentence-transformers/all-mpnet-base-v2",
        "dim": 768
    },
    "InstructorXL": {
        "path": "hkunlp/instructor-xl",
        "dim": 768
    }
}

embedding_config = EMBEDDING_MODELS[EMBEDDING_MODEL_NAME]
embedder = SentenceTransformer(embedding_config["path"])
VECTOR_DIM = embedding_config["dim"]

#index and load metadata
INDEX_PATH = f"faiss_indexes/{EMBEDDING_MODEL_NAME}.index"
METADATA_PATH = f"faiss_indexes/{EMBEDDING_MODEL_NAME}_metadata.json"

try:
    INDEX = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r") as f:
        METADATA = json.load(f)
except Exception as e:
    print(f"Failed to load FAISS index or metadata: {e}")
    exit(1)

#embedding query
def embed_query(query: str) -> np.ndarray:
    return embedder.encode(query).astype(np.float32)

#search similar through FAISS
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
            print(f"FAISS index {idx} exceeds metadata length {len(METADATA)}")
    return results

#create prompt
def format_context(results):
    return "\n\n".join(
        [f"[{i+1}] {chunk}" for i, (_, chunk, _) in enumerate(results)]
    )

def build_prompt(query: str, context: str) -> str:
    return f"""Answer the question using only the following and related context:\n\n{context}\n\nQuestion: {query}\nAnswer:"""

#LLM options: llama3.2, mistral
def generate_response(query: str, model: str = "llama3.2") -> str:
    query_vec = embed_query(query)
    results = search_similar(query_vec, k=5)
    context = format_context(results)
    prompt = build_prompt(query, context)

    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

#interaction and querying
if __name__ == "__main__":
    print(f"{EMBEDDING_MODEL_NAME} + Ollama RAG (FAISS)")
    while True:
        user_query = input("\nAsk a question ('exit' to quit): ")
        if user_query.lower() == "exit":
            break
        start = time.time()
        answer = generate_response(user_query)
        duration = time.time() - start
        print(f"\nResponse (in {duration:.2f} seconds):\n{answer}")
