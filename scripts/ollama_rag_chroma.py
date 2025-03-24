import ollama
import numpy as np
from sentence_transformers import SentenceTransformer
import time
from chroma_db import search_similar 

# models = "MiniLM", "MPNet", "InstructorXL"
EMBEDDING_MODEL_NAME = "InstructorXL"

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

def embed_query(query: str) -> np.ndarray:
    return embedder.encode(query).astype(np.float32)

def format_context(results):
    return "\n\n".join(
        [f"[{i+1}] {chunk}" for i, (_, chunk, _) in enumerate(results)]
    )

def build_prompt(query: str, context: str) -> str:
    return f"""Answer the question using only context related to the query below.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"""

#change LLMs llama3.2 and mistral
def generate_response(query: str, model: str = "mistral") -> str:
    query_vec = embed_query(query)
    results = search_similar(query_vec, k=5)
    context = format_context(results)
    prompt = build_prompt(query, context)
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

if __name__ == "__main__":
    print(f"{EMBEDDING_MODEL_NAME} + Ollama RAG (Chroma)")
    while True:
        user_query = input("\nAsk a question ('exit' to quit): ")
        if user_query.lower() == "exit":
            break
        start = time.time()
        answer = generate_response(user_query)
        duration = time.time() - start
        print(f"\nResponse (in {duration:.2f} seconds):\n{answer}")
