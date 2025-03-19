import os
import numpy as np
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer

#embedding models
EMBEDDING_MODELS = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet": "sentence-transformers/all-mpnet-base-v2",
    "InstructorXL": "hkunlp/instructor-xl"
}

#directories
#folder with preprocessed .txt files
INPUT_FOLDER = "materials"     
#folder to save embeddings    
OUTPUT_FOLDER = "embedded_files"     
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

#load .txt files
def load_text_files(folder):
    """
    loads and returns contents from .txt files in a given folder
    """
    documents = {}
    
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                documents[filename] = f.read().strip()
    
    return documents


#generate embeddings
def generate_embeddings(documents, model_name, batch_size=16):
    """
    generates embeddings for each document using the given model
    for larger model (instructorXL) runs on cpu rather than gpu bc of memory issues
    """
    device = "mps" if torch.backends.mps.is_available() and model_name != "InstructorXL" else "cpu"
    model = SentenceTransformer(EMBEDDING_MODELS[model_name], device=device)
    
    embeddings_dict = {}
    for filename, text in tqdm(documents.items(), desc=f"Embedding with {model_name}"):
        text_chunks = text.split("\n\n")
        #batch embeddings for efficiency
        batch_embeddings = []
        for i in tqdm(range(0, len(text_chunks), batch_size), desc="Processing batches", leave=False):
            batch = text_chunks[i:i+batch_size]
            batch_embeddings.extend(model.encode(batch, show_progress_bar=False))

        embeddings_dict[filename] = np.array(batch_embeddings)

    #save file per model
    output_path = os.path.join(OUTPUT_FOLDER, f"{model_name}_embedded.npy")
    np.save(output_path, embeddings_dict)
    print(f"Saved {model_name} embeddings to {output_path}")

def process_and_embed():
    """
    loads text files then generates and saves embeddings
    """
    documents = load_text_files(INPUT_FOLDER)
    if not documents:
        print("No text files found")
        return
    
    for model_name in EMBEDDING_MODELS.keys():
        generate_embeddings(documents, model_name)

#run pipeline
process_and_embed()
