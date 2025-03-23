import os
import re
import pdfplumber
from typing import List

#clean text using regex
def clean_text(text: str) -> str:
    #normalize bullet spacing
    text = re.sub(r'\s*([●•○■])\s*', r'\n\1 ', text)
    #keep section numbers  
    text = re.sub(r'(?<=\D)(\d+)\s', r'\n\1. ', text)  
    #remove whitespace
    text = re.sub(r'\s+', ' ', text).strip()         
    #paragraph breaks 
    text = re.sub(r'(?<=\.|\?)\s+(?=[A-Z●•○■])', '\n\n', text)
    #new lines
    text = re.sub(r'\n\s*\n', '\n\n', text)            
    return text

#split whitespace
def tokenize(text: str) -> List[str]:
    return text.split()

def detokenize(tokens: List[str]) -> str:
    return " ".join(tokens)

#chunking with overlap
def chunk_tokens(tokens: List[str], chunk_size: int, overlap: int = 0) -> List[str]:
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = tokens[start:end]
        chunks.append(detokenize(chunk))
        start += chunk_size - overlap
    return chunks

#PDF processor
def process_pdf(filepath: str) -> str:
    with pdfplumber.open(filepath) as pdf:
        pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
    return "\n".join(pages)

#process and chunk directory of files
def process_pdfs_in_directory(
    directory_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 0,
    save_to_files: bool = True
):
    output_dir = os.path.join(directory_path, f"chunked_{chunk_size}_overlap{chunk_overlap}")
    if save_to_files:
        os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(directory_path):
        if not filename.lower().endswith(".pdf"):
            continue

        filepath = os.path.join(directory_path, filename)
        try:
            full_text = process_pdf(filepath)
            if not full_text:
                print(f"No extractable text in {filename}")
                continue

            cleaned = clean_text(full_text)
            tokens = tokenize(cleaned)
            chunks = chunk_tokens(tokens, chunk_size, chunk_overlap)

            if save_to_files:
                base_name = os.path.splitext(filename)[0]
                for i, chunk in enumerate(chunks):
                    out_path = os.path.join(output_dir, f"{base_name}_chunk_{i}.txt")
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(chunk)

                print(f"{filename} → {len(chunks)} chunks saved to: {output_dir}")
            else:
                print(f"{filename} → {len(chunks)} chunks processed (not saved)")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    pdf_directory = "/Users/etsub/Desktop/4300/Practical2/materials"

    #felxible edits to chunking and overlap
    process_pdfs_in_directory(
        directory_path=pdf_directory,
        chunk_size=500,
        chunk_overlap=50,
        save_to_files=True
    )
