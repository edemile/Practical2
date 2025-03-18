import pdfplumber
import re
import os  

def clean_text(text):
    # Standardize bullet points with consistent spacing
    text = re.sub(r'\s*([●•○■])\s*', r'\n\1 ', text)  

    # Preserve section numbers with improved spacing
    text = re.sub(r'(?<=\D)(\d+)\s', r'\n\1. ', text)  

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Improved paragraph break handling (as discussed previously)
    text = re.sub(r'(?<=\.|\?)\s+(?=[A-Z]|\n[●•○■])', '\n\n', text)

    # Remove extra blank lines (this is crucial for cleaner output)
    text = re.sub(r'\n\s*\n', '\n\n', text) 

    return text

def process_pdfs_in_directory(directory_path, save_to_files=False):
    """Processes all PDF files in a directory, cleans their text, and saves the results."""

    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory_path, filename)
            try:
                with pdfplumber.open(filepath) as pdf:
                    full_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                    if full_text:
                        cleaned_text = clean_text(full_text)
                        print(f"Processed: {filename}")  
                        if save_to_files:
                            output_filename = os.path.splitext(filename)[0] + "_cleaned.txt"
                            output_filepath = os.path.join(directory_path, output_filename)
                            with open(output_filepath, "w", encoding="utf-8") as outfile:
                                outfile.write(cleaned_text)
                            print(f"Saved cleaned text to: {output_filepath}")
                    else:
                        print(f"Warning: No text extracted from {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


# directory containing the PDF files
pdf_directory = "/Users/etsub/Desktop/4300/Practical2/materials"  

# save the cleaned text to files ---- put false to not save and to just run
process_pdfs_in_directory(pdf_directory, save_to_files=True) 