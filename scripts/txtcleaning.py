import pdfplumber
import re
import os  

def clean_text(text):
    #fix bullet points for consistent spacing
    text = re.sub(r'\s*([●•○■])\s*', r'\n\1 ', text)  

    #section numbers
    text = re.sub(r'(?<=\D)(\d+)\s', r'\n\1. ', text)  

    #spaces
    text = re.sub(r'\s+', ' ', text).strip()

    #paragraph breaks
    text = re.sub(r'(?<=\.|\?)\s+(?=[A-Z]|\n[●•○■])', '\n\n', text)

    #remove blank lines
    text = re.sub(r'\n\s*\n', '\n\n', text) 

    return text

def process_pdfs_in_directory(directory_path, save_to_files=False):
    """processes all PDF files in a directory, cleans text, and saves"""

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
                        print(f"Warning: text not extracted from {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")


#directory for PDF files
pdf_directory = "/Users/etsub/Desktop/4300/Practical2/materials"  

#save the cleaned text to files
#set tp false to not save and to just run
process_pdfs_in_directory(pdf_directory, save_to_files=True) 
