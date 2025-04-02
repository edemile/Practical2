# **Efficient Indexing & Querying of Lecture Notes**

## **Project Overview**
This project evaluates different embedding models and vector databases to determine the best pipeline for efficiently indexing and querying lecture notes. The goal is to measure:

- **Indexing time** – How long it takes to store embeddings in each database.
- **Query time** – How quickly each database retrieves relevant information.
- **Memory usage** – The impact of different models and databases on system resources.
- **Overall efficiency** – Finding the optimal combination for specific use cases.

This allows us to compare performance across **Redis**, **ChromaDB**, and **FAISS** when paired with embedding models like **MiniLM**, **MPNet**, and **InstructorXL**.

## **Installation & Dependencies**

### **Prerequisites**
Ensure you have Python installed. You'll also need:

- **Redis**
- **FAISS**
- **ChromaDB**
- **Ollama** 

### **Required Libraries**
A list of required Python libraries is provided in `requirements.txt`. You can install them manually using `pip`.

## **How to Run**

### **1. Scrape all Lecture Notes**  
Begin by scraping all the necessary lecture notes that will be processed in this pipeline.

### **2. Convert Lecture Notes to PDFs**  
Once the lecture notes are scraped, convert them to PDF format for preprocessing.

### **3. Preprocess PDFs**  
You can preprocess the PDFs using either `txtcleaning.py` or `text_preprocess_flexible.py` depending on your desired cleaning preferences.

### **4. Run Preprocessed Data through Embedding Models**  
Run the preprocessed data through each of the embedding models by executing `embeddings.py`. This script will process the data and save the results to the directory "embedded_files."

### **5. Use Vector Databases**  
You can utilize different vector databases by running the following scripts:
- `chroma_db.py` for Chroma
- `redis_vectordb.py` for Redis
- `faiss_vector_db.py` for FAISS  

You can adjust the parameters to use your desired embedding model by changing the global variable at the top of each script. To switch the model, simply change the value for the `embedding_model` global variable (choose between MiniLM, MPNet, or InstructorXL). Model-specific parameters are automatically adjusted when you modify the model name.

### **6. Querying Capabilities**  
Utilize the querying capabilities for the selected database by running the following scripts:
- `ollama_rag_chroma.py` for ChromaDb
- `ollama_rag_faiss.py` for FAISS
- `ollama_rag_redis.py` for Redis

#### **Example:**
If you want to use **ChromaDB**, **MiniLM**, and **Mistral**, you would:
1. Run `chroma_db.py` with the global variable for the embedding model set to MiniLM.
2. Run `ollama_rag_chroma.py` by changing the global variable to MiniLM and ensuring that the `LLM` parameter in the `generate_response` function is set to `str = 'mistral'`.


## **Conclusion**  
This project aims to benchmark different pipelines with the goal of determining the best model for each use case based on the indexing time and query performance. The final goal is to optimize the pipeline to achieve the fastest and most efficient results.
