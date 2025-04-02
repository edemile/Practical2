Redis Vector Indexing & Query Performance Benchmarking

Overview
This project evaluates the performance of Redis as a vector database for indexing and querying embeddings. The benchmarking compares three models:

InstructorXL

MiniLM

MPNet

The goal is to measure indexing time, query time, and overall efficiency to determine the best model for specific use cases.

How to Run:

1. Scrape all lecture notes
   
2. Convert lecture notes to PDFs
   
3. Preprocess PDFs using txtcleaning.py or text_preprocess_flexible.py depending on the cleaning preferences
   
4. Run the preprocessed data through each embedding model by running embeddings.py (this runs the preprocessed data through each of the embedding models in one script and saved them to the directory "embedded_files"
   
5. Take the embeddings and utilize different vector databases by running the following scripts
   chroma_db.py 
   redis_vectordb.py
   faiss_vector_db.py
   Can adjust the parameters to use the desired embedding models by switching the global variable at the top of the script to match the desired model (MiniLM, MPNet, InstructorXL). model specific parameters are adjusted automatically by just changing the model name.

6. Utlize querying capabilities by running the rag scripts for the desired databse.
   ollama_rag_chroma.py uses ChromaDb
   ollama_rag_faiss.py uses FAISS
   ollama_rag_redis.py uses redis

7. EXAMPLE: If you wanted to use ChromaDB, MiniLM, and Mistral, you would:
   	run chroma_db.py with the global variable for embedding model changed to MiniLM
   	then run ollama_rag_chroma.py by changing the global variable to MiniLM and ensuring the LLM parameter in the "generate_response" 	function is set to "str = 'mistral'"
    
8. Test outputs based on project specific preference for accuracy, speed, memory usage, and ability to provide desired responses
