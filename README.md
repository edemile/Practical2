Redis Vector Indexing & Query Performance Benchmarking

Overview
This project evaluates the performance of Redis as a vector database for indexing and querying embeddings. The benchmarking compares three models:

InstructorXL

MiniLM

MPNet

The goal is to measure indexing time, query time, and overall efficiency to determine the best model for specific use cases.

Experimental Setup
Redis Configuration
Vector Index Type: HNSW (Hierarchical Navigable Small World)

Distance Metric: Cosine Similarity

Index Parameters: 

Data Used
Vector embeddings from different models.

Indexed embeddings were stored in Redis for retrieval and similarity search.

Performance Metrics
Indexing Time
Measures the time taken to add vector embeddings to Redis.

Query Time
Measures the time taken to retrieve the closest vectors from Redis.

Benchmark Results
Indexing Performance
Model	Min Time (s)	Max Time (s)	Avg Time (s)
InstructorXL	0.365	0.372	0.368
MiniLM	0.227	2.428	0.735
MPNet	0.361	0.361	0.361
Query Performance
Query Type	Min Time (s)	Max Time (s)	Avg Time (s)
Redis Query	0.00047	0.0099	0.0042
Key Observations
MiniLM has the most variable indexing time, occasionally spiking.

InstructorXL & MPNet are more stable but slower than MiniLM at its best.

Query times are significantly lower than indexing times across all models.

Conclusion
If speed is the priority → MiniLM is the best option.

If stability is preferred → InstructorXL and MPNet are more predictable.

Redis performs well for fast vector search, but indexing times can vary.
