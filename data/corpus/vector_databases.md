# Vector Databases

A vector database stores high-dimensional embeddings and answers nearest-neighbor queries: given a query vector, return the stored vectors that are closest under a distance metric such as cosine similarity, dot product, or Euclidean distance. They are the storage layer for semantic search, recommendation, and dense retrieval in RAG systems.

Exact nearest-neighbor search compares the query against every stored vector. This brute-force scan is simple and perfectly accurate, and for a few hundred thousand vectors it is often fast enough. Beyond that, approximate nearest neighbor (ANN) indexes trade a little recall for orders of magnitude less work. Popular families include graph-based indexes like HNSW, clustering-based inverted file indexes (IVF), and product quantization, which compresses vectors into short codes.

Metadata filtering is a defining feature that separates a database from a bare index. Queries like "find similar documents, but only those owned by this tenant and created this year" require combining the ANN search with a structured predicate. Pre-filtering restricts the candidate set before the vector search runs; post-filtering runs the vector search first and then discards non-matching results, which can return too few items when the filter is selective.

Operational concerns matter as much as algorithms. Embeddings must be refreshed when the embedding model changes, indexes must be rebuilt or updated incrementally as documents change, and the system needs persistence, replication, and access control like any other database. Systems such as pgvector, Qdrant, Milvus, Weaviate, and Pinecone take different positions on these trade-offs.
