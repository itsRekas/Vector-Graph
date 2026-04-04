from pymilvus import CollectionSchema, FieldSchema, DataType
from VectorDataBase import VectorDataBase

# Initialize VectorDataBase
# Using faster model for better performance
# Note: all-MiniLM-L6-v2 outputs 384 dimensions (same as L12)
# V4: Final version with k parameter forwarding support from Comunica
vdb = VectorDataBase(
    database_name="lubm_db",
    host="localhost", 
    port=19530,
    embedding_model="all-MiniLM-L6-v2",  # Faster alternative
    # embedding_model="paraphrase-multilingual-MiniLM-L12-v2",  # Original (slower)
    # embedding_model="BAAI/bge-large-en-v1.5",
    target_embedding_dim=384  # Model outputs 384 dims (1152D total: 3×384)
)

# Connect to Milvus
vdb.connect()

# Collection name
# V4: Uses the same collection as the endpoint (lubm_graph_v1_normalized)
# collection_name = "vector_graph"
# collection_name = "lubm_graph_v1"
# collection_name = "lubm_graph"
collection_name = "lubm_graph_v1_normalized"
collection_name = "version_5"

# Define collection schema with concatenated embedding field
# Embedding dimension is 3 * target_embedding_dim (subject + predicate + object)
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=vdb.embedding_dimension),
]

schema = CollectionSchema(fields, "V4: Collection for text embeddings with concatenated vector field, k parameter forwarding support")

# Create collection using VectorDataBase class
collection = vdb.add_collection(collection_name, schema, log=True)

# Insert data from file using VectorDataBase class
print("Inserting data from file...")
# insert_result = vdb.insert_data_from_file("sample_data.nt", collection_name, log=True)
# insert_result = vdb.insert_data_from_file("RLUBM.nt", collection_name, log=True)

# insert_result = vdb.insert_data_from_file("RLUBM_cleaned.nt", collection_name, log=True)
# vdb.clear_collection(collection_name)
# vdb.drop_collection(collection_name)

# Create index on embedding field using HNSW for better performance
print("Creating index with HNSW (optimized for speed)...")
# vdb.create_index(
#     collection_name, 
#     "embedding", 
#     index_type="HNSW",  # Use HNSW instead of IVF_FLAT for faster searches
#     M=16,  # Number of bi-directional links (higher = more accurate but slower)
#     ef_construction=200,  # Size of dynamic candidate list (higher = more accurate but slower)
#     log=True
# )

# Load collection into memory using VectorDataBase class
print("Loading collection into memory...")
vdb.load_data(collection_name, log=True)

# Example search using VectorDataBase class
# V4: Search queries use normalized present parts and unnormalized vectors for missing parts
# k parameter is now forwarded from Comunica to the endpoint
print("\n--- Performing similarity search ---")
search_texts = [
    # "SELECT ?X WHERE {<http://www.Department0.University0.edu/GraduateStudent5> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?X}"
    # "SELECT ?X WHERE {?X <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://swat.cse.lehigh.edu/onto/univ-bench.owl#University> }"
    "SELECT ?X WHERE { <http://www.Department4.University1.edu/GraduateStudent28> <http://swat.cse.lehigh.edu/onto/univ-bench.owl#takesCourse> <http://www.Department0.University0.edu/GraduateCourse0> }"
    # "SELECT ?X WHERE{ ?X <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://swat.cse.lehigh.edu/onto/univ-bench.owl#GraduateStudent>}"
    # "SELECT ?X WHERE{ <http://www.Department0.University0.edu/GraduateStudent5> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://swat.cse.lehigh.edu/onto/univ-bench.owl#GraduateStudent>}"
    # "SELECT ?person ?o WHERE { ?person <http://example.org/occupation> \"Software Engineer\" }",
    # "SELECT ?person ?age WHERE { ?person <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Person> . ?person <http://example.org/age> ?age }"
]

# Search using VectorDataBase class
results = vdb.search(
    collection_name=collection_name,
    # field_name="embedding",  # Optional: defaults to "embedding"
    query_texts=search_texts,
    limit=10,
    log=True
)

# Show collection statistics using VectorDataBase class
# print(f"\nCollection statistics:")
# collection = vdb.get_collection(collection_name, log=True)
# print(f"\Search Results:")
print(results)

print("\nVectorDataBase class demo completed successfully!")
print("V4: This version supports k parameter forwarding from Comunica to the vector endpoint")

# Don't drop collection - keep it for the endpoint to use
print(f"Keeping collection '{collection_name}' for the endpoint to use...")
# vdb.drop_collection(collection_name, log=True)

