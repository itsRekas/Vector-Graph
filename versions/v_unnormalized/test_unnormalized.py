from pymilvus import CollectionSchema, FieldSchema, DataType
from lib.VectorDataBase import VectorDataBase

# Initialize VectorDataBase
vdb = VectorDataBase(
    database_name="lubm_db",
    host="localhost", 
    port=19530,
    embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
    # embedding_model="BAAI/bge-large-en-v1.5",
    # target_embedding_dim=512
    target_embedding_dim=384
)

# Connect to Milvus
vdb.connect()

# Collection name
# collection_name = "vector_graph"
collection_name = "lubm_graph_v1"

# Define collection schema with concatenated embedding field
# Embedding dimension is 3 * target_embedding_dim (subject + predicate + object)
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=vdb.embedding_dimension),
]

schema = CollectionSchema(fields, "Collection for text embeddings with concatenated vector field")

# Create collection using VectorDataBase class
collection = vdb.add_collection(collection_name, schema, log=True)

# Insert data from file using VectorDataBase class
print("Inserting data from file...")
# insert_result = vdb.insert_data_from_file("sample_data.nt", collection_name, log=True)
# insert_result = vdb.insert_data_from_file("RLUBM.nt", collection_name, log=True)


# insert_result = vdb.insert_data_from_file("RLUBM_cleaned.nt", collection_name, log=True)
# vdb.clear_collection(collection_name)
# vdb.drop_collection(collection_name)

# Create index on embedding field
print("Creating index...")
vdb.create_index(collection_name, "embedding", log=True)

# Load collection into memory using VectorDataBase class
print("Loading collection into memory...")
vdb.load_data(collection_name, log=True)

# Example search using VectorDataBase class
print("\n--- Performing similarity search ---")
search_texts = [
    # "SELECT ?X WHERE {<http://www.Department0.University0.edu/GraduateStudent5> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?X}"
    "SELECT ?X WHERE{ ?X <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://swat.cse.lehigh.edu/onto/univ-bench.owl#GraduateStudent>}"
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

# Don't drop collection - keep it for the endpoint to use
print("Keeping collection 'text_embeddings' for the endpoint to use...")
# vdb.drop_collection(collection_name, log=True)