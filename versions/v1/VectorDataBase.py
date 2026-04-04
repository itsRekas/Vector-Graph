from pymilvus import connections, utility, MilvusException, Collection, CollectionSchema, FieldSchema, DataType
from pymilvus.orm.mutation import MutationResult
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional, Sequence, Tuple
import numpy as np
import re
import torch

class VectorDataBase:
    
    def __init__(self, database_name : str, host : str, port : int,  embedding_model : str, target_embedding_dim: int):
        self._database_name : str = database_name
        self._host : str = host
        self._port : str = str(port)
        
        # Enable GPU if available for faster embedding generation
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing embedding model on device: {device}")
        self._embedding_model : SentenceTransformer = SentenceTransformer(embedding_model, device=device)
        
        self._collections : set[str] = set()
        self._embedding_dim : int  = target_embedding_dim
        
        # Public properties for schema creation
        # Raw embedding: each triple line embedded as-is (no concatenation)
        self.embedding_dimension = target_embedding_dim
    
    def connect(self):
        try:  
            connections.connect(host = self._host, port = self._port)

            self._collections = set(utility.list_collections())
            print(f"Connected to Milvus. Found {len(self._collections)} existing collections.")
        except MilvusException as e:
            print(f"Error connecting to Milvus: {e}")

    def add_collection(self, collection_name : str, schema : CollectionSchema, log : bool = False) -> Collection:
        try:
            if log:
                print(f"Existing collections: {self._collections}")
            
            if collection_name in self._collections:
                if log:
                    print(f"Collection already exits: {collection_name}")
                return Collection(collection_name)
            
            else:
                collection = Collection(collection_name, schema)
                self._collections.add(collection_name)
                if log:
                    print(f"Added {collection_name} to collections")
                return collection
            
        except MilvusException as e:
            print(f"Error checking collections: {e}")
            return None

    def get_collection(self, collection_name: str, log: bool = False) -> Union[Collection, None]:
        """
        Get an existing collection by name.
        
        Args:
            collection_name: Name of the collection to retrieve
            log: Whether to print log messages
            
        Returns:
            Collection: The collection object if found, None otherwise
        """
        try:
            if collection_name not in self._collections:
                if log:
                    print(f"Collection '{collection_name}' not found in tracked collections")
                return None
            
            collection = Collection(collection_name)
            
            if log:
                print(f"Successfully retrieved collection '{collection_name}'")
                print(f"Collection schema: {collection.schema}")
                print(f"Number of entities: {collection.num_entities}")
            
            return collection
            
        except MilvusException as e:
            print(f"Milvus error while getting collection '{collection_name}': {e}")
            return None
        except Exception as e:
            print(f"Unexpected error while getting collection '{collection_name}': {e}")
            return None


    @staticmethod
    def _format_query_sentence(subject: Optional[str], predicate: Optional[str], obj: Optional[str], object_type: Optional[str] = None) -> str:
        """Create a triple string by concatenating parts as-is (no period)."""
        return f"{subject} {predicate} {obj}"
    
    @staticmethod
    def _extract_triple_from_sparql(sparql_query: str) -> Optional[str]:
        """
        Extract triple pattern from SPARQL query string.
        Returns the triple in format: <subject> <predicate> <object> (no period)
        Returns None if no valid triple pattern found.
        """
        # Pattern to match triple in SPARQL: <s> <p> <o> or <s> <p> "literal" or ?var <p> <o>, etc.
        # Look for patterns like: <uri> <uri> <uri> or <uri> <uri> "literal"
        # This regex matches: <...> <...> (<...>|"...")
        pattern = r'<([^>]+)>\s+<([^>]+)>\s+(?:<([^>]+)>|"([^"]+)")'
        
        # Try to find triple pattern in the query
        match = re.search(pattern, sparql_query)
        if match:
            subject = f"<{match.group(1)}>"
            predicate = f"<{match.group(2)}>"
            # Object can be URI or literal
            if match.group(3):  # URI
                obj = f"<{match.group(3)}>"
            else:  # Literal
                obj = f'"{match.group(4)}"'
            
            # Return in same format as stored: <s> <p> <o> (no period)
            return f"{subject} {predicate} {obj}"
        
        # If no match, try a simpler pattern that handles variables
        # Look for: <uri> <uri> <uri> (ignoring variables)
        simple_pattern = r'(<[^>]+>)\s+(<[^>]+>)\s+(<[^>]+>)'
        match = re.search(simple_pattern, sparql_query)
        if match:
            return f"{match.group(1)} {match.group(2)} {match.group(3)}"
        
        return None

    def _encode_text_batch(self, texts: Sequence[str], normalize: bool = False) -> np.ndarray:
        """Encode a batch of strings to embeddings.
        
        No normalization is applied - embeddings are stored as-is from the model.
        No caching - embeddings are computed fresh each time.
        
        Args:
            texts: List of strings to encode
            normalize: Ignored (kept for compatibility, but normalization is disabled)
        """
        if not texts:
            return np.zeros((0, self._embedding_dim), dtype=float)
        
        # Encode all texts directly
        try:
            embeddings = self._embedding_model.encode(
                texts, 
                convert_to_numpy=True, 
                show_progress_bar=False,
                batch_size=32  # Batch for better GPU utilization
            )
        except TypeError:
            embeddings = self._embedding_model.encode(
                texts, 
                show_progress_bar=False
            )
        
        embeddings = np.asarray(embeddings, dtype=float)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        # No normalization - embeddings stored as-is from the model
        
        return embeddings


        
    def insert_data_from_file(self, filename : str, collection_name : str, log : bool = False, chunk_size : int  = 100 ) -> Union[MutationResult, List]:
        """
        Insert raw triple lines from .nt file as-is into the vector database.
        Each line is embedded directly without parsing, concatenation, or normalization.
        
        Args:
            filename: Path to .nt file containing RDF triples
            collection_name: Name of the collection to insert into
            log: Whether to print log messages
            chunk_size: Number of triples to process in each batch
            
        Returns:
            MutationResult or List of MutationResults
        """
        data = []
        insert_results = []

        try:
            # Read lines and remove trailing period
            with open(filename, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Remove trailing period only
                        line = line.rstrip(' .')
                        data.append(line)
                        
            if log:            
                print(f"Successfully loaded {len(data)} raw triple lines from {filename}")
            
            if collection_name not in self._collections:
                raise ValueError(f"Collection '{collection_name}' does not exist.")
            
            collection = self.get_collection(collection_name, log=log)

            # handle empty file
            if len(data) == 0:
                if log:
                    print("No data to insert.")
                return []

            if collection is None:
                raise ValueError(f"Failed to retrieve collection '{collection_name}'.")

            # process in chunks (slicing handles any remainder)
            for start in range(0, len(data), chunk_size):
                end = min(start + chunk_size, len(data))
                chunk = data[start:end]  # Raw triple lines as-is

                if log:
                    print(f"Generating embeddings for chunk starting at {start} (count={len(chunk)})")

                # Embed each raw triple line directly (no normalization)
                embeddings = self._encode_text_batch(chunk, normalize=False)
                
                if log:
                    print(f"Generated embeddings shape: {embeddings.shape}")
                    print(f"  (raw embedding dimension: {self._embedding_dim})")

                entities = [
                    chunk,                      # text field (raw triple strings as-is)
                    embeddings.tolist(),        # embedding field (single embedding per triple)
                ]

                if log:
                    print(f"Inserting {len(chunk)} entities into collection...")

                try:
                    insert_result = collection.insert(entities)
                    insert_results.append(insert_result)
                    if log:
                        print(f"Inserted {len(chunk)} entities (chunk start={start})")
                except MilvusException as e:
                    # Log and continue with next chunk
                    print(f"Milvus error inserting chunk starting at {start}: {e}")
                except Exception as e:
                    print(f"Unexpected error inserting chunk starting at {start}: {e}")

            # flush to ensure data persisted (best-effort)
            try:
                collection.flush()
            except Exception:
                if log:
                    print("Warning: collection.flush() failed (continuing).")

            # return single MutationResult if only one chunk inserted, else list
            if len(insert_results) == 0:
                return []
            return insert_results[0] if len(insert_results) == 1 else insert_results
            
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            return []
        except IOError as e:
            print(f"Error reading file '{filename}': {e}")
            return []
        except MilvusException as e:
            print(f"Milvus error while processing '{filename}': {e}")
            return []
        except Exception as e:
            print(f"Unexpected error while loading file '{filename}': {e}")
            return []

    def create_index(self, collection_name: str, index_name: str, index_type: str = "HNSW", 
                    metric_type: str = "COSINE", nlist: int = 128, log: bool = False, 
                    M: int = 16, ef_construction: int = 200) -> bool:
        """
        Create an index on a specific field in a collection.
        
        Args:
            collection_name: Name of the collection
            index_name: Name of the field to create index on (usually "embedding")
            index_type: Type of index (IVF_FLAT, IVF_SQ8, HNSW, etc.)
            metric_type: Distance metric (COSINE, L2, IP)
            nlist: Number of clusters for IVF index (used for IVF_FLAT, IVF_SQ8)
            M: Number of bi-directional links for HNSW (default: 16, higher = more accurate but slower)
            ef_construction: Size of dynamic candidate list for HNSW (default: 200, higher = more accurate but slower)
            log: Whether to print log messages
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if collection_name not in self._collections:
                raise ValueError(f"Collection '{collection_name}' does not exist. Please create it first using add_collection().")
            
            collection = self.get_collection(collection_name,log=log)
            
            if log:
                print(f"Creating index on field '{index_name}' for collection '{collection_name}'...")
                print(f"Index type: {index_type}, Metric: {metric_type}")
            
            # Set index parameters based on index type
            if index_type == "HNSW":
                index_params = {
                    "metric_type": metric_type,
                    "index_type": index_type,
                    "params": {
                        "M": M,  # Number of bi-directional links
                        "ef_construction": ef_construction  # Size of dynamic candidate list
                    }
                }
                if log:
                    print(f"HNSW parameters: M={M}, ef_construction={ef_construction}")
            elif index_type in ["IVF_FLAT", "IVF_SQ8", "IVF_PQ"]:
                index_params = {
                    "metric_type": metric_type,
                    "index_type": index_type,
                    "params": {"nlist": nlist}
                }
                if log:
                    print(f"IVF parameters: nlist={nlist}")
            else:
                # Default parameters for other index types
                index_params = {
                    "metric_type": metric_type,
                    "index_type": index_type,
                    "params": {"nlist": nlist}
                }
                if log:
                    print(f"Using default parameters: nlist={nlist}")
            
            collection.create_index(index_name, index_params)
            
            if log:
                print(f"Index created successfully on '{index_name}' with type '{index_type}' and metric '{metric_type}'")
            
            return True
            
        except ValueError as e:
            print(f"Validation error: {e}")
            return False
        except MilvusException as e:
            print(f"Milvus error while creating index: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error while creating index: {e}")
            return False
        
    def load_data(self, collection_name : str, log : bool = False) -> bool:
        """
        Load a collection into memory for searching.
        
        Note: Indexes must be created on vector fields before loading.
        """
        try:
            collection = self.get_collection(collection_name, log=log)
            if collection is None:
                return False
            
            # Check if collection has any indexes (required for loading)
            try:
                indexes = collection.indexes
                if not indexes:
                    print(f"Warning: Collection '{collection_name}' has no indexes. "
                          f"Create indexes on vector fields before loading using create_index().")
            except Exception:
                pass  # If we can't check indexes, try loading anyway
                
            collection.load()
            if log:
                print("Collection loaded into memory")
            return True
            
        except MilvusException as e:
            error_msg = str(e)
            if "index not found" in error_msg.lower():
                print(f"Error: Collection '{collection_name}' cannot be loaded without indexes.")
                print(f"  Please create indexes on vector fields first using create_index().")
                print(f"  Example: vdb.create_index('{collection_name}', 'embedding', log=True)")
            else:
                print(f"Milvus error while loading collection: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error while loading collection: {e}")
            return False

    
    def search(self, collection_name: str, query_texts: Union[str, dict, List[Union[str, dict]]], 
              field_name: Optional[str] = None, limit: int = 5, 
              metric_type: str = "COSINE", nprobe: int = 10, 
              output_fields: Optional[List[str]] = None, log: bool = False) -> List[dict]:
        """
        Perform similarity search on a collection.
        
        Args:
            collection_name: Name of the collection to search
            query_texts: Single query string, dict, or list of query dicts with subject/predicate/object
            field_name: Name of the vector field to search (auto-determined if None)
            limit: Number of top results to return per query
            metric_type: Distance metric (COSINE, L2, IP)
            nprobe: Number of clusters to search (affects speed vs accuracy)
            output_fields: List of fields to return in results (default: ["text"])
            log: Whether to print log messages
            
        Returns:
            List[dict]: List of search results with metadata
        """
        try:
            if collection_name not in self._collections:
                raise ValueError(f"Collection '{collection_name}' does not exist.")
            
            collection = self.get_collection(collection_name)
            if collection is None:
                raise ValueError(f"Collection '{collection_name}' could not be retrieved.")
            
            # Try to load collection if not already loaded
            # This will fail gracefully if indexes don't exist
            try:
                collection.load()
            except MilvusException as e:
                error_msg = str(e)
                if "index not found" in error_msg.lower() or "not loaded" in error_msg.lower():
                    # Check if it's an index issue or just not loaded
                    if "index not found" in error_msg.lower():
                        raise ValueError(
                            f"Collection '{collection_name}' is not ready for search. "
                            f"Indexes must be created on vector fields first.\n"
                            f"  Use: vdb.create_index('{collection_name}', 'embedding', log=True)\n"
                            f"  Then: vdb.load_data('{collection_name}', log=True)"
                        )
                    # If it's just "not loaded", try to continue (might work for some operations)
                # If it's already loaded, that's fine - continue
                pass
            
            # Normalize query inputs to list of strings
            if isinstance(query_texts, str):
                query_strings = [query_texts]
            elif isinstance(query_texts, dict):
                # If dict, try to extract text field or format as triple
                query_strings = [query_texts.get('text', str(query_texts))]
            elif isinstance(query_texts, list):
                query_strings = []
                for item in query_texts:
                    if isinstance(item, str):
                        query_strings.append(item)
                    elif isinstance(item, dict):
                        query_strings.append(item.get('text', str(item)))
                    else:
                        query_strings.append(str(item))
            else:
                query_strings = [str(query_texts)]
            
            if len(query_strings) == 0:
                return []
            
            # Extract triple patterns from SPARQL queries if needed
            processed_queries = []
            for query in query_strings:
                # Check if it looks like a SPARQL query (contains SELECT, WHERE, etc.)
                if isinstance(query, str) and ('SELECT' in query.upper() or 'WHERE' in query.upper()):
                    # Try to extract triple pattern
                    triple = VectorDataBase._extract_triple_from_sparql(query)
                    if triple:
                        processed_queries.append(triple)
                        if log:
                            print(f"Extracted triple from SPARQL: '{triple}'")
                    else:
                        # If extraction fails, use original query
                        processed_queries.append(query)
                        if log:
                            print(f"Could not extract triple from SPARQL, using original query")
                else:
                    # Not a SPARQL query, use as-is
                    processed_queries.append(query)
            
            if log:
                print(f"Performing search on collection '{collection_name}'")
                print(f"Query texts: {len(processed_queries)} queries")
                print(f"Query to be searched is: {processed_queries}")
            
            # Generate embeddings for each processed query string (no normalization)
            query_embeddings = self._encode_text_batch(processed_queries, normalize=False)
            
            # Use single embedding field (default to "embedding" if not specified)
            if field_name is None:
                search_field = "embedding"
            else:
                search_field = field_name
            
            if log:
                print(f"Searching field: {search_field}")
                print(f"Query embedding shape: {query_embeddings.shape}")
                print(f"  (raw embedding dimension: {self._embedding_dim})")
            
            if output_fields is None:
                output_fields = ["text"]
            
            search_params = {
                "metric_type": metric_type,
                "params": {"nprobe": nprobe}
            }
            
            if log:
                print(f"Searching with limit={limit}, metric={metric_type}, nprobe={nprobe}")
            
            results = collection.search(
                query_embeddings.tolist(),
                search_field,
                search_params,
                limit=limit,
                output_fields=output_fields
            )
            
            formatted_results = []
            
            for query_idx, result in enumerate(results):
                query_results = {
                    "query_index": query_idx,
                    "query_text": query_strings[query_idx],
                    "matches": []
                }
                
                for hit in result:
                    match = {
                        "id": hit.id,
                        "distance": hit.distance,
                        "score": 1 - hit.distance if metric_type == "COSINE" else hit.distance,
                    }
                    
                    for field in output_fields:
                        match[field] = hit.entity.get(field)
                    
                    query_results["matches"].append(match)
                
                formatted_results.append(query_results)
                
                if log:
                    print(f"\nQuery {query_idx + 1}: '{query_strings[query_idx]}'")
                    for i, match in enumerate(query_results["matches"]):
                        print(f"  {i+1}. {match.get('text', 'N/A')} (Score: {match['score']:.4f})")
            
            return formatted_results
            
        except ValueError as e:
            print(f"Validation error: {e}")
            return []
        except MilvusException as e:
            print(f"Milvus error during search: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error during search: {e}")
            return []

    def clear_collection(self, collection_name: str, log: bool = False) -> bool:
        """
        Clear all data from a collection (delete all entities).
        
        Args:
            collection_name: Name of the collection to clear
            log: Whether to print log messages
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if collection exists
            if collection_name not in self._collections:
                raise ValueError(f"Collection '{collection_name}' does not exist.")
            
            # Get the collection object
            collection = Collection(collection_name)
            
            if log:
                print(f"Clearing collection '{collection_name}'...")
            
            # Get entity count without loading (doesn't require indexes)
            total_entities = collection.num_entities
            
            if total_entities == 0:
                if log:
                    print("Collection is already empty.")
                return True
            
            if log:
                print(f"Current entity count: {total_entities}")
                print(f"Deleting {total_entities} entities...")
            
            # Use a simple delete expression that matches all entities
            # For Int64 primary key 'id', this will match all entities
            # This approach doesn't require loading the collection or having indexes
            delete_expr = "id >= 0"  # Matches all entities with id >= 0 (all auto-generated IDs)
            
            collection.delete(delete_expr)
            collection.flush()
            
            if log:
                print(f"Successfully cleared collection '{collection_name}'")
                print(f"New entity count: {collection.num_entities}")
            
            return True
            
        except ValueError as e:
            print(f"Validation error: {e}")
            return False
        except MilvusException as e:
            print(f"Milvus error while clearing collection: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error while clearing collection: {e}")
            return False

    def drop_collection(self, collection_name: str, log: bool = False) -> bool:
        """
        Completely drop/delete a collection and remove it from tracking.
        
        Args:
            collection_name: Name of the collection to drop
            log: Whether to print log messages
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if collection exists
            if collection_name not in self._collections:
                if log:
                    print(f"Collection '{collection_name}' does not exist in tracked collections.")
                return True  # Already doesn't exist
            
            if log:
                print(f"Dropping collection '{collection_name}'...")
            
            # Drop the collection using utility function
            utility.drop_collection(collection_name)
            
            # Remove from our tracking set
            self._collections.discard(collection_name)
            
            if log:
                print(f"Successfully dropped collection '{collection_name}'")
            
            return True
            
        except MilvusException as e:
            print(f"Milvus error while dropping collection: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error while dropping collection: {e}")
            return False