from pymilvus import connections, utility, MilvusException, Collection, CollectionSchema, FieldSchema, DataType
from pymilvus.orm.mutation import MutationResult
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional, Sequence, Tuple
import numpy as np
import re

class VectorDataBase:
    
    def __init__(self, database_name : str, host : str, port : int,  embedding_model : str, target_embedding_dim: int):
        self._database_name : str = database_name
        self._host : str = host
        self._port : str = str(port)
        self._embedding_model : SentenceTransformer = SentenceTransformer(embedding_model)
        self._collections : set[str] = set()
        self._embedding_dim : int  = target_embedding_dim
        
        # Public properties for schema creation
        # Concatenated embedding: subject + predicate + object = 3 * embedding_dim
        self.embedding_dimension = target_embedding_dim * 3
    
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
    def _parse_triple_line(line: Optional[str]) -> Optional[dict]:
        """Parse an RDF triple line into structured components."""
        if not line:
            return None

        triple_pattern = r'<([^>]+)>\s+<([^>]+)>\s+(?:"([^"]+)"|<([^>]+)>)\s*\.?'
        match = re.match(triple_pattern, line.strip())
        if not match:
            return None

        subject = f"<{match.group(1)}>"
        predicate = f"<{match.group(2)}>"
        object_literal = match.group(3)
        object_uri = match.group(4)

        if object_literal is not None:
            object_value = object_literal
            object_type = "literal"
        else:
            object_value = f"<{object_uri}>"
            object_type = "uri"

        return {
            "subject": subject,
            "predicate": predicate,
            "object": object_value,
            "object_type": object_type
        }

    @staticmethod
    def _format_query_sentence(subject: Optional[str], predicate: Optional[str], obj: Optional[str], object_type: Optional[str] = None) -> str:
        """Create a descriptive triple string for logging/search context."""
        def normalize_component(value: Optional[str], placeholder: str) -> str:
            return value if value else placeholder

        subject_str = normalize_component(subject, "?s")
        predicate_str = normalize_component(predicate, "?p")
        object_str = normalize_component(obj, "?o")

        return f"{subject_str} {predicate_str} {object_str} ."
    
    def _normalize_triple_record(self, triple: Optional[dict], fallback_text: str) -> dict:
        """Ensure we always have subject/predicate/object fields for downstream processing."""
        if not triple:
            return {
                "subject": fallback_text,
                "predicate": None,
                "object": None,
                "object_type": None,
                "text": fallback_text,
            }
        normalized = {
            "subject": triple.get("subject"),
            "predicate": triple.get("predicate"),
            "object": triple.get("object"),
            "object_type": triple.get("object_type"),
            "text": fallback_text,
        }
        return normalized

    # def _adjust_component_dim(self, embeddings: np.ndarray) -> np.ndarray:
    #     """Truncate or zero-pad embeddings so every component matches self._component_dim."""
    #     if embeddings.shape[1] == self._component_dim:
    #         return embeddings
    #     if embeddings.shape[1] > self._component_dim:
    #         return embeddings[:, :self._component_dim]
    #     pad_width = self._component_dim - embeddings.shape[1]
    #     return np.pad(embeddings, ((0, 0), (0, pad_width)), mode='constant')

    def _encode_text_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a batch of strings and coerce to the configured component dimension."""
        if not texts:
            return np.zeros((0, self._embedding_dim), dtype=float)
        try:
            embeddings = self._embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        except TypeError:
            embeddings = self._embedding_model.encode(texts, show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype=float)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings

    def _encode_component_list(self, components: Sequence[Optional[str]]) -> np.ndarray:
        """Encode each component string independently, substituting zero vectors when missing."""
        total = len(components)
        if total == 0:
            return np.zeros((0, self._embedding_dim), dtype=float)
        result = np.zeros((total, self._embedding_dim), dtype=float)
        texts_to_encode: List[str] = []
        indices: List[int] = []
        for idx, component in enumerate(components):
            if component is None or component == "":
                continue
            texts_to_encode.append(component)
            indices.append(idx)
        if not texts_to_encode:
            return result
        encoded = self._encode_text_batch(texts_to_encode)
        for i, original_idx in enumerate(indices):
            result[original_idx, :] = encoded[i]
        return result

    def _embed_triple_batch(self, triples: Sequence[dict]) -> np.ndarray:
        """Embed triples by embedding subject, predicate, object separately and concatenating.
        
        Missing/variable components are embedded as zero vectors.
        
        Returns a numpy array of shape (batch_size, 3 * embedding_dim) where each row is:
        [embed(subject) | embed(predicate) | embed(object)]
        """
        if not triples:
            return np.zeros((0, self._embedding_dim * 3), dtype=float)
        
        batch_size = len(triples)
        result = np.zeros((batch_size, self._embedding_dim * 3), dtype=float)
        
        # Collect non-empty components for batch encoding
        subjects = []
        predicates = []
        objects = []
        subject_indices = []
        predicate_indices = []
        object_indices = []
        
        for i, triple in enumerate(triples):
            # Subject
            subj = triple.get("subject")
            if subj and subj != "":
                subjects.append(subj)
                subject_indices.append(i)
            
            # Predicate
            pred = triple.get("predicate")
            if pred and pred != "":
                predicates.append(pred)
                predicate_indices.append(i)
            
            # Object
            obj = triple.get("object")
            obj_type = triple.get("object_type")
            if obj and obj != "":
                if obj_type == "literal":
                    objects.append(f"literal:{obj}")
                else:
                    objects.append(obj)
                object_indices.append(i)
        
        # Embed non-empty components in batches
        if subjects:
            subject_embeddings = self._encode_text_batch(subjects)
            for idx, orig_idx in enumerate(subject_indices):
                result[orig_idx, :self._embedding_dim] = subject_embeddings[idx]
        
        if predicates:
            predicate_embeddings = self._encode_text_batch(predicates)
            for idx, orig_idx in enumerate(predicate_indices):
                result[orig_idx, self._embedding_dim:2*self._embedding_dim] = predicate_embeddings[idx]
        
        if objects:
            object_embeddings = self._encode_text_batch(objects)
            for idx, orig_idx in enumerate(object_indices):
                result[orig_idx, 2*self._embedding_dim:] = object_embeddings[idx]
        
        return result

    def _compose_triple_text(self, triple: dict) -> str:
        """Generate a human-readable triple string for logging/results."""
        subject = triple.get("subject") or "?s"
        predicate = triple.get("predicate") or "?p"
        obj = triple.get("object")
        if triple.get("object_type") == "literal":
            obj_repr = f"\"{obj}\"" if obj is not None else "\"\""
        else:
            obj_repr = obj or "?o"
        return f"{subject} {predicate} {obj_repr} ."




    

    # def insert_data_from_file(self, filename : str, collection_name : str, log : bool = False) -> Union[MutationResult, List]:
    #     data = []
        
    #     try:
    #         with open(filename, 'r', encoding='utf-8') as file:
    #             for line in file:
    #                 line = line.strip()
    #                 if line and not line.startswith('#'):
    #                     data.append(line)
    #         if log:            
    #             print(f"Successfully loaded {len(data)} lines from {filename}")
            
    #         if collection_name not in self._collections:
    #             raise ValueError(f"Collection '{collection_name}' does not exist.")
            
    #         collection = self.get_collection(collection_name,log=log)
            
    #         if log:
    #             print("Generating embeddings...")

    #         embeddings = self._embedding_model.encode(data)
            
    #         if log:
    #             print(f"Generated embeddings shape: {embeddings.shape}")

    #         entities = [
    #             data,  # text field
    #             embeddings.tolist()  # embedding field (convert to list)
    #         ]
            
    #         if log:
    #             print("Inserting data into collection...")

    #         insert_result = collection.insert(entities)

    #         if log:
    #             print(f"Inserted {len(data)} entities")
            
    #         return insert_result
            
    #     except FileNotFoundError:
    #         print(f"Error: File '{filename}' not found.")
    #         return []
    #     except IOError as e:
    #         print(f"Error reading file '{filename}': {e}")
    #         return []
    #     except MilvusException as e:
    #         print(f"Milvus error while processing '{filename}': {e}")
    #         return []
    #     except Exception as e:
    #         print(f"Unexpected error while loading file '{filename}': {e}")
    #         return []
        
    def insert_data_from_file(self, filename : str, collection_name : str, log : bool = False, chunk_size : int  = 100 ) -> Union[MutationResult, List]:
        data = []
        insert_results = []

        try:
            with open(filename, 'r', encoding='utf-8') as file:
                for line in file:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        data.append(line)
            if log:            
                print(f"Successfully loaded {len(data)} lines from {filename}")
            
            if collection_name not in self._collections:
                raise ValueError(f"Collection '{collection_name}' does not exist.")
            
            collection = self.get_collection(collection_name,log=log)

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
                chunk = data[start:end]

                normalized_chunk = [self._normalize_triple_record(self._parse_triple_line(raw_line), raw_line)
                                    for raw_line in chunk]
                chunk_texts = [record["text"] for record in normalized_chunk]

                if log:
                    print(f"Generating embeddings for chunk starting at {start} (count={len(chunk)})")

                embeddings = self._embed_triple_batch(normalized_chunk)
                
                if log:
                    print(f"Generated embeddings shape: {embeddings.shape}")
                    print(f"  (concatenated: subject[{self._embedding_dim}] + predicate[{self._embedding_dim}] + object[{self._embedding_dim}] = {self._embedding_dim * 3})")

                entities = [
                    chunk_texts,                    # text field (raw triple strings)
                    embeddings.tolist(),             # embedding field (concatenated)
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

    def create_index(self, collection_name: str, index_name: str, index_type: str = "IVF_FLAT", 
                    metric_type: str = "COSINE", nlist: int = 128, log: bool = False) -> bool:
        """
        Create an index on a specific field in a collection.
        
        Args:
            collection_name: Name of the collection
            field_name: Name of the field to create index on (usually "embedding")
            index_type: Type of index (IVF_FLAT, IVF_SQ8, HNSW, etc.)
            metric_type: Distance metric (COSINE, L2, IP)
            nlist: Number of clusters for IVF index
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
            
            index_params = {
                "metric_type": metric_type,
                "index_type": index_type,
                "params": {"nlist": nlist}
            }
            
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


    @staticmethod
    def _parse_sparql_query(query: str) -> Optional[dict]:
        """Parse a SPARQL SELECT query to extract the triple pattern.
        
        Handles queries like:
        - SELECT ?X WHERE { ?X <predicate> <object> }
        - SELECT ?X WHERE { <subject> <predicate> ?X }
        - SELECT ?X WHERE { <subject> <predicate> <object> }
        """
        if not query or not isinstance(query, str):
            return None
        
        # Extract the WHERE clause pattern
        where_match = re.search(r'WHERE\s*\{([^}]+)\}', query, re.IGNORECASE | re.DOTALL)
        if not where_match:
            return None
        
        pattern = where_match.group(1).strip()
        
        # Parse the triple pattern: subject predicate object
        # Match: ?var or <uri> or "literal"
        triple_match = re.match(r'(\S+)\s+(\S+)\s+(.+)', pattern)
        if not triple_match:
            return None
        
        subject_raw = triple_match.group(1).strip()
        predicate_raw = triple_match.group(2).strip()
        object_raw = triple_match.group(3).strip().rstrip('.')
        
        def normalize_term(term: str) -> Tuple[Optional[str], Optional[str]]:
            """Return (value, type) where type is 'variable', 'uri', or 'literal'"""
            term = term.strip()
            if term.startswith('?'):
                return (None, 'variable')
            elif term.startswith('<') and term.endswith('>'):
                inner = term[1:-1].strip()
                # If it's a very short identifier (likely a variable typo like <X>), treat as variable
                if len(inner) <= 3 and inner.isalnum() and not inner.startswith('http'):
                    return (None, 'variable')
                return (term, 'uri')
            elif term.startswith('"') and term.endswith('"'):
                return (term[1:-1], 'literal')
            else:
                # Try to infer - if it looks like a URI, treat as such
                if term.startswith('http://') or term.startswith('https://'):
                    return (f"<{term}>", 'uri')
                return (term, 'literal')
        
        subject_val, subject_type = normalize_term(subject_raw)
        predicate_val, predicate_type = normalize_term(predicate_raw)
        object_val, object_type = normalize_term(object_raw)
        
        return {
            "subject": subject_val,
            "predicate": predicate_val,
            "object":  object_val,
            "object_type": "literal" if object_type == 'literal' else 'uri',
        }

    def _prepare_query_triples(self, query_items: Union[str, dict, Sequence[Union[str, dict]]]) -> List[dict]:
        """Normalize incoming search queries into structured triples.
        
        Accepts:
        - SPARQL SELECT query strings (e.g., "SELECT ?X WHERE { ?X <p> <o> }")
        - Structured dicts with subject/predicate/object fields
        - Raw triple strings (e.g., "<s> <p> <o> .")
        """
        if query_items is None:
            return []
        
        # Normalize input to a list
        if isinstance(query_items, (str, dict)):
            items: Sequence[Union[str, dict]] = [query_items]
        else:
            items = query_items

        triples: List[dict] = []
        for item in items:
            # Handle dict input (already structured)
            if isinstance(item, dict):
                # If it already has the expected structure, use it
                if "subject" in item or "predicate" in item or "object" in item:
                    normalized = {
                        "subject": item.get("subject"),
                        "predicate": item.get("predicate"),
                        "object": item.get("object"),
                        "object_type": item.get("object_type"),
                        "text": item.get("text") or self._format_query_sentence(
                            item.get("subject"),
                            item.get("predicate"),
                            item.get("object"),
                            item.get("object_type")
                        ),
                    }
                    triples.append(normalized)
                    continue
            
            # Handle string input - try SPARQL first, then raw triple
            if isinstance(item, str):
                # Try parsing as SPARQL query first
                sparql_parsed = self._parse_sparql_query(item)
                if sparql_parsed:
                    normalized = {
                        "subject": sparql_parsed.get("subject"),
                        "predicate": sparql_parsed.get("predicate"),
                        "object": sparql_parsed.get("object"),
                        "object_type": sparql_parsed.get("object_type"),
                        "text": item,  # Keep original SPARQL query as text
                    }
                    triples.append(normalized)
                    continue
                
                # Try parsing as raw triple string
                triple_parsed = self._parse_triple_line(item)
                if triple_parsed:
                    normalized = {
                        "subject": triple_parsed.get("subject"),
                        "predicate": triple_parsed.get("predicate"),
                        "object": triple_parsed.get("object"),
                        "object_type": triple_parsed.get("object_type"),
                        "text": item,  # Keep original triple string as text
                    }
                    triples.append(normalized)
                    continue
                
                # If neither parsing worked, create a fallback with the raw string
                normalized = self._normalize_triple_record(None, item)
                triples.append(normalized)
        
        return triples
    
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
            
            triple_queries = self._prepare_query_triples(query_texts)
            if len(triple_queries) == 0:
                return []
            
            if log:
                print(f"Performing search on collection '{collection_name}'")
                print(f"Query texts: {len(triple_queries)} queries")
                print(f"Query to be searched is: {[q['text'] for q in triple_queries]}")
            
            # Generate concatenated embeddings for each query
            query_embeddings = self._embed_triple_batch(triple_queries)
            
            # Use single embedding field (default to "embedding" if not specified)
            if field_name is None:
                search_field = "embedding"
            else:
                search_field = field_name
            
            if log:
                print(f"Searching field: {search_field}")
                print(f"Query embedding shape: {query_embeddings.shape}")
                print(f"  (concatenated: subject[{self._embedding_dim}] + predicate[{self._embedding_dim}] + object[{self._embedding_dim}] = {self._embedding_dim * 3})")
            
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
                    "query_text": triple_queries[query_idx]["text"],
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
                    print(f"\nQuery {query_idx + 1}: '{triple_queries[query_idx]['text']}'")
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