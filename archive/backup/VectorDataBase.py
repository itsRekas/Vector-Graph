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
        # Get the actual embedding dimension from the model
        # For v3: we embed S, P, O separately and concatenate, so dimension is 3x model dim
        # V3: Embeddings are normalized during insert
        test_embedding = self._embedding_model.encode(["test"], convert_to_numpy=True)
        self._model_embedding_dim = test_embedding.shape[1] if test_embedding.ndim > 1 else len(test_embedding)
        self._embedding_dim = 3 * self._model_embedding_dim  # S + P + O concatenated
        
        # Public properties for schema creation
        # V3: Embedding is concatenation of subject, predicate, object embeddings (3x model dim), normalized
        self.embedding_dimension = self._embedding_dim
    
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
    def _parse_triple(triple_text: str) -> Optional[Tuple[str, str, str]]:
        """
        Parse an RDF triple text into subject, predicate, object components.
        Handles format: <subject> <predicate> <object> or <subject> <predicate> "literal"
        
        Returns:
            Tuple of (subject, predicate, object) or None if parsing fails
        """
        # Pattern to match: <subject> <predicate> "object" or <subject> <predicate> <object> (no period)
        pattern = r'<([^>]+)>\s+<([^>]+)>\s+(?:"([^"]+)"|<([^>]+)>)'
        match = re.match(pattern, triple_text.strip())
        
        if match:
            subject = f"<{match.group(1)}>"
            predicate = f"<{match.group(2)}>"
            object_literal = match.group(3)  # For "literal" values
            object_uri = match.group(4)     # For <URI> values
            obj = f'"{object_literal}"' if object_literal else f"<{object_uri}>"
            return (subject, predicate, obj)
        return None
    
    @staticmethod
    def _format_query_sentence(subject: Optional[str], predicate: Optional[str], obj: Optional[str], object_type: Optional[str] = None) -> str:
        """Create a triple string by concatenating parts as-is (no period)."""
        return f"{subject} {predicate} {obj}"
    
    @staticmethod
    def _extract_triple_from_sparql(sparql_query: str) -> Optional[str]:
        """
        Extract triple pattern from SPARQL query string.
        Handles variables (?var) and URIs/literals in any position (subject, predicate, object).
        Returns the triple in format: <subject> <predicate> <object> (no period)
        For variables, returns "?VAR" format.
        Returns None if no valid triple pattern found.
        """
        # Pattern to match triple in SPARQL with variables in any position:
        # ?var <p> <o>, <s> ?var <o>, <s> <p> ?var, <s> <p> <o>, <s> <p> "literal", etc.
        # This regex matches: (?var|<uri>) (?var|<uri>) (?var|<uri>|"literal")
        # Group 1: subject variable name (if variable)
        # Group 2: subject URI (if URI)
        # Group 3: predicate variable name (if variable)
        # Group 4: predicate URI (if URI)
        # Group 5: object variable name (if variable)
        # Group 6: object URI (if URI)
        # Group 7: object literal (if literal)
        pattern = r'(?:\?(\w+)|<([^>]+)>)\s+(?:\?(\w+)|<([^>]+)>)\s+(?:\?(\w+)|<([^>]+)>|"([^"]+)")'
        
        # Try to find triple pattern in the query
        match = re.search(pattern, sparql_query)
        if match:
            # Subject: variable or URI
            if match.group(1):  # Variable like ?X
                subject = f"?{match.group(1)}"  # Keep as ?VAR for parsing
            else:  # URI
                subject = f"<{match.group(2)}>"
            
            # Predicate: variable or URI
            if match.group(3):  # Variable like ?P
                predicate = f"?{match.group(3)}"
            else:  # URI
                predicate = f"<{match.group(4)}>"
            
            # Object: variable, URI, or literal
            if match.group(5):  # Variable like ?X
                obj = f"?{match.group(5)}"
            elif match.group(6):  # URI
                obj = f"<{match.group(6)}>"
            else:  # Literal
                obj = f'"{match.group(7)}"'
            
            # Return in format: <s> <p> <o> or ?VAR <p> <o> or <s> ?VAR <o> or <s> <p> ?VAR
            return f"{subject} {predicate} {obj}"
        
        # Fallback: try pattern without variables (all URIs)
        # Look for: <uri> <uri> <uri> or <uri> <uri> "literal"
        simple_pattern = r'(<[^>]+>)\s+(<[^>]+>)\s+(?:<([^>]+)>|"([^"]+)")'
        match = re.search(simple_pattern, sparql_query)
        if match:
            subject = match.group(1)
            predicate = match.group(2)
            if match.group(3):  # URI
                obj = f"<{match.group(3)}>"
            else:  # Literal
                obj = f'"{match.group(4)}"'
            return f"{subject} {predicate} {obj}"
        
        return None

    def _encode_text_batch(self, texts: Sequence[str], normalize: bool = False) -> np.ndarray:
        """Encode a batch of strings to embeddings.
        
        V3: Normalize embeddings when normalize=True (for insert operations).
        For search, embeddings are normalized separately in _embed_query_pattern.
        
        Args:
            texts: List of strings to encode
            normalize: If True, normalize embeddings to unit length (for insert)
        """
        if not texts:
            return np.zeros((0, self._model_embedding_dim), dtype=float)
        
        # Encode all texts directly
        # V3: Normalize embeddings if requested (for insert operations)
        try:
            embeddings = self._embedding_model.encode(
                texts, 
                convert_to_numpy=True, 
                show_progress_bar=False,
                batch_size=32,  # Batch for better GPU utilization
                normalize_embeddings=normalize  # V3: Normalize if requested
            )
        except (TypeError, ValueError) as e:
            # Fallback: model doesn't support normalize_embeddings parameter
            embeddings = self._embedding_model.encode(
                texts, 
                convert_to_numpy=True,
                show_progress_bar=False
            )
            # Manually normalize if requested
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                # Avoid division by zero
                norms = np.where(norms == 0, 1, norms)
                embeddings = embeddings / norms
        
        embeddings = np.asarray(embeddings, dtype=float)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        
        # V3: If normalize=False, ensure embeddings are not normalized
        # (for backward compatibility and search operations that need unnormalized embeddings)
        if not normalize:
            norms = np.linalg.norm(embeddings, axis=1)
            if np.allclose(norms, 1.0, atol=0.01):
                # Embeddings are normalized but we don't want them normalized
                # Scale them to match typical non-normalized embedding magnitudes
                scale_factor = 3.7  # Approximate norm of non-normalized embeddings
                embeddings = embeddings * scale_factor
        
        return embeddings
    
    def _embed_triple_parts(self, subject: str, predicate: str, obj: str) -> np.ndarray:
        """
        Embed subject, predicate, and object separately, normalize each part, then concatenate.
        V3 approach: [subject_emb | predicate_emb | object_emb] where each part is normalized separately.
        
        Args:
            subject: Subject string (e.g., "<http://...>")
            predicate: Predicate string (e.g., "<http://...>")
            obj: Object string (e.g., "<http://...>" or '"literal"')
            
        Returns:
            Concatenated embedding vector of shape (3 * model_dim,) where each part is normalized
        """
        # Embed each part separately (unnormalized)
        parts = [subject, predicate, obj]
        embeddings = self._encode_text_batch(parts, normalize=False)
        
        # V3: Normalize each part separately before concatenating
        for i in range(len(embeddings)):
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] = embeddings[i] / norm
        
        # Concatenate: [S | P | O] where each part is normalized
        concatenated = np.concatenate(embeddings, axis=0)
        
        return concatenated
    
    def _embed_query_pattern(self, subject: Optional[str], predicate: Optional[str], obj: Optional[str]) -> np.ndarray:
        """
        Embed a query pattern where missing parts (variables) are replaced with unnormalized zero vectors.
        V3 approach: [subject_emb | predicate_emb | object_emb] with normalized present parts and unnormalized zero vectors for missing parts.
        
        Args:
            subject: Subject string or None if variable
            predicate: Predicate string or None if variable
            obj: Object string or None if variable
            
        Returns:
            Concatenated embedding vector of shape (3 * model_dim,) with normalized present parts and unnormalized zero vectors for missing parts
        """
        # Create unnormalized zero vector for missing parts (shape: model_dim,)
        # V3: Zero vectors remain unnormalized (as per requirement)
        zero_vector = np.zeros(self._model_embedding_dim, dtype=float)
        
        # Track which parts need embedding and their order
        parts_to_embed = []
        parts_list = []  # Order: [subject, predicate, object]
        
        # Subject
        if subject:
            parts_to_embed.append(subject)
            parts_list.append('subject')
        else:
            parts_list.append(None)
        
        # Predicate
        if predicate:
            parts_to_embed.append(predicate)
            parts_list.append('predicate')
        else:
            parts_list.append(None)
        
        # Object
        if obj:
            parts_to_embed.append(obj)
            parts_list.append('object')
        else:
            parts_list.append(None)
        
        # Embed non-None parts (unnormalized first)
        if parts_to_embed:
            embeddings = self._encode_text_batch(parts_to_embed, normalize=False)
            # Ensure 2D: (n_parts, model_dim)
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            
            # V3: Normalize each present part individually (matching insert behavior)
            for i in range(len(embeddings)):
                norm = np.linalg.norm(embeddings[i])
                if norm > 0:
                    embeddings[i] = embeddings[i] / norm
        else:
            # All parts are None - will use all zero vectors
            embeddings = np.zeros((0, self._model_embedding_dim), dtype=float)
        
        # Build concatenated embedding: [S_emb | P_emb | O_emb]
        # Present parts are normalized, missing parts are unnormalized zero vectors
        result_parts = []
        embed_idx = 0  # Index into embeddings array
        
        for part in parts_list:
            if part is None:
                # Missing part: use unnormalized zero vector
                result_parts.append(zero_vector)
            else:
                # Present part: use its normalized embedding
                if embed_idx < len(embeddings):
                    # Get the normalized embedding for this part (shape: model_dim,)
                    part_embedding = embeddings[embed_idx]
                    result_parts.append(part_embedding)
                else:
                    # Fallback: use zero vector if indexing fails
                    result_parts.append(zero_vector)
                embed_idx += 1
        
        # Concatenate: [S | P | O] -> shape (3 * model_dim,)
        concatenated = np.concatenate(result_parts, axis=0)
        
        # Verify final shape
        expected_shape = (3 * self._model_embedding_dim,)
        if concatenated.shape != expected_shape:
            raise ValueError(
                f"Embedding shape mismatch: got {concatenated.shape}, expected {expected_shape}. "
                f"Model dim: {self._model_embedding_dim}"
            )
        
        return concatenated


        
    def insert_data_from_file(self, filename : str, collection_name : str, log : bool = False, chunk_size : int  = 100 ) -> Union[MutationResult, List]:
        """
        Insert triples from .nt file using V3 approach: parse into S, P, O, embed each separately, concatenate, and normalize.
        
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
                print(f"Successfully loaded {len(data)} triple lines from {filename}")
            
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
                chunk = data[start:end]

                if log:
                    print(f"Generating embeddings for chunk starting at {start} (count={len(chunk)})")

                # V3: Parse each triple into S, P, O, embed separately, normalize each part, then concatenate
                embeddings_list = []
                valid_triples = []
                
                for triple_text in chunk:
                    parsed = self._parse_triple(triple_text)
                    if parsed:
                        subject, predicate, obj = parsed
                        # Embed S, P, O separately, normalize each part, then concatenate (done in _embed_triple_parts)
                        embedding = self._embed_triple_parts(subject, predicate, obj)
                        embeddings_list.append(embedding.tolist())
                        valid_triples.append(triple_text)
                    else:
                        if log:
                            print(f"Warning: Could not parse triple: {triple_text[:80]}...")
                
                if log:
                    print(f"Generated embeddings for {len(embeddings_list)} valid triples")
                    print(f"  Embedding dimension: {self._embedding_dim} (3x model dim: {self._model_embedding_dim})")
                    print(f"  V3: Each part (S, P, O) is normalized separately before concatenation")

                if len(embeddings_list) == 0:
                    if log:
                        print(f"No valid triples in chunk starting at {start}, skipping...")
                    continue

                entities = [
                    valid_triples,              # text field (raw triple strings)
                    embeddings_list,            # embedding field (concatenated S|P|O embeddings)
                ]

                if log:
                    print(f"Inserting {len(valid_triples)} entities into collection...")

                try:
                    insert_result = collection.insert(entities)
                    insert_results.append(insert_result)
                    if log:
                        print(f"Inserted {len(valid_triples)} entities (chunk start={start})")
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
            
            # V2: Process queries directly to extract S, P, O patterns
            query_patterns = []
            
            # Normalize query inputs to list format
            if isinstance(query_texts, str):
                # Single string query - will be parsed below
                query_list = [query_texts]
            elif isinstance(query_texts, dict):
                # Single dict query - check if it has S/P/O structure
                if 'subject' in query_texts or 'predicate' in query_texts or 'object' in query_texts:
                    # V2 format: dict with subject/predicate/object
                    pattern = (
                        query_texts.get('subject'),
                        query_texts.get('predicate'),
                        query_texts.get('object')
                    )
                    if log:
                        print(f"DEBUG: Dict query pattern - S={pattern[0]}, P={pattern[1]}, O={pattern[2]}")
                        if pattern[1]:
                            print(f"DEBUG: Dict predicate repr: {repr(pattern[1])}")
                        if pattern[2]:
                            print(f"DEBUG: Dict object repr: {repr(pattern[2])}")
                    query_patterns.append(pattern)
                    query_list = []  # Already processed
                else:
                    # Legacy format: try to extract text field
                    query_list = [query_texts.get('text', str(query_texts))]
            elif isinstance(query_texts, list):
                query_list = []
                for item in query_texts:
                    if isinstance(item, dict):
                        # Check if it's V2 format with subject/predicate/object
                        if 'subject' in item or 'predicate' in item or 'object' in item:
                            # V2 format: process directly
                            query_patterns.append((
                                item.get('subject'),
                                item.get('predicate'),
                                item.get('object')
                            ))
                        else:
                            # Legacy format: extract text field
                            query_list.append(item.get('text', str(item)))
                    elif isinstance(item, str):
                        query_list.append(item)
                    else:
                        query_list.append(str(item))
            else:
                query_list = [str(query_texts)]
            
            # Process string queries to extract S, P, O patterns
            for query in query_list:
                # Check if it looks like a SPARQL query
                if isinstance(query, str) and ('SELECT' in query.upper() or 'WHERE' in query.upper()):
                    # Try to extract triple pattern
                    triple = VectorDataBase._extract_triple_from_sparql(query)
                    if triple:
                        # Split triple into parts (subject, predicate, object)
                        parts = triple.split(None, 2)  # Split on whitespace, max 2 splits
                        if len(parts) == 3:
                            subj_str, pred_str, obj_str = parts
                            
                            # Check each part for variables and convert to None if variable
                            # Subject: if starts with ?, it's a variable
                            subject = None if subj_str.startswith("?") else subj_str
                            
                            # Predicate: if starts with ?, it's a variable
                            predicate = None if pred_str.startswith("?") else pred_str
                            
                            # Object: if starts with ?, it's a variable
                            obj = None if obj_str.startswith("?") else obj_str
                            
                            if log:
                                print(f"DEBUG: SPARQL triple split - parts={parts}, creating pattern: ({subject}, {predicate}, {obj})")
                            
                            query_patterns.append((subject, predicate, obj))
                        else:
                            # Fallback: try parsing (might work for some edge cases)
                            parsed = self._parse_triple(triple)
                            if parsed:
                                query_patterns.append(parsed)
                            else:
                                query_patterns.append((None, None, None))
                    else:
                        # If extraction fails, try to parse as-is
                        parsed = self._parse_triple(query)
                        if parsed:
                            query_patterns.append(parsed)
                        else:
                            # Fallback: treat entire query as subject
                            query_patterns.append((query, None, None))
                else:
                    # Try to parse as triple string
                    parsed = self._parse_triple(query)
                    if parsed:
                        query_patterns.append(parsed)
                    else:
                        # Fallback: treat entire query as subject
                        query_patterns.append((query, None, None))
            
            if len(query_patterns) == 0:
                return []
            
            if log:
                print(f"Performing search on collection '{collection_name}'")
                print(f"Query patterns: {len(query_patterns)} queries")
                for i, (s, p, o) in enumerate(query_patterns):
                    print(f"  Query {i+1}: S={s}, P={p}, O={o}")
            
            # V2: Generate embeddings using _embed_query_pattern (handles variables with zero vectors)
            query_embeddings_list = []
            for subject, predicate, obj in query_patterns:
                if log:
                    print(f"DEBUG: Embedding query pattern - S={subject}, P={predicate}, O={obj}")
                    print(f"DEBUG: Subject type={type(subject)}, Predicate type={type(predicate)}, Object type={type(obj)}")
                    if predicate:
                        print(f"DEBUG: Predicate repr: {repr(predicate)}")
                    if obj:
                        print(f"DEBUG: Object repr: {repr(obj)}")
                embedding = self._embed_query_pattern(subject, predicate, obj)
                if log:
                    # Check the embedding stats
                    zero_part = embedding[:self._model_embedding_dim]  # Subject part
                    pred_part = embedding[self._model_embedding_dim:2*self._model_embedding_dim]  # Predicate part
                    obj_part = embedding[2*self._model_embedding_dim:]  # Object part
                    print(f"DEBUG: Embedding stats - Subject part (should be zeros): max={np.max(np.abs(zero_part)):.6f}, norm={np.linalg.norm(zero_part):.6f}")
                    print(f"DEBUG: Embedding stats - Predicate part: max={np.max(np.abs(pred_part)):.6f}, norm={np.linalg.norm(pred_part):.6f}")
                    print(f"DEBUG: Embedding stats - Object part: max={np.max(np.abs(obj_part)):.6f}, norm={np.linalg.norm(obj_part):.6f}")
                query_embeddings_list.append(embedding)
            
            # Convert to numpy array
            query_embeddings = np.array(query_embeddings_list)
            
            # Use single embedding field (default to "embedding" if not specified)
            if field_name is None:
                search_field = "embedding"
            else:
                search_field = field_name
            
            if log:
                print(f"Searching field: {search_field}")
                print(f"Query embedding shape: {query_embeddings.shape}")
                print(f"  (V2 embedding dimension: {self._embedding_dim} = 3x model dim {self._model_embedding_dim})")
            
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
                query_pattern = query_patterns[query_idx] if query_idx < len(query_patterns) else None
                query_results = {
                    "query_index": query_idx,
                    "query_text": str(query_pattern),  # Show the S, P, O pattern
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
                    s, p, o = query_pattern if query_pattern else (None, None, None)
                    print(f"\nQuery {query_idx + 1}: S={s}, P={p}, O={o}")
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