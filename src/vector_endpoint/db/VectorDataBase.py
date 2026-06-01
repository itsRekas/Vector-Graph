from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    MilvusException,
    connections,
    utility,
)
from pymilvus.orm.mutation import MutationResult
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional, Sequence, Tuple
import numpy as np
import re
import torch
from functools import lru_cache
import hashlib

class VectorDataBase:
    
    def __init__(
        self,
        database_name : str,
        host : str,
        port : int,
        embedding_model : str,
        target_embedding_dim: int,
        dim_adjustment: str = "truncate",
    ):
        self._database_name : str = database_name
        self._host : str = host
        self._port : str = str(port)
        
        # Enable GPU if available for faster embedding generation
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Initializing embedding model on device: {device}")
        self._embedding_model : SentenceTransformer = SentenceTransformer(embedding_model, device=device)
        
        self._collections : set[str] = set()
        self._embedding_dim : int  = target_embedding_dim
        self._dim_adjustment: str = dim_adjustment.strip().lower()
        self._model_output_dim: int = int(self._embedding_model.get_sentence_embedding_dimension())
        self._validate_dim_configuration()
        
        # Public properties for schema creation
        # Concatenated embedding: subject + predicate + object = 3 * embedding_dim
        self.embedding_dimension = target_embedding_dim * 3
        
        # Embedding cache for common patterns
        self._embedding_cache = {}
        self._cache_max_size = 1000  # Cache up to 1000 embeddings

    def _validate_dim_configuration(self) -> None:
        if self._embedding_dim <= 0:
            raise ValueError(f"target_embedding_dim must be > 0, got {self._embedding_dim}")

        if self._dim_adjustment != "truncate":
            raise ValueError(
                f"Unsupported dim_adjustment='{self._dim_adjustment}'. "
                "Only 'truncate' is currently supported."
            )

        if self._embedding_dim > self._model_output_dim:
            raise ValueError(
                f"target_embedding_dim={self._embedding_dim} exceeds model output dim "
                f"{self._model_output_dim} for truncate mode. "
                f"Use <= {self._model_output_dim} or switch to a model with larger native dim."
            )
    
    def clear_cache(self):
        """Clear the embedding cache. Useful for debugging or ensuring fresh embeddings."""
        cache_size = len(self._embedding_cache)
        self._embedding_cache.clear()
        print(f"Cleared embedding cache ({cache_size} entries)")
        return cache_size
    
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

    @staticmethod
    def milvus_storage_stamp(
        collection_name: str,
        *,
        host: str = "localhost",
        port: int = 19530,
        default_embedding_dim: int = 24,
    ) -> str:
        """Report Milvus collection storage via the Milvus API.

        """
        uri = f"http://{host}:{port}"
        try:
            client = MilvusClient(uri=uri)
        except Exception as exc:
            return f"collection={collection_name} error=connect_failed detail={exc}"

        try:
            existing = set(client.list_collections())
        except Exception as exc:
            return f"collection={collection_name} error=list_collections detail={exc}"

        if collection_name not in existing:
            return f"collection={collection_name} error=not_found known={sorted(existing)}"

        try:
            connections.connect(alias="default", host=host, port=str(port))
        except Exception:
            pass

        try:
            collection = Collection(collection_name)
            collection.flush()
            num_entities = int(collection.num_entities)
        except MilvusException as exc:
            return f"collection={collection_name} error=flush_or_count detail={exc}"
        except Exception as exc:
            return f"collection={collection_name} error=flush_or_count detail={exc}"

        stats = client.get_collection_stats(collection_name) or {}
        described = client.describe_collection(collection_name) or {}
        row_count = int(stats.get("row_count", num_entities))

        embedding_dim = default_embedding_dim
        text_max_len = 1000
        for field in described.get("fields", []):
            name = field.get("name")
            params = field.get("params", {}) or {}
            if name == "embedding":
                embedding_dim = int(params.get("dim", embedding_dim))
            elif name == "text":
                text_max_len = int(params.get("max_length", text_max_len))

        embedding_bytes = embedding_dim * 4
        id_bytes = 8
        per_entity_schema_upper_bound = embedding_bytes + id_bytes + text_max_len

        index_type = "unknown"
        index_params: dict = {}
        indexed_rows = row_count
        try:
            idx = client.describe_index(collection_name, "embedding")
            index_type = str(idx.get("index_type", "unknown"))
            index_params = dict(idx.get("params", {}) or {})
            indexed_rows = int(idx.get("indexed_rows", row_count))
        except Exception:
            pass

        segment_rows = 0
        segment_mem_size = 0
        try:
            for seg in utility.get_query_segment_info(collection_name):
                segment_rows += int(getattr(seg, "num_rows", 0) or 0)
                segment_mem_size += int(getattr(seg, "mem_size", 0) or 0)
        except Exception:
            pass

        return (
            f"collection={collection_name} "
            f"entities={num_entities} row_count={row_count} indexed_rows={indexed_rows} "
            f"embedding_dim={embedding_dim} index={index_type} index_params={index_params} "
            f"bytes_per_entity_embedding={embedding_bytes} "
            f"bytes_per_entity_schema_upper_bound={per_entity_schema_upper_bound} "
            f"total_embedding_bytes_est={num_entities * embedding_bytes} "
            f"segment_rows={segment_rows} segment_mem_size_reported={segment_mem_size} "
            f"milvus_stats={stats}"
        )

    def get_storage_stamp(self, collection_name: str) -> str:
        """Instance wrapper using this object's Milvus host/port and embedding dim."""
        return self.milvus_storage_stamp(
            collection_name,
            host=self._host,
            port=int(self._port),
            default_embedding_dim=self.embedding_dimension,
        )

    def _adjust_component_dim(self, embeddings: np.ndarray) -> np.ndarray:
        """Coerce model embeddings to configured component dim using truncation."""
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape={embeddings.shape}")

        current_dim = embeddings.shape[1]
        if current_dim == self._embedding_dim:
            return embeddings

        if self._dim_adjustment == "truncate":
            if current_dim < self._embedding_dim:
                raise ValueError(
                    f"Cannot truncate from dim {current_dim} to larger dim {self._embedding_dim}. "
                    "Use a smaller target_embedding_dim or a higher-dimension model."
                )
            return embeddings[:, : self._embedding_dim]

        raise ValueError(f"Unsupported dim_adjustment='{self._dim_adjustment}'")

    def _encode_text_batch(self, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
        """Encode a batch of strings and coerce to the configured component dimension.
        
        Uses caching for frequently used patterns to avoid recomputation.
        
        Args:
            texts: List of strings to encode
            normalize: If True, L2-normalize embeddings to unit vectors (for cosine similarity)
        """
        if not texts:
            return np.zeros((0, self._embedding_dim), dtype=float)
        
        # Check cache for each text
        cache_key_prefix = "norm_" if normalize else "raw_"
        texts_to_encode = []
        text_indices = []
        cached_embeddings = []
        
        for i, text in enumerate(texts):
            cache_key = cache_key_prefix + hashlib.md5(text.encode('utf-8')).hexdigest()
            if cache_key in self._embedding_cache:
                cached_embeddings.append((i, self._embedding_cache[cache_key]))
            else:
                texts_to_encode.append(text)
                text_indices.append(i)
        
        # Encode texts not in cache
        if texts_to_encode:
            try:
                new_embeddings = self._embedding_model.encode(
                    texts_to_encode, 
                    convert_to_numpy=True, 
                    show_progress_bar=False,
                    batch_size=32  # Batch for better GPU utilization
                )
            except TypeError:
                new_embeddings = self._embedding_model.encode(
                    texts_to_encode, 
                    show_progress_bar=False
                )
            new_embeddings = np.asarray(new_embeddings, dtype=float)
            if new_embeddings.ndim == 1:
                new_embeddings = new_embeddings.reshape(1, -1)
            new_embeddings = self._adjust_component_dim(new_embeddings)
            
            # L2 normalize for cosine similarity (when storing data)
            if normalize:
                norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
                new_embeddings = new_embeddings / norms
            
            # Store in cache (with size limit)
            for idx, text in enumerate(texts_to_encode):
                cache_key = cache_key_prefix + hashlib.md5(text.encode('utf-8')).hexdigest()
                if len(self._embedding_cache) < self._cache_max_size:
                    self._embedding_cache[cache_key] = new_embeddings[idx].copy()
                else:
                    # Simple eviction: remove oldest (first) entry
                    if self._embedding_cache:
                        first_key = next(iter(self._embedding_cache))
                        del self._embedding_cache[first_key]
                    self._embedding_cache[cache_key] = new_embeddings[idx].copy()
        
        # Combine cached and new embeddings in correct order
        if not cached_embeddings and not texts_to_encode:
            return np.zeros((0, self._embedding_dim), dtype=float)
        
        total_count = len(texts)
        result = np.zeros((total_count, self._embedding_dim), dtype=float)
        
        # Place cached embeddings
        for i, emb in cached_embeddings:
            result[i] = emb
        
        # Place new embeddings
        for idx, orig_idx in enumerate(text_indices):
            result[orig_idx] = new_embeddings[idx]
        
        return result

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

    def _embed_triple_batch(self, triples: Sequence[dict], normalize: bool = True) -> np.ndarray:
        """Embed triples by embedding subject, predicate, object separately and concatenating.
        
        Missing/variable components are embedded as zero vectors.
        
        Args:
            triples: List of triple dicts with subject/predicate/object
            normalize: If True, normalize embeddings (for stored data). 
                      If False, keep unnormalized (for query search).
        
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
        
        # Embed non-empty components in batches (unnormalized for queries)
        if subjects:
            subject_embeddings = self._encode_text_batch(subjects, normalize=normalize)
            for idx, orig_idx in enumerate(subject_indices):
                result[orig_idx, :self._embedding_dim] = subject_embeddings[idx]
        
        if predicates:
            predicate_embeddings = self._encode_text_batch(predicates, normalize=normalize)
            for idx, orig_idx in enumerate(predicate_indices):
                result[orig_idx, self._embedding_dim:2*self._embedding_dim] = predicate_embeddings[idx]
        
        if objects:
            object_embeddings = self._encode_text_batch(objects, normalize=normalize)
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

                # Normalize embeddings when storing data (for optimal cosine similarity)
                embeddings = self._embed_triple_batch(normalized_chunk, normalize=True)
                
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
                # Debug: Show parsed triple components
                for i, q in enumerate(triple_queries):
                    print(f"  Query {i+1} parsed as: S={q.get('subject')}, P={q.get('predicate')}, O={q.get('object')}, OType={q.get('object_type')}")
            
            # Normalize query component embeddings so query/search vectors are
            # on the same scale as stored vectors (which are normalized at insert time).
            query_embeddings = self._embed_triple_batch(triple_queries, normalize=True)
            
            # Use single embedding field (default to "embedding" if not specified)
            if field_name is None:
                search_field = "embedding"
            else:
                search_field = field_name
            
            if log:
                print(f"Searching field: {search_field}")
                print(f"Query embedding shape: {query_embeddings.shape}")
                print(f"  (concatenated: subject[{self._embedding_dim}] + predicate[{self._embedding_dim}] + object[{self._embedding_dim}] = {self._embedding_dim * 3})")
                # Debug: Check if subject is zero vector (variable)
                for i, q in enumerate(triple_queries):
                    if q.get('subject') is None:
                        subject_embedding = query_embeddings[i, :self._embedding_dim]
                        is_zero = np.allclose(subject_embedding, 0)
                        print(f"  Query {i+1}: Subject is variable (zero vector: {is_zero}), "
                              f"predicate={q.get('predicate')}, object={q.get('object')}")
            
            if output_fields is None:
                output_fields = ["text"]
            
            # Determine search parameters based on index type
            # Try to get index info to determine if it's HNSW or IVF
            try:
                indexes = collection.indexes
                index_type = None
                if indexes:
                    # Get the index for the search field
                    for idx in indexes:
                        if idx.field_name == search_field:
                            index_type = idx.params.get('index_type', '').upper()
                            break
                
                # For HNSW index, use ef parameter (ef_search)
                # For IVF index, use nprobe parameter
                if index_type == 'HNSW':
                    # ef should be >= limit for HNSW, use 2x limit or 50, whichever is larger
                    ef_value = max(limit * 2, 50)
                    search_params = {
                        "metric_type": metric_type,
                        "params": {"ef": ef_value}
                    }
                    if log:
                        print(f"Searching with limit={limit}, metric={metric_type}, ef={ef_value} (HNSW index)")
                else:
                    # Default to nprobe for IVF or unknown index types
                    search_params = {
                        "metric_type": metric_type,
                        "params": {"nprobe": nprobe}
                    }
                    if log:
                        print(f"Searching with limit={limit}, metric={metric_type}, nprobe={nprobe} (IVF/other index)")
            except Exception as e:
                # Fallback: try HNSW first, then IVF
                if log:
                    print(f"Could not determine index type, trying HNSW parameters: {e}")
                search_params = {
                    "metric_type": metric_type,
                    "params": {"ef": max(limit * 2, 50)}  # Try HNSW first
                }
                if log:
                    print(f"Searching with limit={limit}, metric={metric_type}, ef={search_params['params']['ef']} (fallback)")
            
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