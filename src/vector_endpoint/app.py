"""Flask vector query endpoint for Comunica pattern requests.

Uses Milvus collection ``version_5`` with per-component S|P|O embeddings
(``VectorDataBase``). Accepts JSON POST bodies on ``/vector`` with pattern,
variables, optional value bindings, and optional ``k`` limit.
"""

from flask import Flask, request, jsonify
import json
import os
from typing import Optional
from pathlib import Path
from datetime import datetime
from vector_endpoint.db.VectorDataBase import VectorDataBase
from vector_endpoint.auto_k import CatalogKResolver, milvus_safe_k, _normalize_literal
from vector_endpoint.adaptive_exp import filter_matches_to_rows, adaptive_batch_search
import re
from urllib.parse import unquote


def _adaptive_multipliers_from_request(json_data: dict) -> tuple[int, ...]:
    """Ladder multipliers from POST body or VECTOR_ADAPTIVE_MULTIPLIERS env."""
    raw = json_data.get("adaptive_multipliers")
    if raw is None:
        raw = os.getenv("VECTOR_ADAPTIVE_MULTIPLIERS", "1,10,100,1000")
    if isinstance(raw, (list, tuple)):
        values = [int(m) for m in raw if int(m) > 0]
    else:
        values = [int(tok.strip()) for tok in str(raw).split(",") if tok.strip()]
    if not values:
        return (1, 10, 100, 1000)
    return tuple(values)


def _adaptive_jaccard_from_request(json_data: dict) -> float:
    raw = json_data.get("adaptive_jaccard")
    if raw is None:
        raw = os.getenv("VECTOR_ADAPTIVE_JACCARD", "0.99")
    return float(raw)


def _join_bound_min_k() -> int:
    """Floor k for join-extension POSTs (bound subject/object, open variable).

    Catalog count can be 1 while embedding rank of the true triple is much
    lower (e.g. ``?X telephone ?Y`` with bound ``X``).
    """
    return max(1, int(os.getenv("VECTOR_JOIN_BOUND_MIN_K", "512")))


def _search_query_has_bound_constant(sq: dict) -> bool:
    val = sq.get("subject")
    if isinstance(val, str) and val and not val.startswith("?"):
        return True
    val = sq.get("object")
    if isinstance(val, str) and val and not val.startswith("?"):
        return True
    return False


def _bump_seed_for_join_extension(seed: int, sq: dict, *, values: list[dict]) -> int:
    if not values:
        return seed
    if _search_query_has_bound_constant(sq):
        return max(seed, _join_bound_min_k())
    return seed


def _log_bgp_fetch(
    *,
    search_queries: list[dict],
    values: list[dict],
    returned_count: int,
    k: Optional[int],
) -> None:
    """Log catalog expected vs returned row count after each Comunica BGP POST."""
    if os.getenv("VECTOR_BGP_LOG", "1") == "0":
        return
    sq = search_queries[0] if search_queries else {}
    expected: Optional[int] = None
    if AUTO_K_RESOLVER.available:
        expected = AUTO_K_RESOLVER.catalog_match_count(
            subject=sq.get("subject"),
            predicate=sq.get("predicate"),
            object_value=sq.get("object"),
            object_type=sq.get("object_type"),
        )
    pred = sq.get("predicate") or "?"
    subj = sq.get("subject") or "?"
    obj = sq.get("object") or "?"
    exp_s = str(expected) if expected is not None else "n/a"
    k_s = str(k) if k is not None else "adaptive"
    print(
        f"[BGP] values_in={len(values)} expected={exp_s} returned={returned_count} "
        f"k={k_s} pattern=({subj}, {pred}, {obj})",
        flush=True,
    )


app = Flask(__name__)
VECTOR_COLLECTION_NAME = "version_5"
CATALOG_PATH = Path(os.getenv("VECTOR_CATALOG_PATH", Path(__file__).resolve().parents[2] / "catalog.pkl"))
AUTO_K_RESOLVER = CatalogKResolver(catalog_path=CATALOG_PATH)
if AUTO_K_RESOLVER.available:
    print(f"Catalog auto-k enabled: {CATALOG_PATH}")
else:
    print(f"Catalog auto-k disabled: {AUTO_K_RESOLVER.error}")

# S, P, O embeddings concatenated and normalized (all-MiniLM-L6-v2, dim truncated at load).
vdb = VectorDataBase(
    database_name="lubm_db",
    host="localhost", 
    port=19530,
    embedding_model="all-MiniLM-L6-v2",  # Faster alternative to paraphrase-multilingual-MiniLM-L12-v2
    # embedding_model="paraphrase-multilingual-MiniLM-L12-v2",  # Original (slower, more accurate)
    # embedding_model="BAAI/bge-large-en-v1.5",
    target_embedding_dim=384  # per-component dim; stored vectors are 3 × 384 = 1152
)

# Connect to the vector database
try:
    vdb.connect()
    print("Connected to Vector Database")
except Exception as e:
    print(f"Failed to connect to Vector Database: {e}")
    vdb = None

def extract_sparql_variables(query):
    """Extract variable names from SPARQL SELECT query"""
    # Match SELECT variables (e.g., ?person ?name ?occupation)
    select_match = re.search(r'SELECT\s+(.*?)\s+WHERE', query, re.IGNORECASE | re.DOTALL)
    if select_match:
        select_part = select_match.group(1).strip()
        # Find all variables starting with ?
        variables = re.findall(r'\?(\w+)', select_part)
        return variables
    return []

def parse_rdf_triple(triple_text):
    """Parse an RDF triple text into subject, predicate, object components"""
    # Pattern to match: <subject> <predicate> "object" . or <subject> <predicate> <object> .
    pattern = r'<([^>]+)>\s+<([^>]+)>\s+(?:"([^"]+)"|<([^>]+)>)\s*\.'
    match = re.match(pattern, triple_text.strip())
    
    if match:
        subject = match.group(1)
        predicate = match.group(2)
        object_literal = match.group(3)  # For "literal" values
        object_uri = match.group(4)     # For <URI> values
        
        return {
            'subject': subject,
            'predicate': predicate,
            'object': object_literal if object_literal else object_uri,
            'object_type': 'literal' if object_literal else 'uri'
        }
    return None

def create_sparql_binding_from_triple(triple_data, query_variables):
    """Create SPARQL binding from parsed triple data matching query variables"""
    binding = {}
    
    if not triple_data:
        return binding
        
    # Map common variable names to triple components
    variable_mappings = {
        'person': triple_data['subject'],
        'subject': triple_data['subject'], 
        's': triple_data['subject'],
        'name': triple_data['object'] if 'name' in triple_data['predicate'] else None,
        'occupation': triple_data['object'] if 'occupation' in triple_data['predicate'] else None,
        'age': triple_data['object'] if 'age' in triple_data['predicate'] else None,
        'email': triple_data['object'] if 'email' in triple_data['predicate'] else None,
        'predicate': triple_data['predicate'],
        'p': triple_data['predicate'],
        'object': triple_data['object'],
        'o': triple_data['object']
    }
    
    for var in query_variables:
        if var in variable_mappings and variable_mappings[var] is not None:
            value = variable_mappings[var]
            
            # Determine if it's a URI or literal based on variable name and content
            if var in ['person', 'subject', 's'] or (var in ['predicate', 'p']):
                binding[var] = {"type": "uri", "value": value}
            elif var in ['name', 'occupation', 'age', 'email'] or triple_data['object_type'] == 'literal':
                binding[var] = {"type": "literal", "value": value}
            else:
                binding[var] = {"type": "uri", "value": value}
        elif var == 'person' and triple_data['subject']:
            # Special handling for ?person queries - always map to subject
            binding[var] = {"type": "uri", "value": triple_data['subject']}

    # Fill any SELECT variables missing from the triple mapping.
    for var in query_variables:
        if var in binding:
            continue

        # Infer ?name from the subject URI by taking the last path segment
        if var.lower() in ['name', 'label']:
            subj = triple_data.get('subject', '')
            inferred = None
            if subj:
                # take last segment after slash or hash
                if '/' in subj:
                    inferred = subj.rstrip('/').split('/')[-1]
                elif '#' in subj:
                    inferred = subj.split('#')[-1]

            if inferred:
                inferred = inferred.replace('_', ' ')
                binding[var] = {"type": "literal", "value": inferred}
            else:
                binding[var] = {"type": "literal", "value": ""}
            continue

        # For subject-like variables ensure subject URI is present
        if var in ['s', 'subject']:
            binding[var] = {"type": "uri", "value": triple_data.get('subject', '')}
            continue

        # For predicate-like variables
        if var in ['p', 'predicate']:
            binding[var] = {"type": "uri", "value": triple_data.get('predicate', '')}
            continue

        # For object-like variables
        if var in ['o', 'object']:
            otype = 'literal' if triple_data.get('object_type') == 'literal' else 'uri'
            binding[var] = {"type": otype, "value": triple_data.get('object', '')}
            continue

        binding[var] = {"type": "literal", "value": ""}
    
    return binding


def handle_pattern_query(json_data):
    """Handle JSON pattern query from Comunica vector source."""
    try:
        pattern = json_data.get('pattern', {})
        variables = json_data.get('vars', [])
        values = json_data.get('values', [])
        requested_limit = json_data.get('k') or json_data.get('limit')
        
        # Extract subject, predicate, object from pattern FIRST
        subject = pattern.get('subject')
        predicate = pattern.get('predicate')
        obj = pattern.get('object')
        
        # If no values provided, create a single empty row to process
        if not values:
            values = [{}]

        variable_aliases = set()
        for var in variables:
            if isinstance(var, str):
                variable_aliases.add(var.lstrip('?'))

        def analyze_term(term):
            value = None
            term_type = None
            is_variable = False
            if isinstance(term, dict):
                term_type = term.get('type') or term.get('termType')
                if term_type in ['iri', 'uri', 'literal']:
                    value = term.get('value')
                elif term_type in ['variable', 'Variable']:
                    raw = term.get('value', '')
                    value = raw.lstrip('?')
                    is_variable = True
            elif isinstance(term, str):
                stripped = term.lstrip('?')
                if term.startswith('?') or stripped in variable_aliases:
                    value = stripped
                    is_variable = True
                else:
                    value = term
            elif term is None:
                is_variable = True
            return value, term_type, is_variable

        def format_constant(value, term_type):
            if value is None:
                return None
            if term_type in ['iri', 'uri']:
                return value if value.startswith('<') and value.endswith('>') else f"<{value}>"
            if term_type == 'literal':
                return _normalize_literal(value)
            return value

        def extract_term_value(term_json):
            """Extract value from Comunica term JSON format"""
            if term_json is None:
                return None
            if isinstance(term_json, dict):
                return term_json.get('value')
            if isinstance(term_json, str):
                return term_json
            return None

        # Map pattern variables to their triple positions
        def normalize_variable(var_value):
            if isinstance(var_value, str):
                return var_value.lstrip('?')
            if isinstance(var_value, dict):
                if var_value.get('termType') == 'Variable':
                    return var_value.get('value', '').lstrip('?')
                if var_value.get('type') == 'variable':
                    return var_value.get('value', '').lstrip('?')
            return None

        pattern_variable_roles = {}
        subject_var = normalize_variable(subject)
        if subject_var:
            pattern_variable_roles[subject_var] = 'subject'
        predicate_var = normalize_variable(predicate)
        if predicate_var:
            pattern_variable_roles[predicate_var] = 'predicate'
        object_var = normalize_variable(obj)
        if object_var:
            pattern_variable_roles[object_var] = 'object'

        # Batch vector search: build all query texts, then search in one or few calls.
        # Helper function to build a search query for a given value_row
        # Returns both the search query and the values needed for validation
        def build_search_query_for_row(value_row):
            """Build a search query dict for a single value_row, returns (search_query, validation_info)"""
            subj_value_raw, subj_type, subj_is_var = analyze_term(subject)
            pred_value_raw, pred_type, pred_is_var = analyze_term(predicate)
            obj_value_raw, obj_type, obj_is_var = analyze_term(obj)
            
            # If a pattern variable is bound in value_row, use that value instead
            if subj_is_var and subject_var and subject_var in value_row:
                bound_value = extract_term_value(value_row[subject_var])
                if bound_value:
                    subject_value = format_constant(bound_value, 'iri')
                    subj_is_var = False
                else:
                    subject_value = None
            else:
                subject_value = None if subj_is_var else format_constant(subj_value_raw, subj_type)
            
            if pred_is_var and predicate_var and predicate_var in value_row:
                bound_value = extract_term_value(value_row[predicate_var])
                if bound_value:
                    predicate_value = format_constant(bound_value, 'iri')
                    pred_is_var = False
                else:
                    predicate_value = None
            else:
                predicate_value = None if pred_is_var else format_constant(pred_value_raw, pred_type)
            
            if obj_is_var and object_var and object_var in value_row:
                bound_value = extract_term_value(value_row[object_var])
                if bound_value:
                    term_json = value_row[object_var]
                    if isinstance(term_json, dict):
                        obj_type_from_value = term_json.get('type', 'iri')
                    else:
                        obj_type_from_value = 'iri'
                    object_value = format_constant(bound_value, obj_type_from_value)
                    object_type = 'literal' if obj_type_from_value == 'literal' else 'uri'
                    obj_is_var = False
                else:
                    object_value = None
                    object_type = None
            else:
                object_value = None if obj_is_var else format_constant(obj_value_raw, obj_type)
                object_type = None
                if not obj_is_var and obj_type == 'literal':
                    object_type = 'literal'
                elif not obj_is_var and obj_type in ['iri', 'uri']:
                    object_type = 'uri'
            
            search_query = {
                "subject": subject_value,
                "predicate": predicate_value,
                "object": object_value,
                "object_type": object_type,
            }
            search_query["text"] = VectorDataBase._format_query_sentence(
                subject_value,
                predicate_value,
                object_value,
                object_type
            )
            
            # Return both search query and validation info
            validation_info = {
                "subject_value": subject_value,
                "predicate_value": predicate_value,
                "object_value": object_value,
                "object_type": object_type,
                "subj_is_var": subj_is_var,
                "pred_is_var": pred_is_var,
                "obj_is_var": obj_is_var,
            }
            
            return search_query, validation_info
        
        # Collect all search queries and validation info
        search_queries = []
        validation_info_list = []  # Store validation info for each query
        for value_row in values:
            search_query, validation_info = build_search_query_for_row(value_row)
            search_queries.append(search_query)
            validation_info_list.append(validation_info)

        explicit_k: Optional[int] = None
        if requested_limit is not None:
            try:
                explicit_k = int(requested_limit)
            except (ValueError, TypeError):
                explicit_k = None

        # Batch vector search
        all_results_rows = []
        if vdb is not None and len(search_queries) > 0:
            try:
                def filter_fn(matches, query_idx):
                    vinfo = (
                        validation_info_list[query_idx]
                        if query_idx < len(validation_info_list)
                        else {}
                    )
                    return filter_matches_to_rows(
                        matches,
                        pattern_subject=subject,
                        pattern_predicate=predicate,
                        pattern_object=obj,
                        validation_info=vinfo,
                        variables=variables,
                        pattern_variable_roles=pattern_variable_roles,
                        value_row=values[query_idx],
                        parse_rdf_triple=parse_rdf_triple,
                        log=False,
                    )

                if explicit_k is not None:
                    effective_search_limit = milvus_safe_k(explicit_k)

                    vector_results = vdb.search(
                        collection_name=VECTOR_COLLECTION_NAME,
                        query_texts=search_queries,
                        limit=effective_search_limit,
                        output_fields=["text"],
                        log=False,
                    )

                    for value_row_idx, _value_row in enumerate(values):
                        if value_row_idx < len(vector_results):
                            matches = vector_results[value_row_idx].get("matches", [])
                            rows, _ids = filter_fn(matches, value_row_idx)
                            all_results_rows.extend(rows)
                else:
                    # Per-query seed k from catalog, or fallback by binding count.
                    seed_ks: list[int] = []
                    stability_count_floors: list[Optional[int]] = []
                    for sq in search_queries:
                        seed = None
                        floor: Optional[int] = None
                        if AUTO_K_RESOLVER.available:
                            floor = AUTO_K_RESOLVER.catalog_match_count(
                                subject=sq.get("subject"),
                                predicate=sq.get("predicate"),
                                object_value=sq.get("object"),
                                object_type=sq.get("object_type"),
                            )
                            seed = AUTO_K_RESOLVER.auto_k_for_pattern(
                                subject=sq.get("subject"),
                                predicate=sq.get("predicate"),
                                object_value=sq.get("object"),
                                object_type=sq.get("object_type"),
                            )
                        if seed is None:
                            if len(values) > 10:
                                seed = 10
                            elif len(values) > 5:
                                seed = 25
                            else:
                                seed = 50
                        seed = _bump_seed_for_join_extension(seed, sq, values=values)
                        seed_ks.append(seed)
                        stability_count_floors.append(floor)

                    adaptive_mults = _adaptive_multipliers_from_request(json_data)
                    adaptive_jaccard = _adaptive_jaccard_from_request(json_data)
                    final_rows = adaptive_batch_search(
                        vdb=vdb,
                        collection_name=VECTOR_COLLECTION_NAME,
                        search_queries=search_queries,
                        seed_ks=seed_ks,
                        filter_fn=filter_fn,
                        multipliers=adaptive_mults,
                        jaccard_threshold=adaptive_jaccard,
                        log=False,
                        stability_count_floors=stability_count_floors,
                    )
                    for rows in final_rows:
                        all_results_rows.extend(rows)

            except Exception as e:
                print(f"Error during batch vector search: {e}")
                import traceback
                traceback.print_exc()
        _log_bgp_fetch(
            search_queries=search_queries,
            values=values,
            returned_count=len(all_results_rows),
            k=explicit_k,
        )
        response = {
            "vars": variables,
            "rows": all_results_rows
        }
        return jsonify(response), 200, {'Content-Type': 'application/json'}
    
    except Exception as e:
        print(f"ERROR in handle_pattern_query: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "vars": [], "rows": []}), 500


def log_request(endpoint_name):
    """Log request details when FLASK_DEBUG is enabled."""
    if not os.getenv('FLASK_DEBUG'):
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*50}")
    print(f"[{timestamp}] {endpoint_name}")
    print(f"{'='*50}")
    print(f"Method: {request.method}")
    print(f"URL: {request.url}")
    print(f"Path: {request.path}")
    print(f"Remote Address: {request.remote_addr}")
    
    # Print headers
    print(f"\nHeaders:")
    for header, value in request.headers:
        print(f"  {header}: {value}")
    
    # Print query parameters (always show section)
    print(f"\nQuery Parameters:")
    if request.args:
        for key, value in request.args.items():
            print(f"  {key}: {value}")
    else:
        print(f"  (None)")
    
    # Print form data (always show section)
    print(f"\nForm Data:")
    if request.form:
        for key, value in request.form.items():
            print(f"  {key}: {value}")
    else:
        print(f"  (None)")
    
    # Print content type and length
    print(f"\nContent Info:")
    print(f"  Content-Type: {request.content_type}")
    print(f"  Content-Length: {request.content_length}")
    print(f"  Is JSON: {request.is_json}")
    
    # Print JSON data
    if request.is_json:
        print(f"\nJSON Data:")
        try:
            json_data = request.get_json()
            print(f"  {json.dumps(json_data, indent=2)}")
        except Exception as e:
            print(f"  Error parsing JSON: {e}")
    
    # Print raw data (always show if there's any data)
    print(f"\nRaw Request Body:")
    if request.data:
        try:
            decoded_data = request.data.decode('utf-8')
            print(f"  Length: {len(request.data)} bytes")
            print(f"  Content: {decoded_data}")
        except Exception as e:
            print(f"  Length: {len(request.data)} bytes")
            print(f"  Content (binary): {request.data}")
            print(f"  Decode error: {e}")
    else:
        print(f"  (Empty)")
    
    print(f"{'='*50}\n")

@app.route('/', methods=['POST'])
@app.route('/vector', methods=['POST'])
def vector():
    """Vector pattern query endpoint for Comunica"""
    log_request("VECTOR PATTERN ENDPOINT")
    
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    try:
        json_data = request.get_json()
        if 'pattern' not in json_data:
            return jsonify({"error": "Missing 'pattern' field in request"}), 400
        return handle_pattern_query(json_data)
    except Exception as e:
        print(f"Error handling pattern query: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/', methods=['GET'])
@app.route('/vector', methods=['GET'])
def index():
    """Info endpoint"""
    log_request("VECTOR INFO ENDPOINT (GET)")
    return """Vector Query Endpoint Ready

Supported request format:
- POST /vector (or /)
  Content-Type: application/json
  
  Body:
  {
    "pattern": {
      "subject": "person" | {"type":"iri","value":"..."},
      "predicate": {"type":"iri","value":"..."},
      "object": "name" | {"type":"literal","value":"..."}
    },
    "vars": ["person", "name"],
    "values": [...],  // optional
    "k": 10,          // optional (number of top results to return, Comunica uses 'k')
    "limit": 10       // optional (alias for 'k')
  }

Endpoint responds with:
  {
    "vars": ["person", "name"],
    "rows": [
      {"person": {"type":"iri","value":"..."}, "name": {"type":"literal","value":"..."}}
    ]
  }
""", 200, {'Content-Type': 'text/plain'}


@app.route('/sparql', methods=['GET', 'POST'])
def sparql():
    """Legacy SPARQL endpoint - redirects to info"""
    log_request("SPARQL ENDPOINT (LEGACY)")
    
    query = None
    query_type = "UNKNOWN"
    
    # Extract query based on Comunica's request patterns
    if request.method == 'GET':
        # GET: /sparql?query=SELECT%20*%20WHERE...
        query = request.args.get('query')
        if query:
            query_type = "GET_URL_ENCODED"
            
    elif request.method == 'POST':
        content_type = request.content_type or ""
        
        if 'application/sparql-query' in content_type:
            # POST with raw SPARQL query in body
            query = request.data.decode('utf-8')
            query_type = "POST_RAW_SPARQL"
            
        elif 'application/sparql-update' in content_type:
            # SPARQL UPDATE queries
            query = request.data.decode('utf-8')
            query_type = "POST_SPARQL_UPDATE"
            
        elif 'application/x-www-form-urlencoded' in content_type:
            # POST with form-encoded query parameter
            query = request.form.get('query')
            query_type = "POST_FORM_ENCODED"
            
        elif request.form.get('query'):
            # Fallback for form data
            query = request.form.get('query')
            query_type = "POST_FORM_FALLBACK"
    
    if query:
        print(f"\nSPARQL QUERY DETECTED ({query_type}):")
        print(f"{'='*70}")
        print(f"Content-Type: {request.content_type}")
        print(f"Query Length: {len(query)} characters")
        print(f"{'='*70}")
        print(query)
        print(f"{'='*70}")
        
        # Determine response format based on Accept header and query type
        accept_header = request.headers.get('Accept', '').lower()
        
        # Check what type of query this is
        query_upper = query.upper().strip()
        
        if query_upper.startswith('SELECT'):
            # perform similarity search and return top 3 results
            
            # Extract variables from SPARQL query
            query_variables = extract_sparql_variables(query)
            print(f"Query: {query}")
            print(f"Extracted variables from query: {query_variables}")
            print(f"Variables type: {type(query_variables)}, length: {len(query_variables) if query_variables else 0}")
            
            search_results = []
            if vdb is not None:
                try:
                    collection_name = VECTOR_COLLECTION_NAME
                    vector_results = vdb.search(
                        collection_name=collection_name,
                        query_texts=query,
                        limit=100,  # Return top 100 results
                        output_fields=["text"],
                        log=True
                    )
                    
                    if vector_results and len(vector_results) > 0:
                        # Extract matches from the first query result
                        matches = vector_results[0].get("matches", [])
                        search_results = matches[:3]  # Ensure we only get top 3
                        print(f" Found {len(search_results)} similar results")
                    else:
                        print("No similar results found in vector database")
                        
                except Exception as e:
                    print(f"Error during vector search: {e}")
                    search_results = []
            else:
                print("Vector database not available")
            
            if 'application/sparql-results+json' in accept_header:
                # Return JSON results with vector search results mapped to query variables
                bindings = []
                
                print(f" Decision point: search_results={bool(search_results)}, query_variables={bool(query_variables)}")
                print(f" search_results length: {len(search_results) if search_results else 0}")
                print(f" query_variables content: {query_variables}")
                
                if search_results:
                    print(f" Processing {len(search_results)} search results")
                    # Always try to extract variables, even if we think we have them
                    if not query_variables:
                        print(f"🔧 Re-extracting variables as backup")
                        query_variables = extract_sparql_variables(query)
                    
                    print(f" Final variables: {query_variables}")
                    
                    # Parse each RDF triple and create proper SPARQL bindings
                    for result in search_results:
                        triple_text = result.get("text", "")
                        triple_data = parse_rdf_triple(triple_text)
                        print(f" Processing: {triple_text}")
                        print(f" Parsed data: {triple_data}")
                        
                        if triple_data and query_variables:
                            binding = create_sparql_binding_from_triple(triple_data, query_variables)
                            print(f" Created binding: {binding}")
                            if binding:  # Only add non-empty bindings
                                bindings.append(binding)
                        elif triple_data:
                            # If no query variables, try to infer from the triple
                            # Default to 'person' if subject looks like a person URI
                            if 'person' in triple_data['subject']:
                                binding = {'person': {'type': 'uri', 'value': triple_data['subject']}}
                                bindings.append(binding)
                                print(f" Inferred person binding: {binding}")
                            else:
                                # Generic fallback
                                binding = {
                                    "s": {"type": "uri", "value": triple_data['subject']},
                                    "p": {"type": "uri", "value": triple_data['predicate']},
                                    "o": {"type": "literal" if triple_data['object_type'] == 'literal' else "uri", "value": triple_data['object']}
                                }
                                bindings.append(binding)
                                print(f" Generic binding: {binding}")
                else:
                    # Fallback to sample data if no vector results
                    bindings = [
                        {
                            "s": {"type": "uri", "value": "http://example.org/person/john_doe"},
                            "p": {"type": "uri", "value": "http://example.org/name"},
                            "o": {"type": "literal", "value": "John Doe"}
                        },
                        {
                            "s": {"type": "uri", "value": "http://example.org/person/jane_smith"},
                            "p": {"type": "uri", "value": "http://example.org/name"},
                            "o": {"type": "literal", "value": "Jane Smith"}
                        }
                    ]
                
                # Use query variables for head vars, fallback to inferred or generic
                if query_variables:
                    head_vars = query_variables
                elif bindings and 'person' in bindings[0]:
                    head_vars = ['person']
                elif bindings and 's' in bindings[0]:
                    head_vars = ['s', 'p', 'o']
                else:
                    head_vars = ["s", "p", "o"]
                    
                print(f" Final head vars: {head_vars}")
                print(f" Final bindings count: {len(bindings)}")
                
                response_data = {
                    "head": {"vars": head_vars},
                    "results": {"bindings": bindings}
                }
                return jsonify(response_data), 200, {'Content-Type': 'application/sparql-results+json'}
                
            else:
                # Return XML results (fallback)
                if search_results:
                    xml_results = ""
                    for i, result in enumerate(search_results):
                        xml_results += f'''
                        <result>
                            <binding name="rank"><literal datatype="http://www.w3.org/2001/XMLSchema#integer">{i + 1}</literal></binding>
                            <binding name="text"><literal>{result.get("text", "")}</literal></binding>
                            <binding name="score"><literal datatype="http://www.w3.org/2001/XMLSchema#decimal">{round(result.get("score", 0.0), 4)}</literal></binding>
                            <binding name="similarity"><uri>http://example.org/similarity/{result.get('id', i)}</uri></binding>
                        </result>'''
                    
                    xml_response = f'''<?xml version="1.0"?>
                        <sparql xmlns="http://www.w3.org/2005/sparql-results#">
                        <head>
                            <variable name="rank"/>
                            <variable name="text"/>
                            <variable name="score"/>
                            <variable name="similarity"/>
                        </head>
                        <results>{xml_results}
                        </results>
                        </sparql>'''
                else:
                    xml_response = '''<?xml version="1.0"?>
                        <sparql xmlns="http://www.w3.org/2005/sparql-results#">
                        <head>
                            <variable name="s"/>
                            <variable name="p"/>
                            <variable name="o"/>
                        </head>
                        <results>
                            <result>
                            <binding name="s"><uri>http://example.org/person/john_doe</uri></binding>
                            <binding name="p"><uri>http://example.org/name</uri></binding>
                            <binding name="o"><literal>John Doe</literal></binding>
                            </result>
                            <result>
                            <binding name="s"><uri>http://example.org/person/jane_smith</uri></binding>
                            <binding name="p"><uri>http://example.org/name</uri></binding>
                            <binding name="o"><literal>Jane Smith</literal></binding>
                            </result>
                        </results>
                        </sparql>'''
                return xml_response, 200, {'Content-Type': 'application/sparql-results+xml'}
    
    else:
        accept_header = request.headers.get('Accept', '').lower()
        
        if ('application/n-quads' in accept_header or 
            'application/trig' in accept_header or 
            'application/ld+json' in accept_header or
            'application/n-triples' in accept_header or
            'text/turtle' in accept_header):
            
            service_description = '''@prefix sd: <http://www.w3.org/ns/sparql-service-description#> .
                @prefix ex: <http://example.org/> .

                ex:sparql-service a sd:Service ;
                    sd:endpoint <http://localhost:2222/sparql> ;
                    sd:supportedLanguage sd:SPARQL11Query, sd:SPARQL11Update ;
                    sd:resultFormat <http://www.w3.org/ns/formats/SPARQL_Results_JSON>,
                                <http://www.w3.org/ns/formats/SPARQL_Results_XML> ;
                    sd:feature sd:DereferencesURIs .

                ex:default-graph a sd:Graph ;
                    sd:name ex:default .'''
            
            return service_description, 200, {'Content-Type': 'text/turtle'}
        
        else:
            return """SPARQL Endpoint Ready

            Supported request formats:
            - GET: /sparql?query=<encoded-sparql>
            - POST application/sparql-query: Raw SPARQL in body  
            - POST application/x-www-form-urlencoded: query=<sparql>
            - POST application/sparql-update: SPARQL UPDATE in body

            Supported response formats:
            - application/sparql-results+json (preferred)
            - application/sparql-results+xml
            - text/turtle (for CONSTRUCT/DESCRIBE)

            Service Description: Available in Turtle format with appropriate Accept header.""", 200, {'Content-Type': 'text/plain'}


if __name__ == '__main__':
    print("Starting Vector Endpoint Server...")
    print("Listening on http://localhost:2222")
    print("Press Ctrl+C to stop the server")
    print("="*50)
    
    # Run the Flask app
    app.run(
        host='localhost',
        port=2222,
        debug=True,  # Enable debug mode for auto-reload
        use_reloader=True  # Auto-reload on code changes
    )
