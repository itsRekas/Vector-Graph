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
from vector_endpoint.pattern_query import (
    PatternQueryInput,
    stream_pattern_rows,
)
from vector_endpoint.bgp_log import configure_bgp_logging, log_http_pattern_received
from vector_endpoint.rdf_utils import parse_rdf_triple
from vector_endpoint.server_state import AUTO_K_RESOLVER, VECTOR_COLLECTION_NAME, vdb
import re
from urllib.parse import unquote


app = Flask(__name__)

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
            # Map ?person to subject URI
            binding[var] = {"type": "uri", "value": triple_data['subject']}

    # Fill SELECT variables not covered by the triple mapping.
    for var in query_variables:
        if var in binding:
            continue

        # Infer ?name / ?label from the last URI path segment
        if var.lower() in ['name', 'label']:
            subj = triple_data.get('subject', '')
            inferred = None
            if subj:
                # Last path segment after / or #
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

        # Subject-role variables
        if var in ['s', 'subject']:
            binding[var] = {"type": "uri", "value": triple_data.get('subject', '')}
            continue

        # Predicate-role variables
        if var in ['p', 'predicate']:
            binding[var] = {"type": "uri", "value": triple_data.get('predicate', '')}
            continue

        # Object-role variables
        if var in ['o', 'object']:
            otype = 'literal' if triple_data.get('object_type') == 'literal' else 'uri'
            binding[var] = {"type": otype, "value": triple_data.get('object', '')}
            continue

        binding[var] = {"type": "literal", "value": ""}
    
    return binding


def handle_pattern_pagination(json_data):
    """Session-based pagination request handler."""
    from vector_endpoint.pagination_search import resolve_pagination_page, start_pagination_page
    from vector_endpoint.pagination_sessions import (
        PaginationPageNotCached,
        PaginationSessionGone,
        PaginationSessionNotFound,
        resolve_session_id,
    )

    session_id = resolve_session_id(json_data)
    has_page = json_data.get("page") is not None
    has_next = bool(json_data.get("next"))
    if has_page and has_next:
        return (
            jsonify({"error": "page and next are mutually exclusive"}),
            400,
            {"Content-Type": "application/json"},
        )
    if (has_page or has_next) and not session_id:
        return (
            jsonify({"error": "session is required when using page or next"}),
            400,
            {"Content-Type": "application/json"},
        )
    if "page_index" in json_data or "offset" in json_data:
        return (
            jsonify({"error": "page_index and offset are not supported; use session and page"}),
            400,
            {"Content-Type": "application/json"},
        )

    try:
        if session_id:
            page_num: int | None = None
            if has_page:
                try:
                    page_num = int(json_data["page"])
                except (ValueError, TypeError) as exc:
                    return (
                        jsonify({"error": f"invalid page: {exc}"}),
                        400,
                        {"Content-Type": "application/json"},
                    )
            page = resolve_pagination_page(
                session_id,
                page=page_num,
                cancel=bool(json_data.get("cancel")),
            )
        else:
            query_input = PatternQueryInput.from_json(json_data)
            if query_input.k_mode != "pagination":
                return (
                    jsonify({"error": "k_mode must be 'pagination' to start pagination"}),
                    400,
                    {"Content-Type": "application/json"},
                )
            log_http_pattern_received(query_input)
            page = start_pagination_page(query_input)
        body: dict = {
            "vars": page.vars,
            "rows": page.rows,
            "pagination": page.pagination.to_dict(),
        }
        return jsonify(body), 200, {"Content-Type": "application/json"}
    except PaginationPageNotCached as exc:
        return jsonify({"error": str(exc)}), 404, {"Content-Type": "application/json"}
    except PaginationSessionNotFound as exc:
        return jsonify({"error": str(exc)}), 404, {"Content-Type": "application/json"}
    except PaginationSessionGone as exc:
        return jsonify({"error": str(exc)}), 410, {"Content-Type": "application/json"}
    except ValueError as exc:
        return jsonify({"error": str(exc), "vars": [], "rows": []}), 400, {"Content-Type": "application/json"}
    except Exception as e:
        print(f"ERROR in handle_pattern_pagination: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e), "vars": [], "rows": []}), 500


def handle_pattern_query(json_data):
    """Handle buffered JSON pattern query (single response)."""
    try:
        query_input = PatternQueryInput.from_json(json_data)
        log_http_pattern_received(query_input)
        rows = list(stream_pattern_rows(query_input))
        return jsonify({"vars": query_input.vars, "rows": rows}), 200, {"Content-Type": "application/json"}
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
        from vector_endpoint.pagination_sessions import resolve_session_id

        if resolve_session_id(json_data) or json_data.get("k_mode") == "pagination":
            return handle_pattern_pagination(json_data)
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
                        search_results = matches[:3]  # top 3
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
                    # Extract variables from each triple
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
