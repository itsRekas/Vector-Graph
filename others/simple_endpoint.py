#!/usr/bin/env python3
from flask import Flask, request, jsonify
import json
import re
from datetime import datetime
from lib.VectorDataBase import VectorDataBase

app = Flask(__name__)

# Initialize VectorDataBase
vdb = VectorDataBase("sparql_db", "localhost", 19530, "paraphrase-multilingual-MiniLM-L12-v2")
try:
    vdb.connect()
    print("✅ Connected to Vector Database")
except Exception as e:
    print(f"❌ Failed to connect: {e}")
    vdb = None

def extract_sparql_variables(query):
    """Extract variable names from SPARQL SELECT query"""
    select_match = re.search(r'SELECT\s+(.*?)\s+WHERE', query, re.IGNORECASE | re.DOTALL)
    if select_match:
        select_part = select_match.group(1).strip()
        variables = re.findall(r'\?(\w+)', select_part)
        return variables
    return []

def parse_rdf_triple(triple_text):
    """Parse an RDF triple text into subject, predicate, object components"""
    pattern = r'<([^>]+)>\s+<([^>]+)>\s+(?:"([^"]+)"|<([^>]+)>)\s*\.'
    match = re.match(pattern, triple_text.strip())
    
    if match:
        subject = match.group(1)
        predicate = match.group(2)
        object_literal = match.group(3)
        object_uri = match.group(4)
        
        return {
            'subject': subject,
            'predicate': predicate,
            'object': object_literal if object_literal else object_uri,
            'object_type': 'literal' if object_literal else 'uri'
        }
    return None

@app.route('/sparql', methods=['GET', 'POST'])
def sparql():
    """Simple SPARQL endpoint for Comunica compatibility"""
    
    # Extract query
    query = None
    if request.method == 'GET':
        query = request.args.get('query')
    elif request.method == 'POST':
        if 'application/sparql-query' in (request.content_type or ""):
            query = request.data.decode('utf-8')
        elif request.form.get('query'):
            query = request.form.get('query')
    
    print(f"\n🔍 SPARQL Query: {query}")
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    # Extract variables
    variables = extract_sparql_variables(query)
    print(f"🔍 Variables: {variables}")
    
    # Perform vector search
    search_results = []
    if vdb is not None:
        try:
            vector_results = vdb.search("text_embeddings", query, limit=3, output_fields=["text"])
            if vector_results and len(vector_results) > 0:
                matches = vector_results[0].get("matches", [])
                search_results = matches[:3]
                print(f"🔍 Found {len(search_results)} results")
        except Exception as e:
            print(f"❌ Search error: {e}")
    
    # Create SPARQL bindings
    bindings = []
    
    if search_results and variables:
        for result in search_results:
            triple_text = result.get("text", "")
            triple_data = parse_rdf_triple(triple_text)
            print(f"📋 Processing: {triple_text}")
            print(f"📋 Parsed: {triple_data}")
            
            if triple_data:
                binding = {}
                
                # Map variables to triple components
                for var in variables:
                    if var in ['person', 'subject', 's'] and triple_data['subject']:
                        binding[var] = {"type": "uri", "value": triple_data['subject']}
                    elif var == 'occupation' and 'occupation' in triple_data['predicate']:
                        binding[var] = {"type": "literal", "value": triple_data['object']}
                    elif var == 'name' and 'name' in triple_data['predicate']:
                        binding[var] = {"type": "literal", "value": triple_data['object']}
                    elif var in ['predicate', 'p']:
                        binding[var] = {"type": "uri", "value": triple_data['predicate']}
                    elif var in ['object', 'o']:
                        binding[var] = {"type": "literal" if triple_data['object_type'] == 'literal' else "uri", "value": triple_data['object']}
                
                if binding:
                    bindings.append(binding)
                    print(f"✅ Created binding: {binding}")
    
    # Fallback if no proper bindings created
    if not bindings and search_results:
        for result in search_results:
            triple_text = result.get("text", "")
            triple_data = parse_rdf_triple(triple_text)
            if triple_data:
                bindings.append({
                    "s": {"type": "uri", "value": triple_data['subject']},
                    "p": {"type": "uri", "value": triple_data['predicate']},
                    "o": {"type": "literal" if triple_data['object_type'] == 'literal' else "uri", "value": triple_data['object']}
                })
        variables = ["s", "p", "o"]
    
    # Final response
    response = {
        "head": {"vars": variables or ["s", "p", "o"]},
        "results": {"bindings": bindings}
    }
    
    print(f"🎯 Response vars: {response['head']['vars']}")
    print(f"🎯 Bindings count: {len(bindings)}")
    for i, binding in enumerate(bindings):
        print(f"   Binding {i+1}: {binding}")
    
    return jsonify(response), 200, {'Content-Type': 'application/sparql-results+json'}

@app.route('/health')
def health():
    return jsonify({"status": "ok", "vector_db": "connected" if vdb else "disconnected"})

if __name__ == '__main__':
    print("🚀 Starting Simple SPARQL Vector Endpoint...")
    app.run(host='localhost', port=2222, debug=False)
