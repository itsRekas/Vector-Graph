#!/usr/bin/env python3
"""
Benchmark script that runs queries from query.txt via comunica-vector (with different k values)
and comunica-sparql-file, then compares results with bar charts.
V4 version: Uses lubm_graph_v1_normalized collection. Supports k parameter forwarding from Comunica to vector endpoint.
"""

import re
import subprocess
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple

# Check for optional dependencies
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Visualization will be skipped.")
    print("Install with: pip install matplotlib numpy")

# Configuration
QUERY_FILE = "Query_Types.txt"
RDF_FILE = "RLUBM_cleaned.nt"  # Default RDF file, can be overridden
VECTOR_ENDPOINT = "http://localhost:2222/vector"
K_VALUES = [10, 50, 100, 1000]  # Different k values to test
TIMEOUT = 60  # Timeout in seconds for each query
MAX_RESULT_COUNT = 200  # Only benchmark queries with fewer than this many expected results


def extract_queries(query_file: str, max_results: int = MAX_RESULT_COUNT) -> List[Dict[str, str]]:
    """
    Extract SPARQL queries from Query_Types.txt file.
    Filters queries to only include those with fewer than max_results expected results.
    Returns list of dicts with 'name', 'query', and 'expected_results' fields.
    """
    queries = []
    
    # Try different filename variations
    possible_files = [query_file, query_file.lower(), "Query_Types.txt", "query_types.txt"]
    actual_file = None
    
    for filename in possible_files:
        if os.path.exists(filename):
            actual_file = filename
            break
    
    if not actual_file:
        print(f"Error: Query file not found! Tried: {possible_files}")
        sys.exit(1)
    
    print(f"Reading queries from: {actual_file}")
    print(f"Filtering queries with < {max_results} expected results...")
    
    with open(actual_file, 'r') as f:
        content = f.read()
    
    # Process file line by line for more reliable extraction
    lines = content.split('\n')
    current_query_num = None
    current_query_lines = []
    current_result_count = None
    collecting_query = False
    
    for line in lines:
        line_stripped = line.strip()
        
        # Check for query header (Query 1, Query 2:, etc.)
        q_match = re.match(r'Query\s+(\d+)\s*:?', line_stripped, re.IGNORECASE)
        if q_match:
            # Save previous query if exists and meets criteria
            if current_query_num and current_query_lines:
                query_text = ' '.join(current_query_lines).strip()
                # Normalize whitespace
                query_text = ' '.join(query_text.split())
                if query_text.upper().startswith('SELECT'):
                    # Only add if result count is less than threshold
                    if current_result_count is None or current_result_count < max_results:
                        queries.append({
                            'name': f"Q{current_query_num}",
                            'query': query_text,
                            'expected_results': current_result_count
                        })
            
            # Start new query
            current_query_num = q_match.group(1)
            current_query_lines = []
            current_result_count = None
            collecting_query = False
            continue
        
        # Check for Results line to extract expected result count
        results_match = re.match(r'Results\s*:\s*([\d,]+)\s*tuples?', line_stripped, re.IGNORECASE)
        if results_match:
            # Extract number, remove commas
            result_str = results_match.group(1).replace(',', '')
            try:
                current_result_count = int(result_str)
            except ValueError:
                current_result_count = None
            continue
        
        # Check if we hit a SELECT statement (start of actual query)
        if line_stripped.upper().startswith('SELECT'):
            collecting_query = True
            current_query_lines.append(line_stripped)
        elif collecting_query:
            # Continue collecting query lines
            if line_stripped:  # Only add non-empty lines
                current_query_lines.append(line_stripped)
                # Check if query ends (closing brace on its own line or end of line)
                if line_stripped.endswith('}'):
                    # Query complete - will be saved on next Query header or end of file
                    collecting_query = False
    
    # Save last query
    if current_query_num and current_query_lines:
        query_text = ' '.join(current_query_lines).strip()
        query_text = ' '.join(query_text.split())
        if query_text.upper().startswith('SELECT'):
            # Only add if result count is less than threshold
            if current_result_count is None or current_result_count < max_results:
                queries.append({
                    'name': f"Q{current_query_num}",
                    'query': query_text,
                    'expected_results': current_result_count
                })
    
    print(f"Extracted {len(queries)} queries (filtered from queries with < {max_results} results):")
    for q in queries:
        result_info = f" ({q.get('expected_results', 'unknown')} expected)" if q.get('expected_results') else ""
        print(f"  {q['name']}{result_info}: {q['query'][:70]}...")
    
    return queries


def run_comunica_vector(query: str, k: int) -> Tuple[int, float]:
    """
    Run a query via comunica-vector with specified k value.
    V4: The k parameter is now forwarded from Comunica to the vector endpoint.
    Returns (result_count, execution_time).
    """
    try:
        # Build the command
        cmd = [
            'comunica-vector',
            VECTOR_ENDPOINT,
            '-q', query,
            '-k', str(k)
        ]
        
        print(f"  Running comunica-vector with k={k}...")
        
        # Run command with timeout
        start_time = os.times()[0]  # User time
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT
        )
        
        end_time = os.times()[0]
        execution_time = end_time - start_time
        
        if result.returncode != 0:
            print(f"    Error: {result.stderr}")
            return 0, execution_time
        
        # Parse JSON output (SPARQL Results JSON format)
        try:
            output = json.loads(result.stdout)
            count = 0
            
            # Debug: print structure to understand format
            if k == 10:  # Only print debug for first k value to avoid spam
                print(f"    DEBUG: Response type: {type(output)}")
                if isinstance(output, dict):
                    print(f"    DEBUG: Response keys: {list(output.keys())}")
                    if 'rows' in output:
                        print(f"    DEBUG: rows type: {type(output['rows'])}, length: {len(output['rows']) if isinstance(output['rows'], list) else 'N/A'}")
                        if isinstance(output['rows'], list) and len(output['rows']) > 0:
                            print(f"    DEBUG: First row type: {type(output['rows'][0])}")
                            if isinstance(output['rows'][0], dict):
                                print(f"    DEBUG: First row keys: {list(output['rows'][0].keys())}")
                elif isinstance(output, list):
                    print(f"    DEBUG: Response is a list with {len(output)} items")
            
            # Check for SPARQL Results JSON format
            if isinstance(output, dict):
                if 'results' in output and 'bindings' in output['results']:
                    count = len(output['results']['bindings'])
                # Check for alternative format with 'rows'
                elif 'rows' in output:
                    if isinstance(output['rows'], list):
                        count = len(output['rows'])
                    else:
                        print(f"    WARNING: 'rows' is not a list, type: {type(output['rows'])}")
                        count = 0
                # Check for 'vars' and 'rows' structure (from vector endpoint)
                elif 'vars' in output and 'rows' in output:
                    if isinstance(output['rows'], list):
                        count = len(output['rows'])
                    else:
                        print(f"    WARNING: 'rows' is not a list, type: {type(output['rows'])}")
                        count = 0
                else:
                    # Try to count lines of output as fallback
                    lines = [l for l in result.stdout.strip().split('\n') if l.strip() and not l.strip().startswith('{')]
                    count = len(lines)
            elif isinstance(output, list):
                # Response is directly a list
                count = len(output)
            else:
                print(f"    WARNING: Unexpected response type: {type(output)}")
                count = 0
            
            print(f"    Found {count} results in {execution_time:.2f}s")
            return count, execution_time
            
        except json.JSONDecodeError:
            # Not JSON, try to count result lines (skip header/metadata)
            lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
            # Filter out JSON structure lines and count actual results
            result_lines = [l for l in lines if not (l.strip().startswith('{') or l.strip().startswith('}'))]
            count = len(result_lines) if result_lines else 0
            print(f"    Found {count} results (non-JSON output) in {execution_time:.2f}s")
            return count, execution_time
            
    except subprocess.TimeoutExpired:
        print(f"    Timeout after {TIMEOUT}s")
        return 0, TIMEOUT
    except Exception as e:
        print(f"    Exception: {e}")
        return 0, 0.0


def add_limit_to_query(query: str, limit: int = 1000) -> str:
    """
    Add LIMIT clause to SPARQL query if it doesn't already have one.
    SPARQL format: SELECT ... WHERE { ... } LIMIT 1000
    """
    query_upper = query.upper().strip()
    
    # Check if query already has a LIMIT clause
    if 'LIMIT' in query_upper:
        # Query already has LIMIT, return as-is
        return query
    
    # Remove trailing whitespace
    query = query.rstrip()
    
    # Find the last closing brace (end of WHERE clause) and add LIMIT after it
    # SPARQL format: SELECT ... WHERE { ... } LIMIT 1000
    # We need to find the last } that closes the WHERE clause
    
    # Find the position of the last closing brace
    last_brace_pos = query.rfind('}')
    
    if last_brace_pos != -1:
        # Insert LIMIT after the closing brace
        query = query[:last_brace_pos + 1] + f' LIMIT {limit}'
    else:
        # No closing brace found, just append LIMIT
        query = query + f' LIMIT {limit}'
    
    return query


def run_comunica_sparql_file(query: str, rdf_file: str) -> Tuple[int, float]:
    """
    Run a query via comunica-sparql-file against the RDF file.
    Returns (result_count, execution_time).
    """
    try:
        if not os.path.exists(rdf_file):
            print(f"  Warning: RDF file '{rdf_file}' not found!")
            return 0, 0.0
        
        # Add LIMIT 1000 to the query
        query_with_limit = add_limit_to_query(query, limit=1000)
        
        # Build the command
        cmd = [
            'comunica-sparql-file',
            rdf_file,
            query_with_limit
        ]
        
        print(f"  Running comunica-sparql-file against {rdf_file}...")
        
        # Run command with timeout
        start_time = os.times()[0]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=TIMEOUT
        )
        
        end_time = os.times()[0]
        execution_time = end_time - start_time
        
        if result.returncode != 0:
            print(f"    Error: {result.stderr}")
            return 0, execution_time
        
        # Parse JSON output
        try:
            output = json.loads(result.stdout)
            if 'results' in output and 'bindings' in output['results']:
                count = len(output['results']['bindings'])
                # Subtract 2 to correct for extra items in comunica-sparql-file output
                count = max(0, count - 2)
            else:
                # Try to count lines of output as fallback
                lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
                count = len(lines)
                # Subtract 2 to account for JSON structure lines (opening/closing braces or headers)
                count = max(0, count - 2)
            
            print(f"    Found {count} results in {execution_time:.2f}s")
            return count, execution_time
            
        except json.JSONDecodeError:
            # Not JSON, try to count lines
            lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
            count = len(lines)
            # Subtract 2 to account for header/metadata lines
            count = max(0, count - 2)
            print(f"    Found {count} results (non-JSON output) in {execution_time:.2f}s")
            return count, execution_time
            
    except subprocess.TimeoutExpired:
        print(f"    Timeout after {TIMEOUT}s")
        return 0, TIMEOUT
    except Exception as e:
        print(f"    Exception: {e}")
        return 0, 0.0


def is_simple_query(query_name: str, query_text: str) -> bool:
    """
    Determine if a query is simple (single triple pattern) or complex (multiple triple patterns).
    Simple queries have exactly one triple pattern in the WHERE clause.
    """
    # Extract the WHERE clause content
    where_match = re.search(r'WHERE\s*\{([^}]+)\}', query_text, re.IGNORECASE | re.DOTALL)
    if not where_match:
        # No WHERE clause found, assume complex
        return False
    
    where_content = where_match.group(1).strip()
    
    # Simple queries have exactly one triple pattern
    # Multiple triple patterns are separated by periods (.)
    # Look for periods that separate triples (period followed by variable or URI)
    # Pattern: period, optional whitespace, then variable (?X) or URI (<...>)
    has_multiple_triples = re.search(r'\.\s*(?:\?[A-Za-z]|<[^>]+>)', where_content) is not None
    
    # Simple query: no period separators found (single triple pattern)
    is_simple = not has_multiple_triples
    
    return is_simple


def create_visualization(results: Dict[str, Dict], queries_info: List[Dict[str, str]] = None, output_file: str = "benchmark_results_v4.png"):
    """
    Create bar chart visualizations comparing results across k values and comunica-sparql-file.
    Creates two separate graphs: one for simple queries and one for complex queries.
    """
    if not HAS_MATPLOTLIB:
        print("Skipping visualization (matplotlib not available)")
        return
    
    queries = sorted(results.keys())
    n_queries = len(queries)
    
    if n_queries == 0:
        print("No results to visualize")
        return
    
    # Classify queries as simple or complex
    simple_queries = []
    complex_queries = []
    
    # Create a mapping from query name to query text if provided
    query_text_map = {}
    if queries_info:
        for q_info in queries_info:
            query_text_map[q_info['name']] = q_info['query']
    
    for q in queries:
        query_text = query_text_map.get(q, "")
        if is_simple_query(q, query_text):
            simple_queries.append(q)
        else:
            complex_queries.append(q)
    
    print(f"\nClassifying queries: {len(simple_queries)} simple, {len(complex_queries)} complex")
    print(f"  Simple: {simple_queries}")
    print(f"  Complex: {complex_queries}")
    
    # Create visualization function
    def create_single_plot(query_list, title_suffix, filename_suffix):
        if len(query_list) == 0:
            print(f"Skipping {title_suffix} visualization (no queries)")
            return
        
        fig, ax = plt.subplots(figsize=(max(12, len(query_list) * 2), 6))
        
        # Prepare data
        x = np.arange(len(query_list))
        n_bars = len(K_VALUES) + 1  # k values + comunica-sparql-file
        width = 0.8 / n_bars  # Width of bars to fit all bars in group
        
        # Create bars for each k value
        bars = []
        colors = plt.cm.tab10(np.linspace(0, 1, len(K_VALUES)))
        
        # Calculate bar positions - center all bars around each x position
        for i, k in enumerate(K_VALUES):
            offset = (i - (n_bars - 1) / 2) * width
            pos = x + offset
            values = [results[q].get(f'k_{k}', {}).get('count', 0) for q in query_list]
            bar = ax.bar(pos, values, width, label=f'k={k}', alpha=0.8, color=colors[i])
            bars.append(bar)
        
        # Add comunica-sparql-file bar (last position)
        file_offset = (len(K_VALUES) - (n_bars - 1) / 2) * width
        file_pos = x + file_offset
        file_values = [results[q].get('comunica_sparql_file', {}).get('count', 0) for q in query_list]
        file_bar = ax.bar(file_pos, file_values, width, label='comunica-sparql-file', alpha=0.8, color='red')
        bars.append(file_bar)
        
        # Customize the plot
        ax.set_xlabel('Queries', fontsize=12)
        ax.set_ylabel('Result Count', fontsize=12)
        ax.set_title(f'Query Results Comparison: Vector (k values) vs comunica-sparql-file - {title_suffix} (V4)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(query_list)
        ax.legend(loc='upper left', ncol=2)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars (only if not too many)
        if len(query_list) <= 10:
            for bars_group in bars:
                for bar in bars_group:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}',
                               ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        output_path = output_file.replace('.png', f'_{filename_suffix}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
        plt.close()
    
    # Create both visualizations
    create_single_plot(simple_queries, "Simple Queries", "simple")
    create_single_plot(complex_queries, "Complex Queries", "complex")


def main():
    """Main benchmark execution."""
    print("=" * 70)
    print("Comunica Benchmark Script - V4")
    print("V4: Supports k parameter forwarding from Comunica to vector endpoint")
    print("=" * 70)
    
    # Check if RDF file exists, try alternatives
    rdf_file = RDF_FILE
    if not os.path.exists(rdf_file):
        alternatives = ['RLUBM.nt', 'sample_data.nt', 'test_data.nt']
        for alt in alternatives:
            if os.path.exists(alt):
                rdf_file = alt
                print(f"Using alternative RDF file: {rdf_file}")
                break
        else:
            print(f"Warning: RDF file '{rdf_file}' not found. Benchmark may fail.")
    
    # Extract queries
    print(f"\nStep 1: Extracting queries from {QUERY_FILE}...")
    all_queries = extract_queries(QUERY_FILE, max_results=MAX_RESULT_COUNT)
    
    if not all_queries:
        print("Error: No queries found! Exiting.")
        sys.exit(1)
    
    # Classify queries as simple or complex
    print(f"\nStep 1.5: Classifying queries as simple or complex...")
    simple_queries = []
    complex_queries = []
    
    for query_info in all_queries:
        query_name = query_info['name']
        query_text = query_info['query']
        if is_simple_query(query_name, query_text):
            simple_queries.append(query_info)
        else:
            complex_queries.append(query_info)
    
    print(f"Found {len(simple_queries)} simple queries and {len(complex_queries)} complex queries")
    
    # Select 5 simple and 5 complex queries
    selected_simple = simple_queries[:5]
    selected_complex = complex_queries[:5]
    
    if len(selected_simple) < 5:
        print(f"Warning: Only {len(selected_simple)} simple queries available (requested 5)")
    if len(selected_complex) < 5:
        print(f"Warning: Only {len(selected_complex)} complex queries available (requested 5)")
    
    # Combine selected queries
    queries = selected_simple + selected_complex
    
    print(f"\nSelected queries for benchmarking:")
    print(f"  Simple queries ({len(selected_simple)}): {[q['name'] for q in selected_simple]}")
    print(f"  Complex queries ({len(selected_complex)}): {[q['name'] for q in selected_complex]}")
    
    if not queries:
        print("Error: No queries selected! Exiting.")
        sys.exit(1)
    
    # Run benchmarks
    print(f"\nStep 2: Running benchmarks...")
    results = {}
    
    for query_info in queries:
        query_name = query_info['name']
        query_text = query_info['query']
        
        print(f"\n{'=' * 70}")
        print(f"Query: {query_name}")
        print(f"Query: {query_text[:80]}...")
        print(f"{'=' * 70}")
        
        results[query_name] = {}
        
        # Run with different k values
        for k in K_VALUES:
            count, exec_time = run_comunica_vector(query_text, k)
            results[query_name][f'k_{k}'] = {
                'count': count,
                'time': exec_time
            }
        
        # Run with comunica-sparql-file
        count, exec_time = run_comunica_sparql_file(query_text, rdf_file)
        results[query_name]['comunica_sparql_file'] = {
            'count': count,
            'time': exec_time
        }
    
    # Print summary
    print(f"\n{'=' * 70}")
    print("Benchmark Summary")
    print(f"{'=' * 70}")
    
    for query_name in sorted(results.keys()):
        print(f"\n{query_name}:")
        for k in K_VALUES:
            count = results[query_name][f'k_{k}']['count']
            time = results[query_name][f'k_{k}']['time']
            print(f"  k={k:3d}: {count:5d} results ({time:.2f}s)")
        
        count = results[query_name]['comunica_sparql_file']['count']
        time = results[query_name]['comunica_sparql_file']['time']
        print(f"  comunica-sparql-file: {count:5d} results ({time:.2f}s)")
    
    # Save results to JSON
    output_json = "benchmark_results_v4.json"
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_json}")
    
    # Create visualization
    if HAS_MATPLOTLIB:
        print(f"\nStep 3: Creating visualization...")
        create_visualization(results, queries)
    else:
        print(f"\nStep 3: Skipping visualization (matplotlib not available)")
    
    print(f"\n{'=' * 70}")
    print("Benchmark completed!")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()

