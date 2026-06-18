#!/usr/bin/env python3
"""gRPC client for single-pattern QueryPattern RPC (dim P/R benchmark)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import grpc

from vector_endpoint.grpc_gen.vector.v1 import pattern_pb2_grpc
from vector_endpoint.proto_convert import pattern_query_request_from_json, raw_search_from_proto

_MAX_MESSAGE_BYTES = 128 * 1024 * 1024


def parse_grpc_target(endpoint: str) -> str:
    ep = endpoint.strip()
    if ep.startswith("grpc://"):
        ep = ep[len("grpc://") :]
    if ":" not in ep:
        ep = f"{ep}:50051"
    return ep


def _extract_where_body(query: str) -> str:
    q_upper = query.upper()
    where_idx = q_upper.find("WHERE")
    if where_idx < 0:
        raise ValueError("Query must contain WHERE.")
    open_idx = query.find("{", where_idx)
    close_idx = query.rfind("}")
    if open_idx < 0 or close_idx <= open_idx:
        raise ValueError("Query WHERE clause must include {...}.")
    return query[open_idx + 1 : close_idx].strip()


def _tokenize_pattern(where_body: str) -> List[str]:
    return re.findall(r'(<[^>]+>|"[^"]*"|\?[A-Za-z_][A-Za-z0-9_]*)', where_body)


def _select_vars(query: str) -> List[str]:
    m = re.search(r"SELECT\s+(.*?)\s+WHERE", query, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    return [v.strip().lstrip("?") for v in re.findall(r"\?[A-Za-z_][A-Za-z0-9_]*", m.group(1))]


def _slot_term(token: str) -> Any:
    if token.startswith("?"):
        return token.lstrip("?")
    if token.startswith('"') and token.endswith('"'):
        return {"type": "literal", "value": token[1:-1]}
    return token


@dataclass
class PatternQueryGrpcResult:
    bindings: List[Dict[str, Dict[str, str]]] = field(default_factory=list)
    raw_matches: List[Dict] = field(default_factory=list)
    raw_k_used: Optional[int] = None


@dataclass
class PatternQueryPaginationGrpcResult:
    bindings: List[Dict[str, Dict[str, str]]] = field(default_factory=list)
    raw_matches: List[Dict] = field(default_factory=list)
    pages_fetched: int = 0
    milvus_hits_total: int = 0
    resolved_limit: int = 0
    catalog_k: Optional[int] = None
    page_k: int = 0


def pattern_request_body_from_query(
    query: str,
    *,
    k: Optional[int] = None,
    pagination_limit: Optional[int] = None,
    use_pagination: bool = False,
    adaptive_multipliers: Optional[Sequence[int]] = None,
    adaptive_jaccard: Optional[float] = None,
    include_raw_hits: bool = False,
) -> Dict[str, Any]:
    """Build Comunica-style POST /vector JSON for a single BGP pattern query."""
    body = _extract_where_body(query)
    toks = _tokenize_pattern(body)
    if len(toks) < 3:
        raise ValueError(f"Could not parse triple pattern: {body}")
    vars_list = _select_vars(query)
    req: Dict[str, Any] = {
        "pattern": {
            "subject": _slot_term(toks[0]),
            "predicate": _slot_term(toks[1]),
            "object": _slot_term(toks[2]),
        },
        "vars": vars_list,
        "values": [{}],
    }
    if use_pagination:
        req["k_mode"] = "pagination"
        if k is not None:
            req["k"] = k
        if pagination_limit is not None:
            req["limit"] = pagination_limit
    elif k is not None:
        req["k"] = k
    if adaptive_multipliers is not None:
        req["adaptive_multipliers"] = list(adaptive_multipliers)
    if adaptive_jaccard is not None:
        req["adaptive_jaccard"] = adaptive_jaccard
    if include_raw_hits:
        req["include_raw_hits"] = True
    return req


def _looks_like_uri(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _normalize_literal_value(value: str) -> str:
    raw = value.strip()
    match = re.match(r'^"(.*)"(?:\^\^.+|@[A-Za-z0-9-]+)?$', raw)
    if match:
        return match.group(1)
    return raw


def normalize_binding_row(row: Dict) -> Dict[str, Dict[str, str]]:
    normalized: Dict[str, Dict[str, str]] = {}
    for key, raw in row.items():
        if isinstance(raw, dict) and "value" in raw:
            vtype = str(raw.get("type", "literal"))
            value = str(raw.get("value", ""))
            if vtype != "uri" and vtype != "iri":
                value = _normalize_literal_value(value)
            normalized[key] = {"type": vtype, "value": value}
            continue

        value = str(raw)
        vtype = "uri" if _looks_like_uri(value) else "literal"
        if vtype == "literal":
            value = _normalize_literal_value(value)
        normalized[key] = {"type": vtype, "value": value}
    return normalized


def _binding_row_from_proto(row: Any) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for var, term in row.bindings.items():
        ttype = term.type or "literal"
        if ttype == "iri":
            ttype = "uri"
        out[var] = {"type": ttype, "value": term.value}
    return normalize_binding_row(out)


def query_pattern_grpc(
    endpoint: str,
    body: Dict[str, Any],
    *,
    timeout: Optional[float] = None,
) -> PatternQueryGrpcResult:
    """Run QueryPattern RPC; return post-filter bindings and optional raw Milvus hits."""
    target = parse_grpc_target(endpoint)
    channel = grpc.insecure_channel(
        target,
        options=[
            ("grpc.max_send_message_length", _MAX_MESSAGE_BYTES),
            ("grpc.max_receive_message_length", _MAX_MESSAGE_BYTES),
        ],
    )
    stub = pattern_pb2_grpc.VectorPatternServiceStub(channel)
    request = pattern_query_request_from_json(body)
    result = PatternQueryGrpcResult()
    try:
        for event in stub.QueryPattern(request, timeout=timeout):
            if event.HasField("error"):
                raise RuntimeError(event.error.message or "gRPC pattern query failed")
            if event.HasField("raw_search"):
                raw = raw_search_from_proto(event.raw_search)
                result.raw_matches = list(raw.hits)
                result.raw_k_used = raw.k_used
                continue
            if event.HasField("row"):
                result.bindings.append(_binding_row_from_proto(event.row))
            if event.HasField("row_batch"):
                for row in event.row_batch.rows:
                    result.bindings.append(_binding_row_from_proto(row))
    finally:
        channel.close()
    return result


def query_pattern_pagination_grpc(
    endpoint: str,
    body: Dict[str, Any],
    *,
    timeout: Optional[float] = None,
) -> PatternQueryPaginationGrpcResult:
    """Run QueryPatternPage RPC until pagination.done."""
    from vector_endpoint.grpc_gen.vector.v1 import pattern_pb2
    from vector_endpoint.proto_convert import pattern_page_request_from_json

    target = parse_grpc_target(endpoint)
    channel = grpc.insecure_channel(
        target,
        options=[
            ("grpc.max_send_message_length", _MAX_MESSAGE_BYTES),
            ("grpc.max_receive_message_length", _MAX_MESSAGE_BYTES),
        ],
    )
    stub = pattern_pb2_grpc.VectorPatternServiceStub(channel)
    result = PatternQueryPaginationGrpcResult()
    request = pattern_page_request_from_json(body)
    try:
        while True:
            response = stub.QueryPatternPage(request, timeout=timeout)
            if response.HasField("error"):
                raise RuntimeError(response.error.message or "gRPC pagination query failed")
            for row in response.rows:
                result.bindings.append(_binding_row_from_proto(row))
            if response.HasField("raw_search"):
                for hit in response.raw_search.hits:
                    entry: Dict[str, Any] = {"text": hit.text}
                    if hit.HasField("id"):
                        entry["id"] = hit.id
                    if hit.HasField("distance"):
                        entry["distance"] = hit.distance
                    result.raw_matches.append(entry)
            if not response.HasField("pagination"):
                break
            pag = response.pagination
            result.pages_fetched = pag.page_index
            result.milvus_hits_total = pag.milvus_hits_total
            result.resolved_limit = pag.limit
            result.page_k = pag.k
            if pag.HasField("catalog_k"):
                result.catalog_k = pag.catalog_k
            if pag.done or not pag.HasField("cursor"):
                break
            request = pattern_pb2.PatternPageRequest(cursor=pag.cursor)
    finally:
        channel.close()
    return result
