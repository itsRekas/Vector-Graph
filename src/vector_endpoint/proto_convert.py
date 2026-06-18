"""Convert between JSON-like dicts and protobuf PatternQuery messages."""

from __future__ import annotations

from typing import Any

from vector_endpoint.grpc_gen.vector.v1 import pattern_pb2
from vector_endpoint.pagination_search import PaginationMeta, PaginationPageResult
from vector_endpoint.pattern_query import PatternQueryInput, RawSearchResult


def _looks_like_iri(value: str) -> bool:
    return value.startswith("<") or "://" in value or value.startswith("http:")


def _term_from_proto(
    term: pattern_pb2.Term,
    variable_names: frozenset[str] | None = None,
) -> dict | str | None:
    if not term.type and not term.value:
        return None
    if term.type in ("variable", "Variable") or (
        term.value and term.value.startswith("?")
    ):
        return term.value.lstrip("?")
    name = term.value.lstrip("?")
    if (
        variable_names
        and name in variable_names
        and term.type in ("iri", "uri")
        and not _looks_like_iri(term.value)
    ):
        return name
    out: dict[str, str] = {"type": term.type or "iri", "value": term.value}
    if term.lang:
        out["lang"] = term.lang
    if term.datatype:
        out["datatype"] = term.datatype
    return out


def _term_to_proto(term: Any) -> pattern_pb2.Term:
    msg = pattern_pb2.Term()
    if term is None:
        return msg
    if isinstance(term, str):
        if term.startswith("?"):
            msg.type = "variable"
            msg.value = term.lstrip("?")
        else:
            msg.type = "iri"
            msg.value = term
        return msg
    if isinstance(term, dict):
        msg.type = term.get("type") or term.get("termType") or "iri"
        msg.value = term.get("value", "")
        if term.get("lang"):
            msg.lang = term["lang"]
        if term.get("datatype"):
            msg.datatype = term["datatype"]
    return msg


def pattern_query_input_from_proto(request: pattern_pb2.PatternQueryRequest) -> PatternQueryInput:
    var_names = frozenset(v.lstrip("?") for v in request.vars)
    pattern_dict: dict[str, Any] = {}
    if request.HasField("pattern") or request.pattern.ByteSize() > 0:
        p = request.pattern
        pattern_dict = {
            "subject": _term_from_proto(p.subject, var_names),
            "predicate": _term_from_proto(p.predicate, var_names),
            "object": _term_from_proto(p.object, var_names),
        }
    values: list[dict] = []
    for vrow in request.values:
        values.append({
            k: _term_from_proto(t, var_names) for k, t in vrow.bindings.items()
        })
    json_data: dict[str, Any] = {
        "pattern": pattern_dict,
        "vars": list(request.vars),
        "values": values,
    }
    if request.HasField("k"):
        json_data["k"] = request.k
    if request.adaptive_multipliers:
        json_data["adaptive_multipliers"] = list(request.adaptive_multipliers)
    if request.HasField("adaptive_jaccard"):
        json_data["adaptive_jaccard"] = request.adaptive_jaccard
    if request.HasField("include_raw_hits"):
        json_data["include_raw_hits"] = request.include_raw_hits
    return PatternQueryInput.from_json(json_data)


def row_to_proto(row: dict) -> pattern_pb2.BindingRow:
    binding_row = pattern_pb2.BindingRow()
    for var, term in row.items():
        binding_row.bindings[var.lstrip("?") if isinstance(var, str) else var].CopyFrom(
            _term_to_proto(term)
        )
    return binding_row


def pattern_query_request_from_json(json_data: dict) -> pattern_pb2.PatternQueryRequest:
    """Build protobuf request from Comunica-style JSON body."""
    inp = PatternQueryInput.from_json(json_data)
    req = pattern_pb2.PatternQueryRequest()
    req.vars.extend(inp.vars)
    p = inp.pattern
    req.pattern.subject.CopyFrom(_term_to_proto(p.get("subject")))
    req.pattern.predicate.CopyFrom(_term_to_proto(p.get("predicate")))
    req.pattern.object.CopyFrom(_term_to_proto(p.get("object")))
    for value_row in inp.values:
        vrow = pattern_pb2.ValueRow()
        for k, t in value_row.items():
            vrow.bindings[k].CopyFrom(_term_to_proto(t))
        req.values.append(vrow)
    if inp.k is not None:
        req.k = inp.k
    req.adaptive_multipliers.extend(inp.adaptive_multipliers)
    req.adaptive_jaccard = inp.adaptive_jaccard
    if inp.include_raw_hits:
        req.include_raw_hits = True
    return req


def raw_search_to_proto(raw: RawSearchResult) -> pattern_pb2.RawSearchResult:
    msg = pattern_pb2.RawSearchResult()
    msg.value_row_index = raw.value_row_index
    msg.k_used = raw.k_used
    for hit in raw.hits:
        entry = msg.hits.add()
        if hit.get("id") is not None:
            entry.id = int(hit["id"])
        distance = hit.get("distance")
        if distance is not None:
            entry.distance = float(distance)
        entry.text = str(hit.get("text", ""))
    return msg


def pattern_page_request_to_json(request: pattern_pb2.PatternPageRequest) -> dict[str, Any]:
    var_names = frozenset(v.lstrip("?") for v in request.vars)
    json_data: dict[str, Any] = {}
    if request.HasField("k_mode"):
        json_data["k_mode"] = request.k_mode
    if request.HasField("k"):
        json_data["k"] = request.k
    if request.HasField("limit"):
        json_data["limit"] = request.limit
    if request.HasField("session"):
        json_data["session"] = request.session
    elif request.HasField("cursor"):
        json_data["cursor"] = request.cursor
    if request.HasField("cancel"):
        json_data["cancel"] = request.cancel
    if request.HasField("next"):
        json_data["next"] = request.next
    if request.HasField("page"):
        json_data["page"] = request.page
    if request.HasField("include_raw_hits"):
        json_data["include_raw_hits"] = request.include_raw_hits
    if request.HasField("pattern") or request.pattern.ByteSize() > 0:
        p = request.pattern
        json_data["pattern"] = {
            "subject": _term_from_proto(p.subject, var_names),
            "predicate": _term_from_proto(p.predicate, var_names),
            "object": _term_from_proto(p.object, var_names),
        }
    if request.vars:
        json_data["vars"] = list(request.vars)
    values: list[dict] = []
    for vrow in request.values:
        values.append({
            k: _term_from_proto(t, var_names) for k, t in vrow.bindings.items()
        })
    if values:
        json_data["values"] = values
    return json_data


def pattern_page_request_from_json(json_data: dict) -> pattern_pb2.PatternPageRequest:
    inp = PatternQueryInput.from_json(json_data)
    req = pattern_pb2.PatternPageRequest()
    if inp.k_mode:
        req.k_mode = inp.k_mode
    if inp.k is not None:
        req.k = inp.k
    if inp.pagination_limit is not None:
        req.limit = inp.pagination_limit
    if inp.session:
        req.session = inp.session
    elif inp.cursor:
        req.cursor = inp.cursor
    if inp.advance:
        req.next = True
    if inp.page is not None:
        req.page = inp.page
    if inp.cancel:
        req.cancel = True
    if inp.include_raw_hits:
        req.include_raw_hits = True
    req.vars.extend(inp.vars)
    p = inp.pattern
    req.pattern.subject.CopyFrom(_term_to_proto(p.get("subject")))
    req.pattern.predicate.CopyFrom(_term_to_proto(p.get("predicate")))
    req.pattern.object.CopyFrom(_term_to_proto(p.get("object")))
    for value_row in inp.values:
        vrow = pattern_pb2.ValueRow()
        for k, t in value_row.items():
            vrow.bindings[k].CopyFrom(_term_to_proto(t))
        req.values.append(vrow)
    return req


def pagination_meta_to_proto(meta: PaginationMeta) -> pattern_pb2.PaginationInfo:
    msg = pattern_pb2.PaginationInfo()
    if meta.cursor is not None:
        msg.session = meta.cursor
    msg.done = meta.done
    msg.k = meta.k
    msg.limit = meta.limit
    if meta.catalog_k is not None:
        msg.catalog_k = meta.catalog_k
    msg.page_index = meta.page_index
    msg.milvus_hits_this_page = meta.milvus_hits_this_page
    msg.milvus_hits_total = meta.milvus_hits_total
    msg.value_row_index = meta.value_row_index
    msg.k_mode = meta.k_mode
    if meta.from_cache:
        msg.from_cache = True
    return msg


def pattern_page_result_to_proto(result: PaginationPageResult) -> pattern_pb2.PatternPageResponse:
    resp = pattern_pb2.PatternPageResponse()
    resp.vars.extend(result.vars)
    for row in result.rows:
        resp.rows.append(row_to_proto(row))
    resp.pagination.CopyFrom(pagination_meta_to_proto(result.pagination))
    if result.raw_hits is not None:
        raw = pattern_pb2.RawSearchResult()
        raw.value_row_index = result.pagination.value_row_index
        raw.k_used = result.pagination.k
        for hit in result.raw_hits:
            entry = raw.hits.add()
            if hit.get("id") is not None:
                entry.id = int(hit["id"])
            distance = hit.get("distance")
            if distance is not None:
                entry.distance = float(distance)
            entry.text = str(hit.get("text", ""))
        resp.raw_search.CopyFrom(raw)
    return resp


def raw_search_from_proto(msg: pattern_pb2.RawSearchResult) -> RawSearchResult:
    hits: list[dict] = []
    for hit in msg.hits:
        entry: dict = {"text": hit.text}
        if hit.HasField("id"):
            entry["id"] = hit.id
        if hit.HasField("distance"):
            entry["distance"] = hit.distance
        hits.append(entry)
    return RawSearchResult(
        value_row_index=msg.value_row_index,
        k_used=msg.k_used,
        hits=hits,
    )
