"""Convert between JSON-like dicts and protobuf PatternQuery messages."""

from __future__ import annotations

from typing import Any

from vector_endpoint.grpc_gen.vector.v1 import pattern_pb2
from vector_endpoint.pattern_query import PatternQueryInput


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
    return req
