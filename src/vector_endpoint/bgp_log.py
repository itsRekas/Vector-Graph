"""Thread-safe BGP / gRPC request logging (stderr, same stream as Werkzeug)."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from typing import Any, Optional

_logger = logging.getLogger("vector_endpoint.bgp")
_log_lock = threading.Lock()
_configured = False


def configure_bgp_logging() -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.handlers.clear()
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False
    _configured = True


def bgp_log_enabled() -> bool:
    return os.getenv("VECTOR_BGP_LOG", "1") != "0"


def bgp_verbose_request_log() -> bool:
    return bool(os.getenv("VECTOR_GRPC_DEBUG") or os.getenv("FLASK_DEBUG"))


def bgp_emit(message: str) -> None:
    if not bgp_log_enabled():
        return
    configure_bgp_logging()
    with _log_lock:
        _logger.info(message)


def format_pattern_term(term: Any) -> str:
    if term is None:
        return "?"
    if isinstance(term, str):
        return term if term.startswith("?") else term
    if isinstance(term, dict):
        val = term.get("value", "")
        if term.get("type") in ("variable", "Variable") or (
            isinstance(val, str) and val.startswith("?")
        ):
            return f"?{str(val).lstrip('?')}"
        return str(val) if val else "?"
    return str(term)


def log_pattern_received(
    *,
    transport: str,
    pattern: dict,
    vars: list[str],
    values: list[dict],
    k: Optional[int],
    adaptive_multipliers: tuple[int, ...] = (),
    adaptive_jaccard: float = 0.99,
) -> None:
    values_in = len(values) if values else 1
    k_s = str(k) if k is not None else "adaptive"
    subj = format_pattern_term(pattern.get("subject"))
    pred = format_pattern_term(pattern.get("predicate"))
    obj = format_pattern_term(pattern.get("object"))
    bgp_emit(
        f"[{transport}] pattern received values_in={values_in} vars={vars} "
        f"k={k_s} pattern=({subj}, {pred}, {obj})"
    )
    if bgp_verbose_request_log():
        body = {
            "pattern": pattern,
            "vars": vars,
            "values": values,
            "k": k,
            "adaptive_multipliers": list(adaptive_multipliers),
            "adaptive_jaccard": adaptive_jaccard,
        }
        bgp_emit(f"[{transport}] body: {json.dumps(body, default=str)}")


def log_grpc_rpc_received(*, vars: list[str], value_rows: int) -> None:
    bgp_emit(f"[gRPC] QueryPattern RPC vars={vars} value_rows={value_rows}")


def log_grpc_pattern_received(query_input: Any) -> None:
    log_pattern_received(
        transport="gRPC",
        pattern=query_input.pattern,
        vars=list(query_input.vars),
        values=list(query_input.values),
        k=query_input.k,
        adaptive_multipliers=query_input.adaptive_multipliers,
        adaptive_jaccard=query_input.adaptive_jaccard,
    )


def log_http_pattern_received(query_input: Any) -> None:
    log_pattern_received(
        transport="HTTP",
        pattern=query_input.pattern,
        vars=list(query_input.vars),
        values=list(query_input.values),
        k=query_input.k,
        adaptive_multipliers=query_input.adaptive_multipliers,
        adaptive_jaccard=query_input.adaptive_jaccard,
    )
