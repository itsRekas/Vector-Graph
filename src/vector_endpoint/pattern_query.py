"""Pattern BGP execution for HTTP and gRPC endpoints."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from vector_endpoint.adaptive_exp import filter_matches_to_rows, iter_adaptive_batch_search
from vector_endpoint.auto_k import CatalogKResolver, milvus_safe_k, _normalize_literal
from vector_endpoint.bgp_log import (
    bgp_emit,
    bgp_log_enabled,
    log_grpc_pattern_received,
    log_http_pattern_received,
)
from vector_endpoint.db.VectorDataBase import VectorDataBase
from vector_endpoint.rdf_utils import parse_rdf_triple
from vector_endpoint.pagination_sessions import resolve_session_id
def _collection_name() -> str:
    return os.getenv("VECTOR_COLLECTION", "version_5")


def _default_resolver() -> CatalogKResolver:
    from vector_endpoint.server_state import AUTO_K_RESOLVER

    return AUTO_K_RESOLVER


def _default_vdb() -> VectorDataBase | None:
    from vector_endpoint.server_state import vdb

    return vdb


def adaptive_multipliers_from_request(json_data: dict) -> tuple[int, ...]:
    raw = json_data.get("adaptive_multipliers")
    if raw is None:
        raw = os.getenv("VECTOR_ADAPTIVE_MULTIPLIERS", "1,2,10,100")
    if isinstance(raw, (list, tuple)):
        values = [int(m) for m in raw if int(m) > 0]
    else:
        values = [int(tok.strip()) for tok in str(raw).split(",") if tok.strip()]
    if not values:
        return (1, 10, 100, 1000)
    return tuple(values)


def adaptive_jaccard_from_request(json_data: dict) -> float:
    raw = json_data.get("adaptive_jaccard")
    if raw is None:
        raw = os.getenv("VECTOR_ADAPTIVE_JACCARD", "0.99")
    return float(raw)


def join_bound_min_k() -> int:
    return max(1, int(os.getenv("VECTOR_JOIN_BOUND_MIN_K", "512")))


def search_query_has_bound_constant(sq: dict) -> bool:
    val = sq.get("subject")
    if isinstance(val, str) and val and not val.startswith("?"):
        return True
    val = sq.get("object")
    if isinstance(val, str) and val and not val.startswith("?"):
        return True
    return False


def bump_seed_for_join_extension(seed: int, sq: dict, *, values: list[dict]) -> int:
    return seed
    # if not values:
    #     return seed
    # if search_query_has_bound_constant(sq):
    #     return max(seed, join_bound_min_k())
    # return seed


def _bgp_pattern_context(
    search_queries: list[dict],
    values: list[dict],
    resolver: CatalogKResolver,
) -> tuple[str, str, str, str, str]:
    sq = search_queries[0] if search_queries else {}
    expected: Optional[int] = None
    if resolver.available:
        expected = resolver.catalog_match_count(
            subject=sq.get("subject"),
            predicate=sq.get("predicate"),
            object_value=sq.get("object"),
            object_type=sq.get("object_type"),
        )
    pred = sq.get("predicate") or "?"
    subj = sq.get("subject") or "?"
    obj = sq.get("object") or "?"
    exp_s = str(expected) if expected is not None else "n/a"
    return subj, pred, obj, exp_s, str(len(values))


def log_bgp_start(
    *,
    search_queries: list[dict],
    values: list[dict],
    k: Optional[int],
    resolver: CatalogKResolver,
) -> None:
    if not bgp_log_enabled():
        return
    subj, pred, obj, exp_s, _ = _bgp_pattern_context(search_queries, values, resolver)
    k_s = str(k) if k is not None else "adaptive"
    bgp_emit(
        f"[BGP] start values_in={len(values)} expected={exp_s} k={k_s} "
        f"pattern=({subj}, {pred}, {obj})"
    )


def log_bgp_progress(message: str) -> None:
    bgp_emit(f"[BGP] {message}")


def log_bgp_fetch(
    *,
    search_queries: list[dict],
    values: list[dict],
    returned_count: int,
    k: Optional[int],
    resolver: CatalogKResolver,
) -> None:
    if not bgp_log_enabled():
        return
    subj, pred, obj, exp_s, _ = _bgp_pattern_context(search_queries, values, resolver)
    k_s = str(k) if k is not None else "adaptive"
    bgp_emit(
        f"[BGP] done values_in={len(values)} expected={exp_s} returned={returned_count} "
        f"k={k_s} pattern=({subj}, {pred}, {obj})"
    )


def resolve_k_mode(json_data: dict, query_input: Optional["PatternQueryInput"] = None) -> str:
    if resolve_session_id(json_data):
        return "pagination"
    if query_input and query_input.session:
        return "pagination"
    raw = json_data.get("k_mode") or (query_input.k_mode if query_input else None)
    if raw in ("fixed", "adaptive", "pagination"):
        return str(raw)
    if json_data.get("k_mode") == "pagination" or (query_input and query_input.k_mode == "pagination"):
        return "pagination"
    if json_data.get("k") is not None or (query_input and query_input.k is not None):
        if raw == "pagination":
            return "pagination"
        return "fixed"
    return "adaptive"


@dataclass
class PatternQueryInput:
    pattern: dict
    vars: list[str] = field(default_factory=list)
    values: list[dict] = field(default_factory=list)
    k: Optional[int] = None
    adaptive_multipliers: tuple[int, ...] = (1, 10, 100, 1000)
    adaptive_jaccard: float = 0.99
    include_raw_hits: bool = False
    k_mode: Optional[str] = None
    pagination_limit: Optional[int] = None
    session: Optional[str] = None
    cursor: Optional[str] = None
    page: Optional[int] = None
    advance: bool = False
    cancel: bool = False

    @classmethod
    def from_json(cls, json_data: dict) -> PatternQueryInput:
        k_mode_raw = json_data.get("k_mode")
        k_mode = str(k_mode_raw) if k_mode_raw in ("fixed", "adaptive", "pagination") else None
        explicit_k: Optional[int] = None
        pagination_limit: Optional[int] = None
        if k_mode == "pagination":
            if json_data.get("k") is not None:
                try:
                    explicit_k = int(json_data["k"])
                except (ValueError, TypeError):
                    explicit_k = None
            if json_data.get("limit") is not None:
                try:
                    pagination_limit = int(json_data["limit"])
                except (ValueError, TypeError):
                    pagination_limit = None
        else:
            requested = json_data.get("k") or json_data.get("limit")
            if requested is not None:
                try:
                    explicit_k = int(requested)
                except (ValueError, TypeError):
                    explicit_k = None
        session_val = json_data.get("session") or json_data.get("cursor")
        session = str(session_val) if session_val else None
        page: Optional[int] = None
        if json_data.get("page") is not None:
            try:
                page = int(json_data["page"])
            except (ValueError, TypeError):
                page = None
        return cls(
            pattern=json_data.get("pattern", {}),
            vars=list(json_data.get("vars", [])),
            values=list(json_data.get("values", [])),
            k=explicit_k,
            adaptive_multipliers=adaptive_multipliers_from_request(json_data),
            adaptive_jaccard=adaptive_jaccard_from_request(json_data),
            include_raw_hits=bool(json_data.get("include_raw_hits", False)),
            k_mode=k_mode,
            pagination_limit=pagination_limit,
            session=session,
            cursor=session,
            page=page,
            advance=bool(json_data.get("next", False)),
            cancel=bool(json_data.get("cancel", False)),
        )


@dataclass
class RawSearchResult:
    value_row_index: int
    k_used: int
    hits: list[dict]


@dataclass
class PatternStreamEvent:
    row: Optional[dict] = None
    raw_search: Optional[RawSearchResult] = None


@dataclass
class BgpExecutionState:
    variables: list[str]
    values: list[dict]
    search_queries: list[dict]
    validation_info_list: list[dict]
    subject: Any
    predicate: Any
    obj: Any
    pattern_variable_roles: dict[str, str]


def build_bgp_execution_state(query_input: PatternQueryInput) -> BgpExecutionState:
    pattern = query_input.pattern
    variables = query_input.vars
    values = list(query_input.values) if query_input.values else [{}]
    subject = pattern.get("subject")
    predicate = pattern.get("predicate")
    obj = pattern.get("object")

    variable_aliases = set()
    for var in variables:
        if isinstance(var, str):
            variable_aliases.add(var.lstrip("?"))

    def analyze_term(term):
        value = None
        term_type = None
        is_variable = False
        if isinstance(term, dict):
            term_type = term.get("type") or term.get("termType")
            if term_type in ["iri", "uri", "literal"]:
                value = term.get("value")
            elif term_type in ["variable", "Variable"]:
                raw = term.get("value", "")
                value = raw.lstrip("?")
                is_variable = True
        elif isinstance(term, str):
            stripped = term.lstrip("?")
            if term.startswith("?") or stripped in variable_aliases:
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
        if term_type in ["iri", "uri"]:
            return value if value.startswith("<") and value.endswith(">") else f"<{value}>"
        if term_type == "literal":
            return _normalize_literal(value)
        return value

    def extract_term_value(term_json):
        if term_json is None:
            return None
        if isinstance(term_json, dict):
            return term_json.get("value")
        if isinstance(term_json, str):
            return term_json
        return None

    def normalize_variable(var_value):
        if isinstance(var_value, str):
            return var_value.lstrip("?")
        if isinstance(var_value, dict):
            if var_value.get("termType") == "Variable":
                return var_value.get("value", "").lstrip("?")
            if var_value.get("type") == "variable":
                return var_value.get("value", "").lstrip("?")
        return None

    pattern_variable_roles = {}
    subject_var = normalize_variable(subject)
    if subject_var:
        pattern_variable_roles[subject_var] = "subject"
    predicate_var = normalize_variable(predicate)
    if predicate_var:
        pattern_variable_roles[predicate_var] = "predicate"
    object_var = normalize_variable(obj)
    if object_var:
        pattern_variable_roles[object_var] = "object"

    def build_search_query_for_row(value_row):
        subj_value_raw, subj_type, subj_is_var = analyze_term(subject)
        pred_value_raw, pred_type, pred_is_var = analyze_term(predicate)
        obj_value_raw, obj_type, obj_is_var = analyze_term(obj)

        if subj_is_var and subject_var and subject_var in value_row:
            bound_value = extract_term_value(value_row[subject_var])
            if bound_value:
                subject_value = format_constant(bound_value, "iri")
                subj_is_var = False
            else:
                subject_value = None
        else:
            subject_value = None if subj_is_var else format_constant(subj_value_raw, subj_type)

        if pred_is_var and predicate_var and predicate_var in value_row:
            bound_value = extract_term_value(value_row[predicate_var])
            if bound_value:
                predicate_value = format_constant(bound_value, "iri")
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
                    obj_type_from_value = term_json.get("type", "iri")
                else:
                    obj_type_from_value = "iri"
                object_value = format_constant(bound_value, obj_type_from_value)
                object_type = "literal" if obj_type_from_value == "literal" else "uri"
                obj_is_var = False
            else:
                object_value = None
                object_type = None
        else:
            object_value = None if obj_is_var else format_constant(obj_value_raw, obj_type)
            object_type = None
            if not obj_is_var and obj_type == "literal":
                object_type = "literal"
            elif not obj_is_var and obj_type in ["iri", "uri"]:
                object_type = "uri"

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
            object_type,
        )
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

    search_queries: list[dict] = []
    validation_info_list: list[dict] = []
    for value_row in values:
        search_query, validation_info = build_search_query_for_row(value_row)
        search_queries.append(search_query)
        validation_info_list.append(validation_info)

    return BgpExecutionState(
        variables=variables,
        values=values,
        search_queries=search_queries,
        validation_info_list=validation_info_list,
        subject=subject,
        predicate=predicate,
        obj=obj,
        pattern_variable_roles=pattern_variable_roles,
    )


def make_bgp_filter_fn(state: BgpExecutionState) -> Callable[[list[dict], int], tuple[list[dict], set[int]]]:
    def filter_fn(matches, query_idx):
        vinfo = (
            state.validation_info_list[query_idx]
            if query_idx < len(state.validation_info_list)
            else {}
        )
        return filter_matches_to_rows(
            matches,
            pattern_subject=state.subject,
            pattern_predicate=state.predicate,
            pattern_object=state.obj,
            validation_info=vinfo,
            variables=state.variables,
            pattern_variable_roles=state.pattern_variable_roles,
            value_row=state.values[query_idx],
            parse_rdf_triple=parse_rdf_triple,
            log=False,
        )

    return filter_fn


def stream_pattern_rows(
    query_input: PatternQueryInput,
    **kwargs: Any,
) -> Iterator[dict]:
    """Yield post-filter binding rows."""
    for event in stream_pattern_events(query_input, **kwargs):
        if event.row is not None:
            yield event.row


def stream_pattern_events(
    query_input: PatternQueryInput,
    *,
    collection_name: str | None = None,
    database: VectorDataBase | None = None,
    resolver: CatalogKResolver | None = None,
) -> Iterator[PatternStreamEvent]:
    """Yield binding rows and optional raw Milvus hits per value row."""
    if collection_name is None:
        collection_name = _collection_name()
    if database is None:
        database = _default_vdb()
    if resolver is None:
        resolver = _default_resolver()
    bgp = build_bgp_execution_state(query_input)
    variables = bgp.variables
    values = bgp.values
    search_queries = bgp.search_queries
    subject = bgp.subject
    predicate = bgp.predicate
    obj = bgp.obj

    explicit_k = query_input.k
    returned_count = 0
    log_bgp_start(
        search_queries=search_queries,
        values=values,
        k=explicit_k,
        resolver=resolver,
    )

    if database is None or len(search_queries) == 0:
        log_bgp_fetch(
            search_queries=search_queries,
            values=values,
            returned_count=0,
            k=explicit_k,
            resolver=resolver,
        )
        return

    filter_fn = make_bgp_filter_fn(bgp)

    def emit_rows(rows: list[dict]) -> Iterator[PatternStreamEvent]:
        nonlocal returned_count
        for row in rows:
            returned_count += 1
            yield PatternStreamEvent(row=row)

    def emit_raw(value_row_index: int, k_used: int, matches: list[dict]) -> Iterator[PatternStreamEvent]:
        if not query_input.include_raw_hits:
            return
        yield PatternStreamEvent(
            raw_search=RawSearchResult(
                value_row_index=value_row_index,
                k_used=k_used,
                hits=list(matches),
            )
        )

    try:
        if explicit_k is not None:
            effective_search_limit = milvus_safe_k(explicit_k)
            log_bgp_progress(
                f"search k={effective_search_limit} queries={len(search_queries)}"
            )
            vector_results = database.search(
                collection_name=collection_name,
                query_texts=search_queries,
                limit=effective_search_limit,
                output_fields=["text"],
                log=False,
            )
            for value_row_idx, _value_row in enumerate(values):
                if value_row_idx < len(vector_results):
                    matches = vector_results[value_row_idx].get("matches", [])
                    yield from emit_raw(value_row_idx, effective_search_limit, matches)
                    rows, _ids = filter_fn(matches, value_row_idx)
                    yield from emit_rows(rows)
        else:
            seed_ks: list[int] = []
            stability_count_floors: list[Optional[int]] = []
            for sq in search_queries:
                seed = None
                floor: Optional[int] = None
                if resolver.available:
                    floor = resolver.catalog_match_count(
                        subject=sq.get("subject"),
                        predicate=sq.get("predicate"),
                        object_value=sq.get("object"),
                        object_type=sq.get("object_type"),
                    )
                    seed = resolver.auto_k_for_pattern(
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
                seed = bump_seed_for_join_extension(seed, sq, values=values)
                seed_ks.append(seed)
                stability_count_floors.append(floor)

            bgp_log = bgp_log_enabled()
            if bgp_log:
                mult_s = ",".join(str(m) for m in query_input.adaptive_multipliers)
                log_bgp_progress(
                    f"adaptive queries={len(search_queries)} "
                    f"seed_k min={min(seed_ks)} max={max(seed_ks)} "
                    f"multipliers={mult_s} jaccard={query_input.adaptive_jaccard}"
                )

            raw_emitted: set[int] = set()

            # Finalization order, not value-row index order.
            pending_raw: list[tuple[int, int, list[dict]]] = []

            def _capture_final_round(query_idx: int, k_used: int, matches: list[dict]) -> None:
                if query_idx in raw_emitted:
                    return
                raw_emitted.add(query_idx)
                pending_raw.append((query_idx, k_used, matches))

            for query_idx, rows in iter_adaptive_batch_search(
                vdb=database,
                collection_name=collection_name,
                search_queries=search_queries,
                seed_ks=seed_ks,
                filter_fn=filter_fn,
                multipliers=query_input.adaptive_multipliers,
                jaccard_threshold=query_input.adaptive_jaccard,
                log=bgp_log,
                stability_count_floors=stability_count_floors,
                on_final_round=_capture_final_round,
            ):
                for idx, k_used, matches in pending_raw:
                    yield from emit_raw(idx, k_used, matches)
                pending_raw.clear()
                yield from emit_rows(rows)

    except Exception as e:  # noqa: BLE001
        print(f"Error during batch vector search: {e}")
        import traceback

        traceback.print_exc()
        raise

    log_bgp_fetch(
        search_queries=search_queries,
        values=values,
        returned_count=returned_count,
        k=explicit_k,
        resolver=resolver,
    )


def collect_pattern_rows(
    query_input: PatternQueryInput | dict,
    **kwargs: Any,
) -> tuple[list[str], list[dict]]:
    """Return all BGP binding rows as a list."""
    if isinstance(query_input, dict):
        query_input = PatternQueryInput.from_json(query_input)
    rows = list(stream_pattern_rows(query_input, **kwargs))
    return query_input.vars, rows
