"""Milvus search-iterator pagination."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from vector_endpoint.adaptive_exp import filter_matches_to_rows
from vector_endpoint.auto_k import CatalogKResolver, milvus_safe_k, resolve_pagination_limit
from vector_endpoint.bgp_log import bgp_emit, bgp_log_enabled
from vector_endpoint.db.VectorDataBase import SearchIteratorHandle, VectorDataBase
from vector_endpoint.pagination_sessions import (
    PAGINATION_SESSION_STORE,
    PaginationPageNotCached,
    PaginationSession,
    PaginationSessionGone,
    PaginationSessionNotFound,
    RowFilterContext,
    new_cursor,
)
from vector_endpoint.pattern_query import PatternQueryInput, build_bgp_execution_state
from vector_endpoint.rdf_utils import parse_rdf_triple
def _collection_name() -> str:
    return os.getenv("VECTOR_COLLECTION", "version_5")


def _default_resolver() -> CatalogKResolver:
    from vector_endpoint.server_state import AUTO_K_RESOLVER

    return AUTO_K_RESOLVER


def _default_vdb() -> VectorDataBase | None:
    from vector_endpoint.server_state import vdb

    return vdb


@dataclass
class PaginationMeta:
    cursor: Optional[str]
    done: bool
    k: int
    limit: int
    catalog_k: Optional[int]
    page_index: int
    milvus_hits_this_page: int
    milvus_hits_total: int
    value_row_index: int
    k_mode: str = "pagination"
    from_cache: bool = False

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "done": self.done,
            "k": self.k,
            "limit": self.limit,
            "page_index": self.page_index,
            "milvus_hits_this_page": self.milvus_hits_this_page,
            "milvus_hits_total": self.milvus_hits_total,
            "value_row_index": self.value_row_index,
            "k_mode": self.k_mode,
        }
        if self.catalog_k is not None:
            out["catalog_k"] = self.catalog_k
        if self.cursor is not None:
            out["session"] = self.cursor
        if self.from_cache:
            out["from_cache"] = True
        return out


@dataclass
class PaginationPageResult:
    vars: list[str]
    rows: list[dict]
    pagination: PaginationMeta
    raw_hits: Optional[list[dict]] = None


def _catalog_k_for_query(
    sq: dict,
    resolver: CatalogKResolver,
) -> Optional[int]:
    if not resolver.available:
        return None
    return resolver.auto_k_for_pattern(
        subject=sq.get("subject"),
        predicate=sq.get("predicate"),
        object_value=sq.get("object"),
        object_type=sq.get("object_type"),
    )


def _open_iterators_for_state(
    *,
    database: VectorDataBase,
    collection_name: str,
    bgp_state,
    k: int,
    limit: int,
) -> list[Optional[SearchIteratorHandle]]:
    handles: list[Optional[SearchIteratorHandle]] = []
    for sq in bgp_state.search_queries:
        handles.append(
            database.open_search_iterator(
                collection_name,
                sq,
                batch_size=k,
                limit=limit,
                output_fields=["text"],
                log=False,
            )
        )
    return handles


def _row_contexts_from_state(bgp_state) -> list[RowFilterContext]:
    contexts: list[RowFilterContext] = []
    for idx, vinfo in enumerate(bgp_state.validation_info_list):
        contexts.append(
            RowFilterContext(
                subject=bgp_state.subject,
                predicate=bgp_state.predicate,
                obj=bgp_state.obj,
                variables=bgp_state.variables,
                pattern_variable_roles=bgp_state.pattern_variable_roles,
                value_row=bgp_state.values[idx],
                validation_info=vinfo,
            )
        )
    return contexts


def _cache_current_page(
    session: PaginationSession,
    rows: list[dict],
    raw_hits: Optional[list[dict]],
) -> None:
    idx = session.pages_fetched
    session.page_cache[idx] = list(rows)
    if raw_hits is not None:
        session.raw_hits_cache[idx] = list(raw_hits)


def _filter_row(session: PaginationSession, row_idx: int, matches: list[dict]) -> list[dict]:
    ctx = session.row_contexts[row_idx]
    rows, _ids = filter_matches_to_rows(
        matches,
        pattern_subject=ctx.subject,
        pattern_predicate=ctx.predicate,
        pattern_object=ctx.obj,
        validation_info=ctx.validation_info,
        variables=ctx.variables,
        pattern_variable_roles=ctx.pattern_variable_roles,
        value_row=ctx.value_row,
        parse_rdf_triple=parse_rdf_triple,
        log=False,
    )
    return rows


def _fetch_one_page(session: PaginationSession) -> PaginationPageResult:
    if session.done:
        return _terminal_page_result(session)

    remaining = session.limit - session.milvus_hits_total
    if remaining <= 0:
        return _finalize_session(session)

    batch_size = min(session.k, remaining)
    row_idx = session.current_row
    handle = session.iterators[row_idx]
    if handle is None or handle.closed:
        return _advance_row_or_finish(session)

    hits = handle.next_page()
    if not hits:
        handle.close()
        session.iterators[row_idx] = None
        return _advance_row_or_finish(session)

    if len(hits) > batch_size:
        hits = hits[:batch_size]

    session.pages_fetched += 1
    session.milvus_hits_total += len(hits)
    rows = _filter_row(session, row_idx, hits)
    raw_hits = list(hits) if session.include_raw_hits else None
    _cache_current_page(session, rows, raw_hits)

    if session.milvus_hits_total >= session.limit:
        return _finalize_session(
            session,
            rows=rows,
            milvus_hits_this_page=len(hits),
            raw_hits=raw_hits,
        )

    if len(hits) < batch_size:
        handle.close()
        session.iterators[row_idx] = None
        if row_idx + 1 < len(session.iterators):
            session.current_row += 1
            return PaginationPageResult(
                vars=session.vars,
                rows=rows,
                raw_hits=raw_hits,
                pagination=PaginationMeta(
                    cursor=session.cursor,
                    done=False,
                    k=session.k,
                    limit=session.limit,
                    catalog_k=session.catalog_k,
                    page_index=session.pages_fetched,
                    milvus_hits_this_page=len(hits),
                    milvus_hits_total=session.milvus_hits_total,
                    value_row_index=row_idx,
                ),
            )
        return _finalize_session(
            session,
            rows=rows,
            milvus_hits_this_page=len(hits),
            raw_hits=raw_hits,
        )

    return PaginationPageResult(
        vars=session.vars,
        rows=rows,
        raw_hits=raw_hits,
        pagination=PaginationMeta(
            cursor=session.cursor,
            done=False,
            k=session.k,
            limit=session.limit,
            catalog_k=session.catalog_k,
            page_index=session.pages_fetched,
            milvus_hits_this_page=len(hits),
            milvus_hits_total=session.milvus_hits_total,
            value_row_index=row_idx,
        ),
    )


def _advance_row_or_finish(session: PaginationSession) -> PaginationPageResult:
    if session.current_row + 1 < len(session.iterators):
        session.current_row += 1
        return _fetch_one_page(session)
    return _finalize_session(session)


def _finalize_session(
    session: PaginationSession,
    *,
    rows: Optional[list[dict]] = None,
    milvus_hits_this_page: int = 0,
    raw_hits: Optional[list[dict]] = None,
) -> PaginationPageResult:
    session.done = True
    PAGINATION_SESSION_STORE.close(session.cursor)
    return PaginationPageResult(
        vars=session.vars,
        rows=rows or [],
        raw_hits=raw_hits,
        pagination=PaginationMeta(
            cursor=None,
            done=True,
            k=session.k,
            limit=session.limit,
            catalog_k=session.catalog_k,
            page_index=session.pages_fetched,
            milvus_hits_this_page=milvus_hits_this_page,
            milvus_hits_total=session.milvus_hits_total,
            value_row_index=session.current_row,
        ),
    )


def _terminal_page_result(session: PaginationSession) -> PaginationPageResult:
    return PaginationPageResult(
        vars=session.vars,
        rows=[],
        pagination=PaginationMeta(
            cursor=None,
            done=True,
            k=session.k,
            limit=session.limit,
            catalog_k=session.catalog_k,
            page_index=session.pages_fetched,
            milvus_hits_this_page=0,
            milvus_hits_total=session.milvus_hits_total,
            value_row_index=session.current_row,
        ),
    )


def start_pagination_page(
    query_input: PatternQueryInput,
    *,
    collection_name: str | None = None,
    database: VectorDataBase | None = None,
    resolver: CatalogKResolver | None = None,
) -> PaginationPageResult:
    if collection_name is None:
        collection_name = _collection_name()
    if database is None:
        database = _default_vdb()
    if resolver is None:
        resolver = _default_resolver()
    if query_input.k_mode != "pagination":
        raise ValueError("k_mode must be 'pagination' to start pagination")
    if query_input.k is None or query_input.k <= 0:
        raise ValueError("pagination requires positive k (batch size)")
    if database is None:
        raise ValueError("vector database is not available")

    bgp_state = build_bgp_execution_state(query_input)
    if not bgp_state.search_queries:
        raise ValueError("pattern produced no search queries")

    sq0 = bgp_state.search_queries[0]
    catalog_k = _catalog_k_for_query(sq0, resolver)
    safe_k = milvus_safe_k(int(query_input.k))
    resolved_limit = resolve_pagination_limit(
        safe_k,
        catalog_k=catalog_k,
        explicit_limit=query_input.pagination_limit,
    )

    if bgp_log_enabled() and catalog_k is not None:
        default_cap = 2 * catalog_k
        if query_input.pagination_limit is None and resolved_limit > default_cap:
            bgp_emit(
                f"[BGP] pagination limit bumped to k={resolved_limit} "
                f"(catalog_k={catalog_k})"
            )

    cursor = new_cursor()
    iterators = _open_iterators_for_state(
        database=database,
        collection_name=collection_name,
        bgp_state=bgp_state,
        k=safe_k,
        limit=resolved_limit,
    )
    session = PaginationSession(
        cursor=cursor,
        vars=list(query_input.vars),
        collection_name=collection_name,
        iterators=iterators,
        row_contexts=_row_contexts_from_state(bgp_state),
        k=safe_k,
        limit=resolved_limit,
        catalog_k=catalog_k,
        include_raw_hits=query_input.include_raw_hits,
    )
    PAGINATION_SESSION_STORE.create(session)
    return _fetch_one_page(session)


def _cached_page_result(session: PaginationSession, page: int) -> PaginationPageResult:
    rows = session.page_cache[page]
    raw_hits = session.raw_hits_cache.get(page) if session.include_raw_hits else None
    return PaginationPageResult(
        vars=session.vars,
        rows=list(rows),
        raw_hits=list(raw_hits) if raw_hits is not None else None,
        pagination=PaginationMeta(
            cursor=session.cursor,
            done=session.done,
            k=session.k,
            limit=session.limit,
            catalog_k=session.catalog_k,
            page_index=page,
            milvus_hits_this_page=0,
            milvus_hits_total=session.milvus_hits_total,
            value_row_index=session.current_row,
            from_cache=True,
        ),
    )


def resolve_pagination_page(
    session_id: str,
    *,
    page: int | None = None,
    cancel: bool = False,
) -> PaginationPageResult:
    if cancel:
        try:
            session = PAGINATION_SESSION_STORE.get(session_id)
        except PaginationSessionNotFound:
            raise
        except PaginationSessionGone:
            raise
        PAGINATION_SESSION_STORE.close(session_id)
        return _terminal_page_result(session)

    try:
        session = PAGINATION_SESSION_STORE.get(session_id)
    except PaginationSessionNotFound:
        raise
    except PaginationSessionGone:
        raise

    if page is not None:
        if page < 1:
            raise ValueError("page must be >= 1")
        if page not in session.page_cache:
            raise PaginationPageNotCached(f"page {page} is not cached")
        return _cached_page_result(session, page)

    return _fetch_one_page(session)


def next_pagination_page(
    cursor: str,
    *,
    cancel: bool = False,
) -> PaginationPageResult:
    return resolve_pagination_page(cursor, cancel=cancel)


def collect_pagination_pages(
    query_input: PatternQueryInput,
    *,
    collection_name: str | None = None,
    database: VectorDataBase | None = None,
    resolver: CatalogKResolver | None = None,
) -> tuple[list[dict], PaginationPageResult, list[dict]]:
    """Drain all pagination pages in-process (benchmark helper)."""
    if collection_name is None:
        collection_name = _collection_name()
    if database is None:
        database = _default_vdb()
    if resolver is None:
        resolver = _default_resolver()
    all_rows: list[dict] = []
    all_raw: list[dict] = []
    page = start_pagination_page(
        query_input,
        collection_name=collection_name,
        database=database,
        resolver=resolver,
    )
    all_rows.extend(page.rows)
    if page.raw_hits:
        all_raw.extend(page.raw_hits)
    while not page.pagination.done and page.pagination.cursor:
        page = next_pagination_page(page.pagination.cursor)
        all_rows.extend(page.rows)
        if page.raw_hits:
            all_raw.extend(page.raw_hits)
    return all_rows, page, all_raw
