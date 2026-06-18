"""In-memory pagination session store for Milvus search iterators."""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from vector_endpoint.db.VectorDataBase import SearchIteratorHandle


def pagination_session_ttl_sec() -> float:
    return float(os.getenv("VECTOR_PAGINATION_SESSION_TTL_SEC", "300"))


@dataclass
class RowFilterContext:
    subject: object
    predicate: object
    obj: object
    variables: list[str]
    pattern_variable_roles: dict[str, str]
    value_row: dict
    validation_info: dict


@dataclass
class PaginationSession:
    cursor: str
    vars: list[str]
    collection_name: str
    iterators: list[Optional[SearchIteratorHandle]]
    row_contexts: list[RowFilterContext]
    k: int
    limit: int
    catalog_k: Optional[int]
    current_row: int = 0
    pages_fetched: int = 0
    milvus_hits_total: int = 0
    done: bool = False
    include_raw_hits: bool = False
    page_cache: dict[int, list[dict]] = field(default_factory=dict)
    raw_hits_cache: dict[int, list[dict]] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_access = time.time()


class PaginationSessionNotFound(Exception):
    """Raised when session id does not match any live session."""


class PaginationSessionGone(Exception):
    """Raised when session expired or was already closed."""


class PaginationPageNotCached(Exception):
    """Raised when a cached page number was requested but is not available."""


def resolve_session_id(json_data: dict) -> Optional[str]:
    """Return session id from ``session`` or legacy ``cursor`` field."""
    session_val = json_data.get("session") or json_data.get("cursor")
    return str(session_val) if session_val else None


class PaginationSessionStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, PaginationSession] = {}

    def _purge_expired(self) -> None:
        ttl = pagination_session_ttl_sec()
        now = time.time()
        expired = [
            cursor
            for cursor, session in self._sessions.items()
            if now - session.last_access > ttl
        ]
        for cursor in expired:
            self._close_unlocked(cursor, remove=True)

    def create(self, session: PaginationSession) -> str:
        with self._lock:
            self._purge_expired()
            self._sessions[session.cursor] = session
            return session.cursor

    def get(self, cursor: str) -> PaginationSession:
        with self._lock:
            self._purge_expired()
            session = self._sessions.get(cursor)
            if session is None:
                raise PaginationSessionNotFound("pagination session not found")
            if session.done:
                raise PaginationSessionGone("pagination session expired")
            if time.time() - session.last_access > pagination_session_ttl_sec():
                self._close_unlocked(cursor, remove=True)
                raise PaginationSessionGone("pagination session expired")
            session.touch()
            return session

    def close(self, cursor: str) -> None:
        with self._lock:
            self._close_unlocked(cursor, remove=True)

    def _close_unlocked(self, cursor: str, *, remove: bool) -> None:
        session = self._sessions.pop(cursor, None) if remove else self._sessions.get(cursor)
        if session is None:
            return
        session.page_cache.clear()
        session.raw_hits_cache.clear()
        for handle in session.iterators:
            if handle is not None and not handle.closed:
                handle.close()
        session.done = True


PAGINATION_SESSION_STORE = PaginationSessionStore()


def new_cursor() -> str:
    return str(uuid.uuid4())
