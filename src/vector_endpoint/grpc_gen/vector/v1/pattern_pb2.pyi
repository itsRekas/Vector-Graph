from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Term(_message.Message):
    __slots__ = ("type", "value", "lang", "datatype")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    LANG_FIELD_NUMBER: _ClassVar[int]
    DATATYPE_FIELD_NUMBER: _ClassVar[int]
    type: str
    value: str
    lang: str
    datatype: str
    def __init__(self, type: _Optional[str] = ..., value: _Optional[str] = ..., lang: _Optional[str] = ..., datatype: _Optional[str] = ...) -> None: ...

class Pattern(_message.Message):
    __slots__ = ("subject", "predicate", "object")
    SUBJECT_FIELD_NUMBER: _ClassVar[int]
    PREDICATE_FIELD_NUMBER: _ClassVar[int]
    OBJECT_FIELD_NUMBER: _ClassVar[int]
    subject: Term
    predicate: Term
    object: Term
    def __init__(self, subject: _Optional[_Union[Term, _Mapping]] = ..., predicate: _Optional[_Union[Term, _Mapping]] = ..., object: _Optional[_Union[Term, _Mapping]] = ...) -> None: ...

class ValueRow(_message.Message):
    __slots__ = ("bindings",)
    class BindingsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Term
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Term, _Mapping]] = ...) -> None: ...
    BINDINGS_FIELD_NUMBER: _ClassVar[int]
    bindings: _containers.MessageMap[str, Term]
    def __init__(self, bindings: _Optional[_Mapping[str, Term]] = ...) -> None: ...

class PatternQueryRequest(_message.Message):
    __slots__ = ("pattern", "vars", "values", "k", "adaptive_multipliers", "adaptive_jaccard", "include_raw_hits")
    PATTERN_FIELD_NUMBER: _ClassVar[int]
    VARS_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    ADAPTIVE_MULTIPLIERS_FIELD_NUMBER: _ClassVar[int]
    ADAPTIVE_JACCARD_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_RAW_HITS_FIELD_NUMBER: _ClassVar[int]
    pattern: Pattern
    vars: _containers.RepeatedScalarFieldContainer[str]
    values: _containers.RepeatedCompositeFieldContainer[ValueRow]
    k: int
    adaptive_multipliers: _containers.RepeatedScalarFieldContainer[int]
    adaptive_jaccard: float
    include_raw_hits: bool
    def __init__(self, pattern: _Optional[_Union[Pattern, _Mapping]] = ..., vars: _Optional[_Iterable[str]] = ..., values: _Optional[_Iterable[_Union[ValueRow, _Mapping]]] = ..., k: _Optional[int] = ..., adaptive_multipliers: _Optional[_Iterable[int]] = ..., adaptive_jaccard: _Optional[float] = ..., include_raw_hits: _Optional[bool] = ...) -> None: ...

class MilvusHit(_message.Message):
    __slots__ = ("id", "distance", "text")
    ID_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    id: int
    distance: float
    text: str
    def __init__(self, id: _Optional[int] = ..., distance: _Optional[float] = ..., text: _Optional[str] = ...) -> None: ...

class RawSearchResult(_message.Message):
    __slots__ = ("value_row_index", "k_used", "hits")
    VALUE_ROW_INDEX_FIELD_NUMBER: _ClassVar[int]
    K_USED_FIELD_NUMBER: _ClassVar[int]
    HITS_FIELD_NUMBER: _ClassVar[int]
    value_row_index: int
    k_used: int
    hits: _containers.RepeatedCompositeFieldContainer[MilvusHit]
    def __init__(self, value_row_index: _Optional[int] = ..., k_used: _Optional[int] = ..., hits: _Optional[_Iterable[_Union[MilvusHit, _Mapping]]] = ...) -> None: ...

class BindingRow(_message.Message):
    __slots__ = ("bindings",)
    class BindingsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Term
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Term, _Mapping]] = ...) -> None: ...
    BINDINGS_FIELD_NUMBER: _ClassVar[int]
    bindings: _containers.MessageMap[str, Term]
    def __init__(self, bindings: _Optional[_Mapping[str, Term]] = ...) -> None: ...

class PatternQueryMetadata(_message.Message):
    __slots__ = ("vars",)
    VARS_FIELD_NUMBER: _ClassVar[int]
    vars: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, vars: _Optional[_Iterable[str]] = ...) -> None: ...

class PatternQueryDone(_message.Message):
    __slots__ = ("total_rows", "returned_count", "k_mode")
    TOTAL_ROWS_FIELD_NUMBER: _ClassVar[int]
    RETURNED_COUNT_FIELD_NUMBER: _ClassVar[int]
    K_MODE_FIELD_NUMBER: _ClassVar[int]
    total_rows: int
    returned_count: int
    k_mode: str
    def __init__(self, total_rows: _Optional[int] = ..., returned_count: _Optional[int] = ..., k_mode: _Optional[str] = ...) -> None: ...

class PatternQueryError(_message.Message):
    __slots__ = ("message",)
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    message: str
    def __init__(self, message: _Optional[str] = ...) -> None: ...

class PatternQueryEvent(_message.Message):
    __slots__ = ("metadata", "row", "done", "error", "row_batch", "raw_search")
    METADATA_FIELD_NUMBER: _ClassVar[int]
    ROW_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    ROW_BATCH_FIELD_NUMBER: _ClassVar[int]
    RAW_SEARCH_FIELD_NUMBER: _ClassVar[int]
    metadata: PatternQueryMetadata
    row: BindingRow
    done: PatternQueryDone
    error: PatternQueryError
    row_batch: BindingRowBatch
    raw_search: RawSearchResult
    def __init__(self, metadata: _Optional[_Union[PatternQueryMetadata, _Mapping]] = ..., row: _Optional[_Union[BindingRow, _Mapping]] = ..., done: _Optional[_Union[PatternQueryDone, _Mapping]] = ..., error: _Optional[_Union[PatternQueryError, _Mapping]] = ..., row_batch: _Optional[_Union[BindingRowBatch, _Mapping]] = ..., raw_search: _Optional[_Union[RawSearchResult, _Mapping]] = ...) -> None: ...

class BindingRowBatch(_message.Message):
    __slots__ = ("rows",)
    ROWS_FIELD_NUMBER: _ClassVar[int]
    rows: _containers.RepeatedCompositeFieldContainer[BindingRow]
    def __init__(self, rows: _Optional[_Iterable[_Union[BindingRow, _Mapping]]] = ...) -> None: ...

class PatternPageRequest(_message.Message):
    __slots__ = ("k_mode", "k", "limit", "cursor", "cancel", "pattern", "vars", "values", "include_raw_hits", "next", "page", "session")
    K_MODE_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    CURSOR_FIELD_NUMBER: _ClassVar[int]
    CANCEL_FIELD_NUMBER: _ClassVar[int]
    PATTERN_FIELD_NUMBER: _ClassVar[int]
    VARS_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_RAW_HITS_FIELD_NUMBER: _ClassVar[int]
    NEXT_FIELD_NUMBER: _ClassVar[int]
    PAGE_FIELD_NUMBER: _ClassVar[int]
    SESSION_FIELD_NUMBER: _ClassVar[int]
    k_mode: str
    k: int
    limit: int
    cursor: str
    cancel: bool
    pattern: Pattern
    vars: _containers.RepeatedScalarFieldContainer[str]
    values: _containers.RepeatedCompositeFieldContainer[ValueRow]
    include_raw_hits: bool
    next: bool
    page: int
    session: str
    def __init__(self, k_mode: _Optional[str] = ..., k: _Optional[int] = ..., limit: _Optional[int] = ..., cursor: _Optional[str] = ..., cancel: _Optional[bool] = ..., pattern: _Optional[_Union[Pattern, _Mapping]] = ..., vars: _Optional[_Iterable[str]] = ..., values: _Optional[_Iterable[_Union[ValueRow, _Mapping]]] = ..., include_raw_hits: _Optional[bool] = ..., next: _Optional[bool] = ..., page: _Optional[int] = ..., session: _Optional[str] = ...) -> None: ...

class PaginationInfo(_message.Message):
    __slots__ = ("cursor", "done", "k", "limit", "catalog_k", "page_index", "milvus_hits_this_page", "milvus_hits_total", "value_row_index", "k_mode", "from_cache", "session")
    CURSOR_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    CATALOG_K_FIELD_NUMBER: _ClassVar[int]
    PAGE_INDEX_FIELD_NUMBER: _ClassVar[int]
    MILVUS_HITS_THIS_PAGE_FIELD_NUMBER: _ClassVar[int]
    MILVUS_HITS_TOTAL_FIELD_NUMBER: _ClassVar[int]
    VALUE_ROW_INDEX_FIELD_NUMBER: _ClassVar[int]
    K_MODE_FIELD_NUMBER: _ClassVar[int]
    FROM_CACHE_FIELD_NUMBER: _ClassVar[int]
    SESSION_FIELD_NUMBER: _ClassVar[int]
    cursor: str
    done: bool
    k: int
    limit: int
    catalog_k: int
    page_index: int
    milvus_hits_this_page: int
    milvus_hits_total: int
    value_row_index: int
    k_mode: str
    from_cache: bool
    session: str
    def __init__(self, cursor: _Optional[str] = ..., done: _Optional[bool] = ..., k: _Optional[int] = ..., limit: _Optional[int] = ..., catalog_k: _Optional[int] = ..., page_index: _Optional[int] = ..., milvus_hits_this_page: _Optional[int] = ..., milvus_hits_total: _Optional[int] = ..., value_row_index: _Optional[int] = ..., k_mode: _Optional[str] = ..., from_cache: _Optional[bool] = ..., session: _Optional[str] = ...) -> None: ...

class PatternPageResponse(_message.Message):
    __slots__ = ("vars", "rows", "pagination", "raw_search", "error")
    VARS_FIELD_NUMBER: _ClassVar[int]
    ROWS_FIELD_NUMBER: _ClassVar[int]
    PAGINATION_FIELD_NUMBER: _ClassVar[int]
    RAW_SEARCH_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    vars: _containers.RepeatedScalarFieldContainer[str]
    rows: _containers.RepeatedCompositeFieldContainer[BindingRow]
    pagination: PaginationInfo
    raw_search: RawSearchResult
    error: PatternQueryError
    def __init__(self, vars: _Optional[_Iterable[str]] = ..., rows: _Optional[_Iterable[_Union[BindingRow, _Mapping]]] = ..., pagination: _Optional[_Union[PaginationInfo, _Mapping]] = ..., raw_search: _Optional[_Union[RawSearchResult, _Mapping]] = ..., error: _Optional[_Union[PatternQueryError, _Mapping]] = ...) -> None: ...
