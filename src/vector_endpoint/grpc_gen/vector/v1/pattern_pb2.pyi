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
    __slots__ = ("pattern", "vars", "values", "k", "adaptive_multipliers", "adaptive_jaccard")
    PATTERN_FIELD_NUMBER: _ClassVar[int]
    VARS_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    ADAPTIVE_MULTIPLIERS_FIELD_NUMBER: _ClassVar[int]
    ADAPTIVE_JACCARD_FIELD_NUMBER: _ClassVar[int]
    pattern: Pattern
    vars: _containers.RepeatedScalarFieldContainer[str]
    values: _containers.RepeatedCompositeFieldContainer[ValueRow]
    k: int
    adaptive_multipliers: _containers.RepeatedScalarFieldContainer[int]
    adaptive_jaccard: float
    def __init__(self, pattern: _Optional[_Union[Pattern, _Mapping]] = ..., vars: _Optional[_Iterable[str]] = ..., values: _Optional[_Iterable[_Union[ValueRow, _Mapping]]] = ..., k: _Optional[int] = ..., adaptive_multipliers: _Optional[_Iterable[int]] = ..., adaptive_jaccard: _Optional[float] = ...) -> None: ...

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
    __slots__ = ("metadata", "row", "done", "error", "row_batch")
    METADATA_FIELD_NUMBER: _ClassVar[int]
    ROW_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    ROW_BATCH_FIELD_NUMBER: _ClassVar[int]
    metadata: PatternQueryMetadata
    row: BindingRow
    done: PatternQueryDone
    error: PatternQueryError
    row_batch: BindingRowBatch
    def __init__(self, metadata: _Optional[_Union[PatternQueryMetadata, _Mapping]] = ..., row: _Optional[_Union[BindingRow, _Mapping]] = ..., done: _Optional[_Union[PatternQueryDone, _Mapping]] = ..., error: _Optional[_Union[PatternQueryError, _Mapping]] = ..., row_batch: _Optional[_Union[BindingRowBatch, _Mapping]] = ...) -> None: ...

class BindingRowBatch(_message.Message):
    __slots__ = ("rows",)
    ROWS_FIELD_NUMBER: _ClassVar[int]
    rows: _containers.RepeatedCompositeFieldContainer[BindingRow]
    def __init__(self, rows: _Optional[_Iterable[_Union[BindingRow, _Mapping]]] = ...) -> None: ...
