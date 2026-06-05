"""gRPC server-streaming servicer for vector pattern queries."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import grpc

from vector_endpoint.grpc_gen.vector.v1 import pattern_pb2, pattern_pb2_grpc
from vector_endpoint.bgp_log import log_grpc_pattern_received, log_grpc_rpc_received
from vector_endpoint.pattern_query import stream_pattern_rows
from vector_endpoint.proto_convert import pattern_query_input_from_proto, row_to_proto


def _row_batch_size() -> int:
    return max(1, int(os.getenv("VECTOR_GRPC_ROW_BATCH", "100")))


def _max_message_bytes() -> int:
    return int(os.getenv("VECTOR_GRPC_MAX_MESSAGE_BYTES", str(128 * 1024 * 1024)))


class VectorPatternServicer(pattern_pb2_grpc.VectorPatternServiceServicer):
    def QueryPattern(self, request, context):  # noqa: N802
        log_grpc_rpc_received(
            vars=list(request.vars),
            value_rows=len(request.values),
        )
        try:
            query_input = pattern_query_input_from_proto(request)
        except Exception as exc:  # noqa: BLE001
            yield pattern_pb2.PatternQueryEvent(
                error=pattern_pb2.PatternQueryError(message=str(exc))
            )
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(exc))
            return

        log_grpc_pattern_received(query_input)

        yield pattern_pb2.PatternQueryEvent(
            metadata=pattern_pb2.PatternQueryMetadata(vars=query_input.vars)
        )

        k_mode = "fixed" if query_input.k is not None else "adaptive"
        total = 0
        batch_size = _row_batch_size()
        batch: list[pattern_pb2.BindingRow] = []

        try:
            for row in stream_pattern_rows(query_input):
                total += 1
                batch.append(row_to_proto(row))
                if len(batch) >= batch_size:
                    yield pattern_pb2.PatternQueryEvent(
                        row_batch=pattern_pb2.BindingRowBatch(rows=batch)
                    )
                    batch = []
            if batch:
                yield pattern_pb2.PatternQueryEvent(
                    row_batch=pattern_pb2.BindingRowBatch(rows=batch)
                )
            yield pattern_pb2.PatternQueryEvent(
                done=pattern_pb2.PatternQueryDone(
                    total_rows=total,
                    returned_count=total,
                    k_mode=k_mode,
                )
            )
        except Exception as exc:  # noqa: BLE001
            yield pattern_pb2.PatternQueryEvent(
                error=pattern_pb2.PatternQueryError(message=str(exc))
            )
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))


def create_grpc_server() -> grpc.Server:
    max_bytes = _max_message_bytes()
    server = grpc.server(
        ThreadPoolExecutor(max_workers=int(os.getenv("VECTOR_GRPC_WORKERS", "4"))),
        options=[
            ("grpc.max_send_message_length", max_bytes),
            ("grpc.max_receive_message_length", max_bytes),
        ],
    )
    pattern_pb2_grpc.add_VectorPatternServiceServicer_to_server(
        VectorPatternServicer(), server
    )
    return server
