"""gRPC-only entrypoint for vector pattern queries."""

from __future__ import annotations

import os
import signal
import socket
import sys

from vector_endpoint.bgp_log import configure_bgp_logging
from vector_endpoint.grpc_server import create_grpc_server


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def main() -> None:
    configure_bgp_logging()
    port = int(os.getenv("VECTOR_GRPC_PORT", "50051"))
    host = os.getenv("VECTOR_GRPC_HOST", "[::]")

    if _port_in_use(port):
        print(
            f"ERROR: port {port} is already in use (another grpc_app may be running).\n"
            "  Stop it first: pkill -f 'vector_endpoint.grpc_app'\n"
            "  Then restart a single server in this terminal.",
            file=sys.stderr,
        )
        sys.exit(1)

    server = create_grpc_server()
    listen_addr = f"{host}:{port}"
    bound = server.add_insecure_port(listen_addr)
    if bound == 0:
        print(f"ERROR: gRPC failed to bind {listen_addr}", file=sys.stderr)
        sys.exit(1)
    server.start()
    print(f"gRPC Vector Pattern Service listening on {listen_addr}", file=sys.stderr)
    print("Press Ctrl+C to stop", file=sys.stderr)
    if os.getenv("VECTOR_BGP_LOG", "1") != "0":
        print(
            "BGP logs -> stderr (same as Flask/Werkzeug). Set VECTOR_BGP_LOG=0 to disable.",
            file=sys.stderr,
        )
    if os.getenv("VECTOR_GRPC_DEBUG") or os.getenv("FLASK_DEBUG"):
        print(
            "Verbose request bodies enabled (VECTOR_GRPC_DEBUG or FLASK_DEBUG)",
            file=sys.stderr,
        )

    def _shutdown(signum, frame):  # noqa: ARG001
        server.stop(grace=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
