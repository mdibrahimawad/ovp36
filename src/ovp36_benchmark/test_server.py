"""Deterministic loopback HTTP test plumbing; no model or production serving."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from socketserver import TCPServer
from threading import Event, Thread
from time import monotonic
from typing import Any

from .identity import canonical_json_bytes
from .requests import PreparedRequest


MAX_REQUEST_BYTES = 1024 * 1024
READ_TIMEOUT_SECONDS = 2.0
JOIN_TIMEOUT_SECONDS = 5.0


class StubServerError(ValueError):
    """Safe code only; never retain HTTP payloads or underlying exceptions."""


@dataclass(frozen=True)
class StubExchange:
    expected_request: PreparedRequest = field(repr=False)
    response_body: Mapping[str, Any] = field(repr=False)
    status_code: int = 200


class _Server(HTTPServer):
    def server_bind(self):
        # HTTPServer.server_bind calls getfqdn(), including for numeric loopback.
        TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(READ_TIMEOUT_SECONDS)
        return connection, address

    def handle_error(self, request, client_address):
        self.owner._fail("server_error")


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_key")
        value[key] = item
    return value


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_error(self, code, message=None, explain=None):
        # BaseHTTPRequestHandler otherwise echoes invalid methods/request lines.
        self.server.owner._received += 1
        self._reject("wrong_method" if code == 501 else "malformed_http")

    def _reply(self, status, body):
        self.close_connection = True
        self.send_response_only(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _reject(self, code):
        self.server.owner._fail(code)
        self._reply(400, canonical_json_bytes({"error": {"code": code}}))

    def do_POST(self):
        owner = self.server.owner
        owner._received += 1
        if self.path != "/v1/chat/completions":
            self._reject("wrong_path")
            return
        lengths = self.headers.get_all("Content-Length", [])
        if (len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit()
                or self.headers.get("Transfer-Encoding") is not None):
            self._reject("invalid_content_length")
            return
        # Limit decimal width before conversion, including pathological headers.
        if len(lengths[0]) > 10 or int(lengths[0]) > MAX_REQUEST_BYTES:
            self._reject("request_too_large")
            return
        remaining = int(lengths[0])
        chunks = []
        deadline = monotonic() + READ_TIMEOUT_SECONDS
        try:
            while remaining:
                timeout = deadline - monotonic()
                if timeout <= 0:
                    raise TimeoutError
                self.connection.settimeout(timeout)
                chunk = self.rfile.read1(remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
        except OSError:
            remaining = max(remaining, 1)
        self.connection.settimeout(READ_TIMEOUT_SECONDS)
        if remaining:
            self._reject("truncated_body")
            return
        try:
            body = json.loads(b"".join(chunks).decode("utf-8"), object_pairs_hook=_unique_object)
            encoded = canonical_json_bytes(body)
        except (ValueError, RecursionError):
            self._reject("malformed_json")
            return
        if type(body) is not dict:
            self._reject("request_shape_mismatch")
            return
        if owner._matched == len(owner._exchanges):
            self._reject("unexpected_request")
            return
        expected_bytes, response, status = owner._exchanges[owner._matched]
        if encoded != expected_bytes:
            expected = json.loads(expected_bytes)
            if body.keys() != expected.keys():
                code = "request_shape_mismatch"
            elif canonical_json_bytes(body["model"]) != canonical_json_bytes(expected["model"]):
                code = "model_mismatch"
            elif canonical_json_bytes(body["messages"]) != canonical_json_bytes(expected["messages"]):
                code = "messages_mismatch"
            else:
                code = "generation_mismatch"
            self._reject(code)
            return
        owner._matched += 1
        self._reply(status, response)
        owner._responded += 1


class DeterministicChatServer:
    """Single-use, ordered exchanges on one ephemeral IPv4 loopback listener."""

    def __init__(self, exchanges: Sequence[StubExchange]):
        snapshots = []
        failure = None
        try:
            for exchange in exchanges:
                if (not isinstance(exchange, StubExchange)
                        or not isinstance(exchange.expected_request, PreparedRequest)
                        or not isinstance(exchange.response_body, Mapping)
                        or type(exchange.status_code) is not int
                        or not 200 <= exchange.status_code <= 599):
                    raise ValueError
                snapshots.append((canonical_json_bytes(exchange.expected_request.to_request_body()),
                                  canonical_json_bytes(dict(exchange.response_body)), exchange.status_code))
        except (ValueError, TypeError, RecursionError):
            failure = StubServerError("invalid_exchange")
        if failure is not None:
            raise failure
        self._exchanges = tuple(snapshots)
        self._received = self._matched = self._responded = 0
        self._failure = None
        self._server = self._thread = None
        self._ready = Event()
        self._entered = False

    @property
    def base_url(self) -> str:
        if self._server is None or self._server.socket.fileno() < 0:
            raise StubServerError("server_not_running")
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    @property
    def received_count(self) -> int:
        return self._received

    @property
    def matched_count(self) -> int:
        return self._matched

    @property
    def responded_count(self) -> int:
        return self._responded

    def _fail(self, code):
        if self._failure is None:
            self._failure = code

    def assert_complete(self) -> None:
        if self._failure is not None:
            raise StubServerError(self._failure)
        if self._matched != len(self._exchanges) or self._responded != len(self._exchanges):
            raise StubServerError("unconsumed_exchanges")

    def _serve(self):
        self._ready.set()
        try:
            self._server.serve_forever(poll_interval=0.05)
        except Exception:
            self._fail("server_error")

    def __enter__(self):
        if self._entered:
            raise StubServerError("server_already_entered")
        self._entered = True
        try:
            self._server = _Server(("127.0.0.1", 0), _Handler)
            self._server.owner = self
            self._thread = Thread(target=self._serve, name="ovp36-local-stub")
            self._thread.start()
            if not self._ready.wait(JOIN_TIMEOUT_SECONDS):
                self._fail("startup_timeout")
        except Exception:
            self._fail("startup_failed")
        if self._failure is not None:
            self._close()
            raise StubServerError(self._failure)
        return self

    def _close(self):
        try:
            if self._thread is not None and self._thread.is_alive():
                self._server.shutdown()
        except Exception:
            self._fail("shutdown_failed")
        finally:
            if self._server is not None:
                try:
                    self._server.server_close()
                except Exception:
                    self._fail("shutdown_failed")
            if self._thread is not None and self._thread.ident is not None:
                self._thread.join(JOIN_TIMEOUT_SECONDS)
                if self._thread.is_alive():
                    self._fail("shutdown_timeout")

    def __exit__(self, exc_type, exc_value, traceback):
        self._close()
        if exc_type is None:
            self.assert_complete()
