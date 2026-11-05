# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loopback proxy that sits on both sides of the Switchyard routes.

``/front/*``    Hermes -> Switchyard. Each request is one routed turn.
``/local/*``    Switchyard -> local tier. Each request is one routing stage.
``/cloud/*``    Switchyard -> cloud tier. Each request is one routing stage.
``/monitor/*``  The live monitor state read by the dashboard.

Front and monitor routes require the sidecar's bearer token. The tier routes
carry whatever credential Switchyard attaches, except a tier served by a
resolved endpoint (a Hermes provider, or Hermes's llama.cpp runtime), which gets
that endpoint's own credential in place of anything Switchyard sent.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import select
import socket
import sys
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple
from urllib.parse import urlsplit

import httpx

from . import failure, stages
from .initiator import REQUEST_ID_HEADER as HERMES_REQUEST_HEADER
from .monitor import CLIENT_CLOSED, Monitor

logger = logging.getLogger(__name__)

SELECTED_MODEL_HEADER = "x-model-router-selected-model"
SESSION_HEADER = "x-switchyard-session-id"
# The provider profile's mark on every Hermes client; Hermes's main-agent calls also carry SESSION_HEADER.
CLIENT_HEADER = "x-switchyard-client"
HERMES_CLIENT = "hermes"
_IMAGE_PARTS = frozenset({"image_url", "input_image", "image"})
# Switchyard copies a request's correlation id onto every upstream call it makes
# for that request, so the tap stamps each turn's id here.
REQUEST_ID_HEADER = "x-request-id"
_TURN_PREFIX = "vgr-turn-"
_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer",
    "transfer-encoding", "upgrade", "host", "content-length", "accept-encoding", "content-encoding",
})
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]", "::1"})
_MAX_BODY_BYTES = 64 * 1024 * 1024
# How long the cloud tier may send nothing, before its first byte or between two, before the tap
# gives up on it. Hermes waits as long as a VGR decision can take, so nothing else would end a
# stalled cloud reply. The front route and the local tier stay unbounded: a VGR turn and an
# unbounded local model are silent while they work.
CLOUD_IDLE_TIMEOUT_S = 300.0
_CLOUD_TIMEOUT = httpx.Timeout(connect=15.0, read=CLOUD_IDLE_TIMEOUT_S, write=60.0, pool=None)

# ``(base_url, credential headers)`` of a tier whose address is not fixed, or None while it is down.
EndpointResolver = Callable[[], Optional[Tuple[str, Dict[str, str]]]]
_CREDENTIAL_HEADERS = ("authorization", "x-api-key")
_COMPLETION_PATHS = ("/chat/completions", "/v1/messages", "/responses")
# Responses stream events that carry generated output: text, thinking and its summary, tool-call arguments.
_STREAMED_RESPONSES_DELTAS = (".output_text.delta", ".reasoning_text.delta", ".reasoning_summary_text.delta",
                              ".function_call_arguments.delta")
# ``(preset name, route model id)`` a request's ``model`` selects, or None for an unknown model.
PresetResolver = Callable[[Any], Optional[Tuple[str, str]]]


def _turn_of(request_id: Optional[str]) -> Optional[int]:
    if request_id and request_id.startswith(_TURN_PREFIX):
        suffix = request_id[len(_TURN_PREFIX):]
        return int(suffix) if suffix.isdigit() else None
    return None


def _stall_message(seconds: int) -> str:
    return f"the cloud model sent nothing for {seconds} s"


def _completed(response: Optional[Dict[str, Any]]) -> Tuple[int, bytes]:
    """``(status, body)`` answering unstreamed a request the tap streamed: the stream's final response."""
    if response is None:
        return 502, json.dumps({"error": {"message": "the cloud model's stream ended without a response"}}).encode()
    if response.get("status") == "failed":
        return 502, json.dumps({"error": response.get("error") or {"message": "the cloud model's response failed"}}).encode()
    return 200, json.dumps(response).encode()


class _Accumulator:
    """Collects the assistant text, tool-call presence, usage and logprobs of one response."""

    def __init__(self) -> None:
        self.text_parts: list[str] = []
        self.tool_call = False
        self.usage: Optional[Dict[str, Any]] = None
        self.model: Optional[str] = None
        self.p_yes: Optional[float] = None
        # The final ``response`` object of a Responses stream: what the same request answers unstreamed.
        self.response: Optional[Dict[str, Any]] = None
        # Whether a stream reached its terminal event, after which a caller may hang up without losing anything.
        self.ended = False
        self._buffer = b""
        # What a stream has sent so far, thinking and tool-call arguments included, for its live token count.
        self._streamed_chars = 0
        self._streamed_deltas = 0

    @property
    def text(self) -> str:
        return "".join(self.text_parts)

    @property
    def streamed_tokens(self) -> int:
        """The output tokens a stream has sent so far, estimated: a local server sends about one a delta,
        and a provider that sends several at once about one per four characters."""
        return max(self._streamed_deltas, -(-self._streamed_chars // 4))

    def _streamed(self, piece: Any) -> None:
        if isinstance(piece, str) and piece:
            self._streamed_chars += len(piece)
            self._streamed_deltas += 1

    def feed_json(self, raw: bytes) -> None:
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return
        if not isinstance(body, dict):
            return
        self.model = body.get("model") or self.model
        self._merge_usage(body.get("usage"))
        self.p_yes = stages.p_yes(body)
        for choice in body.get("choices") or ():
            message = choice.get("message") or {}
            if isinstance(message.get("content"), str):
                self.text_parts.append(message["content"])
            if message.get("tool_calls"):
                self.tool_call = True
        if body.get("type") == "message":
            for block in body.get("content") or ():
                self._anthropic_block(block)
        if body.get("object") == "response":
            self._response_output(body.get("output"))

    def _response_output(self, output: Any) -> None:
        for item in output or ():
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function_call":
                self.tool_call = True
            for part in item.get("content") or ():
                if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                    self.text_parts.append(part["text"])

    def _responses_event(self, event: Dict[str, Any]) -> None:
        kind = str(event.get("type"))
        if kind.endswith(_STREAMED_RESPONSES_DELTAS):
            self._streamed(event.get("delta"))
        if kind == "response.output_text.delta" and isinstance(event.get("delta"), str):
            self.text_parts.append(event["delta"])
        elif kind == "response.output_item.added" and (event.get("item") or {}).get("type") == "function_call":
            self.tool_call = True
        elif kind in ("response.completed", "response.incomplete", "response.failed") and isinstance(
            event.get("response"), dict
        ):
            self.response = event["response"]
            self.model = event["response"].get("model") or self.model
            self._merge_usage(event["response"].get("usage"))

    def _merge_usage(self, usage: Any) -> None:
        # Anthropic splits usage across message_start (input) and message_delta (output).
        if isinstance(usage, dict) and usage:
            self.usage = {**(self.usage or {}), **{k: v for k, v in usage.items() if v is not None}}

    def _anthropic_block(self, block: Any) -> None:
        if not isinstance(block, dict):
            return
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            self.text_parts.append(block["text"])
        elif block.get("type") == "tool_use":
            self.tool_call = True

    def _anthropic_event(self, event: Dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "message_start" and isinstance(event.get("message"), dict):
            message = event["message"]
            self.model = message.get("model") or self.model
            self._merge_usage(message.get("usage"))
        elif kind == "content_block_start":
            self._anthropic_block(event.get("content_block"))
        elif kind == "content_block_delta":
            delta = event.get("delta") or {}
            self._streamed(delta.get("text") or delta.get("thinking") or delta.get("partial_json"))
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                self.text_parts.append(delta["text"])
        elif kind == "message_delta":
            self._merge_usage(event.get("usage"))

    def feed_sse(self, chunk: bytes) -> None:
        self._buffer += chunk
        *lines, self._buffer = self._buffer.split(b"\n")
        for line in lines:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                self.ended = True
                continue
            try:
                event = json.loads(payload)
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(event, dict):
                continue
            if str(event.get("type", "")).startswith(("message_", "content_block_")):
                self._anthropic_event(event)
                self.ended = self.ended or event.get("type") == "message_stop"
                continue
            if str(event.get("type", "")).startswith("response."):
                self._responses_event(event)
                self.ended = self.ended or self.response is not None
                continue
            self.model = event.get("model") or self.model
            self._merge_usage(event.get("usage"))
            for choice in event.get("choices") or ():
                delta = choice.get("delta") or {}
                for key in ("content", "reasoning_content", "reasoning"):
                    self._streamed(delta.get(key))
                for tool_call in delta.get("tool_calls") or ():
                    if isinstance(tool_call, dict):
                        self._streamed((tool_call.get("function") or {}).get("arguments"))
                if isinstance(delta.get("content"), str):
                    self.text_parts.append(delta["content"])
                if delta.get("tool_calls"):
                    self.tool_call = True


class _Drafts:
    """Finished local drafts, kept for a repeat of the same request within one Hermes API call.

    When the cloud tier fails a step VGR escalated, Switchyard asks the local tier again with
    the request the local draft already answered, and Hermes may retry the whole call after
    that. Replaying the draft spares the local model generating the same answer two or three
    times. Hermes gives a retry the call's request id and a user's regenerate a new one, so a
    regenerate is never answered from here. An engine that serves the draft itself sends no
    repeat, and nothing is replayed.
    """

    MAX_ENTRIES = 16
    MAX_BYTES = 4 * 1024 * 1024
    TTL_S = 900.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: "OrderedDict[str, Tuple[float, str, bytes]]" = OrderedDict()

    @staticmethod
    def key(scope: str, body: Mapping[str, Any]) -> str:
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
        return f"{scope}\n{hashlib.sha256(canonical.encode()).hexdigest()}"

    def get(self, key: str) -> Optional[Tuple[str, bytes]]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or time.monotonic() - entry[0] > self.TTL_S:
                self._entries.pop(key, None)
                return None
            return entry[1], entry[2]

    def put(self, key: str, content_type: str, data: bytes) -> None:
        if len(data) > self.MAX_BYTES:
            return
        with self._lock:
            self._entries[key] = (time.monotonic(), content_type, data)
            self._entries.move_to_end(key)
            while len(self._entries) > self.MAX_ENTRIES:
                self._entries.popitem(last=False)


class _HangUp:
    """Shuts the upstream connection when the caller hangs up, so the cancel reaches the upstream.

    Switchyard buffers VGR's answer and writes nothing until it is done, so a failed
    write would only reveal the hang-up after the whole local draft and any cloud call.
    """

    def __init__(self, caller: socket.socket) -> None:
        self.happened = False
        self._caller = caller
        self._upstream: Any = None
        self._lock = threading.Lock()
        self._done = threading.Event()
        threading.Thread(target=self._watch, daemon=True).start()

    def trace(self, event: str, info: Dict[str, Any]) -> None:
        if event in ("connection.connect_tcp.complete", "connection.start_tls.complete"):
            self.attach(info.get("return_value"))

    def attach(self, stream: Any) -> None:
        with self._lock:
            self._upstream = stream
            if self.happened:
                self._shut()

    def finish(self) -> None:
        """Stops watching; the connection may go back to the pool after this."""
        with self._lock:
            self._done.set()
            self._upstream = None

    def _watch(self) -> None:
        while not self._done.is_set():
            try:
                readable, _, _ = select.select([self._caller], [], [], 0.25)
                if not readable:
                    continue
                if self._caller.recv(1, socket.MSG_PEEK):
                    return  # the caller sent more data, which is not a hang-up
            except (OSError, ValueError):
                pass
            with self._lock:
                if not self._done.is_set():
                    self.happened = True
                    self._shut()
            return

    def _shut(self) -> None:
        upstream = self._upstream.get_extra_info("socket") if self._upstream is not None else None
        if upstream is not None:
            try:
                upstream.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


class TapServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        port: int,
        *,
        token: str,
        local_base_url: str,
        cloud_base_url: str,
        monitor: Monitor,
        presets: PresetResolver,
        on_shutdown: Optional[Callable[[], None]] = None,
        endpoints: Optional[Dict[str, EndpointResolver]] = None,
        stream_only: Iterable[str] = (),
    ) -> None:
        super().__init__(("127.0.0.1", port), _Handler)
        self.token = token
        self.monitor = monitor
        self.switchyard_url: Optional[str] = None
        self.on_shutdown = on_shutdown
        self.upstreams = {
            "local": local_base_url.rstrip("/"),
            "cloud": cloud_base_url.rstrip("/"),
        }
        self.endpoints = dict(endpoints or {})
        self.stream_only = frozenset(stream_only)
        self.presets = presets
        self.drafts = _Drafts()
        timeout = httpx.Timeout(connect=15.0, read=None, write=60.0, pool=None)
        self.client = httpx.Client(timeout=timeout)
        # A pooled connection is only reachable once its response headers arrive; a fresh one
        # can be shut while the upstream is still working on a response it has not started.
        self.loopback_client = httpx.Client(timeout=timeout, limits=httpx.Limits(max_keepalive_connections=0))

    @property
    def port(self) -> int:
        return self.server_address[1]

    @staticmethod
    def pinned(tier: str) -> Tuple[str, str]:
        """``(preset name, route model id)`` of the built-in route pinned to ``tier``."""
        from .settings import pin_preset

        preset = pin_preset(tier)
        return preset.name, preset.model

    @property
    def pin_presets(self) -> frozenset:
        from .settings import PIN_TARGETS, pin_preset

        return frozenset(pin_preset(tier).name for tier in PIN_TARGETS)

    def resolve(self, path: str) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, str]]]:
        """``(route, upstream URL, credential headers)`` for a request path; route is None when unknown.

        The upstream URL is None for the monitor routes and for a resolved tier that is down.
        The headers are set only when the tap, not Switchyard, owns the tier's credential.
        """
        parts = urlsplit(path)
        segments = parts.path.lstrip("/").split("/", 1)
        route, rest = segments[0], (segments[1] if len(segments) > 1 else "")
        query = f"?{parts.query}" if parts.query else ""
        if route == "front" and self.switchyard_url:
            return route, f"{self.switchyard_url}/{rest}{query}", None
        if route in self.endpoints:
            try:
                endpoint = self.endpoints[route]()
            except Exception as error:  # a resolver failure is a tier that is down
                logger.warning("switchyard tap: the %s tier's endpoint did not resolve: %s", route, error)
                endpoint = None
            if endpoint is None:
                return route, None, {}
            base_url, credentials = endpoint
            return route, f"{base_url.rstrip('/')}/{rest}{query}", dict(credentials)
        if route in self.upstreams:
            return route, f"{self.upstreams[route]}/{rest}{query}", None
        if route == "monitor":
            return route, None, None
        return None, None, None

    def handle_error(self, request: Any, client_address: Any) -> None:
        # A client that hangs up (Hermes cancelling a stream, a health probe) is not a tap fault.
        if isinstance(sys.exc_info()[1], (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)

    def server_close(self) -> None:
        super().server_close()
        self.client.close()
        self.loopback_client.close()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: TapServer

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("tap: " + format, *args)

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def _dispatch(self, method: str) -> None:
        if not self._trusted_origin():
            self._send_json(403, {"error": "forbidden"})
            return
        route, upstream, credentials = self.server.resolve(self.path)
        if route is None:
            self._send_json(404, {"error": "not found"})
            return
        if route in ("front", "monitor") and not self._authorized():
            self._send_json(401, {"error": "unauthorized"})
            return
        if route == "monitor":
            self._serve_monitor(method)
            return
        if upstream is None:
            self._send_json(503, {"error": f"the {route} model server is not running"})
            return
        length = int(self.headers.get("content-length") or 0)
        if length > _MAX_BODY_BYTES:
            self._send_json(413, {"error": "request too large"})
            return
        raw = self.rfile.read(length) if length else b""
        self._proxy(method, route, upstream, raw, credentials)

    def _trusted_origin(self) -> bool:
        # Browsers always send Origin on cross-site requests; no legitimate caller of
        # this loopback proxy is a web page, and a loopback Host blocks DNS rebinding.
        if self.headers.get("origin"):
            return False
        host = (self.headers.get("host") or "").rsplit(":", 1)[0].lower()
        return host in _LOOPBACK_HOSTS

    def _authorized(self) -> bool:
        supplied = self.headers.get("authorization", "")
        scheme, _, credential = supplied.partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(credential.strip(), self.server.token)

    def _serve_monitor(self, method: str) -> None:
        path = urlsplit(self.path).path
        if method == "GET" and path == "/monitor/state":
            self._send_json(200, self.server.monitor.snapshot())
        elif method == "GET" and path == "/monitor/health":
            self._send_json(200, {"ok": True, "switchyard": bool(self.server.switchyard_url), "pid": os.getpid()})
        elif method == "POST" and path == "/monitor/pin":
            self._set_pin()
        elif method == "POST" and path == "/monitor/shutdown":
            self._send_json(200, {"ok": True})
            if self.server.on_shutdown:
                threading.Thread(target=self.server.on_shutdown, daemon=True).start()
        else:
            self._send_json(404, {"error": "not found"})

    def _set_pin(self) -> None:
        from .settings import SettingsError, parse_pin

        length = int(self.headers.get("content-length") or 0)
        if length > 4096:
            self.close_connection = True
            self._send_json(413, {"error": "request too large"})
            return
        try:
            body = json.loads(self.rfile.read(length) or b"{}") if length else {}
            pin = parse_pin(body.get("pin") if isinstance(body, dict) else None)
        except (ValueError, SettingsError) as error:
            self._send_json(400, {"error": str(error)})
            return
        self.server.monitor.pin = pin
        self._send_json(200, {"ok": True, "pin": pin})

    def _proxy(self, method: str, route: str, upstream: str, raw: bytes, credentials: Optional[Dict[str, str]]) -> None:
        body: Dict[str, Any] = {}
        if raw:
            try:
                parsed = json.loads(raw)
                body = parsed if isinstance(parsed, dict) else {}
            except (ValueError, UnicodeDecodeError):
                body = {}
        upstream_path = urlsplit(upstream).path
        is_completion = method == "POST" and upstream_path.endswith(_COMPLETION_PATHS)
        monitor = self.server.monitor
        preset = None
        if route == "front" and is_completion:
            selected = self.server.presets(body.get("model"))
            if selected is None:
                self._send_json(404, {"error": {"message": f"unknown Switchyard preset {body.get('model')!r}"}})
                return
            preset, route_model = selected
            if monitor.pin:
                preset, route_model = self.server.pinned(monitor.pin)
            elif self._side_task(body) and preset not in self.server.pin_presets:
                # Titles and approval checks run under Hermes's 30 s auxiliary timeout, which a VGR turn can outlast.
                preset, route_model = self.server.pinned("local")
            if body.get("model") != route_model:
                body["model"] = route_model
                raw = json.dumps(body).encode()
        turn_id = monitor.begin_turn(
            body, self.headers.get(SESSION_HEADER), preset, hermes_request_id=self.headers.get(HERMES_REQUEST_HEADER),
        ) if preset else None
        stalled = monitor.cloud_stalled(_turn_of(self.headers.get(REQUEST_ID_HEADER))) if route == "cloud" else None
        if stalled:
            # Switchyard retries a 502, and waiting out the idle limit again on each retry multiplies the stall.
            self._send_json(502, {"error": {"message": _stall_message(stalled)}})
            return
        call = (
            monitor.begin_call(route, body, _turn_of(self.headers.get(REQUEST_ID_HEADER)))
            if route != "front" and is_completion else None
        )
        assemble = (
            route in self.server.stream_only and is_completion
            and upstream_path.endswith("/responses") and body.get("stream") is not True
        )
        if assemble:
            body["stream"] = True
            raw = json.dumps(body).encode()
        draft_key = None
        if call is not None and route == "local" and call["stage"] == stages.LOCAL_DRAFT and not assemble:
            scope = monitor.request_scope(_turn_of(self.headers.get(REQUEST_ID_HEADER)))
            draft_key = _Drafts.key(scope, body) if scope else None
            replay = self.server.drafts.get(draft_key) if draft_key else None
            if replay is not None:
                self._replay(call, *replay)
                return

        headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in _HOP_BY_HOP and k.lower() not in (REQUEST_ID_HEADER, HERMES_REQUEST_HEADER)
        }
        if route == "front" or credentials is not None:
            headers = {k: v for k, v in headers.items() if k.lower() not in _CREDENTIAL_HEADERS}
        headers.update(credentials or {})
        if route == "front":
            if turn_id is not None:
                headers[REQUEST_ID_HEADER] = f"{_TURN_PREFIX}{turn_id}"
        elif self.headers.get(REQUEST_ID_HEADER):
            headers[REQUEST_ID_HEADER] = self.headers[REQUEST_ID_HEADER]
        headers["accept-encoding"] = "identity"
        status, served_model, accumulator = 502, None, _Accumulator()
        no_credit: Optional[str] = None  # a tier's provider message refusing for lack of credit
        headers_sent = False
        delivered = False  # the caller has the stream's terminal event, so a hang-up after it cancels nothing
        hang_up = _HangUp(self.connection)
        loopback = urlsplit(upstream).hostname in _LOOPBACK_HOSTS
        client = self.server.loopback_client if loopback else self.server.client
        captured: Optional[list] = [] if draft_key else None
        call_open = call is not None

        def close_call(final_status: int) -> None:
            nonlocal call_open
            if not call_open:
                return
            call_open = False
            monitor.end_call(
                call,
                status=final_status,
                usage=accumulator.usage,
                text=accumulator.text,
                tool_call=accumulator.tool_call,
                p_yes=accumulator.p_yes,
                out_of_credits=no_credit,
            )

        try:
            with client.stream(
                method, upstream, headers=headers, content=raw, extensions={"trace": hang_up.trace},
                **({"timeout": _CLOUD_TIMEOUT} if route == "cloud" else {}),
            ) as response:
                hang_up.attach(response.extensions.get("network_stream"))
                status = response.status_code
                served_model = response.headers.get(SELECTED_MODEL_HEADER)
                content_type = response.headers.get("content-type", "")
                # The Codex backend sends its streams with no Content-Type.
                streaming = "text/event-stream" in content_type or (
                    route in self.server.stream_only and is_completion and status < 400 and not content_type
                )
                data = None
                assembled = assemble and streaming
                if assembled:
                    for chunk in response.iter_bytes():
                        accumulator.feed_sse(chunk)
                        if call is not None:
                            monitor.stream_progress(call, accumulator.usage, accumulator.streamed_tokens)
                        if accumulator.ended:
                            break
                    accumulator.feed_sse(b"\n")
                    streaming = False
                    status, data = _completed(accumulator.response)
                if not streaming and status >= 400 and data is None:
                    data = response.read()
                    if call is not None:
                        no_credit = failure.out_of_credits(status, data)
                        if no_credit:
                            # Switchyard retries a 429 and serves nothing in place of a 402; on a 403 it
                            # goes straight to the other tier, held steps included.
                            status = 403
                    elif preset is not None and failure.annotates(status):
                        refused = monitor.out_of_credits(turn_id)
                        data = failure.annotate(data, body, refused)
                        if refused:
                            status = 402  # billing to Hermes, which does not retry it
                self.send_response(status)
                headers_sent = True
                for key, value in response.headers.items():
                    if key.lower() not in _HOP_BY_HOP and not (assembled and key.lower() == "content-type"):
                        self.send_header(key, value)
                if assembled:
                    self.send_header("Content-Type", "application/json")
                if streaming:
                    if not content_type:
                        self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    for chunk in response.iter_bytes():
                        accumulator.feed_sse(chunk)
                        if call is not None:
                            monitor.stream_progress(call, accumulator.usage, accumulator.streamed_tokens)
                        if captured is not None:
                            captured.append(chunk)
                        if accumulator.ended:
                            # Recorded before the caller sees the end, which can finish the turn at once.
                            close_call(status)
                            if captured is not None and status == 200:
                                self.server.drafts.put(draft_key, response.headers.get("content-type", ""),
                                                       b"".join(captured))
                        self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                        self.wfile.flush()
                        delivered = accumulator.ended
                        # An upstream or proxy (llama-swap) can hold a finished stream open; nothing follows its end.
                        if delivered:
                            break
                    self.wfile.write(b"0\r\n\r\n")
                else:
                    if data is None:
                        data = response.read()
                    if not assembled:
                        accumulator.feed_json(data)
                    if captured is not None and status == 200:
                        self.server.drafts.put(draft_key, response.headers.get("content-type", ""), data)
                    close_call(status)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                self.wfile.flush()
                hang_up.finish()
        except httpx.HTTPError as error:
            if hang_up.happened:
                status = status if delivered else CLIENT_CLOSED
                self.close_connection = True
            else:
                logger.warning("vgr tap: %s upstream failed: %s", route, error)
                idle = route == "cloud" and isinstance(error, httpx.ReadTimeout)
                if idle:
                    monitor.note_cloud_stall(_turn_of(self.headers.get(REQUEST_ID_HEADER)), int(_CLOUD_TIMEOUT.read))
                if headers_sent:
                    self.close_connection = True
                else:
                    status = 502
                    message = (_stall_message(int(_CLOUD_TIMEOUT.read)) if idle
                               else f"upstream unavailable: {type(error).__name__}")
                    self._send_json(502, {"error": {"message": message}})
        except (BrokenPipeError, ConnectionResetError):
            status = status if delivered else CLIENT_CLOSED
        finally:
            hang_up.finish()
            close_call(status)
            if turn_id is not None:
                monitor.end_turn(turn_id, status=status, served_model=served_model or accumulator.model)

    def _replay(self, call: Dict[str, Any], content_type: str, data: bytes) -> None:
        """Answer a local draft request with the reply the local tier gave the identical request."""
        logger.info("vgr tap: replaying the local draft for a repeated request")
        accumulator = _Accumulator()
        streaming = "text/event-stream" in content_type
        if streaming:
            accumulator.feed_sse(data)
            accumulator.feed_sse(b"\n")
        else:
            accumulator.feed_json(data)
        self.server.monitor.end_call(
            call, status=200, usage=None, text=accumulator.text, tool_call=accumulator.tool_call,
            p_yes=accumulator.p_yes, replayed=True,
        )
        try:
            self.send_response(200)
            self.send_header("Content-Type", content_type or "application/json")
            if streaming:
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n0\r\n\r\n")
            else:
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def _side_task(self, body: Dict[str, Any]) -> bool:
        """A Hermes side task (Hermes's client mark, no session id) the local tier can read: no images."""
        if self.headers.get(CLIENT_HEADER) != HERMES_CLIENT or self.headers.get(SESSION_HEADER):
            return False
        items = body.get("messages") or body.get("input")
        return not isinstance(items, list) or not any(
            isinstance(part, dict) and part.get("type") in _IMAGE_PARTS
            for item in items if isinstance(item, dict) and isinstance(item.get("content"), list)
            for part in item["content"]
        )

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass
