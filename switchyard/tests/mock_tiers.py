# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible stand-ins for the local and cloud tiers of a VGR route.

Every verifier says ``yes`` unless the user's task contains ``REFUTE``, in
which case every verifier says ``no``.
"""

from __future__ import annotations

import json
import math
import select
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from plugins.switchyard import stages


class MockTier(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, name: str, answer: str) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.name = name
        self.answer = answer
        self.requests: List[Dict[str, Any]] = []
        # Seconds a local draft waits for its caller to hang up before answering.
        self.hold = 0.0
        # Seconds a stream waits, once its headers are sent, for its caller to hang up before going on.
        self.stall = 0.0
        # Seconds a chat stream stays open after its [DONE], as a proxy such as llama-swap may keep it.
        self.linger = 0.0
        # Set to an event to hold a chat stream after its first content event until the event is set.
        self.midstream: Optional[threading.Event] = None
        # An HTTP error status every request fails with; 200 answers normally.
        self.status = 200
        # The JSON body a failing status carries; empty for a generic error.
        self.error_body = ""
        # False plays a server whose replies carry no logprobs, as a thinking model's do.
        self.logprobs = True
        # True plays the Codex backend, which refuses a Responses request without stream: true.
        self.stream_only = False
        # (token, probability) pairs a verdict's first token offers in place of the yes/no pair.
        self.alternatives: List[tuple[str, float]] = []
        self.abandoned = threading.Event()
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"

    def stop(self) -> None:
        self.shutdown()
        self.server_close()


def _reply(server: MockTier, body: Dict[str, Any]) -> tuple[str, bool]:
    """``(text, refuted)`` for one request."""
    stage = stages.classify("local" if server.name == "local" else "cloud", body)
    task = json.dumps(body.get("messages", []))
    refuted = "REFUTE" in task
    if stage == stages.TYPING:
        return "answer", refuted
    if stage in (stages.LOCAL_DRAFT, stages.CLOUD_SERVE):
        return server.answer, refuted
    return ("no" if refuted else "yes"), refuted


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: MockTier

    def log_message(self, *_: Any) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
        self.server.requests.append({
            "body": body,
            "path": self.path,
            "authorization": self.headers.get("authorization"),
            "x_api_key": self.headers.get("x-api-key"),
            "originator": self.headers.get("originator"),
        })
        refused = self.server.stream_only and self.path.endswith("/responses") and not body.get("stream")
        if self.server.status != 200 or refused:
            payload = (json.dumps({"detail": "Stream must be set to true"}) if refused else self.server.error_body
                       or json.dumps({"error": {"message": "mock tier failure"}})).encode()
            self.send_response(400 if refused else self.server.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        text, refuted = _reply(self.server, body)
        if self.server.stall and not body.get("stream") and self._caller_hung_up(self.server.stall):
            self.server.abandoned.set()
            return
        if self.server.hold and stages.classify("local", body) == stages.LOCAL_DRAFT and self._caller_hung_up():
            self.server.abandoned.set()
            return
        if self.path.endswith("/v1/messages"):
            self._anthropic(body, text)
            return
        if self.path.endswith("/responses"):
            self._responses(body, self.server.answer)
            return
        usage = {"prompt_tokens": 50, "completion_tokens": max(1, len(text) // 4), "total_tokens": 0}
        choice: Dict[str, Any] = {
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }
        if body.get("logprobs") and self.server.logprobs:
            yes, no = (0.05, 0.95) if refuted else (0.97, 0.03)
            alternatives = self.server.alternatives or [("yes", yes), ("no", no)]
            choice["logprobs"] = {"content": [{
                "token": text, "logprob": math.log(yes if text == "yes" else no),
                "top_logprobs": [{"token": token, "logprob": math.log(p)} for token, p in alternatives],
            }]}
        if body.get("stream"):
            self._stream(body, text, usage)
            return
        payload = json.dumps({
            "id": "mock", "object": "chat.completion", "model": body.get("model"),
            "choices": [choice], "usage": usage,
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _caller_hung_up(self, seconds: float = 0.0) -> bool:
        readable, _, _ = select.select([self.connection], [], [], seconds or self.server.hold)
        return bool(readable) and not self.connection.recv(1, socket.MSG_PEEK)

    def _stream(self, body: Dict[str, Any], text: str, usage: Dict[str, Any]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        if self.server.stall:
            self.wfile.flush()
            if self._caller_hung_up(self.server.stall):
                self.server.abandoned.set()
                return
        model = body.get("model")
        events = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": usage},
        ]
        for event in events:
            event.update({"id": "mock", "object": "chat.completion.chunk", "model": model})
            self._chunk(f"data: {json.dumps(event)}\n\n".encode())
            if self.server.midstream is not None and event is events[0]:
                self.server.midstream.wait(10)
        self._chunk(b"data: [DONE]\n\n")
        if self.server.linger and self._caller_hung_up(self.server.linger):
            return
        self.wfile.write(b"0\r\n\r\n")

    def _anthropic(self, body: Dict[str, Any], text: str) -> None:
        usage = {"input_tokens": 10, "cache_read_input_tokens": 40, "output_tokens": max(1, len(text) // 4)}
        message = {
            "id": "mock", "type": "message", "role": "assistant", "model": body.get("model"),
            "content": [{"type": "text", "text": text}], "stop_reason": "end_turn", "usage": usage,
        }
        if not body.get("stream"):
            payload = json.dumps(message).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        start = {**message, "content": [], "usage": {**usage, "output_tokens": 0}}
        events = [
            {"type": "message_start", "message": start},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": usage["output_tokens"]}},
            {"type": "message_stop"},
        ]
        for event in events:
            self._chunk(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())
        self.wfile.write(b"0\r\n\r\n")

    def _responses(self, body: Dict[str, Any], text: str) -> None:
        """OpenAI Responses API, as the Codex backend speaks it."""
        usage = {"input_tokens": 50, "output_tokens": max(1, len(text) // 4), "total_tokens": 0}
        item = {"type": "message", "id": "msg_mock", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}]}
        response = {"id": "resp_mock", "object": "response", "created_at": 0, "model": body.get("model"),
                    "status": "completed", "output": [item], "usage": usage}
        if not body.get("stream"):
            payload = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(200)
        if not self.server.stream_only:  # the Codex backend sends its streams with no Content-Type
            self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        pending = {**response, "status": "in_progress", "output": [], "usage": None}
        events = [
            {"type": "response.created", "response": pending},
            {"type": "response.output_item.added", "output_index": 0, "item": {**item, "status": "in_progress", "content": []}},
            {"type": "response.content_part.added", "output_index": 0, "content_index": 0, "item_id": "msg_mock",
             "part": {"type": "output_text", "text": "", "annotations": []}},
            {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "item_id": "msg_mock", "delta": text},
            {"type": "response.output_text.done", "output_index": 0, "content_index": 0, "item_id": "msg_mock", "text": text},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.completed", "response": response},
        ]
        for number, event in enumerate(events):
            event["sequence_number"] = number
            self._chunk(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode())
        self.wfile.write(b"0\r\n\r\n")

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()
