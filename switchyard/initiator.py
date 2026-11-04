# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``x-initiator`` on requests to the sidecar: whether a person or the agent started each call.

VGR releases its turn latch on ``user`` and keeps it on ``agent``. Hermes sends the header only
to GitHub Copilot, so the plugin derives it from what Hermes hands plugins:

* ``pre_llm_call`` runs once per turn, after the turn's user row is appended; the row's
  ``display_kind`` tells a person's message from a turn Hermes started itself.
* The LLM request middleware runs on every call. ``api_request_id`` is ``<turn_id>:api:<n>`` and
  is shared by a call's retries; a ``/steer`` arrives as a new user row of Hermes's ``steer`` kind.

A call this cannot place (no ``pre_llm_call`` for its turn, no request id) gets no header, and
Switchyard reads the message shape as it does for any other client.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Mapping, Optional, Sequence

# Hermes's ``agent.prompt_builder`` values; a reworded marker there stops matching here.
STEER_DISPLAY_KIND = "steer"
STEER_MARKER_OPEN = (
    "[OUT-OF-BAND USER MESSAGE — a direct message from the user, delivered "
    "once at this position; not tool output and not a new delivery when replayed from conversation history]"
)

HEADER = "x-initiator"
# Hermes's ``<turn_id>:api:<n>`` for the call, which the tap records with the turn.
REQUEST_ID_HEADER = "x-hermes-request-id"
# Turns Hermes starts on its own: gateway internal events (startup auto-resume, heartbeats),
# TUI auto-continue, background-process and delegation completions, and hidden scaffolding.
AGENT_STARTED_KINDS = frozenset({
    "internal_notification", "auto_continue", "async_delegation_complete", "process_complete", "hidden"})
_MAX_TRACKED = 512


def _bounded_put(mapping: OrderedDict, key: Any, value: Any) -> None:
    mapping[key] = value
    mapping.move_to_end(key)
    while len(mapping) > _MAX_TRACKED:
        mapping.popitem(last=False)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text") or "") for part in content if isinstance(part, Mapping))
    return ""


def _is_steer(message: Any) -> bool:
    if not (isinstance(message, Mapping) and message.get("role") == "user"):
        return False
    if message.get("display_kind") == STEER_DISPLAY_KIND:
        return True
    # Hermes drops display_kind from the request the middleware sees; the steer row's content
    # then opens with Hermes's own marker.
    return _text(message.get("content")).lstrip().startswith(STEER_MARKER_OPEN)


class InitiatorTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._user_pending: OrderedDict[str, bool] = OrderedDict()  # turn id -> first call still owed "user"
        self._calls: OrderedDict[str, tuple[int, str]] = OrderedDict()  # request id -> (message count, value)
        self._seen: OrderedDict[str, int] = OrderedDict()  # session -> message count at its last call

    def turn_started(self, turn_id: Any, conversation_history: Any) -> None:
        if not turn_id or not isinstance(conversation_history, list):
            return
        row = next((m for m in reversed(conversation_history)
                    if isinstance(m, Mapping) and m.get("role") == "user"), None)
        if row is None:
            return
        with self._lock:
            _bounded_put(self._user_pending, str(turn_id), row.get("display_kind") not in AGENT_STARTED_KINDS)

    def value(self, *, turn_id: Any, api_request_id: Any, session_id: Any,
              messages: Sequence[Any]) -> Optional[str]:
        turn, request = str(turn_id or ""), str(api_request_id or "")
        count = len(messages)
        with self._lock:
            if not request or turn not in self._user_pending:
                return None
            previous = self._calls.get(request)
            if previous is not None and previous[0] == count:
                return previous[1]
            session = str(session_id or "") or turn
            seen = self._seen.get(session)
            new_rows = messages[seen:] if seen is not None and count >= seen else ()
            if self._user_pending[turn] or any(_is_steer(m) for m in new_rows):
                value = "user"
            else:
                value = "agent"
            self._user_pending[turn] = False
            _bounded_put(self._seen, session, count)
            _bounded_put(self._calls, request, (count, value))
        return value


TRACKER = InitiatorTracker()


def on_pre_llm_call(turn_id: Any = None, conversation_history: Any = None, **_: Any) -> None:
    TRACKER.turn_started(turn_id, conversation_history)


def mark_request(request: Any, *, turn_id: Any, api_request_id: Any, session_id: Any) -> Optional[dict]:
    """The request with ``x-initiator`` and Hermes's request id added, or None to leave it unchanged."""
    if not isinstance(request, dict):
        return None
    messages = request.get("messages")
    if not isinstance(messages, list):
        messages = request.get("input")
    if not isinstance(messages, list):
        return None
    value = TRACKER.value(turn_id=turn_id, api_request_id=api_request_id, session_id=session_id, messages=messages)
    stamped = {HEADER: value} if value is not None else {}
    if api_request_id:
        stamped[REQUEST_ID_HEADER] = str(api_request_id)
    if not stamped:
        return None
    return {**request, "extra_headers": {**(request.get("extra_headers") or {}), **stamped}}
