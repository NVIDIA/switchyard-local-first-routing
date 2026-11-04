# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where Switchyard moved a chat to the cloud model or back to the local one, and why.

The monitor records every request with the Hermes session id the provider profile stamps on it, so
one chat's requests can be picked out of its state. A change is a request served on the other tier
than the chat's request before it, with the routing preset choosing both. Hermes Desktop shows each
change in the chat: the plugin opens the turn's reply with a ``::switchyard-route`` directive per
change, which the desktop half renders in place, and strips them from every request the model is
sent. In the classic CLI, :class:`CliNotice` prints a line.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Mapping, Optional

from .settings import ROLES

logger = logging.getLogger(__name__)

DIRECTIVE = "switchyard-route"
# VGR keeps the rest of a user message on the cloud once one step escalates; those steps say only that.
LATCHED = "turn_verification_latched"
_MAX_TRACKED = 512
_MAX_CHANGES = 5
# The tap records a request just after its last byte reaches Hermes, so a hook can run first.
_SETTLE_TRIES = 10
_SETTLE_S = 0.2
_CLOCK_SLACK_S = 0.25
_LEADING_DIRECTIVES = re.compile(r"\A(?:::switchyard-route\{[^{}\n]*\}[ \t]*(?:\n[ \t]*)*)+")
_UNSAFE_ATTR = str.maketrans({'"': "'", "{": "(", "}": ")"})


def _finished(state: Mapping[str, Any], session_id: str) -> List[Mapping[str, Any]]:
    return [t for t in state.get("turns") or [] if t.get("session_id") == session_id]


def _pinned(state: Mapping[str, Any], turn: Mapping[str, Any]) -> bool:
    """Whether the user's single-tier model choice sent ``turn`` to its tier."""
    return bool(((state.get("presets") or {}).get(turn.get("preset")) or {}).get("pinned"))


def _change(state: Mapping[str, Any], older: Optional[Mapping[str, Any]],
            newer: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    tier, before = newer.get("served_tier"), (older or {}).get("served_tier")
    why = newer.get("why")
    # A pin, for every chat or by picking a single-tier model, is the user's choice, not a routing change.
    if state.get("pin") or _pinned(state, newer):
        return None
    refused = (newer.get("out_of_credits") or {}).get("tier")
    if tier in ROLES and refused in ROLES and refused != tier:
        return {
            "escalated": tier == "cloud",
            "tier": tier,
            "model": state.get(f"{tier}_model"),
            "algorithm": newer.get("algorithm"),
            "why": {"headline": f"{ROLES[refused].capitalize()} model is out of credits",
                    "summary": f"The {ROLES[tier]} model answered in its place. Add credits, or set another "
                               f"{ROLES[refused]} model with `hermes switchyard setup`."},
        }
    if tier == "cloud":
        if before == "cloud" and (why or {}).get("reason") == LATCHED:
            return None
    elif tier != "local" or before != "cloud" or _pinned(state, older or {}):
        return None
    return {
        "escalated": tier == "cloud",
        "tier": tier,
        "model": state.get(f"{tier}_model"),
        "algorithm": newer.get("algorithm"),
        "why": {"headline": why.get("headline"), "summary": why.get("summary")} if why else None,
    }


def since(state: Optional[Mapping[str, Any]], session_id: str, started_at: float) -> Optional[Dict[str, Any]]:
    """The session's requests sent from ``started_at`` on, and the changes among them; None until one is recorded.

    Each change carries ``step``, the 1-based position of the request that made it. A preset that
    decides every request afresh can repeat a change; only the first of a run counts."""
    if not state:
        return None
    finished = _finished(state, session_id)  # newest first
    count = sum(1 for t in finished if (t.get("started") or 0) >= started_at - _CLOCK_SLACK_S)
    if not count:
        return None
    window = finished[count - 1::-1]
    before = finished[count] if count < len(finished) else None
    changes: List[Dict[str, Any]] = []
    for step, (older, newer) in enumerate(zip([before, *window[:-1]], window), 1):
        change = _change(state, older, newer)
        if change is not None and not (changes and changes[-1]["escalated"] == change["escalated"]):
            changes.append({**change, "step": step})
    return {"requests": count, "changes": changes}


def wait_since(fetch_state: Callable[[], Optional[Mapping[str, Any]]], session_id: str, started_at: float,
               requests: int = 1) -> Optional[Dict[str, Any]]:
    """:func:`since`, once the monitor holds ``requests`` of them or the settle time runs out."""
    found = None
    for attempt in range(_SETTLE_TRIES):
        state = fetch_state()
        if state is None:  # the sidecar is not answering; nothing is worth holding a turn for
            return found
        found = since(state, session_id, started_at)
        if found is not None and found["requests"] >= requests:
            return found
        if attempt + 1 < _SETTLE_TRIES:
            time.sleep(_SETTLE_S)
    return found


def _attr(value: Any, limit: int) -> str:
    return " ".join(str(value).translate(_UNSAFE_ATTR).split())[:limit]


def directive(change: Mapping[str, Any], steps: int) -> str:
    """``change`` as the transcript directive the desktop half renders; values are cut to fit its parser."""
    why = change.get("why") or {}
    attrs = {
        "change": "escalated" if change["escalated"] else "deescalated",
        "tier": change["tier"],
        "model": _attr(change.get("model") or "", 120),
        "algorithm": _attr(change.get("algorithm") or "", 32),
        "step": str(change["step"]) if steps > 1 else "",
        "steps": str(steps) if steps > 1 else "",
        "reason": _attr(why.get("headline") or "", 160),
        "summary": _attr(why.get("summary") or "", 400),
    }
    body = " ".join(f'{key}="{value}"' for key, value in attrs.items() if value)
    return f"::{DIRECTIVE}{{{body}}}"


def mark_reply(text: str, found: Optional[Mapping[str, Any]]) -> Optional[str]:
    """``text`` opened with a directive per change in ``found``; None when nothing changed."""
    changes = (found or {}).get("changes") or []
    if not text or not changes:
        return None
    lines = [directive(change, found["requests"]) for change in changes[:_MAX_CHANGES]]
    return "\n\n".join([*lines, text])


def unmark(text: str) -> str:
    """``text`` without the directives :func:`mark_reply` put in front of it."""
    return _LEADING_DIRECTIVES.sub("", text, count=1)


def _unmark_message(message: Any) -> Any:
    if not isinstance(message, Mapping) or message.get("role") != "assistant":
        return message
    content = message.get("content")
    if isinstance(content, str):
        text = unmark(content)
        return message if text == content else {**message, "content": text}
    if not isinstance(content, list):
        return message
    # The reply's text is its first text part; reasoning parts can come before it.
    index = next((i for i, part in enumerate(content) if isinstance(part, Mapping) and isinstance(part.get("text"), str)),
                 None)
    if index is None:
        return message
    text = unmark(content[index]["text"])
    if text == content[index]["text"]:
        return message
    return {**message, "content": [*content[:index], {**content[index], "text": text}, *content[index + 1:]]}


def unmark_request(request: Any) -> Optional[Dict[str, Any]]:
    """An LLM request with :func:`mark_reply`'s directives taken out of its history; None when it holds none."""
    if not isinstance(request, dict):
        return None
    key = "messages" if isinstance(request.get("messages"), list) else "input"
    messages = request.get(key)
    if not isinstance(messages, list):
        return None
    unmarked = [_unmark_message(message) for message in messages]
    if all(new is old for new, old in zip(unmarked, messages)):
        return None
    return {**request, key: unmarked}


def notice_text(change: Mapping[str, Any]) -> str:
    if change.get("escalated"):
        verb = "escalated" if change.get("algorithm") == "vgr" else "routed"
        line = f"Switchyard {verb} this turn to {change.get('model') or 'the cloud model'}"
    else:
        line = f"Switchyard returned this turn to {change.get('model') or 'the local model'}"
    why = change.get("why") or {}
    return f"{line}: {why['headline']}" if why.get("headline") else f"{line}."


class CliNotice:
    """Shows one line when a turn goes to the cloud model, and one when a turn comes back to the local one."""

    def __init__(self, fetch_state: Callable[[], Optional[Mapping[str, Any]]], show: Callable[[str], None]) -> None:
        self._fetch_state = fetch_state
        self._show = show
        self._shown: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()

    def watch(self, session_id: str, turn_id: str, started_at: float) -> None:
        """Check the request in the background, so the agent never waits on the sidecar."""
        threading.Thread(
            target=self._check_quietly, args=(session_id, turn_id, started_at), daemon=True, name="switchyard-notice"
        ).start()

    def _check_quietly(self, session_id: str, turn_id: str, started_at: float) -> None:
        try:
            self.check(session_id, turn_id, started_at)
        except Exception:  # a failed notice must not print a traceback into the user's terminal
            logger.debug("Switchyard escalation notice failed", exc_info=True)

    def check(self, session_id: str, turn_id: str, started_at: float) -> None:
        found = wait_since(self._fetch_state, session_id, started_at)
        for change in (found or {}).get("changes") or []:
            key = f"{session_id}:{turn_id}:{'escalated' if change['escalated'] else 'deescalated'}"
            with self._lock:
                if key in self._shown:
                    continue
                self._shown[key] = None
                while len(self._shown) > _MAX_TRACKED:
                    self._shown.popitem(last=False)
            self._show(notice_text(change))
