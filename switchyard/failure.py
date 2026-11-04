# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a failed turn leaves done: the tool calls of the user's current message that already ran.

Nothing a turn did is undone when a later call of it fails, but Hermes reports only the
provider's error. The tap puts the turn's finished tool calls ahead of that error's message,
which Hermes shows under "Provider said:". Only tool names go in: the calls themselves are in
the chat already, and Hermes classifies an error by its message, so it must not carry the
calls' arbitrary text.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .settings import ROLES

_MAX_NAMES = 4
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]")
_MAX_PROVIDER_MESSAGE = 300
# An account that cannot pay, as cloud providers word it; a retry cannot clear any of these.
_NO_CREDIT_CODES = frozenset({
    "insufficient_quota", "insufficient_credits", "billing_not_active", "payment_required", "balance_depleted",
    "credit_balance_exhausted", "no_usable_credits",
})
_NO_CREDIT_WORDS = (
    "insufficient_quota", "exceeded your current quota", "insufficient credits", "insufficient balance",
    "credit balance", "credits exhausted", "no credits", "out of credits", "top up your credits",
    "payment required", "billing hard limit", "budget limit exceeded",
)


def annotates(status: int) -> bool:
    """Whether a failure with ``status`` is annotated. Hermes reads a 400's message length to tell a
    context overflow from a format error, so request errors are left as they are."""
    return status >= 500 or status in (401, 403, 429)


def out_of_credits(status: int, data: bytes) -> Optional[str]:
    """The provider's message when a failed call says its account has no credit left, else None."""
    if status not in (402, 403, 429):
        return None
    try:
        payload = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        payload = None
    error = payload.get("error") if isinstance(payload, dict) else None
    error = error if isinstance(error, dict) else {}
    codes = {str(error.get(key) or "").lower() for key in ("code", "type")}
    message = str(error.get("message") or (data[:_MAX_PROVIDER_MESSAGE].decode("utf-8", "replace") if not error else ""))
    if status == 402 or codes & _NO_CREDIT_CODES or any(word in message.lower() for word in _NO_CREDIT_WORDS):
        return " ".join(message.split())[:_MAX_PROVIDER_MESSAGE] or "the account has no credit left"
    return None


def out_of_credits_message(no_credit: Mapping[str, str]) -> str:
    """What the chat shows ahead of the error when a turn failed after a tier refused for lack of credit;
    worded so Hermes classifies it as billing, which it does not retry. ``no_credit`` is ``{"tier", "message"}``,
    with ``tiers`` naming every tier that refused."""
    if len(set(no_credit.get("tiers") or ())) > 1:
        return ("Both of Switchyard's models, efficient and capable, are out of credits (insufficient_quota). "
                "Add credits or set other models with `hermes switchyard setup`.")
    role = ROLES.get(no_credit["tier"], no_credit["tier"])
    return (f"Switchyard's {role} model is out of credits (insufficient_quota), and no other model answered in "
            f"its place. Add credits or set another {role} model with `hermes switchyard setup`.")


def _items(body: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    items = body.get("messages")
    if not isinstance(items, list):
        items = body.get("input")
    return [item for item in items if isinstance(item, Mapping)] if isinstance(items, list) else []


def _calls_and_results(item: Mapping[str, Any]) -> Tuple[List[Tuple[str, str]], List[str]]:
    """``([(call id, tool name)], [ids of the results])`` one message or Responses item carries."""
    calls = [(str(call.get("id")), str((call.get("function") or {}).get("name") or ""))
             for call in item.get("tool_calls") or () if isinstance(call, Mapping)]
    results = [str(item.get("tool_call_id"))] if item.get("role") == "tool" else []
    if item.get("type") == "function_call":
        calls.append((str(item.get("call_id")), str(item.get("name") or "")))
    elif item.get("type") == "function_call_output":
        results.append(str(item.get("call_id")))
    content = item.get("content")
    for block in content if isinstance(content, list) else ():
        if not isinstance(block, Mapping):
            continue
        if block.get("type") == "tool_use":
            calls.append((str(block.get("id")), str(block.get("name") or "")))
        elif block.get("type") == "tool_result":
            results.append(str(block.get("tool_use_id")))
    return calls, results


def ran_this_turn(body: Mapping[str, Any]) -> List[str]:
    """Names of the tool calls since the latest user message that have a result, in order."""
    names: Dict[str, str] = {}
    results: List[str] = []
    for item in _items(body):
        calls, done = _calls_and_results(item)
        # Anthropic carries tool results as a user message; that continues the turn.
        if item.get("role") == "user" and not done:
            names, results = {}, []
            continue
        names.update(calls)
        results.extend(done)
    return [names[call_id] for call_id in results if call_id in names]


def note(body: Mapping[str, Any]) -> Optional[str]:
    """The sentence naming what already ran, or None when no tool call of the turn has."""
    ran = ran_this_turn(body)
    if not ran:
        return None
    counts = Counter(_UNSAFE_NAME.sub("", name) or "tool" for name in ran)
    named = [f"{name} ({count})" if count > 1 else name for name, count in counts.most_common(_MAX_NAMES)]
    more = len(counts) - len(named)
    listed = ", ".join(named) + (f" and {more} more" if more > 0 else "")
    steps = "1 tool call" if len(ran) == 1 else f"{len(ran)} tool calls"
    return f"Before this error, {steps} for this message had already run, and nothing was undone: {listed}."


def annotate(data: bytes, body: Mapping[str, Any], no_credit: Optional[Mapping[str, str]] = None) -> bytes:
    """``data``, a failed completion's JSON error, with :func:`note` ahead of its message, and a tier's
    lack of credit ahead of it too when that is what the turn ran into, quoting the provider unless the
    message already does."""
    text = " ".join(part for part in (
        note(body), out_of_credits_message(no_credit) if no_credit else None,
    ) if part) or None
    if text is None:
        return data
    try:
        payload = json.loads(data)
    except (ValueError, UnicodeDecodeError):
        return data
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return data
    message = error.get("message") if isinstance(error.get("message"), str) else ""
    if no_credit and no_credit["message"] not in message:
        text = f"{text} Its provider said: {no_credit['message']}"
    error["message"] = f"{text} {message}" if message else text
    return json.dumps(payload).encode()
