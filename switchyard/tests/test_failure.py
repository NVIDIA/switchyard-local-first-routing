# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which tool calls a failed turn had already run, in each wire format Hermes can send."""

from __future__ import annotations

import json

from plugins.switchyard import failure


def _chat_call(call_id: str, name: str) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}]}


def test_only_the_current_messages_finished_calls_count():
    chat = {"messages": [
        {"role": "user", "content": "earlier task"},
        _chat_call("old", "terminal"),
        {"role": "tool", "tool_call_id": "old", "content": "done"},
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "now this"},
        _chat_call("a", "terminal"),
        {"role": "tool", "tool_call_id": "a", "content": ""},
        _chat_call("b", "write_file"),
        {"role": "tool", "tool_call_id": "b", "content": ""},
        _chat_call("c", "terminal"),
        {"role": "tool", "tool_call_id": "c", "content": ""},
        _chat_call("pending", "patch"),
    ]}
    anthropic = {"messages": [
        {"role": "user", "content": "now this"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "terminal", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": ""}]},
    ]}
    responses = {"input": [
        {"role": "user", "content": [{"type": "input_text", "text": "now this"}]},
        {"type": "function_call", "call_id": "a", "name": "terminal", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "a", "output": ""},
    ]}
    assert failure.ran_this_turn(chat) == ["terminal", "write_file", "terminal"]
    assert failure.note(chat) == ("Before this error, 3 tool calls for this message had already run, "
                                  "and nothing was undone: terminal (2), write_file.")
    assert failure.ran_this_turn(anthropic) == failure.ran_this_turn(responses) == ["terminal"]
    assert failure.note({"messages": [{"role": "user", "content": "hi"}]}) is None


def test_the_note_leads_the_error_message_and_request_errors_are_left_alone():
    body = {"messages": [{"role": "user", "content": "go"}, _chat_call("a", "terminal"),
                         {"role": "tool", "tool_call_id": "a", "content": ""}]}
    error = json.dumps({"type": "error", "error": {"type": "api_error", "message": "upstream failed"}}).encode()
    message = json.loads(failure.annotate(error, body))["error"]["message"]
    assert message.endswith(". upstream failed")
    assert message.startswith("Before this error, 1 tool call")
    assert failure.annotate(b"not json", body) == b"not json"
    assert [status for status in (400, 401, 403, 404, 413, 429, 500, 503) if failure.annotates(status)] == [
        401, 403, 429, 500, 503]


def test_a_tier_out_of_credits_is_named_and_its_provider_quoted_once():
    said = "You exceeded your current quota."
    no_credit = failure.out_of_credits(429, json.dumps({"error": {"message": said, "code": "insufficient_quota"}}).encode())
    assert no_credit == said
    assert failure.out_of_credits(429, b'{"error": {"message": "Rate limit reached", "code": "rate_limit_exceeded"}}') is None
    user = {"messages": [{"role": "user", "content": "hi"}]}

    for engine_said, quoted in ((f"upstream error: {said}", 1), ("backend temporarily unavailable", 1)):
        data = json.dumps({"error": {"message": engine_said}}).encode()
        message = json.loads(failure.annotate(data, user, {"tier": "local", "message": said}))["error"]["message"]
        assert message.startswith("Switchyard's efficient model is out of credits (insufficient_quota), "
                                  "and no other model answered in its place.")
        assert message.count(said) == quoted
        assert message.endswith(engine_said)
