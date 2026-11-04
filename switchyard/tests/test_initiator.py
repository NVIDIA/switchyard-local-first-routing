# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the plugin changes on requests and replies in a real Hermes turn loop: x-initiator and route markers."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from plugins.switchyard import _on_llm_request, escalation, initiator
from plugins.switchyard.tests.test_escalation import _monitor, _request

FRONT = "http://127.0.0.1:4110/front/v1"


def _response(tool: bool) -> SimpleNamespace:
    calls = [SimpleNamespace(id="c1", type="function", function=SimpleNamespace(
        name="todo", arguments=json.dumps({"todos": []})))] if tool else None
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="" if tool else "done", tool_calls=calls),
        finish_reason="tool_calls" if tool else "stop")], model="switchyard/vgr", usage=None)


@pytest.fixture
def agent(monkeypatch):
    from hermes_cli import plugins
    from run_agent import AIAgent

    manager = plugins.get_plugin_manager()
    monkeypatch.setitem(manager._hooks, "pre_llm_call", [initiator.on_pre_llm_call])
    monkeypatch.setitem(manager._middleware, "llm_request", [_on_llm_request])
    monkeypatch.setattr(initiator, "TRACKER", initiator.InitiatorTracker())
    with patch("agent.process_bootstrap.OpenAI"), patch("agent.model_metadata.fetch_model_metadata", return_value={}):
        agent = AIAgent(model="switchyard/vgr", api_key="k", base_url=FRONT, enabled_toolsets=["todo"],
                        quiet_mode=True, skip_context_files=True, skip_memory=True, max_iterations=5)
    agent.client = MagicMock()
    agent._cached_system_prompt = "Use the tools."
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _sent(agent, replies, on_call=None):
    """The x-initiator of each request one turn sends."""
    seen = []

    def create(**kwargs):
        seen.append((kwargs.get("extra_headers") or {}).get(initiator.HEADER))
        if on_call:
            on_call(len(seen))
        return replies.pop(0)

    agent.client.chat.completions.create.side_effect = create
    return seen


def test_a_turn_marks_its_first_call_and_steers_user_and_the_rest_agent(agent):
    seen = _sent(agent, [_response(True), _response(True), _response(False)],
                 on_call=lambda n: n == 1 and agent.steer("use the staging config instead"))
    agent.run_conversation("tidy the todo list")
    assert seen == ["user", "user", "agent"]


def test_a_steer_is_hermess_steer_row_not_text_that_looks_like_its_marker():
    from agent.prompt_builder import steer_user_row

    row = steer_user_row("use the staging config instead")
    assert initiator._is_steer(row)
    assert initiator._is_steer({key: value for key, value in row.items() if key != "display_kind"})
    assert not initiator._is_steer({"role": "user", "content": "[OUT-OF-BAND USER MESSAGE] do it now"})
    assert not initiator._is_steer({**row, "role": "tool"})


def test_turns_hermes_starts_itself_are_agent_and_the_next_person_turn_is_user(agent):
    seen = _sent(agent, [_response(False), _response(False)])
    agent.run_conversation("[System note: resume the interrupted task]", persist_user_display_kind="auto_continue")
    agent.run_conversation("thanks, now add a test")
    assert seen == ["agent", "user"]


def test_each_call_carries_hermes_request_id_for_the_event_record(agent):
    replies, ids = [_response(True), _response(False)], []

    def create(**kwargs):
        ids.append((kwargs.get("extra_headers") or {}).get(initiator.REQUEST_ID_HEADER))
        return replies.pop(0)

    agent.client.chat.completions.create.side_effect = create
    agent.run_conversation("tidy the todo list")
    assert len(ids) == len(set(ids)) == 2
    assert all(isinstance(i, str) and ":api:" in i for i in ids)


def test_a_retry_keeps_its_first_attempts_value_and_a_reused_id_with_new_messages_does_not():
    tracker = initiator.InitiatorTracker()
    tracker.turn_started("t1", [{"role": "user", "content": "fix the build"}])
    call = dict(turn_id="t1", api_request_id="t1:api:1", session_id="s")
    assert tracker.value(**call, messages=[{}, {}]) == "user"
    assert tracker.value(**call, messages=[{}, {}]) == "user"
    assert tracker.value(**call, messages=[{}, {}, {}]) == "agent"


def test_desktop_replies_open_with_where_the_turn_changed_tier_and_the_model_never_sees_it(agent, monkeypatch):
    import plugins.switchyard as plugin
    from hermes_cli import plugins
    from tui_gateway.server import _resolve_agent_platform

    manager = plugins.get_plugin_manager()
    monkeypatch.setitem(manager._hooks, "post_api_request", [plugin._on_post_api_request])
    monkeypatch.setitem(manager._hooks, "transform_llm_output", [plugin._on_transform_llm_output])
    monkeypatch.setattr(escalation, "_CLOCK_SLACK_S", 0)
    monitor = _monitor()
    monkeypatch.setattr(plugin, "_fetch_state", monitor.snapshot)
    monkeypatch.setenv("HERMES_DESKTOP", "1")
    monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)
    agent.platform = _resolve_agent_platform(None)
    tiers, sent = ["local", "cloud", "local", "cloud"], []

    def create(**kwargs):
        sent.append(kwargs["messages"])
        _request(monitor, agent.session_id, tiers.pop(0), "decided")
        return _response(len(tiers) == 3)

    agent.client.chat.completions.create.side_effect = create
    plugin.sidecar.note_renderer(True)
    first = agent.run_conversation("tidy the todo list")
    assert first["final_response"].startswith(
        '::switchyard-route{change="escalated" tier="cloud" model="aws/anthropic/opus" algorithm="vgr" step="2" '
        'steps="2" reason="No check confirmed the local answer" summary="')
    assert first["final_response"].endswith('"}\n\ndone')

    second = agent.run_conversation("thanks", conversation_history=first["messages"])
    assert second["final_response"].startswith('::switchyard-route{change="deescalated" tier="local" model="qwen"')
    assert [m["content"] for m in sent[-1] if m["role"] == "assistant"][-1] == "done"
    assert escalation.unmark(second["final_response"]) == "done"

    # With the Desktop half off nothing draws a note, and Desktop would show it as raw text.
    plugin.sidecar.note_renderer(False)
    third = agent.run_conversation("again", conversation_history=second["messages"])
    assert third["final_response"] == "done"


def test_an_unplaced_call_or_another_provider_is_left_unchanged(agent):
    request = {"messages": [{"role": "user", "content": "hi"}]}
    marked = initiator.mark_request(request, turn_id="unknown", api_request_id="unknown:api:1", session_id="s")
    assert marked["extra_headers"] == {initiator.REQUEST_ID_HEADER: "unknown:api:1"}
    assert initiator.mark_request(request, turn_id="unknown", api_request_id=None, session_id="s") is None
    assert _on_llm_request(request=request, provider="openrouter", base_url="https://openrouter.ai/api/v1") is None
