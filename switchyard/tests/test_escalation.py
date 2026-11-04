# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the chat and the CLI notice say when a chat moves between the local and the cloud model."""

from __future__ import annotations

import time
from typing import Optional

from plugins.switchyard import escalation
from plugins.switchyard.monitor import Monitor

_VGR = {"vgr": {"algorithm": "vgr", "mode": None, "route": {}}}
_TOOL_STEP = [{"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                                                                   "function": {"name": "ls", "arguments": "{}"}}]},
              {"role": "tool", "tool_call_id": "c1", "content": "README.md"}]


def _monitor() -> Monitor:
    return Monitor(local_model="qwen", cloud_model="aws/anthropic/opus", presets=_VGR, default_preset="vgr")


def _begin(monitor: Monitor, session: str, tier: str, reason: Optional[str] = None, step: bool = False):
    messages = [{"role": "user", "content": "fix the build"}, *(_TOOL_STEP if step else [])]
    turn = monitor.begin_turn({"messages": messages}, session, "vgr")
    call = monitor.begin_call(tier, {"messages": []}, turn)
    monitor.end_call(call, status=200, usage={"prompt_tokens": 10, "completion_tokens": 2},
                     text="ok", tool_call=False, p_yes=None)
    if reason:
        monitor.record_decision({"session_id": session, "reason": reason, "branch": "chat", "served": tier,
                                 "at": time.time()})
    return turn


def _request(monitor: Monitor, session: str, tier: str, reason: Optional[str] = None, step: bool = False) -> None:
    turn = _begin(monitor, session, tier, reason, step)
    monitor.end_turn(turn, status=200, served_model="qwen" if tier == "local" else "aws/anthropic/opus")


def _headline(state, index: int) -> str:
    return state["turns"][index]["why"]["headline"]


def _changes(state, started_at: float):
    return [(c["step"], c["escalated"], c["model"]) for c in escalation.since(state, "s1", started_at)["changes"]]


def test_a_turn_reports_the_step_that_escalated_and_a_turn_back_on_the_local_model(monkeypatch):
    monkeypatch.setattr(escalation, "_CLOCK_SLACK_S", 0)
    monitor = _monitor()
    _request(monitor, "s1", "local", "decided")
    turn = time.time()
    _request(monitor, "s1", "local", "decided")
    _request(monitor, "s1", "cloud", "decided", step=True)
    _request(monitor, "s1", "cloud", escalation.LATCHED, step=True)
    _request(monitor, "s2", "local", "decided")
    state = monitor.snapshot()

    assert _changes(state, turn) == [(2, True, "aws/anthropic/opus")]
    assert escalation.since(state, "s1", turn)["changes"][0]["why"]["headline"] == "No check confirmed the local answer"
    assert escalation.since(state, "s2", turn)["changes"] == []
    assert escalation.since(state, "unknown", turn) is None
    assert escalation.since({**state, "pin": "cloud"}, "s1", turn)["changes"] == []

    escalated_again = time.time()
    _request(monitor, "s1", "cloud", "decided")
    back = time.time()
    _request(monitor, "s1", "local", "decided")
    _request(monitor, "s1", "local", "decided", step=True)
    state = monitor.snapshot()
    assert _changes(state, escalated_again) == [(1, True, "aws/anthropic/opus"), (2, False, "qwen")]
    assert _changes(state, back) == [(1, False, "qwen")]
    assert escalation.since(state, "s1", back)["changes"][0]["why"]["headline"] == _headline(state, 1)
    assert escalation.since({**state, "pin": "local"}, "s1", back)["changes"] == []


def test_a_tier_without_credit_is_named_when_the_other_one_answers(monkeypatch):
    monkeypatch.setattr(escalation, "_CLOCK_SLACK_S", 0)
    monitor = _monitor()
    for refused, served, model in (("cloud", "local", "qwen"), ("local", "cloud", "aws/anthropic/opus")):
        started_at = time.time()
        turn = monitor.begin_turn({"messages": [{"role": "user", "content": "fix the build"}]}, "s1", "vgr")
        call = monitor.begin_call(refused, {"messages": []}, turn)
        monitor.end_call(call, status=403, usage=None, text="", tool_call=False, p_yes=None,
                         out_of_credits="You exceeded your current quota")
        served_call = monitor.begin_call(served, {"messages": []}, turn)
        monitor.end_call(served_call, status=200, usage=None, text="ok", tool_call=False, p_yes=None)
        monitor.end_turn(turn, status=200, served_model=model)
        state = monitor.snapshot()

        change = escalation.since(state, "s1", started_at)["changes"][0]
        role, other = ("Capable", "efficient") if refused == "cloud" else ("Efficient", "capable")
        assert (change["escalated"], change["model"]) == (served == "cloud", model)
        assert change["why"]["headline"] == f"{role} model is out of credits"
        assert change["why"]["summary"].startswith(f"The {other} model answered in its place.")
        assert state["credit_alerts"] == {refused: {"message": "You exceeded your current quota",
                                                    "at": state["credit_alerts"][refused]["at"]}}
        assert state["turns"][0]["stages"][0]["status"] == "out_of_credits"
        # The tier's next answer clears its alert.
        clear = monitor.begin_call(refused, {"messages": []})
        monitor.end_call(clear, status=200, usage=None, text="ok", tool_call=False, p_yes=None)
        assert monitor.snapshot()["credit_alerts"] == {}


def test_the_cli_notice_waits_for_the_request_and_shows_each_change_once(monkeypatch):
    monkeypatch.setattr(escalation, "_SETTLE_S", 0)
    monkeypatch.setattr(escalation, "_CLOCK_SLACK_S", 0)
    monitor = _monitor()
    _request(monitor, "s1", "cloud", "decided")
    started_at = time.time()
    turn = _begin(monitor, "s1", "local", "decided")
    states = [monitor.snapshot()]
    monitor.end_turn(turn, status=200, served_model="qwen")
    states.append(monitor.snapshot())
    shown = []
    notice = escalation.CliNotice(lambda: states.pop(0) if len(states) > 1 else states[0], shown.append)

    notice.check("s1", "t1", started_at)
    assert shown == [f"Switchyard returned this turn to qwen: {_headline(states[0], 0)}"]
    later = time.time()
    _request(monitor, "s1", "cloud", "decided")
    states[0] = monitor.snapshot()
    notice.check("s1", "t2", later)
    _request(monitor, "s1", "cloud", escalation.LATCHED, step=True)
    states[0] = monitor.snapshot()
    notice.check("s1", "t2", later)
    assert shown[1:] == ["Switchyard escalated this turn to aws/anthropic/opus: No check confirmed the local answer"]
