# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The monitor's totals keep background tasks apart, as the timeline does."""

from __future__ import annotations

import json

from plugins.switchyard.monitor import Monitor
from plugins.switchyard.tap import _Accumulator


def _turn(monitor: Monitor, session, calls) -> None:
    turn = monitor.begin_turn({"messages": [{"role": "user", "content": "hi"}]}, session, "r")
    for tier, prompt in calls:
        call = monitor.begin_call(tier, {"messages": []}, turn)
        monitor.end_call(call, status=200, usage={"prompt_tokens": prompt, "completion_tokens": 1},
                         text="ok", tool_call=False, p_yes=None)
    monitor.end_turn(turn, status=200, served_model="qwen")


def test_background_tasks_are_totalled_apart_from_the_chat():
    monitor = Monitor(local_model="qwen", cloud_model="opus",
                      presets={"r": {"algorithm": "random", "mode": None, "route": {}}}, default_preset="r")
    _turn(monitor, "s1", [("local", 100)])
    _turn(monitor, None, [("local", 40), ("local", 7)])

    totals = monitor.snapshot()["totals"]
    background = totals["background"]
    assert (totals["turns"], totals["local"]["calls"], totals["local"]["prompt"]) == (2, 3, 147)
    assert (background["turns"], background["local"]["calls"], background["local"]["prompt"]) == (1, 2, 47)
    assert background["served"] == {"local": 1, "cloud": 0}
    assert sum(stage["calls"] for stage in background["by_stage"].values()) == 2
    preset = totals["by_preset"]["r"]
    assert (preset["turns"], preset["background"]["turns"], preset["background"]["local"]["prompt"]) == (2, 1, 47)


def test_a_stream_without_usage_keeps_the_thinking_it_streamed_in_its_count():
    monitor = Monitor(local_model="qwen", cloud_model="opus",
                      presets={"r": {"algorithm": "random", "mode": None, "route": {}}}, default_preset="r")
    turn = monitor.begin_turn({"messages": [{"role": "user", "content": "hi"}]}, "s1", "r")
    call = monitor.begin_call("local", {"messages": [], "stream": True}, turn)
    stream = _Accumulator()
    for delta in ({"reasoning_content": "Let me add these up."}, {"reasoning_content": " Two"}, {"content": "4"}):
        stream.feed_sse(f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n".encode())
        monitor.stream_progress(call, stream.usage, stream.streamed_tokens)
    live = monitor.snapshot()["current"]["tokens"]
    monitor.end_call(call, status=200, usage=stream.usage, text=stream.text, tool_call=False, p_yes=None)
    monitor.end_turn(turn, status=200, served_model="qwen")

    # 25 characters in three deltas; the reply text alone, "4", would count one token.
    assert live == {"prompt": 0, "completion": 7}
    assert monitor.snapshot()["turns"][0]["tokens"]["local"]["completion"] == 7


def test_a_calls_prompt_is_estimated_from_its_request_until_the_usage_comes():
    monitor = Monitor(local_model="qwen", cloud_model="opus",
                      presets={"r": {"algorithm": "random", "mode": None, "route": {}}}, default_preset="r")
    turn = monitor.begin_turn({"messages": [{"role": "user", "content": "hi"}]}, "s1", "r")
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 40_000}}
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "x" * 400}, image]}]}
    served = monitor.begin_call("local", body, turn)
    failed = monitor.begin_call("cloud", body, turn)
    live = monitor.snapshot()["current"]["tokens"]["prompt"]
    monitor.end_call(failed, status=502, usage=None, text="", tool_call=False, p_yes=None)
    monitor.end_call(served, status=200, usage={"prompt_tokens": 120, "completion_tokens": 3},
                     text="ok", tool_call=False, p_yes=None)
    monitor.end_turn(turn, status=200, served_model="qwen")

    # Both requests read before either model answers: the 400 characters of text, not the image.
    assert 200 <= live <= 260
    tokens = monitor.snapshot()["turns"][0]["tokens"]
    assert (tokens["local"]["prompt"], tokens["cloud"]["prompt"]) == (120, 0)


def test_watch_finishes_a_turn_that_ends_while_a_newer_one_is_shown(monkeypatch, capsys):
    import time
    from types import SimpleNamespace

    from plugins.switchyard import _cmd_watch, sidecar

    monitor = Monitor(local_model="qwen", cloud_model="opus",
                      presets={"r": {"algorithm": "random", "mode": None, "route": {}}}, default_preset="r")
    turns, calls, snapshots = [], [], []
    for session in ("cli", "desktop"):
        turns.append(monitor.begin_turn({"messages": [{"role": "user", "content": "hi"}]}, session, "r"))
        calls.append(monitor.begin_call("local", {"messages": []}, turns[-1]))
        snapshots.append(monitor.snapshot())
    # The older turn ends while the newer one is still the current one, then the newer ends.
    for turn, call in zip(turns, calls):
        monitor.end_call(call, status=200, usage={"prompt_tokens": 5, "completion_tokens": 1},
                         text="ok", tool_call=False, p_yes=None)
        monitor.end_turn(turn, status=200, served_model="qwen")
        snapshots.append(monitor.snapshot())

    def sleep(_seconds):
        if not snapshots:
            raise KeyboardInterrupt

    monkeypatch.setattr(sidecar, "load_settings", lambda: SimpleNamespace(tap_url="http://127.0.0.1:1"))
    monkeypatch.setattr(sidecar, "fetch_state", lambda _cfg: snapshots.pop(0))
    monkeypatch.setattr(time, "sleep", sleep)
    assert _cmd_watch() == 0

    finished = [line for line in capsys.readouterr().out.splitlines() if " finished" in line]
    assert finished == ["#1 finished: served local, served by the local model, 5 in / 1 out tokens",
                        "#2 finished: served local, served by the local model, 5 in / 1 out tokens"]


def test_event_records_stay_unique_and_ordered_across_sidecar_restarts(tmp_path):
    events = tmp_path / "events.jsonl"
    for _ in range(2):
        monitor = Monitor(local_model="qwen", cloud_model="opus", events_path=events,
                          presets={"r": {"algorithm": "random", "mode": None, "route": {}}}, default_preset="r")
        for _ in range(3):
            turn = monitor.begin_turn({"messages": [{"role": "user", "content": "hi"}]}, "s1", "r",
                                      hermes_request_id="t1:api:1")
            monitor.end_turn(turn, status=200, served_model="qwen")

    records = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in records] == [1, 2, 3, 1, 2, 3]
    uids = [r["uid"] for r in records]
    assert len(set(uids)) == 6
    assert uids == sorted(uids)
    assert len({r["run_id"] for r in records}) == 2
    assert {r["hermes_request_id"] for r in records} == {"t1:api:1"}
