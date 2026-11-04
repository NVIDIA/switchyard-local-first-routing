# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Why VGR served a turn where it did: the router's logged reason, matched to its turn and explained."""

from __future__ import annotations

from plugins.switchyard import reasons
from plugins.switchyard.decisions import LogFollower, parse
from plugins.switchyard.monitor import Monitor

# Captured from Switchyard 2231a5ea (the vendored wheel's commit) serving a VGR route.
CLOUD_LINE = (
    '2026-10-05T18:32:18.334482Z  INFO switchyard.request{otel.kind="server" openinference.span.kind="CHAIN"}:'
    'libsy.run{algorithm="vgr" switchyard.algorithm="vgr" openinference.span.kind="CHAIN" '
    'switchyard.route="switchyard/vgr" session_id="3152655bfa0c4c2aaa2f8e70f4c1d0e0" '
    'session.id="3152655bfa0c4c2aaa2f8e70f4c1d0e0"}: libsy: vgr decision reason="decided" branch=Some(Agentic) served=Cloud'
)
TIMEOUT_LINE = (
    '2026-10-05T18:39:20.715245Z  INFO switchyard.request{otel.kind="server" openinference.span.kind="CHAIN"}:'
    'libsy.run{algorithm="vgr" switchyard.algorithm="vgr" openinference.span.kind="CHAIN" '
    'switchyard.route="switchyard/vgr" session_id="e08f8241" session.id="e08f8241"}: '
    'libsy: vgr decision reason="local_timed_out" branch=None served=Cloud'
)
# VGR's two deliberation calls (``vgr/rungs.rs``): reasoning, then a verdict scored from logprobs.
DELIBERATION_PROMPT = ("You are a strict verifier. Judge whether the record shows the task was completed. "
                       "Reason carefully about whether the evidence supports the result.")
VERDICT_PROMPT = ("You are a strict verifier. Judge whether the record shows the task was completed. You are "
                  "also given a verifier's analysis of the record, which may be cut off before its conclusion; "
                  "weigh it, but judge the record yourself. Reply with exactly one word: yes or no.")
VGR_ROUTE = {"deadline_seconds": 300, "local_turn_budget_seconds": 600, "mode": "active"}


def _monitor() -> Monitor:
    return Monitor(
        local_model="qwen", cloud_model="opus",
        presets={"vgr": {"algorithm": "vgr", "mode": "active", "route": VGR_ROUTE}},
        default_preset="vgr",
    )


def _call(monitor: Monitor, turn_id: int, tier: str, system: str = "", *, text: str = "", p_yes=None,
          tool_call: bool = False, logprobs: bool = False) -> None:
    body = {"model": tier, "messages": [{"role": "system", "content": system}, {"role": "user", "content": "x"}]}
    if logprobs:
        body["logprobs"] = True
    call = monitor.begin_call(tier, body, turn_id)
    monitor.end_call(call, status=200, usage={"prompt_tokens": 10, "completion_tokens": 2},
                     text=text, tool_call=tool_call, p_yes=p_yes)


def _turn(monitor: Monitor, session: str = "s1") -> int:
    return monitor.begin_turn({"messages": [{"role": "user", "content": "fix the build"}]}, session, "vgr")


def _finished(monitor: Monitor, turn_id: int) -> dict:
    return next(t for t in monitor.snapshot()["turns"] if t["id"] == turn_id)


def test_parse_reads_reason_branch_tier_session_and_time():
    decision = parse(CLOUD_LINE)
    assert decision["reason"] == "decided"
    assert decision["branch"] == "agentic"
    assert decision["served"] == "cloud"
    assert decision["session_id"] == "3152655bfa0c4c2aaa2f8e70f4c1d0e0"
    assert abs(decision["at"] - 1791225138.334482) < 1e-3
    assert parse(TIMEOUT_LINE)["branch"] is None
    assert parse("2026-10-05T18:32:18Z  INFO switchyard: LLM request handled status=200") is None


def test_parse_reads_upstream_switchyard_tier_names_as_the_plugins():
    upstream = TIMEOUT_LINE.replace('"local_timed_out"', '"efficient_timed_out"')
    assert (parse(upstream)["reason"], parse(upstream)["served"]) == ("local_timed_out", "cloud")
    served_efficient = CLOUD_LINE.replace("served=Cloud", 'served=Efficient policy_version="1.1"')
    assert parse(served_efficient)["served"] == "local"
    assert parse(CLOUD_LINE.replace("served=Cloud", "served=Capable"))["served"] == "cloud"


def test_follower_reports_only_lines_appended_after_it_starts(tmp_path):
    log = tmp_path / "sidecar.log"
    log.write_text(TIMEOUT_LINE + "\n", encoding="utf-8")
    seen = []
    follower = LogFollower(log, seen.append)
    follower.start()
    follower.stop()
    with log.open("a", encoding="utf-8") as handle:
        handle.write("unrelated line\n" + CLOUD_LINE[:40])
    follower.poll()
    assert seen == []
    with log.open("a", encoding="utf-8") as handle:
        handle.write(CLOUD_LINE[40:] + "\n")
    follower.poll()
    assert [d["reason"] for d in seen] == ["decided"]


def test_decision_logged_before_the_response_reaches_its_turn_when_the_turn_closes(tmp_path):
    monitor = _monitor()
    log = tmp_path / "sidecar.log"
    log.write_text("", encoding="utf-8")
    follower = LogFollower(log, monitor.record_decision)
    follower.start()
    follower.stop()
    monitor.attach_decisions(follower.poll)
    other = _turn(monitor, session="other")
    turn = _turn(monitor, session="e08f8241")
    _call(monitor, turn, "local")
    _call(monitor, turn, "cloud")
    with log.open("a", encoding="utf-8") as handle:
        handle.write(TIMEOUT_LINE.replace("2026-10-05T18:39:20.715245Z", _now_stamp()) + "\n")
    monitor.end_turn(turn, status=200, served_model="opus")
    monitor.end_turn(other, status=200, served_model="qwen")
    finished = _finished(monitor, turn)
    assert finished["decision"] == {"reason": "local_timed_out", "branch": None, "served": "cloud", "policy_version": None}
    assert finished["why"]["headline"] == "Local model ran out of time"
    assert "300s" in finished["why"]["summary"]
    assert finished["outcome"] == "escalated_on_time_limit"
    assert _finished(monitor, other)["decision"] is None


def _now_stamp() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def test_agentic_escalation_after_passing_local_checks_blames_earlier_tool_errors():
    monitor = _monitor()
    turn = _turn(monitor)
    _call(monitor, turn, "local", text="Done.")
    _call(monitor, turn, "local", "You are a strict verifier. Evidence...", text="yes", p_yes=0.81, logprobs=True)
    monitor.record_decision({"reason": "decided", "branch": "agentic", "served": "cloud", "session_id": "s1", "at": None})
    monitor.end_turn(turn, status=200, served_model="opus")
    why = _finished(monitor, turn)["why"]
    assert why["headline"] == "Earlier tool errors blocked a local finish"
    assert why["checks"] == [{"label": "Local confidence check", "result": "81% sure the task is done (needs 20%)", "ok": True}]


def test_answer_escalation_lists_each_failed_check():
    monitor = _monitor()
    turn = _turn(monitor)
    _call(monitor, turn, "local", text="42")
    _call(monitor, turn, "local", "You are a strict verifier. Evidence...", text="no", p_yes=0.35, logprobs=True)
    _call(monitor, turn, "local", DELIBERATION_PROMPT, text="The record shows 42 but the task asked")
    _call(monitor, turn, "local", VERDICT_PROMPT, text="no", p_yes=0.3, logprobs=True)
    _call(monitor, turn, "cloud", "You are a strict answer verifier", text="no")
    _call(monitor, turn, "cloud", "You are a strict verifier. Evidence...", text="maybe")
    _call(monitor, turn, "cloud", text="43")
    monitor.record_decision({"reason": "decided", "branch": "answer", "served": "cloud", "session_id": "s1", "at": None})
    monitor.end_turn(turn, status=200, served_model="opus")
    why = _finished(monitor, turn)["why"]
    assert why["headline"] == "No check confirmed the local answer"
    assert [(c["label"], c["ok"]) for c in why["checks"]] == [
        ("Local confidence check", False),
        ("Local self-review", False),
        ("Cloud answer check", False),
        ("Cloud evidence check", None),
    ]
    assert why["checks"][0]["result"] == "35% sure the answer is right (needs 70%)"
    assert why["checks"][1]["result"] == "30% sure the answer is right (needs 50%)"
    assert [c["label"] for c in _finished(monitor, turn)["stages"][2:4]] == [
        "Local deliberation", "Deliberation verdict (logprob)"]


def test_policy_1_1_holds_a_typed_answer_to_the_self_graded_bar():
    line = CLOUD_LINE.replace("branch=Some(Agentic) served=Cloud", 'branch=Some(Answer) served=Local policy_version="1.1"')
    assert parse(line)["policy_version"] == "1.1"
    assert parse(CLOUD_LINE)["policy_version"] is None

    monitor = _monitor()
    turn = _turn(monitor)
    _call(monitor, turn, "local", text="2")
    _call(monitor, turn, "local", "You are a strict answer verifier", text="yes", p_yes=0.95, logprobs=True)
    monitor.record_decision({**parse(line), "session_id": "s1", "at": None})
    monitor.end_turn(turn, status=200, served_model="qwen")
    assert _finished(monitor, turn)["why"]["checks"] == [
        {"label": "Local confidence check", "result": "95% sure the answer is right (needs 90%)", "ok": True}]


def test_a_shadow_tool_step_the_judge_allowed_is_labelled_allowed_though_the_cloud_ran_it():
    monitor = Monitor(
        local_model="qwen", cloud_model="opus",
        presets={"vgr": {"algorithm": "vgr", "mode": "shadow", "route": {**VGR_ROUTE, "mode": "shadow"}}},
        default_preset="vgr",
    )
    turn = _turn(monitor)
    _call(monitor, turn, "local", tool_call=True)
    _call(monitor, turn, "local", "You are an escalation judge", text="no", p_yes=0.07, logprobs=True)
    _call(monitor, turn, "cloud", tool_call=True)
    monitor.record_decision({"reason": "turn_verification_complete", "branch": None, "served": "cloud",
                             "session_id": "s1", "at": None})
    monitor.end_turn(turn, status=200, served_model="opus")
    finished = _finished(monitor, turn)
    assert finished["outcome"] == "shadow_tool_call_allowed"
    assert finished["why"]["headline"] == "Tool call approved"
    assert finished["why"]["summary"].endswith("Shadow mode ran the cloud model's step instead.")


def test_coding_bar_is_the_unconditional_one_unless_the_cloud_check_approved():
    monitor = _monitor()
    alone = _turn(monitor)
    _call(monitor, alone, "local", text="def reverse_string(s): ...")
    _call(monitor, alone, "local", "You are a strict verifier. Evidence...", text="yes", p_yes=0.72, logprobs=True)
    monitor.record_decision({"reason": "decided", "branch": "coding", "served": "local", "session_id": "s1", "at": None})
    monitor.end_turn(alone, status=200, served_model="qwen")
    assert _finished(monitor, alone)["why"]["checks"][0] == {
        "label": "Local confidence check",
        "result": "72% sure the answer is right (needs 60%, or 20% if the cloud check approves)",
        "ok": True,
    }

    confirmed = _turn(monitor, session="s2")
    _call(monitor, confirmed, "local", text="def reverse_string(s): ...")
    _call(monitor, confirmed, "local", "You are a strict verifier. Evidence...", text="yes", p_yes=0.4, logprobs=True)
    _call(monitor, confirmed, "cloud", "You are a strict verifier. Evidence...", text="yes")
    monitor.record_decision({"reason": "decided", "branch": "coding", "served": "local", "session_id": "s2", "at": None})
    monitor.end_turn(confirmed, status=200, served_model="qwen")
    assert _finished(monitor, confirmed)["why"]["checks"][0] == {
        "label": "Local confidence check",
        "result": "40% sure the answer is right (needs 20% with the cloud check approving)",
        "ok": True,
    }


def test_a_coding_turn_whose_local_check_gave_no_verdict_says_so():
    monitor = _monitor()
    turn = _turn(monitor)
    _call(monitor, turn, "local", text="def reverse_string(s): ...")
    _call(monitor, turn, "local", "You are a strict verifier. Evidence...", text="maybe", logprobs=True)
    _call(monitor, turn, "cloud", text="def reverse_string(s): return s[::-1]")
    monitor.record_decision({"reason": "decided", "branch": "coding", "served": "cloud", "session_id": "s1", "at": None})
    monitor.end_turn(turn, status=200, served_model="opus")
    assert _finished(monitor, turn)["why"]["headline"] == "Local check gave no clear answer"


def test_a_cloud_error_served_to_the_user_is_named_as_one():
    monitor = _monitor()
    turn = _turn(monitor)
    _call(monitor, turn, "local", text="def reverse_string(s): ...")
    _call(monitor, turn, "local", "You are a strict verifier. Evidence...", text="no", p_yes=0.1, logprobs=True)
    body = {"model": "cloud", "messages": [{"role": "user", "content": "x"}]}
    monitor.end_call(monitor.begin_call("cloud", body, turn), status=400, usage=None,
                     text="Stream must be set to true", tool_call=False, p_yes=None)
    monitor.record_decision({"reason": "decided", "branch": "coding", "served": "cloud", "session_id": "s1", "at": None})
    monitor.end_turn(turn, status=400, served_model="opus")
    why = _finished(monitor, turn)["why"]
    assert why["headline"] == "Cloud model refused the request"
    assert "HTTP 400" in why["summary"]


def test_an_empty_prompt_has_its_own_explanation():
    monitor = _monitor()
    turn = _turn(monitor)
    _call(monitor, turn, "cloud", text="Hello! How can I help?")
    monitor.record_decision({"reason": "empty_prompt", "branch": None, "served": "cloud", "session_id": "s1", "at": None})
    monitor.end_turn(turn, status=200, served_model="opus")
    why = _finished(monitor, turn)["why"]
    assert why["headline"] == "Empty message"
    assert "recorded the reason" not in why["summary"]


def test_trajectory_escalation_cites_both_consecutive_votes():
    monitor = _monitor()
    first = _turn(monitor)
    _call(monitor, first, "local", tool_call=True)
    _call(monitor, first, "local", "You are an escalation judge", text="yes", p_yes=0.62, logprobs=True)
    monitor.record_decision({"reason": "turn_verification_complete", "branch": None, "served": "local",
                             "session_id": "s1", "at": None})
    monitor.end_turn(first, status=200, served_model="qwen")
    second = _turn(monitor)
    _call(monitor, second, "local", tool_call=True)
    _call(monitor, second, "local", "You are an escalation judge", text="yes", p_yes=0.77, logprobs=True)
    _call(monitor, second, "cloud", text="taking over")
    monitor.record_decision({"reason": "turn_verification_escalated", "branch": None, "served": "cloud",
                             "session_id": "s1", "at": None})
    monitor.end_turn(second, status=200, served_model="opus")
    assert _finished(monitor, first)["why"]["headline"] == "Tool call allowed with a warning"
    why = _finished(monitor, second)["why"]
    assert why["headline"] == "Escalation judge flagged the local work twice in a row"
    assert "77% this step and 62% on the step before" in why["summary"]


def test_a_cloud_decision_answered_by_the_local_model_is_a_cloud_failure_fallback():
    monitor = _monitor()
    turn = _turn(monitor)
    _call(monitor, turn, "cloud", text="")
    _call(monitor, turn, "local", text="Done.")
    monitor.record_decision({"reason": "turn_verification_latched", "branch": None, "served": "cloud",
                             "session_id": "s1", "at": None})
    monitor.end_turn(turn, status=200, served_model="qwen")
    finished = _finished(monitor, turn)
    assert finished["outcome"] == "local_after_cloud_failure"
    assert finished["why"]["headline"] == "Cloud model failed, so the local model answered"

    turn = _turn(monitor)
    call = monitor.begin_call("cloud", {"model": "cloud", "messages": [{"role": "user", "content": "x"}]}, turn)
    monitor.end_call(call, status=403, usage=None, text="", tool_call=False, p_yes=None,
                     out_of_credits="You exceeded your current quota")
    _call(monitor, turn, "local", text="Done.")
    monitor.record_decision({"reason": "turn_verification_latched", "branch": None, "served": "cloud",
                             "session_id": "s1", "at": None})
    monitor.end_turn(turn, status=200, served_model="qwen")
    assert _finished(monitor, turn)["why"]["headline"] == "Cloud model is out of credits, so the local model answered"

    turn = _turn(monitor)
    call = monitor.begin_call("cloud", {"model": "cloud", "messages": [{"role": "user", "content": "x"}]}, turn)
    monitor.note_cloud_stall(turn, 300)
    monitor.end_call(call, status=502, usage=None, text="", tool_call=False, p_yes=None)
    _call(monitor, turn, "local", text="Done.")
    monitor.record_decision({"reason": "decided", "branch": "answer", "served": "cloud", "session_id": "s1", "at": None})
    monitor.end_turn(turn, status=200, served_model="qwen")
    why = _finished(monitor, turn)["why"]
    assert why["headline"] == "Cloud model stopped responding, so the local model answered"
    assert "sent nothing for 300 s" in why["summary"]


def test_a_shadow_step_held_on_the_cloud_says_it_was_not_checked():
    monitor = Monitor(
        local_model="qwen", cloud_model="opus",
        presets={"vgr": {"algorithm": "vgr", "mode": "shadow", "route": {**VGR_ROUTE, "mode": "shadow"}}},
        default_preset="vgr",
    )
    turn = _turn(monitor)
    _call(monitor, turn, "cloud", tool_call=True)
    monitor.record_decision({"reason": "turn_verification_latched", "branch": None, "served": "cloud",
                             "session_id": "s1", "at": None})
    monitor.end_turn(turn, status=200, served_model="opus")
    finished = _finished(monitor, turn)
    assert finished["outcome"] == "shadow_unchecked"
    assert finished["why"]["headline"] == "Shadow mode stopped checking this message"
    assert "escalated" not in finished["why"]["summary"]


def test_turn_budget_reason_uses_the_preset_budget():
    turn = {"algorithm": "vgr", "served_tier": "cloud", "stages": [], "task_type": None,
            "decision": {"reason": "local_turn_budget", "branch": None, "served": "cloud"}}
    why = reasons.explain(turn, {"local_turn_budget_seconds": 900})
    assert why["headline"] == "This message used up its local time"
    assert "900s" in why["summary"]
    assert why["source"] == "router"


def test_without_a_logged_reason_the_account_is_marked_inferred():
    monitor = _monitor()
    turn = _turn(monitor)
    _call(monitor, turn, "cloud", text="answer")
    monitor.end_turn(turn, status=200, served_model="opus")
    why = _finished(monitor, turn)["why"]
    assert why["source"] == "inferred"
    assert why["headline"] == "Sent to the cloud without a local attempt"


def test_inferred_coding_turn_with_tool_history_is_judged_as_tool_using_work():
    monitor = _monitor()
    session = {"messages": [
        {"role": "user", "content": "fix the build"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]}
    turn = monitor.begin_turn(session, "s1", "vgr")
    _call(monitor, turn, "local", "You classify a user request so a router...", text="coding")
    _call(monitor, turn, "local", text="Done.")
    _call(monitor, turn, "local", "You are a strict verifier. Evidence...", text="no", p_yes=0.1, logprobs=True)
    _call(monitor, turn, "cloud", text="Fixed.")
    monitor.end_turn(turn, status=200, served_model="opus")
    why = _finished(monitor, turn)["why"]
    assert why["source"] == "inferred"
    assert why["checks"][0]["result"] == "10% sure the task is done (needs 20%)"


def test_other_algorithms_get_no_account():
    monitor = Monitor(local_model="qwen", cloud_model="opus",
                      presets={"r": {"algorithm": "random", "mode": None, "route": {}}}, default_preset="r")
    turn = monitor.begin_turn({"messages": []}, "s1", "r")
    monitor.end_turn(turn, status=200, served_model="qwen")
    assert monitor.snapshot()["turns"][0]["why"] is None
