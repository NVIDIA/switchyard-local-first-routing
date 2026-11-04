# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain-language account of why VGR kept a turn local or sent it to the cloud.

The router's own reason (see :mod:`decisions`) says which rule fired; the
verifier calls the tap recorded say what each check found. The thresholds
mirror ``crates/libsy/src/algorithms/vgr/decide.rs`` and ``runtime.rs``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from . import stages

# Readout score each branch needs before the local answer may stand (decide.rs `dial`).
READOUT_BAR = {"answer": 0.7, "default_verified": 0.7, "chat": 0.3, "agentic": 0.2, "coding": 0.2}
# From policy 1.1 the local model grades its own typed answer, so the bar is the engine's readout bar.
ANSWER_SELF_GRADED = 0.9
ANSWER_SELF_GRADED_POLICY = (1, 1)
# Verdict-call score a deliberation needs to affirm (decide.rs `DELIBERATION`).
DELIBERATION_BAR = 0.5
# Readout a coding answer needs when no cloud check approves it (decide.rs `CODING_CONFIDENT`).
CODING_CONFIDENT = 0.6
# A proposed tool call counts as an escalation vote at this P(escalate) (runtime.rs).
TURN_ESCALATE_AT = 0.5
TURN_CONFIRMATIONS = 2
# A local check whose first token was not mostly "yes" or "no" (stages.score).
NO_VERDICT = "no clear yes or no answer, which never counts as approval"
_TYPE_BRANCH = {"coding": "coding", "answer": "answer", "chat": "chat", "agentic": "agentic", "abstain": "default_verified"}
_BRANCH_NAMES = {
    "coding": "coding",
    "answer": "a question to answer",
    "chat": "conversation",
    "agentic": "tool-using work",
    "default_verified": "a general request",
}


def _pct(p: float) -> str:
    return f"{round(p * 100)}%"


def _last(calls: List[Mapping[str, Any]], stage: str, tier: Optional[str] = None) -> Optional[Mapping[str, Any]]:
    for call in reversed(calls):
        if call["stage"] == stage and (tier is None or call["tier"] == tier):
            return call
    return None


def _policy(turn: Mapping[str, Any]) -> tuple:
    version = str((turn.get("decision") or {}).get("policy_version") or "")
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError:
        return ()


def _readout_bar(turn: Mapping[str, Any], branch: Optional[str]) -> Optional[float]:
    if branch == "answer" and _policy(turn) >= ANSWER_SELF_GRADED_POLICY:
        return ANSWER_SELF_GRADED
    return READOUT_BAR.get(branch or "")


def _deliberation_p(calls: List[Mapping[str, Any]]) -> Optional[float]:
    verdict = _last(calls, stages.DELIBERATION_VERDICT)
    return verdict.get("p_yes") if verdict else None


def _verdict_check(label: str, call: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if call is None:
        return None
    verdict = call.get("verdict")
    if verdict == "yes":
        return {"label": label, "result": "approved", "ok": True}
    if verdict == "no":
        return {"label": label, "result": "rejected", "ok": False}
    if call.get("status") not in (None, "ok", "running"):
        return {"label": label, "result": "failed, which never counts as approval", "ok": False}
    return {"label": label, "result": "no clear verdict, which never counts as approval", "ok": None}


def checks(turn: Mapping[str, Any], branch: Optional[str]) -> List[Dict[str, Any]]:
    """What each verifier that ran on this turn's final answer found, cheapest first."""
    calls = turn["stages"]
    found: List[Dict[str, Any]] = []
    readout = _last(calls, stages.READOUT)
    subject = "the task is done" if branch == "agentic" else "the answer is right"
    if readout is not None:
        bar = _readout_bar(turn, branch)
        if readout.get("p_yes") is None:
            found.append({"label": "Local confidence check", "result": NO_VERDICT, "ok": None})
        else:
            p = readout["p_yes"]
            needs = f" (needs {_pct(bar)})" if bar is not None else ""
            if branch == "coding" and bar is not None:
                if _cloud_approved(calls):
                    needs = f" (needs {_pct(bar)} with the cloud check approving)"
                else:
                    needs = f" (needs {_pct(CODING_CONFIDENT)}, or {_pct(bar)} if the cloud check approves)"
                    bar = CODING_CONFIDENT
            found.append({
                "label": "Local confidence check",
                "result": f"{_pct(p)} sure {subject}{needs}",
                "ok": (p >= bar) if bar is not None else None,
            })
    if _last(calls, stages.DELIBERATION) is not None:
        p = _deliberation_p(calls)
        found.append({
            "label": "Local self-review",
            "result": NO_VERDICT if p is None else f"{_pct(p)} sure {subject} (needs {_pct(DELIBERATION_BAR)})",
            "ok": None if p is None else p >= DELIBERATION_BAR,
        })
    for stage, label in ((stages.ANSWER_VERIFIER, "Cloud answer check"), (stages.EVIDENCE_VERIFIER, "Cloud evidence check")):
        check = _verdict_check(label, _last(calls, stage, "cloud"))
        if check is not None:
            found.append(check)
    return found


def _cloud_approved(calls: List[Mapping[str, Any]]) -> bool:
    return any(
        (call := _last(calls, stage, "cloud")) is not None and call.get("verdict") == "yes"
        for stage in (stages.ANSWER_VERIFIER, stages.EVIDENCE_VERIFIER)
    )


def _trajectory_p(turn: Mapping[str, Any]) -> Optional[float]:
    judge = _last(turn["stages"], stages.TRAJECTORY_JUDGE)
    return judge.get("p_yes") if judge else None


def _seconds(route: Mapping[str, Any], key: str, default: int) -> int:
    value = route.get(key, default)
    return int(value) if isinstance(value, (int, float)) else default


def _result(reason: Optional[str], served: Optional[str], headline: str, summary: str,
            found: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    return {
        "reason": reason,
        "served": served,
        "headline": headline,
        "summary": summary,
        "checks": found or [],
        "source": "router" if reason else "inferred",
    }


def _decided(turn: Mapping[str, Any], branch: Optional[str], served: Optional[str], route: Mapping[str, Any]) -> Dict[str, Any]:
    found = checks(turn, branch)
    kind = _BRANCH_NAMES.get(branch or "", "the request")
    if served == "local":
        passed = [c for c in found if c["ok"]]
        reason = f"{passed[0]['label']}: {passed[0]['result']}." if passed else "The checks VGR ran approved it."
        return _result("decided", served, "Local answer approved", f"VGR typed this turn as {kind}. {reason}", found)
    if route.get("mode") == "shadow":
        return _result("decided", served, "Shadow mode always serves the cloud",
                       "VGR is in shadow mode: it records what it would have decided but always answers from the cloud.", found)
    if branch in (None, "unknown"):
        return _result("decided", served, "Nothing to check",
                       "The local model's attempt was empty, so there was no answer to verify.", found)
    readout = _last(turn["stages"], stages.READOUT)
    if readout is not None and readout.get("p_yes") is None and not any(c["ok"] for c in found):
        return _result("decided", served, "Local check gave no clear answer",
                       f"VGR typed this turn as {kind}. When the local model checked its own answer, it did not "
                       "reply with a clear yes or no: most of its first word went to something else. VGR never "
                       "counts that as approval, so the cloud model answered instead.", found)
    if branch == "coding":
        return _result("decided", served, "No check confirmed the local code",
                       "VGR typed this as a coding request with no tool calls, so nothing ran the code and VGR "
                       "judged the code itself. Local code needs the local model to be at least 60% sure it is "
                       "right, or at least 20% sure with the cloud model confirming it. Neither held, so the "
                       "cloud model answered instead.", found)
    if branch == "agentic":
        bar = READOUT_BAR["agentic"]
        readout = _last(turn["stages"], stages.READOUT)
        deliberation = _deliberation_p(turn["stages"])
        local_ok = bool(readout and readout.get("p_yes") is not None and readout["p_yes"] >= bar) or (
            deliberation is not None and deliberation >= DELIBERATION_BAR)
        if local_ok:
            confirm = _last(turn["stages"], stages.EVIDENCE_VERIFIER, "cloud")
            if confirm is not None:
                return _result("decided", served, "Cloud double-check rejected the local work",
                               "Tool calls earlier in this session hit errors, so after the local checks passed VGR "
                               "asked the cloud model to confirm the work, and it did not.", found)
            return _result("decided", served, "Earlier tool errors blocked a local finish",
                           "The local checks passed, but tool calls earlier in this session failed and the run did "
                           "not end with enough clean steps for VGR to trust the local result.", found)
        return _result("decided", served, "Local checks weren't confident the task is done",
                       "VGR typed this turn as tool-using work. Neither local check confirmed the work was finished, "
                       "so the cloud model took over.", found)
    note = " These replies also need a session with no tool errors." if branch in ("chat", "default_verified") else ""
    return _result("decided", served, "No check confirmed the local answer",
                   f"VGR typed this turn as {kind}. None of its checks approved the local answer, "
                   f"so the cloud model answered instead.{note}", found)


def _inferred_branch(turn: Mapping[str, Any]) -> Optional[str]:
    """The router's branch rule (vgr.rs ``derive_capabilities``) applied to what the tap saw."""
    task_type = turn.get("task_type") or ""
    if task_type == "agentic" or (turn.get("tool_history") and task_type not in ("answer", "chat")):
        return "agentic"
    return _TYPE_BRANCH.get(task_type)


def explain(
    turn: Mapping[str, Any],
    route: Optional[Mapping[str, Any]] = None,
    previous: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Why a finished VGR turn was served where it was; None for other algorithms or failed turns.

    ``route`` is the preset's VGR route settings; ``previous`` is the session's prior turn,
    whose trajectory vote counts toward a two-in-a-row escalation.
    """
    if turn.get("algorithm") != "vgr":
        return None
    route = route or {}
    decision = turn.get("decision") or {}
    reason = decision.get("reason")
    served = turn.get("served_tier") or decision.get("served")
    if served not in ("local", "cloud"):
        return None
    branch = decision.get("branch") or _inferred_branch(turn)
    deadline = _seconds(route, "deadline_seconds", 300)
    budget = _seconds(route, "local_turn_budget_seconds", 600)
    p = _trajectory_p(turn)

    if served == "local" and decision.get("served") == "cloud" and "cloud" in (
            (turn.get("out_of_credits") or {}).get("tiers") or ()):
        return _result(reason, served, "Cloud model is out of credits, so the local model answered",
                       "VGR sent this step to the cloud model, but its provider refused for lack of credit. The local "
                       "model answered instead, and VGR did not check that answer.")
    if served == "local" and decision.get("served") == "cloud" and turn.get("cloud_stalled_s"):
        return _result(reason, served, "Cloud model stopped responding, so the local model answered",
                       f"VGR sent this step to the cloud model, but it sent nothing for {turn['cloud_stalled_s']} s, "
                       "so VGR stopped waiting for it. The local model answered instead, and VGR did not check "
                       "that answer.")
    if served == "local" and decision.get("served") == "cloud":
        return _result(reason, served, "Cloud model failed, so the local model answered",
                       "VGR sent this step to the cloud model, but the cloud model returned an error or could not "
                       "be reached. The local model answered instead, and VGR did not check that answer.")
    serve = _last(turn["stages"], stages.CLOUD_SERVE, "cloud")
    if served == "cloud" and serve is not None and str(serve.get("status")).startswith("http_"):
        refused = serve["status"].startswith("http_4")
        return _result(reason, served, "Cloud model refused the request" if refused else "Cloud model returned an error",
                       f"VGR sent this step to the cloud model, which returned an error (HTTP {serve['status'][5:]}), "
                       "so nothing answered. `hermes switchyard setup` checks the cloud tier's model, address "
                       "and key.", checks(turn, branch))
    if reason == "decided":
        return _decided(turn, branch, served, route)
    if reason == "local_timed_out":
        drafted = _last(turn["stages"], stages.LOCAL_DRAFT) is not None
        summary = (f"VGR gives each request {deadline}s to produce and check a local answer. The local model "
                   "did not finish in time, so the cloud model took over." if drafted else
                   f"There was no time left in the {deadline}s decision window to try locally.")
        return _result(reason, served, "Local model ran out of time", summary)
    if reason == "local_turn_budget":
        return _result(reason, served, "This message used up its local time",
                       f"VGR allows {budget}s of local work per user message, counted on the clock from the "
                       "message's first step, so time spent running tools and waiting for you to approve "
                       "them counts too. That was used up, so the rest of this message goes to the cloud. "
                       "The budget resets with the next user message.")
    if reason == "local_image_unsupported":
        return _result(reason, served, "Image in the request",
                       "The request contains an image. VGR treats the local model as text-only, so it went "
                       "straight to the cloud.")
    if reason == "local_unavailable":
        return _result(reason, served, "Local model unavailable",
                       "The local model server returned an error or could not be reached.")
    if reason == "malformed_tool_call":
        return _result(reason, served, "Broken tool call from the local model",
                       "The local model's tool call was cut off or had invalid arguments, so VGR sent this step "
                       "to the cloud instead of running it.")
    if reason == "breaker_open":
        return _result(reason, served, "Local model paused after repeated failures",
                       "Several recent local calls failed or timed out, so VGR is skipping the local model for a while.")
    if reason == "turn_verification_escalated":
        prior = _trajectory_p(previous) if previous else None
        votes = f"{_pct(p)} this step" if p is not None else "a warning this step"
        if prior is not None and prior >= TURN_ESCALATE_AT:
            votes += f" and {_pct(prior)} on the step before"
        return _result(reason, served, "Escalation judge flagged the local work twice in a row",
                       "Before each tool call the local model proposes, a judge estimates the odds that the work "
                       f"is going wrong. It gave {votes}. {TURN_CONFIRMATIONS} warnings in a row at "
                       f"{_pct(TURN_ESCALATE_AT)} or more hand the rest of this user message to the cloud.")
    if reason == "turn_verification_latched" and route.get("mode") == "shadow":
        return _result(reason, served, "Shadow mode stopped checking this message",
                       "Shadow mode answers every step from the cloud model. Once it has served one tool step of "
                       "a user message, VGR does not check that message's later steps, so it recorded no decision "
                       "for this one. Checking starts again with the next user message.")
    if reason == "turn_verification_latched":
        return _result(reason, served, "Already handed to the cloud for this message",
                       "An earlier step of this user message was escalated. The cloud model stays in charge "
                       "until the next user message.")
    if reason == "turn_verification_complete":
        shadow = " Shadow mode ran the cloud model's step instead." if route.get("mode") == "shadow" else ""
        if p is None:
            return _result(reason, served, "Tool call allowed",
                           "The escalation judge gave no usable score, and VGR does not escalate on a missing "
                           f"score.{shadow}")
        if p >= TURN_ESCALATE_AT:
            return _result(reason, served, "Tool call allowed with a warning",
                           f"The escalation judge gave {_pct(p)} odds that the work is going wrong. That counts as "
                           f"one warning; one more in a row hands the work to the cloud.{shadow}")
        return _result(reason, served, "Tool call approved",
                       f"The escalation judge gave {_pct(p)} odds that the work is going wrong; it escalates at "
                       f"{_pct(TURN_ESCALATE_AT)} or more, {TURN_CONFIRMATIONS} steps in a row.{shadow}")
    if reason == "kill_switch":
        return _result(reason, served, "Local routing switched off",
                       "VGR's kill switch is on, so every request goes to the cloud.")
    if reason == "empty_prompt":
        return _result(reason, served, "Empty message",
                       "The message was empty, so there was nothing for the local model to answer or for VGR "
                       "to check. It went to the cloud without a local attempt.")
    if reason == "mode_off":
        return _result(reason, served, "VGR is off for this preset",
                       "This preset's VGR mode is off, so every request goes to the cloud.")
    if reason:
        return _result(reason, served, "Sent to the cloud" if served == "cloud" else "Kept local",
                       f"Switchyard recorded the reason as {reason.replace('_', ' ')}.")
    return _inferred(turn, branch, served, p)


def _inferred(turn: Mapping[str, Any], branch: Optional[str], served: str, p: Optional[float]) -> Dict[str, Any]:
    """Best account from the recorded calls alone, when the router's reason was not in the log."""
    outcome = turn.get("outcome")
    if outcome == "local_tool_call":
        detail = f" The escalation judge gave {_pct(p)} odds that the work is going wrong." if p is not None else ""
        return _result(None, served, "Tool call approved", "The local model's tool call was allowed." + detail)
    if outcome == "local_committed":
        return _result(None, served, "Local answer approved", "VGR's checks approved the local answer.",
                       checks(turn, branch))
    if outcome == "escalated_by_trajectory_judge":
        detail = f" It gave {_pct(p)} odds this step." if p is not None else ""
        return _result(None, served, "Escalation judge flagged the local work",
                       "The judge that reviews each proposed tool call decided the work was going wrong." + detail)
    if outcome == "escalated_without_attempt":
        return _result(None, served, "Sent to the cloud without a local attempt",
                       "Usually because this message's local time budget ran out, an earlier step already "
                       "escalated, or the request had an image.")
    return _result(None, served, "No check confirmed the local answer",
                   "The local answer was not approved, so the cloud model answered instead.", checks(turn, branch))
