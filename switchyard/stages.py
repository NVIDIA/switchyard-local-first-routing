# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classify the upstream calls a Switchyard route makes into routing stages.

Switchyard's judge and verifier calls are fresh requests with fixed system
prompts (``crates/libsy/src/algorithms``), so the stage of any call that
reaches a tier can be read from the request alone. Every other call is task
work: the local tier's answer (VGR's candidate draft) or the cloud tier's.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

LOCAL_DRAFT = "local_draft"
CLOUD_SERVE = "cloud_serve"
# VGR
TYPING = "typing"
READOUT = "readout"
DELIBERATION = "deliberation"
DELIBERATION_VERDICT = "deliberation_verdict"
ANSWER_VERIFIER = "answer_verifier"
EVIDENCE_VERIFIER = "evidence_verifier"
TRAJECTORY_JUDGE = "trajectory_judge"
# Other algorithms
CLASSIFIER = "classifier"
ESCALATION_JUDGE = "escalation_judge"
ADVISOR_REVIEW = "advisor_review"

WORK_STAGES = frozenset({LOCAL_DRAFT, CLOUD_SERVE})

STAGE_LABELS = {
    LOCAL_DRAFT: "Drafting on local tier",
    TYPING: "Typing the task",
    READOUT: "Confidence readout (logprob)",
    DELIBERATION: "Local deliberation",
    DELIBERATION_VERDICT: "Deliberation verdict (logprob)",
    ANSWER_VERIFIER: "Answer verifier",
    EVIDENCE_VERIFIER: "Evidence verifier",
    TRAJECTORY_JUDGE: "Trajectory judge (tool call)",
    CLOUD_SERVE: "Serving from cloud tier",
    CLASSIFIER: "Capability classifier",
    ESCALATION_JUDGE: "Escalation judge",
    ADVISOR_REVIEW: "Advisor review",
}
# Outside VGR a local call is the answer itself, not a draft awaiting verification.
_GENERIC_LABELS = {LOCAL_DRAFT: "Answering on local tier", CLOUD_SERVE: "Answering on cloud tier"}

# Pipeline phase of each stage. VGR runs attempt -> type -> verify -> serve;
# every other algorithm routes, then serves.
VGR_PHASES = ("attempt", "type", "verify", "serve")
STAGE_PHASE = {
    LOCAL_DRAFT: "attempt",
    TYPING: "type",
    READOUT: "verify",
    DELIBERATION: "verify",
    DELIBERATION_VERDICT: "verify",
    ANSWER_VERIFIER: "verify",
    EVIDENCE_VERIFIER: "verify",
    TRAJECTORY_JUDGE: "verify",
    CLOUD_SERVE: "serve",
    CLASSIFIER: "route",
    ESCALATION_JUDGE: "route",
    ADVISOR_REVIEW: "route",
}

_TYPING_PREFIX = "You classify a user request so a router"
_ESCALATION_PREFIX = "You are an escalation judge"
_ANSWER_PREFIX = "You are a strict answer verifier"
_EVIDENCE_PREFIX = "You are a strict verifier."
# VGR deliberates in two calls on the evidence prompt: one reasons with no verdict, then
# one is given that reasoning and scored from its first token (``vgr/rungs.rs``).
_DELIBERATION_MARK = "Reason carefully about whether the evidence supports the result."
_DELIBERATION_VERDICT_MARK = "You are also given a verifier's analysis of the record"
_CLASSIFIER_PREFIX = "You are a task-level probability forecaster"
_ADVISOR_PREFIX = "You are a senior reviewer acting as a quality gate"


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, Mapping)
        )
    return ""


def system_prompt(body: Mapping[str, Any]) -> str:
    """The system prompt of an OpenAI chat or Anthropic Messages request."""
    if body.get("system"):
        return _text(body["system"])
    for message in body.get("messages") or ():
        if isinstance(message, Mapping) and message.get("role") in ("system", "developer"):
            return _text(message.get("content"))
    instructions = body.get("instructions")
    return instructions if isinstance(instructions, str) else ""


def classify(tier: str, body: Mapping[str, Any], algorithm: str = "vgr") -> str:
    """The stage of one upstream request a route sent to ``tier`` (``local`` or ``cloud``)."""
    prompt = system_prompt(body).lstrip()
    if prompt.startswith(_TYPING_PREFIX):
        return TYPING
    if prompt.startswith(_ESCALATION_PREFIX):
        return TRAJECTORY_JUDGE if algorithm == "vgr" else ESCALATION_JUDGE
    if prompt.startswith(_ANSWER_PREFIX):
        return READOUT if body.get("logprobs") else ANSWER_VERIFIER
    if prompt.startswith(_EVIDENCE_PREFIX):
        if _DELIBERATION_MARK in prompt:
            return DELIBERATION
        if _DELIBERATION_VERDICT_MARK in prompt:
            return DELIBERATION_VERDICT
        return READOUT if body.get("logprobs") else EVIDENCE_VERIFIER
    if prompt.startswith(_CLASSIFIER_PREFIX):
        return CLASSIFIER
    if prompt.startswith(_ADVISOR_PREFIX):
        return ADVISOR_REVIEW
    return LOCAL_DRAFT if tier == "local" else CLOUD_SERVE


def label(stage: str, algorithm: str) -> str:
    if algorithm != "vgr" and stage in _GENERIC_LABELS:
        return _GENERIC_LABELS[stage]
    return STAGE_LABELS[stage]


def phase(stage: str, algorithm: str) -> str:
    if algorithm != "vgr" and stage in WORK_STAGES:
        return "serve"
    return STAGE_PHASE[stage]


def verdict(stage: str, text: str) -> Optional[str]:
    """A VGR judge's reading as VGR parses it: the last non-empty line, or the typed task.

    A deliberation's reasoning call has none; its verdict call carries the result.
    """
    if stage in WORK_STAGES or stage == DELIBERATION or STAGE_PHASE[stage] == "route":
        return None
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return None
    last = lines[-1].lower().rstrip(".!")
    if stage == TYPING:
        return last if last in ("coding", "agentic", "answer", "chat", "abstain") else "abstain"
    return last if last in ("yes", "no") else "unknown"


_YES, _NO = ("yes", "y", "true"), ("no", "n", "false")
# The least probability "yes" and "no" hold together for Switchyard to read the first token as a verdict.
MIN_VERDICT_MASS = 0.5


def score(alternatives: Any) -> Optional[float]:
    """P(yes) from a first token's alternatives, as Switchyard's readout (``vgr/readout.rs``) scores it:
    None unless both "yes" and "no" are among them, holding at least :data:`MIN_VERDICT_MASS` together."""
    yes: Optional[float] = None
    no: Optional[float] = None
    for candidate in alternatives if isinstance(alternatives, list) else ():
        if not isinstance(candidate, Mapping):
            continue
        try:
            probability = math.exp(float(candidate.get("logprob")))
        except (TypeError, ValueError, OverflowError):
            continue
        token = str(candidate.get("token") or "").strip().strip("\"'.,").lower()
        if token in _YES:
            yes = (yes or 0.0) + probability
        elif token in _NO:
            no = (no or 0.0) + probability
    if yes is None or no is None or yes + no < MIN_VERDICT_MASS:
        return None
    return yes / (yes + no)


def p_yes(response: Mapping[str, Any]) -> Optional[float]:
    """:func:`score` of the first generated token of a Chat Completions response."""
    try:
        first = response["choices"][0]["logprobs"]["content"][0]
    except (KeyError, IndexError, TypeError):
        return None
    return score(first.get("top_logprobs") if isinstance(first, Mapping) else None)
