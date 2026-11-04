# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live record of routed turns, the stages each went through, and the tokens each tier spent.

A turn is one request Hermes sends to a Switchyard preset. Every upstream call
Switchyard makes while that request is open belongs to it.
"""

from __future__ import annotations

import itertools
import json
import secrets
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from . import reasons, stages

CLIENT_CLOSED = 499  # the caller hung up before the response finished
_MAX_EVENTS_BYTES = 5 * 1024 * 1024
_PREVIEW_CHARS = 120
# Slack between the router's log clock and the turn's start and end when matching a decision.
_DECISION_SLACK_S = 2.0


def _empty_tokens() -> Dict[str, int]:
    return {"prompt": 0, "completion": 0, "calls": 0, "ms": 0}


def _add_tokens(into: Dict[str, int], prompt: int, completion: int, elapsed_ms: int) -> None:
    into["prompt"] += prompt
    into["completion"] += completion
    into["calls"] += 1
    into["ms"] += elapsed_ms


def _empty_totals() -> Dict[str, Any]:
    return {
        "local": _empty_tokens(),
        "cloud": _empty_tokens(),
        "by_stage": {},
        "served": {"local": 0, "cloud": 0},
        "turns": 0,
    }


def _empty_preset_totals() -> Dict[str, Any]:
    return {"turns": 0, "served": {"local": 0, "cloud": 0}, "local": _empty_tokens(), "cloud": _empty_tokens()}


def _count_call(totals: Dict[str, Any], call: Mapping[str, Any]) -> None:
    tier, prompt, completion, elapsed = call["tier"], call["prompt_tokens"], call["completion_tokens"], call["ms"]
    _add_tokens(totals[tier], prompt, completion, elapsed)
    _add_tokens(totals["by_stage"].setdefault(call["stage"], _empty_tokens()), prompt, completion, elapsed)


def _count_preset_turn(totals: Dict[str, Any], turn: Mapping[str, Any]) -> None:
    totals["turns"] += 1
    for tier in ("local", "cloud"):
        for key in ("prompt", "completion", "calls", "ms"):
            totals[tier][key] += turn["tokens"][tier][key]
    if turn["served_tier"] in ("local", "cloud"):
        totals["served"][turn["served_tier"]] += 1


def _is_background(turn: Optional[Mapping[str, Any]]) -> bool:
    """Titles, compression and memory calls; the timeline files a call with no turn here too."""
    return turn is None or turn["kind"] == "auxiliary"


def _latest_user_preview(body: Mapping[str, Any]) -> tuple[str, str]:
    """``(role of the last message, preview of the last user message)``."""
    messages = [m for m in body.get("messages") or () if isinstance(m, Mapping)]
    last_role = str(messages[-1].get("role", "")) if messages else ""
    for message in reversed(messages):
        if message.get("role") == "user":
            text = " ".join(stages._text(message.get("content")).split())
            return last_role, text[:_PREVIEW_CHARS]
    return last_role, ""


def _has_tool_history(body: Mapping[str, Any]) -> bool:
    """Whether the conversation already holds a tool call or result (OpenAI or Anthropic shape)."""
    for message in body.get("messages") or ():
        if not isinstance(message, Mapping):
            continue
        if message.get("role") == "tool" or message.get("tool_calls"):
            return True
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(block, Mapping) and block.get("type") in ("tool_use", "tool_result") for block in content
        ):
            return True
    return False


class Monitor:
    """Thread-safe turn/stage/token state, mirrored to an append-only JSONL file."""

    def __init__(
        self,
        *,
        local_model: str,
        cloud_model: str,
        presets: Mapping[str, Mapping[str, Any]],
        default_preset: str,
        events_path: Optional[Path] = None,
        max_turns: int = 200,
        max_segments: int = 1000,
        pin: str = "",
        runtime: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """``presets`` maps each preset name to ``{"algorithm": ..., "mode": <VGR mode or None>, "route": {...}}``.

        ``runtime`` describes the Switchyard build serving the routes (version, location).
        """
        self.local_model = local_model
        self.cloud_model = cloud_model
        # The tier every request is pinned to, or "" while presets route normally.
        self.pin = pin
        self.runtime = dict(runtime or {})
        self.presets = {name: dict(info) for name, info in presets.items()}
        self.default_preset = default_preset
        # Per tier, set while its latest answer was that its account has no credit left.
        self.credit_alerts: Dict[str, Dict[str, Any]] = {}
        self.started_at = time.time()
        self.run_id = _time_ordered_uid()
        self._events_path = events_path
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self._read_decisions: Optional[Callable[[], None]] = None
        self._active: Dict[int, Dict[str, Any]] = {}
        self._finished: deque = deque(maxlen=max_turns)
        self._timeline: deque = deque(maxlen=max_segments)
        # Everything, with the background share kept apart so the dashboard can leave it out.
        self._totals = {**_empty_totals(), "by_preset": {}, "background": _empty_totals()}

    def _algorithm(self, preset: Optional[str]) -> str:
        info = self.presets.get(preset or self.default_preset) or {}
        return str(info.get("algorithm") or "")

    def begin_turn(self, body: Mapping[str, Any], session_id: Optional[str] = None, preset: Optional[str] = None,
                   hermes_request_id: Optional[str] = None) -> int:
        last_role, preview = _latest_user_preview(body)
        preset = preset or self.default_preset
        turn = {
            # Restarts with the sidecar; ``uid`` and ``run_id`` identify a record across restarts.
            "id": next(self._ids),
            "uid": _time_ordered_uid(),
            "run_id": self.run_id,
            # Hermes's ``<turn_id>:api:<n>`` for the call, shared by its retries.
            "hermes_request_id": hermes_request_id,
            "preset": preset,
            "algorithm": self._algorithm(preset),
            "session_id": session_id,
            "started": time.time(),
            "ended": None,
            # The provider profile stamps a session id on every main-agent call;
            # auxiliary tasks (titles, compression, memory) arrive without one.
            "kind": "auxiliary" if not session_id else "tool_step" if last_role == "tool" else "user_turn",
            "preview": preview,
            "tool_history": _has_tool_history(body),
            "stream": bool(body.get("stream")),
            "stages": [],
            "tokens": {"local": _empty_tokens(), "cloud": _empty_tokens()},
            "served_tier": None,
            "served_model": None,
            "outcome": None,
            "task_type": None,
            # The router's own reason (decisions.py) and the plain-language account built from it.
            "decision": None,
            "why": None,
            # ``{"tier", "message"}`` when a tier refused one of the turn's calls for lack of credit.
            "out_of_credits": None,
            # Seconds the cloud tier sent nothing before the tap gave up on one of the turn's calls.
            "cloud_stalled_s": None,
            "status": "running",
        }
        with self._lock:
            self._active[turn["id"]] = turn
        return turn["id"]

    def begin_call(self, tier: str, body: Mapping[str, Any], turn_id: Optional[int] = None) -> Dict[str, Any]:
        with self._lock:
            turn = self._active.get(turn_id) if turn_id is not None else self._newest_active()
            algorithm = turn["algorithm"] if turn is not None else self._algorithm(None)
        stage = stages.classify(tier, body, algorithm)
        call = {
            "stage": stage,
            "label": stages.label(stage, algorithm),
            "phase": stages.phase(stage, algorithm),
            "tier": tier,
            "model": str(body.get("model", "")),
            "started": time.time(),
            "ms": None,
            # Estimated until the upstream reports its usage; a long prompt can take a local model
            # many seconds to read before it streams anything.
            "prompt_tokens": _prompt_estimate(body),
            "completion_tokens": 0,
            "estimated": True,
            "verdict": None,
            "p_yes": None,
            "tool_call": False,
            "status": "running",
        }
        with self._lock:
            if turn is not None and turn["id"] in self._active:
                turn["stages"].append(call)
        return call

    def end_call(
        self,
        call: Dict[str, Any],
        *,
        status: int,
        usage: Optional[Mapping[str, Any]],
        text: str,
        tool_call: bool,
        p_yes: Optional[float],
        out_of_credits: Optional[str] = None,
        replayed: bool = False,
    ) -> None:
        prompt, completion, estimated = (0, 0, False) if replayed else _usage_counts(usage, text)
        with self._lock:
            if estimated:
                # The streamed count also holds the thinking and tool-call arguments the text leaves out.
                completion = max(completion, call["completion_tokens"])
                prompt = prompt or (call["prompt_tokens"] if 200 <= status < 300 else 0)
            if replayed:
                # Served from the tap's copy of an earlier identical draft; no model ran.
                call["replayed"] = True
            call["ms"] = int((time.time() - call["started"]) * 1000)
            call["status"] = "ok" if 200 <= status < 300 else "out_of_credits" if out_of_credits else f"http_{status}"
            if out_of_credits:
                self.credit_alerts[call["tier"]] = {"message": out_of_credits, "at": time.time()}
            elif call["status"] == "ok":
                self.credit_alerts.pop(call["tier"], None)
            call["prompt_tokens"] = prompt
            call["completion_tokens"] = completion
            call["estimated"] = estimated
            call["verdict"] = stages.verdict(call["stage"], text)
            call["p_yes"] = p_yes
            call["tool_call"] = tool_call
            turn = self._turn_owning(call)
            _count_call(self._totals, call)
            if _is_background(turn):
                _count_call(self._totals["background"], call)
            if turn is not None:
                if out_of_credits:
                    earlier = (turn["out_of_credits"] or {}).get("tiers") or []
                    turn["out_of_credits"] = {"tier": call["tier"], "message": out_of_credits,
                                              "tiers": [*(t for t in earlier if t != call["tier"]), call["tier"]]}
                _add_tokens(turn["tokens"][call["tier"]], prompt, completion, call["ms"])
                if call["stage"] == stages.TYPING:
                    turn["task_type"] = call["verdict"]
            self._timeline.append(_segment(call, turn))

    def stream_progress(self, call: Dict[str, Any], usage: Optional[Mapping[str, Any]], streamed: int) -> None:
        """Count a call still streaming: its prompt once the stream reports it, and the output sent so far.
        :meth:`end_call` replaces both with the call's usage."""
        prompt = _prompt_count(usage)
        with self._lock:
            if call["status"] == "running":
                if prompt is not None:
                    call["prompt_tokens"] = prompt
                call["completion_tokens"] = streamed

    def request_scope(self, turn_id: Optional[int]) -> Optional[str]:
        """The Hermes API call the running turn ``turn_id`` serves, shared by Hermes's retries of it;
        the turn itself when Hermes sent no request id."""
        with self._lock:
            turn = self._active.get(turn_id) if turn_id is not None else None
            if turn is None:
                return None
            if turn["hermes_request_id"]:
                return f"hermes:{turn['session_id']}:{turn['hermes_request_id']}"
            return f"turn:{turn['uid']}"

    def out_of_credits(self, turn_id: Optional[int]) -> Optional[Dict[str, str]]:
        """``{"tier", "message", "tiers"}`` if a tier refused a call of the running turn ``turn_id`` for lack
        of credit: the last tier to refuse, its provider's message, and every tier that refused."""
        with self._lock:
            turn = self._active.get(turn_id) if turn_id is not None else None
            return turn["out_of_credits"] if turn is not None else None

    def note_cloud_stall(self, turn_id: Optional[int], seconds: int) -> None:
        """Record that the cloud tier went silent for ``seconds`` on a call of the running turn ``turn_id``."""
        with self._lock:
            turn = self._active.get(turn_id) if turn_id is not None else None
            if turn is not None:
                turn["cloud_stalled_s"] = seconds

    def cloud_stalled(self, turn_id: Optional[int]) -> Optional[int]:
        """The seconds the cloud tier stalled for on the running turn ``turn_id``, or None if it has not."""
        with self._lock:
            turn = self._active.get(turn_id) if turn_id is not None else None
            return turn["cloud_stalled_s"] if turn is not None else None

    def attach_decisions(self, read: Callable[[], None]) -> None:
        """``read`` delivers any decision lines logged so far to :meth:`record_decision`."""
        self._read_decisions = read

    def record_decision(self, decision: Mapping[str, Any]) -> None:
        """Attach the router's reason for one request to the VGR turn it belongs to."""
        with self._lock:
            turn = self._decision_turn(decision)
            if turn is None:
                return
            turn["decision"] = {key: decision.get(key) for key in ("reason", "branch", "served", "policy_version")}
            if turn["ended"] is not None:
                turn["outcome"] = _outcome(turn, self._mode(turn))
                turn["why"] = self._explain(turn)

    def _decision_turn(self, decision: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        at = decision.get("at")
        candidates = [
            turn
            for turn in [*self._active.values(), *list(self._finished)[-20:]]
            if turn["algorithm"] == "vgr"
            and turn["decision"] is None
            and turn["session_id"] == decision.get("session_id")
            and (at is None or (
                turn["started"] <= at + _DECISION_SLACK_S
                and (turn["ended"] is None or turn["ended"] >= at - _DECISION_SLACK_S)
            ))
        ]
        return min(candidates, key=lambda t: t["id"]) if candidates else None

    def _mode(self, turn: Mapping[str, Any]) -> Optional[str]:
        return (self.presets.get(turn["preset"]) or {}).get("mode")

    def _explain(self, turn: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        route = (self.presets.get(turn["preset"]) or {}).get("route") or {}
        previous = next(
            (t for t in reversed(self._finished)
             if t["id"] < turn["id"] and t["session_id"] == turn["session_id"] and t["session_id"]),
            None,
        )
        return reasons.explain(turn, route, previous)

    def end_turn(self, turn_id: int, *, status: int, served_model: Optional[str]) -> None:
        if self._read_decisions is not None:
            self._read_decisions()
        with self._lock:
            turn = self._active.pop(turn_id, None)
            if turn is None:
                return
            turn["ended"] = time.time()
            # A call still open now outlived its turn, as a local draft does when VGR stops waiting for it;
            # its record would otherwise stay "running" for good.
            for call in turn["stages"]:
                if call["status"] == "running":
                    call["status"] = "abandoned"
                    call["ms"] = int((turn["ended"] - call["started"]) * 1000)
            turn["served_model"] = served_model
            succeeded = 200 <= status < 300
            turn["status"] = "ok" if succeeded else f"http_{status}"
            turn["served_tier"] = self._tier_of(served_model, turn) if succeeded else None
            turn["outcome"] = _outcome(turn, self._mode(turn))
            turn["why"] = self._explain(turn)
            self._finished.append(turn)
            preset = self._totals["by_preset"].setdefault(turn["preset"], {
                **_empty_preset_totals(), "background": _empty_preset_totals(),
            })
            _count_preset_turn(preset, turn)
            counted = [self._totals]
            if _is_background(turn):
                _count_preset_turn(preset["background"], turn)
                counted.append(self._totals["background"])
            for totals in counted:
                totals["turns"] += 1
                if turn["served_tier"] in ("local", "cloud"):
                    totals["served"][turn["served_tier"]] += 1
        self._append_event(turn)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            foreground = [t for t in self._active.values() if t["kind"] != "auxiliary"]
            current = max(foreground, key=lambda t: t["id"]) if foreground else self._newest_active()
            running = [
                _segment(call, turn)
                for turn in self._active.values()
                for call in turn["stages"]
                if call["status"] == "running"
            ]
            timeline = sorted([*self._timeline, *running], key=lambda s: s["started"])
            return json.loads(json.dumps({
                "now": time.time(),
                "started_at": self.started_at,
                "presets": self.presets,
                "default_preset": self.default_preset,
                "pin": self.pin,
                "runtime": self.runtime,
                "local_model": self.local_model,
                "cloud_model": self.cloud_model,
                "credit_alerts": self.credit_alerts,
                "current": _current_view(current) if current else None,
                "turns": list(self._finished)[-50:][::-1],
                "totals": self._totals,
                "timeline": timeline,
            }))

    def _newest_active(self) -> Optional[Dict[str, Any]]:
        return self._active[max(self._active)] if self._active else None

    def _turn_owning(self, call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for turn in self._active.values():
            if any(stage is call for stage in turn["stages"]):
                return turn
        return None

    def _tier_of(self, served_model: Optional[str], turn: Mapping[str, Any]) -> Optional[str]:
        # Two endpoints may serve the same model id; then only the stages tell them apart.
        if self.local_model != self.cloud_model:
            if served_model == self.local_model:
                return "local"
            if served_model == self.cloud_model:
                return "cloud"
        served = [s["tier"] for s in turn["stages"] if s["stage"] in (stages.LOCAL_DRAFT, stages.CLOUD_SERVE)]
        return served[-1] if served else None

    def _append_event(self, turn: Mapping[str, Any]) -> None:
        if self._events_path is None:
            return
        try:
            self._events_path.parent.mkdir(parents=True, exist_ok=True)
            if self._events_path.exists() and self._events_path.stat().st_size > _MAX_EVENTS_BYTES:
                self._events_path.replace(self._events_path.with_suffix(".jsonl.1"))
            with self._events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(turn, separators=(",", ":")) + "\n")
        except OSError:
            pass


_uid_lock = threading.Lock()
_last_uid_ms = 0


def _time_ordered_uid() -> str:
    """A UUIDv7: ids sort by creation time, across sidecar runs too."""
    global _last_uid_ms
    with _uid_lock:
        # Never step back, so ids made in the same or a rewound millisecond keep their order.
        ms = _last_uid_ms = max(int(time.time() * 1000), _last_uid_ms + 1)
    value = (ms << 80) | (0x7 << 76) | (secrets.randbits(12) << 64) | (0b10 << 62) | secrets.randbits(62)
    return str(uuid.UUID(int=value))


def _prompt_count(usage: Optional[Mapping[str, Any]]) -> Optional[int]:
    if not isinstance(usage, Mapping):
        return None
    prompt = usage.get("prompt_tokens")
    if prompt is None and isinstance(usage.get("input_tokens"), int):
        # Anthropic's input_tokens excludes the prompt it read from or wrote to cache.
        cached = (usage.get("cache_read_input_tokens") or 0) + (usage.get("cache_creation_input_tokens") or 0)
        prompt = usage["input_tokens"] + (cached if isinstance(cached, int) else 0)
    return prompt if isinstance(prompt, int) else None


def _prompt_estimate(body: Mapping[str, Any]) -> int:
    """A request's prompt tokens at four characters a token, tool schemas included and images left out."""
    def chars(value: Any) -> int:
        if isinstance(value, str):
            return 0 if value.startswith("data:") else len(value)
        if isinstance(value, Mapping):
            # Anthropic carries a base64 image as a source's ``data``.
            return sum(len(str(key)) + chars(item) for key, item in value.items() if key != "data")
        if isinstance(value, list):
            return sum(chars(item) for item in value)
        return 0

    return -(-sum(chars(body.get(key)) for key in ("system", "instructions", "messages", "input", "tools")) // 4)


def _usage_counts(usage: Optional[Mapping[str, Any]], text: str) -> tuple[int, int, bool]:
    if isinstance(usage, Mapping):
        prompt = _prompt_count(usage)
        completion = usage.get("completion_tokens", usage.get("output_tokens"))
        if isinstance(prompt, int) and isinstance(completion, int):
            return prompt, completion, False
    # The upstream reported no usage; approximate the completion from its text.
    return 0, max(1, len(text) // 4) if text else 0, True


# Shadow mode serves every reply from the cloud, so a turn is labelled by what VGR decided
# where its logged reason says. A shadow "decided" logs the cloud whichever way VGR decided.
_SHADOW_OUTCOMES = {"turn_verification_complete": "shadow_tool_call_allowed", "decided": "shadow_checked",
                    "turn_verification_latched": "shadow_unchecked"}
# VGR's reasons for handing a step to the cloud because the request or the message ran out of time.
_TIME_LIMITS = ("local_timed_out", "local_turn_budget")


def _outcome(turn: Mapping[str, Any], mode: Optional[str] = None) -> str:
    calls: List[Mapping[str, Any]] = turn["stages"]
    if turn["algorithm"] != "vgr":
        if turn["served_tier"] in ("local", "cloud"):
            return f"served_{turn['served_tier']}"
        return "cancelled" if turn["status"] == f"http_{CLIENT_CLOSED}" else "failed"
    drafted = [c for c in calls if c["stage"] == stages.LOCAL_DRAFT]
    if turn["served_tier"] == "local":
        if (turn.get("decision") or {}).get("served") == "cloud":
            return "local_after_cloud_failure"
        return "local_tool_call" if drafted and drafted[-1]["tool_call"] else "local_committed"
    if turn["served_tier"] == "cloud":
        shadow = _SHADOW_OUTCOMES.get((turn.get("decision") or {}).get("reason")) if mode == "shadow" else None
        if shadow:
            return shadow
        if (turn.get("decision") or {}).get("reason") in _TIME_LIMITS:
            return "escalated_on_time_limit"
        if not drafted:
            return "escalated_without_attempt"
        if any(c["stage"] == stages.TRAJECTORY_JUDGE for c in calls):
            return "escalated_by_trajectory_judge"
        return "escalated_after_verification"
    return "cancelled" if turn["status"] == f"http_{CLIENT_CLOSED}" else "failed"


def _segment(call: Mapping[str, Any], turn: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """One upstream call as a span on the routing timeline."""
    running = call["status"] == "running"
    return {
        "turn_id": turn["id"] if turn else None,
        "preset": turn["preset"] if turn else None,
        "kind": turn["kind"] if turn else "auxiliary",
        "tier": call["tier"],
        "stage": call["stage"],
        "label": call["label"],
        "started": call["started"],
        "ms": int((time.time() - call["started"]) * 1000) if running else call["ms"],
        "tokens": call["prompt_tokens"] + call["completion_tokens"],
        "running": running,
    }


def _current_view(turn: Mapping[str, Any]) -> Dict[str, Any]:
    running = [c for c in turn["stages"] if c["status"] == "running"]
    latest = running[-1] if running else (turn["stages"][-1] if turn["stages"] else None)
    if latest is None:
        stage, label, phase, tier = None, "Routing", None, None
    elif latest["status"] == "running":
        stage, label, phase, tier = latest["stage"], latest["label"], latest["phase"], latest["tier"]
    else:
        decide = "verify" if turn["algorithm"] == "vgr" else "route"
        stage, label, phase, tier = None, "Deciding", decide, latest["tier"]
    finished = [turn["tokens"][t] for t in ("local", "cloud")]
    return {
        "turn": turn,
        "stage": stage,
        "label": label,
        "phase": phase,
        "tier": tier,
        "elapsed_ms": int((time.time() - turn["started"]) * 1000),
        # The turn's tokens so far: finished calls' usage, plus what the calls still streaming have sent.
        "tokens": {
            "prompt": sum(t["prompt"] for t in finished) + sum(c["prompt_tokens"] for c in running),
            "completion": sum(t["completion"] for t in finished) + sum(c["completion_tokens"] for c in running),
        },
    }
