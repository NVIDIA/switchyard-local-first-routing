# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VGR's own reason for each routing decision, read from the sidecar log.

Switchyard logs one ``vgr decision`` line per routed request
(``crates/libsy/src/algorithms/vgr/runtime.rs``) to stderr, which the detached
sidecar writes to ``sidecar.log``. The line names the reason, the verification
branch and the tier served; its span carries the request's session id, which
is how the monitor finds the turn it belongs to. Newer engines also log the
decision policy's version. Decisions are reported with the plugin's tier names,
whichever names the engine logs.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

_DECISION = re.compile(
    r'vgr decision reason="?(?P<reason>[a-z_]+)"?\s+branch=(?:Some\((?P<branch>\w+)\)|None)\s+served=(?P<served>\w+)'
    r'(?:\s+policy_version="?(?P<policy>[\d.]+)"?)?'
)
_SESSION = re.compile(r'\bsession_id="(?P<session>[^"]*)"')
_TIMESTAMP = re.compile(r"^\s*(?P<ts>\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?)Z")
_CAMEL = re.compile(r"(?<!^)(?=[A-Z])")
# Upstream Switchyard names the tiers efficient and capable, in ``served`` and in reasons.
_UPSTREAM_TIERS = {"efficient": "local", "capable": "cloud"}
_UPSTREAM_REASON = re.compile(r"^efficient_")


def _epoch(stamp: str) -> Optional[float]:
    whole, _, fraction = stamp.partition(".")
    try:
        moment = dt.datetime.strptime(whole, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None
    return moment.timestamp() + (float(f"0.{fraction}") if fraction.isdigit() else 0.0)


def parse(line: str) -> Optional[Dict[str, Any]]:
    """The decision a ``vgr decision`` log line records, or None for any other line."""
    match = _DECISION.search(line)
    if match is None:
        return None
    session = _SESSION.search(line, 0, match.start())
    stamp = _TIMESTAMP.match(line)
    branch = match.group("branch")
    served = match.group("served").lower()
    return {
        "reason": _UPSTREAM_REASON.sub("local_", match.group("reason")),
        "branch": _CAMEL.sub("_", branch).lower() if branch else None,
        "served": _UPSTREAM_TIERS.get(served, served),
        "policy_version": match.group("policy"),
        "session_id": session.group("session") if session else None,
        "at": _epoch(stamp.group("ts")) if stamp else None,
    }


class LogFollower:
    """Follows a growing log file from its current end and reports each decision line.

    :meth:`poll` reads whatever was appended since the last read; the background
    thread calls it periodically, and the monitor calls it before closing a turn
    so a decision logged just before the response is not missed.
    """

    def __init__(self, path: Path, on_decision: Callable[[Dict[str, Any]], None], interval: float = 0.25) -> None:
        self.path = path
        self.on_decision = on_decision
        self.interval = interval
        self._lock = threading.Lock()
        self._offset: Optional[int] = None
        self._partial = b""
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        with self._lock:
            self._offset = self._size()
        self._thread = threading.Thread(target=self._run, name="switchyard-decisions", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _size(self) -> int:
        try:
            return os.stat(self.path).st_size
        except OSError:
            return 0

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.poll()

    def poll(self) -> None:
        decisions = []
        with self._lock:
            size = self._size()
            if self._offset is None or size < self._offset:
                self._offset, self._partial = 0, b""
            if size == self._offset:
                return
            try:
                with open(self.path, "rb") as handle:
                    handle.seek(self._offset)
                    data = handle.read(size - self._offset)
            except OSError:
                return
            self._offset += len(data)
            *lines, self._partial = (self._partial + data).split(b"\n")
            for raw in lines:
                decision = parse(raw.decode("utf-8", errors="replace"))
                if decision is not None:
                    decisions.append(decision)
        for decision in decisions:
            self.on_decision(decision)
