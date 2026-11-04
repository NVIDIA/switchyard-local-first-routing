# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Starting the detached sidecar."""

from __future__ import annotations

import json

from plugins.switchyard import settings as sy_settings
from plugins.switchyard import sidecar


def _settings(port: int) -> sy_settings.SwitchyardSettings:
    return sy_settings.load({
        "port": port,
        "local": {"base_url": "http://127.0.0.1:9931/v1", "model": "mock/local"},
        "cloud": {"base_url": "https://cloud.example/v1", "model": "mock/cloud"},
    })


def test_a_sidecar_on_an_older_port_is_stopped_before_the_new_one_starts(monkeypatch):
    directory = sidecar.state_dir()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sidecar.json").write_text(json.dumps({"pid": 1, "port": 4110}), encoding="utf-8")
    stopped, spawned, up = [], [], set()
    monkeypatch.setattr(sidecar, "stop", lambda cfg: stopped.append(cfg.port) or True)
    monkeypatch.setattr(sidecar, "spawn_detached", lambda *args, **kwargs: spawned.append(args) or up.add(4211))
    monkeypatch.setattr(sidecar, "health", lambda cfg, timeout=1.0: {"switchyard": True} if cfg.port in up else None)
    monkeypatch.setattr(sidecar, "ensure_token", lambda: "token")

    assert sidecar.ensure_running(_settings(4211))
    assert stopped == [4110]
    assert len(spawned) == 1


def test_a_restart_does_not_pass_off_the_old_sidecar_as_its_replacement(monkeypatch):
    running = {"pid": 41}
    killed = []
    monkeypatch.setattr(sidecar, "health", lambda cfg, timeout=1.0: {"switchyard": True, **running})
    monkeypatch.setattr(sidecar, "stop", lambda cfg: True)
    monkeypatch.setattr(sidecar, "_RESTART_WAIT_S", 0.0)
    monkeypatch.setattr(sidecar.time, "sleep", lambda seconds: None)

    monkeypatch.setattr(sidecar, "_kill_recorded", lambda pid=None: killed.append(pid))
    assert sidecar.restart(_settings(4110)) is False
    assert killed == [41]

    monkeypatch.setattr(sidecar, "_kill_recorded", lambda pid=None: running.update(pid=42))
    assert sidecar.restart(_settings(4110)) is True
