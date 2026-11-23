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


def test_a_url_local_tier_reads_images_when_its_server_reports_vision():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

    answers = {"modalities": {"vision": True}}

    class Server(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(answers).encode() if self.path.startswith("/props") else b"{}"
            self.send_response(200 if self.path.startswith("/props") else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Server)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        tier = sy_settings.load({
            "local": {"base_url": f"http://127.0.0.1:{server.server_port}/v1", "model": "mock/local"},
            "cloud": {"base_url": "https://cloud.example/v1", "model": "mock/cloud"},
        }).local
        assert sidecar.local_reads_images(tier) is True
        answers["modalities"] = {"vision": False}
        assert sidecar.local_reads_images(tier) is False
        answers.pop("modalities")
        assert sidecar.local_reads_images(tier) is False
    finally:
        server.shutdown()
    # A server that is not up counts as text-only.
    assert sidecar.local_reads_images(tier) is False


def test_a_local_server_that_reads_images_later_gets_them_without_a_restart(monkeypatch, tmp_path):
    import tomllib
    import types

    cfg = _settings(4110)
    assert sidecar.may_read_images_later(cfg, frozenset())
    # A Switchyard that can't let the local model take images has nothing to switch to.
    assert not sidecar.may_read_images_later(cfg, frozenset({sy_settings.LOCAL_ATTACHMENTS_KEY}))

    reads = {"now": False}
    monkeypatch.setattr(sidecar, "local_reads_images", lambda tier: reads["now"])

    class Engine:
        def __init__(self, path, port):
            self.toml = open(path, encoding="utf-8").read()
            self.base_url = "http://127.0.0.1:2"

    tap = types.SimpleNamespace(switchyard_url="http://127.0.0.1:1")
    path = tmp_path / "switchyard.toml"
    assert sidecar.adopt_local_vision(cfg, Engine, path, frozenset(), tap) is None
    assert tap.switchyard_url == "http://127.0.0.1:1"

    reads["now"] = True
    engine = sidecar.adopt_local_vision(cfg, Engine, path, frozenset(), tap)
    assert tap.switchyard_url == engine.base_url
    route = tomllib.loads(engine.toml)["routes"][cfg.default_preset]
    assert route[sy_settings.LOCAL_ATTACHMENTS_KEY] is True
    # `hermes switchyard config` reads this file while the sidecar runs.
    assert path.read_text(encoding="utf-8") == engine.toml


def test_a_sidecar_too_busy_to_answer_is_not_taken_for_stopped(monkeypatch):
    import httpx

    clock = {"now": 0.0}
    killed = []

    def get(url, **kwargs):
        clock["now"] += 0.5
        if killed:
            raise httpx.ConnectError("refused")
        raise httpx.ReadTimeout("busy")

    monkeypatch.setattr(sidecar.httpx, "get", get)
    monkeypatch.setattr(sidecar.httpx, "post", lambda *args, **kwargs: None)
    monkeypatch.setattr(sidecar.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(sidecar.time, "sleep", lambda seconds: clock.update(now=clock["now"] + seconds))
    monkeypatch.setattr(sidecar, "_kill_recorded", lambda pid=None: killed.append(pid) or True)

    assert sidecar.stop(_settings(4110)) is True
    assert killed == [None]

    # One that keeps its port after the kill is reported, not passed off as stopped.
    killed.clear()
    monkeypatch.setattr(sidecar, "_kill_recorded", lambda pid=None: True)
    assert sidecar.stop(_settings(4110)) is False
