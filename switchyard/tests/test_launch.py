# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The detached sidecar starts from the saved settings and the profile's ``.env``."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

import pytest

pytest.importorskip("switchyard_rust.server")

import plugins.switchyard as plugin  # noqa: E402
from plugins.switchyard import sidecar  # noqa: E402
from plugins.switchyard import settings as sy_settings  # noqa: E402

from .mock_tiers import MockTier  # noqa: E402


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _save_settings(home: Path, monkeypatch, local: MockTier, cloud: MockTier) -> None:
    (home / "config.yaml").write_text(f"""
plugins:
  enabled: [{sy_settings.PLUGIN_ID}]
  entries:
    {sy_settings.PLUGIN_ID}:
      settings:
        port: {_free_port()}
        presets: [{{name: vgr, algorithm: vgr}}]
        local: {{base_url: "{local.base_url}", model: mock/local, api_key_env: ""}}
        cloud: {{base_url: "{cloud.base_url}", model: mock/cloud, api_key_env: LAUNCH_TEST_CLOUD_KEY}}
""", encoding="utf-8")
    # The child reads the keys from the .env file, not from this process.
    (home / ".env").write_text(f"{sy_settings.TOKEN_ENV}=launch-token\nLAUNCH_TEST_CLOUD_KEY=secret\n")
    monkeypatch.setenv(sy_settings.TOKEN_ENV, "launch-token")
    monkeypatch.delenv("LAUNCH_TEST_CLOUD_KEY", raising=False)


def test_the_sidecar_launches_from_the_saved_settings_and_env_file(_hermes_home, monkeypatch):
    local, cloud = MockTier("local", "4"), MockTier("cloud", "4")
    _save_settings(_hermes_home, monkeypatch, local, cloud)

    assert sidecar._serve_argv()[-2:] == ["switchyard", "serve"]
    try:
        assert sidecar.ensure_running(), (sidecar.state_dir() / "sidecar.log").read_text()
    finally:
        sidecar.stop()
        local.stop()
        cloud.stop()


def test_setup_with_a_new_local_server_restarts_the_running_sidecar_on_it(_hermes_home, monkeypatch, capsys):
    local, cloud, other = MockTier("local", "4"), MockTier("cloud", "4"), MockTier("other", "4")
    _save_settings(_hermes_home, monkeypatch, local, cloud)
    parser = argparse.ArgumentParser()
    plugin._setup_parser(parser)
    try:
        assert sidecar.ensure_running(), (sidecar.state_dir() / "sidecar.log").read_text()

        args = parser.parse_args(["setup", "--local-url", other.base_url, "--local-model", "mock/other",
                                  "--local-format", "openai_chat", "--bounded-local"])
        assert plugin._handle(args) == 0

        assert sidecar.fetch_state(sidecar.load_settings())["local_model"] == "mock/other"
        assert "Restarted the running sidecar" in capsys.readouterr().out
        assert plugin._handle(parser.parse_args(["status"])) == 0
        assert "local thinking time: bounded" in capsys.readouterr().out

        from tools.approval_smart import _smart_approve

        other.answer, seen = "APPROVE", len(other.requests)
        assert _smart_approve("python -c \"print('hi')\"", "script execution via -c flag") == "approve"
        verdict = other.requests[seen:][-1]["body"]
        assert (verdict["reasoning_effort"], verdict["chat_template_kwargs"]) == ("none", {"enable_thinking": False})
        assert not any("APPROVE" in json.dumps(request["body"]) for request in cloud.requests)

        args = parser.parse_args(["setup", "--cloud-url", other.base_url, "--cloud-model", "mock/unkeyed",
                                  "--cloud-format", "openai_chat", "--cloud-key-env", "LAUNCH_TEST_UNSET_KEY"])
        assert plugin._handle(args) == 1
        assert "LAUNCH_TEST_UNSET_KEY is not set, so the running sidecar keeps its previous settings" in (
            capsys.readouterr().out)
        assert sidecar.fetch_state(sidecar.load_settings())["cloud_model"] == "mock/cloud"
    finally:
        sidecar.stop()
        for tier in (local, cloud, other):
            tier.stop()
