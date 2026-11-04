# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The plugin's copies of Hermes's private constants and checks still agree with Hermes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from plugins.switchyard import initiator, providers, tap


def test_hermes_side_task_clients_carry_the_client_header_and_no_session(monkeypatch):
    from agent import auxiliary_client
    from hermes_cli.config import set_config_value

    # Registers this profile in place of any copy a Hermes checkout bundles.
    spec = importlib.util.spec_from_file_location(
        "switchyard_provider_profile", Path(__file__).parents[1] / "provider" / "__init__.py")
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    monkeypatch.setenv("SWITCHYARD_API_KEY", "test-key")
    set_config_value("model.provider", "switchyard")
    set_config_value("model.default", "switchyard/vgr")
    client, model, _ = auxiliary_client._resolve_auto_route(task="title_generation")
    assert model == "switchyard/vgr"
    assert client._custom_headers.get(tap.CLIENT_HEADER) == tap.HERMES_CLIENT
    assert tap.SESSION_HEADER not in client._custom_headers


def test_the_steer_marker_matches_hermes():
    from agent.prompt_builder import STEER_DISPLAY_KIND, STEER_MARKER_OPEN

    assert (initiator.STEER_DISPLAY_KIND, initiator.STEER_MARKER_OPEN) == (STEER_DISPLAY_KIND, STEER_MARKER_OPEN)


@pytest.mark.parametrize("key", [
    "sk-ant-oat01-x", "sk-ant-api03-x", "sk-ant-usr-x", "sk-ant-svc-x", "sk-ant-admin01-x", "sk-ant-future-x",
    "eyJhbGciOi", "cc-x", "sk-proj-x", "",
])
def test_claude_subscription_tokens_are_told_apart_as_hermes_does(key):
    from agent.anthropic_credentials import _is_oauth_token

    if key.startswith(("sk-ant-usr", "sk-ant-svc", "sk-ant-admin")) and _is_oauth_token("sk-ant-svc-x"):
        pytest.skip("this Hermes predates the Console API-key families (hermes-agent #133856)")
    assert providers._is_anthropic_oauth(key) == _is_oauth_token(key)


@pytest.mark.parametrize("url", [
    "http://localhost:8080/v1", "http://127.0.0.1:1234/v1", "http://[::1]:8000/v1", "http://0.0.0.0:8000",
    "http://host.docker.internal:11434/v1", "http://ollama:11434", "http://box.local:8000",
    "http://10.0.0.5/v1", "http://172.20.1.1/v1", "http://172.32.1.1/v1", "http://192.168.1.9:8000/v1",
    "http://169.254.1.1/v1", "http://100.100.1.1/v1", "http://100.128.1.1/v1", "http://172.026.1.1/v1",
    "https://api.openai.com/v1", "https://openrouter.ai/api/v1", "http://[2001:4860::8888]/v1",
    "http://LOCALHOST:8080/v1/", "  http://127.0.0.1:8000/v1/  ",
])
def test_local_endpoints_are_recognised_as_hermes_does(url):
    from agent.model_metadata import is_local_endpoint

    assert providers._is_local({"api_url": url}) == is_local_endpoint(url)
