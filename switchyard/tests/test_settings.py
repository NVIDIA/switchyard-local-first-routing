# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier sources, pins and compatibility checks in the plugin settings."""

from __future__ import annotations

import pytest

from plugins.switchyard import providers
from plugins.switchyard import settings as sy_settings


def _load(local=None, cloud=None, **extra):
    return sy_settings.load({
        "local": local or {"base_url": "http://127.0.0.1:8080/v1", "model": "qwen"},
        "cloud": cloud or {"base_url": "https://api.example.com/v1", "model": "big", "api_key_env": "BIG_KEY"},
        **extra,
    })


def test_provider_tier_takes_its_format_from_the_provider_and_has_no_url():
    cfg = _load(cloud={"source": "provider", "provider": "openai-codex", "model": "gpt-5.5"})
    assert (cfg.cloud.source, cfg.cloud.base_url, cfg.cloud.format) == ("provider", "", "openai_responses")
    assert cfg.cloud.extra_body == {"store": False}
    rendered = sy_settings.render_switchyard_toml(cfg)
    assert "BIG_KEY" not in rendered and "openai-codex" not in rendered


@pytest.mark.parametrize("cloud, extra_body", [
    ({"base_url": "https://gateway.example.com/v1", "model": "anthropic/claude-opus-5-5"},
     sy_settings.CLAUDE_EXTRA_BODY),
    ({"source": "provider", "provider": "anthropic", "model": "claude-sonnet-5"}, sy_settings.CLAUDE_EXTRA_BODY),
    ({"base_url": "https://gateway.example.com/v1", "model": "anthropic/claude-opus-5-5",
      "extra_body": {}}, {}),
    ({"base_url": "https://gateway.example.com/v1", "model": "claude-opus-5-5", "format": "openai_chat"}, {}),
])
def test_claude_over_anthropic_messages_thinks_adaptively_unless_the_tier_says_otherwise(cloud, extra_body):
    assert _load(cloud=cloud).cloud.extra_body == extra_body


def test_any_two_models_route_but_not_the_same_endpoint_twice():
    two_locals = _load(cloud={"base_url": "http://192.168.1.20:8000/v1", "model": "qwen-big"})
    assert two_locals.cloud.format == "openai_chat"
    same_model_two_providers = _load(
        local={"source": "provider", "provider": "nous", "model": "hermes-4"},
        cloud={"source": "provider", "provider": "openrouter", "model": "hermes-4"},
    )
    assert same_model_two_providers.local.model == same_model_two_providers.cloud.model
    with pytest.raises(sy_settings.SettingsError):
        _load(cloud={"base_url": "http://127.0.0.1:8080/v1", "model": "qwen"})


@pytest.mark.parametrize("tier", [
    {"base_url": "http://cloud.example.com/v1", "model": "m"},
    {"source": "provider", "provider": "not a provider", "model": "m"},
    {"source": "managed", "model": "m", "format": "anthropic_messages"},
])
def test_invalid_tiers_are_rejected(tier):
    with pytest.raises(sy_settings.SettingsError):
        _load(cloud=tier)


def test_pin_parses_and_renders_hidden_passthrough_routes():
    assert [sy_settings.parse_pin(v) for v in ("off", "", None, "smart", "LOCAL", "cloud")] == ["", "", "", "", "local", "cloud"]
    with pytest.raises(sy_settings.SettingsError):
        sy_settings.parse_pin("both")
    cfg = _load(pin="cloud")
    assert cfg.pin == "cloud"
    assert [p.name for p in cfg.presets] == [sy_settings.RECOMMENDED]
    rendered = sy_settings.render_switchyard_toml(cfg)
    assert "pin-local" in rendered and "pin-cloud" in rendered
    with pytest.raises(sy_settings.SettingsError):
        _load(presets={"pin-x": {"algorithm": "passthrough", "options": {"target": "cloud"}}})


def test_vgr_routes_use_upstream_switchyards_key_names_for_a_build_of_upstream():
    cfg = _load(bounded_local=True)
    legacy = sy_settings.render_switchyard_toml(cfg, upstream_names=False)
    upstream = sy_settings.render_switchyard_toml(cfg, upstream_names=True)
    for ours, theirs in sy_settings.UPSTREAM_VGR_KEYS.items():
        assert f"\n{ours} = " in legacy and f"\n{ours} = " not in upstream
        assert f"\n{theirs} = " in upstream
    assert '\njudge_target = "local"' in upstream


def test_vgr_warns_when_its_judge_tier_cannot_return_logprobs():
    assert sy_settings.compatibility_warnings(_load()) == []
    responses_judge = _load(local={"source": "provider", "provider": "openai-codex", "model": "gpt-5.5-mini"})
    assert "token probabilities" in sy_settings.compatibility_warnings(responses_judge)[0]


@pytest.mark.parametrize("runtime, headers", [
    ({"provider": "nous", "api_mode": "chat_completions", "api_key": "jwt"}, {"Authorization": "Bearer jwt"}),
    ({"provider": "anthropic", "api_mode": "anthropic_messages", "api_key": "sk-ant-api03-key"}, {"x-api-key": "sk-ant-api03-key"}),
])
def test_provider_credentials_become_request_headers(runtime, headers):
    assert providers.credential_headers(runtime) == headers


@pytest.mark.parametrize("runtime, message", [
    ({"provider": "nous", "api_mode": "chat_completions", "api_key": ""}, "hermes auth add nous"),
    ({"provider": "anthropic", "api_mode": "anthropic_messages", "api_key": "sk-ant-oat01-token"}, "subscription"),
    ({"provider": "openai-codex", "api_mode": "codex_app_server", "api_key": "t"}, "cannot route"),
])
def test_unservable_provider_logins_are_refused_with_a_reason(runtime, message):
    with pytest.raises(providers.ProviderUnavailable, match=message):
        providers.credential_headers(runtime)


def test_a_provider_that_now_resolves_to_switchyard_or_another_format_is_reported_not_misrouted():
    tier = _load(cloud={"source": "provider", "provider": "custom", "model": "m"}).cloud
    with pytest.raises(providers.ProviderUnavailable, match="Switchyard itself"):
        providers.endpoint(tier, {"provider": "custom", "api_mode": "chat_completions",
                                  "base_url": "http://127.0.0.1:4110/front/v1", "api_key": "k"})
    with pytest.raises(providers.ProviderUnavailable, match="run setup again"):
        providers.endpoint(tier, {"provider": "custom", "api_mode": "anthropic_messages",
                                  "base_url": "https://x.example/v1", "api_key": "k"})


def test_vgr_tells_a_new_user_how_long_a_reply_can_take():
    note = sy_settings.setup_note("vgr", _load().default.route)
    assert "up to 30 minutes" in note and "images always go to the cloud" in note
    assert "up to 90 seconds" in sy_settings.setup_note("vgr", {"deadline_seconds": 90})
    bounded = _load(bounded_local=True).default.route
    assert "up to 5 minutes" in sy_settings.setup_note("vgr", bounded)
    assert sy_settings.setup_note("llm_classifier", {}) is None


def test_vgr_is_unbounded_unless_bounded_local_or_the_preset_sets_a_limit():
    def limits(route):
        return route["deadline_seconds"], route.get("local_turn_budget_seconds")

    assert limits(_load().default.route) == (sy_settings.UNBOUNDED_DEADLINE_SECONDS, None)
    assert limits(_load(unbounded_local=False).default.route) == (sy_settings.UNBOUNDED_DEADLINE_SECONDS, None)
    assert limits(_load(bounded_local=True).default.route) == (300, 600)
    quick = {"quick": {"algorithm": "vgr", "options": {"deadline_seconds": 120}}}
    assert limits(_load(presets=quick).default.route) == (120, None)
    with pytest.raises(sy_settings.SettingsError, match="bounded_local must be true or false"):
        _load(bounded_local="yes")


def test_plugin_yaml_declares_every_hook_and_middleware_the_plugin_registers(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    import hermes_yaml

    import plugins.switchyard as plugin

    monkeypatch.setattr(plugin, "_ensure_front_token", lambda: None)
    monkeypatch.setattr(plugin, "_start_early", lambda: None)
    # register keeps the context it is given; this stand-in must not outlive the test.
    monkeypatch.setattr(sy_settings, "_context", None)
    hooks, middleware = set(), set()
    plugin.register(SimpleNamespace(
        register_cli_command=lambda *_, **__: None,
        register_hook=lambda name, _fn: hooks.add(name),
        register_middleware=lambda name, _fn: middleware.add(name),
    ))
    manifest = hermes_yaml.safe_load((Path(plugin.__file__).parent / "plugin.yaml").read_text())
    assert set(manifest["provides_hooks"]) == hooks
    assert set(manifest["provides_middleware"]) == middleware
