# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The onboarding endpoint that points Switchyard at a local and a cloud tier."""

from __future__ import annotations

import json
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from plugins.switchyard import settings as sy_settings
from plugins.switchyard import sidecar
from plugins.switchyard.dashboard import plugin_api


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(sidecar, "restart", lambda cfg: True)
    monkeypatch.setattr(sidecar, "account_warnings", lambda cfg: [])
    # setup writes the token to this Hermes's .env only when none is set yet.
    monkeypatch.delenv(sy_settings.TOKEN_ENV, raising=False)
    app = FastAPI()
    app.include_router(plugin_api.router)
    return TestClient(app)


def _config():
    from hermes_cli.config import load_config_readonly

    return load_config_readonly() or {}


def test_managed_setup_routes_hermes_through_the_vgr_preset_with_the_quickstart_model(client):
    from agent.secret_scope import get_secret
    from hermes_cli.config import set_config_value

    set_config_value("model.provider", "llamacpp")
    set_config_value("model.default", "Qwen3.6-35B-A3B-UD-Q4_K_M")

    response = client.post("/setup", json={"local": {"runtime": "managed"}, "cloud": {"api_key": "sk-test-cloud-key"}})

    assert response.status_code == 200, response.text
    cfg = sy_settings.load(sy_settings.plugin_settings(_config()))
    assert cfg.local.managed and cfg.local.model == "Qwen3.6-35B-A3B-UD-Q4_K_M"
    assert sy_settings.PLUGIN_ID in _config()["plugins"]["enabled"]
    assert _config()["model"]["provider"] == sy_settings.PROVIDER
    assert _config()["model"]["default"] == "switchyard/vgr"
    assert get_secret(cfg.cloud.api_key_env) == "sk-test-cloud-key"


def _external_setup(client, **extra):
    return client.post("/setup", json={
        "local": {"runtime": "external", "base_url": "http://127.0.0.1:9931/v1", "model": "local-m"},
        "cloud": {"api_key": "sk-test-cloud-key"},
        **extra,
    })


def test_a_keyless_local_server_names_no_key_for_the_sidecar_to_require(client):
    assert _external_setup(client).status_code == 200

    cfg = sy_settings.load(sy_settings.plugin_settings(_config()))
    assert cfg.local.api_key_env == ""
    sidecar._export_tier_keys(cfg)


def test_session_titles_and_smart_approval_go_straight_to_the_local_tier(client):
    from agent.auxiliary_client import _get_task_extra_body, _resolve_task_provider_model

    assert _external_setup(client).status_code == 200

    cfg = sy_settings.load(sy_settings.plugin_settings(_config()))
    for task in ("title_generation", "approval"):
        provider, model, base_url, _, _ = _resolve_task_provider_model(task)
        assert (provider, model, base_url) == (sy_settings.PROVIDER, "switchyard/only-efficient", cfg.front_base_url)
        assert sy_settings.PIN_MODELS[model] == "local"
    for task in ("title_generation", "approval"):
        assert _get_task_extra_body(task) == {"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}


def test_a_session_title_waits_for_a_local_model_busy_with_the_turn_and_keeps_a_timeout_the_user_set(client):
    from agent.auxiliary_client import _get_task_timeout
    from hermes_cli.config import set_config_value

    assert _external_setup(client).status_code == 200
    assert _get_task_timeout("title_generation") == sidecar._HERMES_API_TIMEOUT_S + sidecar._TITLE_SLACK_S
    assert _external_setup(client, bounded_local=True).status_code == 200
    assert _get_task_timeout("title_generation") == 300 + sidecar._TITLE_SLACK_S

    set_config_value("auxiliary.title_generation.timeout", "45")
    assert _external_setup(client, bounded_local=False).status_code == 200
    assert _get_task_timeout("title_generation") == 45


def test_a_session_title_waits_for_the_local_model_when_switchyard_is_picked_per_chat(monkeypatch):
    from agent.auxiliary_client import _get_task_timeout, _resolve_task_provider_model
    from hermes_cli.config import unset_config_value
    from hermes_cli.plugins_state import save_plugin_setting

    monkeypatch.setenv("SWITCHYARD_TEST_CLOUD_KEY", "sk-test-cloud-key")
    sidecar.configure({
        "local": {"base_url": "http://127.0.0.1:9931/v1", "model": "local-m"},
        "cloud": {"api_key_env": "SWITCHYARD_TEST_CLOUD_KEY"},
    }, make_default=False)
    waits = sidecar._HERMES_API_TIMEOUT_S + sidecar._TITLE_SLACK_S

    model = _config().get("model")
    assert not (isinstance(model, dict) and model.get("provider") == sy_settings.PROVIDER)
    assert _resolve_task_provider_model("title_generation")[0] != sy_settings.PROVIDER
    assert _get_task_timeout("title_generation") == waits

    unset_config_value("auxiliary.title_generation.timeout")
    for key in (sy_settings.HERMES_SETTINGS_KEY, sy_settings.TITLE_TIMEOUT_KEY):
        save_plugin_setting(sy_settings.PLUGIN_ID, (key,), None)
    assert sidecar.update_hermes_settings() is True
    assert _get_task_timeout("title_generation") == waits


def test_an_upgrade_writes_the_hermes_settings_an_older_setup_did_not(client):
    from agent.auxiliary_client import _get_task_extra_body, _get_task_timeout, _resolve_task_provider_model
    from hermes_cli.config import unset_config_value
    from hermes_cli.plugins_state import save_plugin_setting
    from hermes_cli.timeouts import get_provider_stale_timeout

    assert _external_setup(client).status_code == 200
    for task in ("provider", "model", "base_url"):
        unset_config_value(f"auxiliary.approval.{task}")
    unset_config_value("auxiliary.title_generation.timeout")
    unset_config_value("auxiliary.title_generation.extra_body")
    for key in sidecar._STREAM_TIMEOUT_KEYS:
        unset_config_value(f"providers.{sy_settings.PROVIDER}.{key}")
    for key in (sy_settings.HERMES_SETTINGS_KEY, sy_settings.TITLE_TIMEOUT_KEY, sy_settings.STREAM_TIMEOUT_KEY):
        save_plugin_setting(sy_settings.PLUGIN_ID, (key,), None)

    assert sidecar.update_hermes_settings() is True
    assert _resolve_task_provider_model("approval")[1] == sy_settings.LOCAL_PIN_MODEL
    assert _get_task_timeout("title_generation") == sidecar._HERMES_API_TIMEOUT_S + sidecar._TITLE_SLACK_S
    assert _get_task_extra_body("title_generation") == {"reasoning_effort": "none",
                                                         "chat_template_kwargs": {"enable_thinking": False}}
    wait = sy_settings.UNBOUNDED_DEADLINE_SECONDS + sidecar._CLOUD_START_S
    assert get_provider_stale_timeout(sy_settings.PROVIDER, "switchyard/vgr") == wait
    assert sidecar.update_hermes_settings() is False


def test_a_cloud_efficient_tier_is_not_sent_llama_cpp_thinking_fields(client):
    from agent.auxiliary_client import _get_task_extra_body, _resolve_task_provider_model

    response = client.post("/setup", json={
        "local": {"runtime": "external", "base_url": "https://efficient.example/v1", "model": "small",
                  "api_key": "sk-test-efficient-key"},
        "cloud": {"api_key": "sk-test-cloud-key"},
    })

    assert response.status_code == 200, response.text
    assert _resolve_task_provider_model("approval")[1] == "switchyard/only-efficient"
    assert _get_task_extra_body("approval") == {}
    assert _config()["auxiliary"]["title_generation"]["timeout"] == sidecar._HERMES_TITLE_TIMEOUT_S


def test_a_title_model_the_user_chose_is_kept(client):
    from hermes_cli.config import set_config_value

    set_config_value("auxiliary.title_generation.provider", "openrouter")
    set_config_value("auxiliary.title_generation.model", "google/gemini-flash")

    response = _external_setup(client)

    assert response.status_code == 200, response.text

    title = _config()["auxiliary"]["title_generation"]
    assert (title["provider"], title["model"], title.get("base_url")) == ("openrouter", "google/gemini-flash", "")


def test_setup_warns_when_the_local_model_gives_a_verifier_no_score(client):
    from .mock_tiers import MockTier

    local = MockTier("local", "hi")
    try:
        setup = {"local": {"runtime": "external", "base_url": local.base_url, "model": "local-m"},
                 "cloud": {"api_key": "sk-test-cloud-key"}}
        scored = client.post("/setup", json=setup)
        assert scored.status_code == 200, scored.text
        assert scored.json()["config"]["warnings"] == []
        probe = local.requests[-1]["body"]
        assert (probe["logprobs"], probe["reasoning_effort"], probe["chat_template_kwargs"]) == (
            True, "none", {"enable_thinking": False})

        local.logprobs = False
        unscored = client.post("/setup", json=setup)
        assert unscored.status_code == 200, unscored.text
        assert ["escalate every turn" in note for note in unscored.json()["config"]["warnings"]] == [True]

        # Switchyard scores a verdict only when "yes" and "no" both hold real probability.
        local.logprobs = True
        for alternatives in ([("Okay", 0.99), ("No", 0.001)], [("Okay", 0.95), ("yes", 0.03), ("no", 0.02)]):
            local.alternatives = alternatives
            stray = client.post("/setup", json=setup)
            assert ["escalate every turn" in note for note in stray.json()["config"]["warnings"]] == [True], alternatives
    finally:
        local.stop()


def test_setup_warns_about_a_tier_whose_account_cannot_answer(monkeypatch):
    from .mock_tiers import MockTier

    local, cloud = MockTier("local", "ok"), MockTier("cloud", "ok")
    monkeypatch.setenv("SETUP_TEST_KEY", "k")
    try:
        cfg = sy_settings.load({
            "local": {"base_url": local.base_url, "model": "local-m", "api_key_env": "SETUP_TEST_KEY"},
            "cloud": {"base_url": cloud.base_url, "model": "cloud-m", "api_key_env": "SETUP_TEST_KEY"},
        })
        assert sidecar.account_warnings(cfg) == []
        assert cloud.requests[-1]["authorization"] == "Bearer k"

        cloud.status = 429
        cloud.error_body = json.dumps({"error": {"message": "You exceeded your current quota.", "code": "insufficient_quota"}})
        local.status = 401
        warnings = sidecar.account_warnings(cfg)
        assert len(warnings) == 2
        assert warnings[0].startswith("The efficient model local-m refused its API key (HTTP 401)")
        assert warnings[1].startswith("The capable model cloud-m is out of credits, so Switchyard answers with the "
                                      "efficient model")
        assert warnings[1].endswith("You exceeded your current quota.")

        cloud.status = 429
        cloud.error_body = json.dumps({"error": {"message": "Rate limit reached, retry in 2s", "code": "rate_limit_exceeded"}})
        local.status = 200
        assert sidecar.account_warnings(cfg) == []

        cloud.status = 400
        cloud.error_body = json.dumps({"error": {"message": "The model `cloud-m` does not exist"}})
        [warning] = sidecar.account_warnings(cfg)
        assert warning.startswith("The capable model cloud-m refused a test request (HTTP 400: The model `cloud-m` "
                                  "does not exist)")
    finally:
        local.stop()
        cloud.stop()


def _load_provider_profile():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "switchyard_provider_profile_restore", Path(__file__).parents[1] / "provider" / "__init__.py")
    profile = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(profile)
    return profile


def _provider_copy():
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "plugins" / "model-providers" / sy_settings.PROVIDER


def test_the_plugin_puts_its_provider_profile_where_hermes_discovers_it():
    from pathlib import Path

    import plugins.switchyard as plugin

    copy = _provider_copy()
    copy.mkdir(parents=True)
    (copy / "__init__.py").write_text("stale\n")
    plugin._install_provider()

    bundled = Path(plugin.__file__).with_name("provider")
    assert {path.name: path.read_bytes() for path in copy.iterdir()} == {
        name: (bundled / name).read_bytes() for name in plugin._PROVIDER_FILES}


def test_a_removed_plugins_provider_copy_undoes_setup_and_deletes_itself(client, monkeypatch, tmp_path):
    import importlib.util

    import providers
    import plugins.switchyard as plugin
    from hermes_cli.config import load_config, save_config, set_config_value
    from hermes_cli.plugins_cmd import _forget_plugin_config

    set_config_value("model.provider", "openrouter")
    set_config_value("model.default", "qwen/qwen3.6-plus")
    assert _external_setup(client).status_code == 200
    written = _config()["auxiliary"]["title_generation"]["timeout"]
    # As on a Hermes set up by a plugin that kept the record only in its settings.
    (sidecar.state_dir() / sidecar.RESTORE_FILE).unlink()
    plugin._update_hermes_settings()
    plugin._install_provider()
    # Hermes's remove deletes the plugin's settings, and leaves the id the plugin had before its rename.
    _forget_plugin_config({sy_settings.PLUGIN_ID})
    config = load_config()
    config.setdefault("plugins", {})["enabled"] = ["memory-x", sy_settings.LEGACY_PLUGIN_ID]
    save_config(config)
    registered = []
    monkeypatch.setattr(providers, "register_provider", registered.append)
    # The plugin under test sits in this Hermes tree's own plugins folder.
    monkeypatch.setattr(providers, "__file__", str(tmp_path / "elsewhere" / "providers" / "__init__.py"))
    spec = importlib.util.spec_from_file_location("switchyard_provider_copy", _provider_copy() / "__init__.py")
    spec.loader.exec_module(importlib.util.module_from_spec(spec))

    assert (_config()["model"]["provider"], _config()["model"]["default"]) == ("openrouter", "qwen/qwen3.6-plus")
    assert _config()["auxiliary"]["title_generation"]["timeout"] != written
    assert _config()["plugins"]["enabled"] == ["memory-x"]
    assert sy_settings.PLUGIN_ID not in (_config()["plugins"].get("entries") or {})
    assert registered == []
    assert not _provider_copy().exists()
    assert not (sidecar.state_dir() / sidecar.RESTORE_FILE).exists()


def test_disabling_the_plugin_stops_its_sidecar_while_the_secret_scope_is_empty(client, monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    import agent.secret_scope
    from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled

    assert _external_setup(client).status_code == 200
    token = sidecar.get_token()
    seen = []

    class Sidecar(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append((self.path, self.headers.get("Authorization")))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Sidecar)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        (sidecar.state_dir() / "sidecar.json").write_text(json.dumps({"port": server.server_address[1]}))
        assert dashboard_set_agent_plugin_enabled(sy_settings.PLUGIN_ID, enabled=False)["ok"]
        # As while Hermes loads its providers.
        monkeypatch.setattr(agent.secret_scope, "get_secret", lambda name: None)
        monkeypatch.delenv(sy_settings.TOKEN_ENV, raising=False)
        _load_provider_profile()
    finally:
        server.shutdown()
    # Hermes's own provider load and the one above each ask.
    assert set(seen) == {("/monitor/shutdown", f"Bearer {token}")}


def test_setup_after_a_remove_takes_back_the_title_timeout_the_remove_left(client):
    from hermes_cli.config import unset_config_value
    from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled

    assert _external_setup(client).status_code == 200
    written = _config()["auxiliary"]["title_generation"]["timeout"]
    unset_config_value(f"plugins.entries.{sy_settings.PLUGIN_ID}.settings.{sy_settings.TITLE_TIMEOUT_KEY}")
    (sidecar.state_dir() / sidecar.RESTORE_FILE).unlink()
    assert _external_setup(client).status_code == 200
    assert _config()["auxiliary"]["title_generation"]["timeout"] == written

    assert dashboard_set_agent_plugin_enabled(sy_settings.PLUGIN_ID, enabled=False)["ok"]
    _load_provider_profile()
    assert _config()["auxiliary"]["title_generation"]["timeout"] != written


def test_disabling_the_plugin_gives_hermes_back_the_model_setup_replaced(client):
    from hermes_cli.config import set_config_value
    from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled

    set_config_value("model.provider", "openrouter")
    set_config_value("model.default", "qwen/qwen3.6-plus")
    set_config_value("auxiliary.approval.extra_body.chat_template_kwargs.keep", "true")
    before = {task: dict(_config()["auxiliary"][task]) for task in sy_settings.PINNED_TASKS}
    assert _external_setup(client).status_code == 200
    assert _external_setup(client).status_code == 200
    profile = _load_provider_profile()
    assert (profile.PLUGIN_ID, profile.PREVIOUS_MODEL_KEY, profile.TITLE_TIMEOUT_KEY, profile.LOCAL_PIN_MODEL,
            profile.PINNED_TASKS, profile.THINKING_OFF) == (
        sy_settings.PLUGIN_ID, sy_settings.PREVIOUS_MODEL_KEY, sy_settings.TITLE_TIMEOUT_KEY,
        sy_settings.LOCAL_PIN_MODEL, sy_settings.PINNED_TASKS, sy_settings.THINKING_OFF)
    assert _config()["model"]["provider"] == sy_settings.PROVIDER
    assert _config()["auxiliary"]["title_generation"]["timeout"] != before["title_generation"]["timeout"]

    assert dashboard_set_agent_plugin_enabled(sy_settings.PLUGIN_ID, enabled=False)["ok"]
    _load_provider_profile()

    model = _config()["model"]
    assert (model["provider"], model["default"], model.get("base_url") or "") == ("openrouter", "qwen/qwen3.6-plus", "")
    for task, node in before.items():
        keys = ("provider", "model", "base_url", "extra_body", "timeout")
        assert {key: _config()["auxiliary"][task].get(key) for key in keys} == {key: node.get(key) for key in keys}
    assert _load_provider_profile().restore_previous_model() is False


def test_disabling_the_plugin_unpins_titles_when_setup_ran_after_the_model_moved_off_switchyard(client):
    from hermes_cli.config import set_config_value
    from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled

    assert _external_setup(client).status_code == 200
    set_config_value("model.provider", "openrouter")
    set_config_value("model.default", "qwen/qwen3.6-plus")
    assert _external_setup(client).status_code == 200

    assert dashboard_set_agent_plugin_enabled(sy_settings.PLUGIN_ID, enabled=False)["ok"]
    _load_provider_profile()

    model = _config()["model"]
    assert (model["provider"], model["default"]) == ("openrouter", "qwen/qwen3.6-plus")
    for task in sy_settings.PINNED_TASKS:
        node = _config()["auxiliary"][task]
        assert {key: node.get(key) or "" for key in ("provider", "model", "base_url")} == {
            "provider": "", "model": "", "base_url": ""}
        assert not node.get("extra_body")


def test_disabling_the_plugin_unpins_side_tasks_with_no_record_of_what_setup_replaced(client):
    from hermes_cli.config import set_config_value, unset_config_value
    from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled

    assert _external_setup(client).status_code == 200
    unset_config_value(f"plugins.entries.{sy_settings.PLUGIN_ID}.settings.{sy_settings.PREVIOUS_MODEL_KEY}")
    (sidecar.state_dir() / sidecar.RESTORE_FILE).unlink()
    set_config_value("model.provider", "openrouter")

    assert dashboard_set_agent_plugin_enabled(sy_settings.PLUGIN_ID, enabled=False)["ok"]
    _load_provider_profile()

    assert _config()["model"]["provider"] == "openrouter"
    for task in sy_settings.PINNED_TASKS:
        node = _config()["auxiliary"][task]
        assert {key: node.get(key) or "" for key in ("provider", "model", "base_url")} == {
            "provider": "", "model": "", "base_url": ""}


def test_disabling_the_plugin_leaves_saved_replies_as_they_are(client):
    from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled
    from hermes_constants import get_hermes_home
    from hermes_state_registry import acquire, release_or_close

    from plugins.switchyard import escalation

    assert _external_setup(client).status_code == 200
    change = {"escalated": True, "tier": "cloud", "model": "opus", "step": 2}
    marked = escalation.mark_reply("done", {"changes": [change, {**change, "escalated": False}], "requests": 2})
    assert marked.startswith("::switchyard-route{") and marked.endswith("\n\ndone")
    db = acquire(get_hermes_home() / "state.db")
    try:
        db.create_session("s1", "desktop")
        db.append_message("s1", "assistant", marked)
    finally:
        release_or_close(db)

    assert dashboard_set_agent_plugin_enabled(sy_settings.PLUGIN_ID, enabled=False)["ok"]
    _load_provider_profile()

    db = acquire(get_hermes_home() / "state.db")
    try:
        assert [message["content"] for message in db.get_messages("s1")] == [marked]
    finally:
        release_or_close(db)


def test_settings_saved_under_the_old_plugin_id_move_to_the_new_one():
    import plugins.switchyard as plugin
    from hermes_cli.config import set_config_value

    legacy = f"plugins.entries.{sy_settings.LEGACY_PLUGIN_ID}.settings"
    set_config_value(f"{legacy}.default_preset", "vgr")
    set_config_value(f"{legacy}.local.model", "local-m")

    plugin._adopt_legacy_settings()
    entries = _config()["plugins"]["entries"]
    assert sy_settings.LEGACY_PLUGIN_ID not in entries
    assert entries[sy_settings.PLUGIN_ID]["settings"] == {"default_preset": "vgr", "local": {"model": "local-m"}}

    set_config_value(f"{legacy}.default_preset", "stale")
    plugin._adopt_legacy_settings()
    entries = _config()["plugins"]["entries"]
    assert sy_settings.LEGACY_PLUGIN_ID not in entries
    assert entries[sy_settings.PLUGIN_ID]["settings"]["default_preset"] == "vgr"


def test_the_old_plugin_id_is_taken_out_of_the_plugin_lists_on_load():
    import plugins.switchyard as plugin
    from hermes_cli.config import load_config, save_config

    config = load_config()
    config.setdefault("plugins", {})["enabled"] = ["memory-x", sy_settings.LEGACY_PLUGIN_ID, sy_settings.PLUGIN_ID]
    save_config(config)

    plugin._forget_legacy_id()
    assert _config()["plugins"]["enabled"] == ["memory-x", sy_settings.PLUGIN_ID]


def test_a_model_chosen_after_setup_is_left_alone_when_the_plugin_is_disabled(client):
    from hermes_cli.config import set_config_value
    from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled

    set_config_value("model.provider", "openrouter")
    set_config_value("model.default", "qwen/qwen3.6-plus")
    before = {task: dict(_config()["auxiliary"][task]) for task in sy_settings.PINNED_TASKS}
    assert _external_setup(client).status_code == 200
    set_config_value("model.provider", "anthropic")
    set_config_value("model.default", "claude-sonnet-5-5")

    assert dashboard_set_agent_plugin_enabled(sy_settings.PLUGIN_ID, enabled=False)["ok"]
    assert _load_provider_profile().restore_previous_model() is False
    assert (_config()["model"]["provider"], _config()["model"]["default"]) == ("anthropic", "claude-sonnet-5-5")
    for task, node in before.items():
        keys = ("provider", "model", "base_url", "extra_body", "timeout")
        assert {key: _config()["auxiliary"][task].get(key) for key in keys} == {key: node.get(key) for key in keys}


def _finished_job(client, body):
    started = client.post("/setup/jobs", json=body)
    assert started.status_code == 200, started.text
    for _ in range(600):
        job = client.get(f"/setup/jobs/{started.json()['job_id']}").json()
        if job["status"] != "running":
            return job
        time.sleep(0.2)
    raise AssertionError("the setup job did not finish")


def test_the_setup_job_reports_its_steps_and_the_result(client, monkeypatch):
    steps = []
    configure = sidecar.configure

    def recording(updates, **kwargs):
        progress = kwargs.pop("progress")
        return configure(updates, progress=lambda step: (steps.append(step), progress(step)), **kwargs)

    monkeypatch.setattr(sidecar, "configure", recording)
    job = _finished_job(client, {
        "local": {"runtime": "external", "base_url": "http://127.0.0.1:9931/v1", "model": "local-m"},
        "cloud": {"api_key": "sk-test-cloud-key"},
    })

    assert job["status"] == "done", job
    assert job["result"]["running"] and job["step"] == job["steps"]
    assert steps == ["enable", "save"]
    assert sy_settings.PLUGIN_ID in _config()["plugins"]["enabled"]


def test_the_setup_job_reads_keys_under_multi_profile_hosting(monkeypatch):
    from fastapi import Depends

    from agent.secret_scope import set_multiplex_active
    from hermes_cli.web_server_dashboard import _plugin_route_secret_scope

    monkeypatch.setattr(sidecar, "restart", lambda cfg: bool(sidecar.ensure_token()))
    monkeypatch.setattr(sidecar, "account_warnings", lambda cfg: [])
    app = FastAPI()
    app.include_router(plugin_api.router, dependencies=[Depends(_plugin_route_secret_scope)])
    set_multiplex_active(True)
    try:
        job = _finished_job(TestClient(app), {
            "local": {"runtime": "external", "base_url": "http://127.0.0.1:9931/v1", "model": "local-m"},
            "cloud": {"api_key": "sk-test-cloud-key"},
        })
    finally:
        set_multiplex_active(False)

    assert job["status"] == "done", job
    assert job["result"]["running"]


def test_a_rejected_setup_job_reports_why_and_frees_the_next_setup(client):
    job = _finished_job(client, {"local": {"runtime": "external", "model": "m"}, "cloud": {"api_key": "sk-test-cloud-key"}})

    assert job["status"] == "error" and "base url" in job["error"]
    assert _external_setup(client).status_code == 200


def test_presets_are_created_kept_across_setup_and_deleted(client):
    assert _external_setup(client).status_code == 200

    created = client.put("/presets/quick", json={
        "algorithm": "vgr", "options": {"deadline_seconds": 120}, "make_default": True,
    })
    assert created.status_code == 200, created.text
    assert _config()["model"]["default"] == "switchyard/quick"

    rerun = _external_setup(client, preset_name="quick", preset={"algorithm": "vgr"})
    assert rerun.status_code == 200, rerun.text
    cfg = sy_settings.load(sy_settings.plugin_settings(_config()))
    assert cfg.preset("quick").route["deadline_seconds"] == 120

    before = _config()
    bad = client.put("/presets/quick", json={"algorithm": "vgr", "options": {"mode": "sideways"}})
    assert bad.status_code == 400
    assert _config() == before

    assert client.delete("/presets/quick").status_code == 200
    cfg = sy_settings.load(sy_settings.plugin_settings(_config()))
    assert [p.name for p in cfg.presets] == ["vgr"]
    assert _config()["model"]["default"] == "switchyard/vgr"


def test_algorithms_other_than_vgr_are_neither_offered_nor_shown(client):
    from hermes_cli.plugins_state import save_plugin_setting

    assert _external_setup(client).status_code == 200
    before = _config()

    refused = client.put("/presets/split", json={"algorithm": "random", "options": {"weights": [3, 1]}})
    refused_setup = _external_setup(client, preset_name="split", preset={"algorithm": "llm_classifier"})

    assert refused.status_code == 400 and "offers Verification-Gated Routing only" in refused.json()["detail"]
    assert refused_setup.status_code == 400
    assert _config() == before
    assert [a["name"] for a in client.get("/state").json()["algorithms"]] == ["vgr"]

    # A preset an earlier build saved with another algorithm, even as the default, is not loaded or listed.
    saved = list(sy_settings.plugin_settings(_config()).get("presets") or [])
    save_plugin_setting(sy_settings.PLUGIN_ID, ("presets",), [*saved, {"name": "split", "algorithm": "random"}])
    save_plugin_setting(sy_settings.PLUGIN_ID, ("default_preset",), "split")
    cfg = sy_settings.load(sy_settings.plugin_settings(_config()))
    assert ([p.name for p in cfg.presets], cfg.default_preset) == (["vgr"], "vgr")
    assert [p["name"] for p in client.get("/state").json()["config"]["presets"]] == ["vgr"]
    assert _load_provider_profile()._preset_names() == ["vgr"]


def test_vgr_starts_unbounded_and_bounded_local_applies_the_time_limits_until_turned_off(client):
    def vgr_route():
        return sy_settings.load(sy_settings.plugin_settings(_config())).preset("vgr").route

    assert _external_setup(client).status_code == 200
    assert vgr_route()["deadline_seconds"] == sy_settings.UNBOUNDED_DEADLINE_SECONDS
    assert "local_turn_budget_seconds" not in vgr_route()
    assert client.get("/state").json()["config"]["bounded_local"] is False

    assert _external_setup(client, bounded_local=True).status_code == 200
    assert (vgr_route()["deadline_seconds"], vgr_route()["local_turn_budget_seconds"]) == (300, 600)
    assert client.get("/state").json()["config"]["bounded_local"] is True

    assert _external_setup(client).status_code == 200
    assert vgr_route()["local_turn_budget_seconds"] == 600

    assert _external_setup(client, bounded_local=False).status_code == 200
    assert vgr_route()["deadline_seconds"] == sy_settings.UNBOUNDED_DEADLINE_SECONDS
    assert "local_turn_budget_seconds" not in vgr_route()


def test_hermes_waits_out_an_unbounded_vgr_decision_and_keeps_timeouts_the_user_set(client):
    from agent.chat_completion_helpers import _cloud_stale_timeout_for
    from hermes_cli.config import set_config_value
    from hermes_cli.timeouts import get_provider_request_timeout, get_provider_stale_timeout

    class Agent:
        provider, model = sy_settings.PROVIDER, "switchyard/vgr"

    def timeouts():
        return get_provider_stale_timeout(Agent.provider, Agent.model), get_provider_request_timeout(Agent.provider, Agent.model)

    wait = sy_settings.UNBOUNDED_DEADLINE_SECONDS + sidecar._CLOUD_START_S
    assert _external_setup(client, bounded_local=False).status_code == 200
    assert timeouts() == (wait, wait)
    # An explicit provider timeout replaces Hermes's 900 s ceiling for local endpoints.
    assert _cloud_stale_timeout_for(Agent, {}) == wait

    assert _external_setup(client, bounded_local=True).status_code == 200
    assert timeouts() == (None, None)

    set_config_value(f"providers.{sy_settings.PROVIDER}.stale_timeout_seconds", "5000")
    assert _external_setup(client, bounded_local=False).status_code == 200
    assert timeouts() == (5000, wait)
    assert _external_setup(client, bounded_local=True).status_code == 200
    assert timeouts() == (5000, None)


def test_rejected_setup_leaves_config_untouched(client):
    before = _config()

    missing_url = client.post("/setup", json={"local": {"runtime": "external", "model": "m"}, "cloud": {"api_key": "sk-test-cloud-key"}})
    plain_http_cloud = client.post("/setup", json={
        "local": {"runtime": "external", "base_url": "http://127.0.0.1:9931/v1", "model": "local-m"},
        "cloud": {"base_url": "http://cloud.example/v1", "api_key": "sk-test-cloud-key"},
    })
    no_local_model = client.post("/setup", json={"local": {"runtime": "managed"}, "cloud": {"api_key": "sk-test-cloud-key"}})

    assert missing_url.status_code == 400
    assert plain_http_cloud.status_code == 400 and "https" in plain_http_cloud.json()["detail"]
    assert no_local_model.status_code == 409
    assert _config() == before


def test_switching_a_tier_to_a_subscription_records_its_format_and_drops_the_url(client, monkeypatch):
    from plugins.switchyard import providers

    assert _external_setup(client).status_code == 200
    monkeypatch.setattr(providers, "_runtime", lambda tier: {
        "provider": "openai-codex", "api_mode": "codex_responses",
        "base_url": "https://chatgpt.com/backend-api/codex", "api_key": "codex-oauth-sentinel",
    })
    response = client.post("/setup", json={
        "local": {"source": "url", "base_url": "http://127.0.0.1:9931/v1", "model": "local-m"},
        "cloud": {"source": "provider", "provider": "openai-codex", "model": "gpt-5.5"},
    })
    assert response.status_code == 200, response.text
    saved = sy_settings.plugin_settings(_config())["cloud"]
    assert (saved["source"], saved["provider"], saved["format"], saved["base_url"]) == (
        "provider", "openai-codex", "openai_responses", None)
    assert "codex-oauth-sentinel" not in str(_config())
    # VGR judges with the efficient tier, which still speaks Chat Completions: no warning.
    assert response.json()["config"]["warnings"] == []

    monkeypatch.setattr(providers, "_runtime", lambda tier: {"provider": "nous", "api_mode": "chat_completions", "api_key": ""})
    signed_out = client.post("/setup", json={"cloud": {"source": "provider", "provider": "nous", "model": "hermes-4"}})
    assert signed_out.status_code == 400 and "hermes auth add nous" in signed_out.json()["detail"]


def test_hermes_current_custom_endpoint_is_saved_as_its_url_before_switchyard_takes_the_model(client):
    from agent.secret_scope import get_secret
    from hermes_cli.config import set_config_value

    set_config_value("model.provider", "custom")
    set_config_value("model.base_url", "http://127.0.0.1:9931/v1")
    set_config_value("model.default", "local-m")
    set_config_value("model.api_key", "sk-local-test-key-123")

    response = client.post("/setup", json={
        "local": {"source": "provider", "provider": "custom", "model": "local-m"},
        "cloud": {"api_key": "sk-test-cloud-key"},
    })

    assert response.status_code == 200, response.text
    cfg = sy_settings.load(sy_settings.plugin_settings(_config()))
    # A provider tier would now resolve to the Switchyard front that replaced model.base_url.
    assert (cfg.local.source, cfg.local.base_url, cfg.local.model) == ("url", "http://127.0.0.1:9931/v1", "local-m")
    assert get_secret(cfg.local.api_key_env) == "sk-local-test-key-123"
    assert "sk-local-test-key-123" not in str(sy_settings.plugin_settings(_config()))
    assert _config()["model"]["base_url"] == cfg.front_base_url


_ROWS = [
    {"slug": "openrouter", "name": "OpenRouter", "authenticated": True, "source": "hermes",
     "models": ["vendor/big", "vendor/small"]},
    {"slug": "moa", "name": "Mixture of Agents", "authenticated": True, "source": "virtual", "models": ["default"]},
    {"slug": "switchyard", "name": "Switchyard", "authenticated": True, "models": ["switchyard/vgr"]},
    {"slug": "custom:lan-vllm", "name": "LAN vLLM", "authenticated": True, "source": "user-config",
     "api_url": "http://192.168.1.20:8000/v1", "models": ["qwen3-32b"]},
    {"slug": "llamacpp", "name": "Local", "authenticated": True, "source": "local-runtime",
     "models": ["Qwen3.6-35B-A3B-UD-Q4_K_M"]},
    {"slug": "anthropic", "name": "Anthropic", "authenticated": True, "source": "hermes", "models": []},
]


@pytest.fixture
def inventory(monkeypatch):
    from hermes_cli import inventory as hermes_inventory

    rows = [dict(row) for row in _ROWS]
    monkeypatch.setattr(hermes_inventory, "build_model_options_payload", lambda ctx, **kw: {"providers": rows})
    return rows


def test_hermes_models_default_the_efficient_tier_to_the_one_click_local_model(client, inventory):
    listed = client.get("/providers").json()

    assert [(p["provider"], p["source"], p["local"]) for p in listed["providers"]] == [
        ("openrouter", "provider", False), ("custom:lan-vllm", "provider", True), ("llamacpp", "managed", True)]
    assert listed["defaults"]["local"] == {"source": "managed", "provider": "llamacpp", "model": "Qwen3.6-35B-A3B-UD-Q4_K_M"}
    assert listed["defaults"]["cloud"]["provider"] == "openrouter"


def test_hermes_current_model_fills_the_tier_it_belongs_to(client, inventory):
    from hermes_cli.config import set_config_value

    set_config_value("model.provider", "custom:lan-vllm")
    set_config_value("model.default", "qwen3-32b")
    inventory[3]["is_current"] = True
    local_current = client.get("/providers").json()
    assert local_current["defaults"]["local"] == {"source": "provider", "provider": "custom:lan-vllm", "model": "qwen3-32b"}

    set_config_value("model.provider", "openrouter")
    set_config_value("model.default", "vendor/small")
    inventory[3]["is_current"], inventory[0]["is_current"] = False, True
    remote_current = client.get("/providers").json()
    assert remote_current["defaults"]["cloud"] == {"source": "provider", "provider": "openrouter", "model": "vendor/small"}
    assert remote_current["defaults"]["local"]["provider"] == "llamacpp"


def test_cli_setup_without_tier_flags_takes_hermes_models_and_then_keeps_them(inventory, monkeypatch):
    import argparse

    from plugins.switchyard import _cmd_setup, providers

    monkeypatch.setattr(providers, "_runtime", lambda tier: {
        "provider": "openrouter", "api_mode": "chat_completions", "base_url": "https://openrouter.ai/api/v1", "api_key": "sk-or",
    })
    monkeypatch.setattr(sidecar, "account_warnings", lambda cfg: [])
    args = argparse.Namespace(algorithm=None, preset=None, no_default=False, port=None)

    assert _cmd_setup(args) == 0
    cfg = sy_settings.load(sy_settings.plugin_settings(_config()))
    assert (cfg.local.source, cfg.local.model) == ("managed", "Qwen3.6-35B-A3B-UD-Q4_K_M")
    assert (cfg.cloud.source, cfg.cloud.provider) == ("provider", "openrouter")

    inventory.pop()
    inventory.pop(4)
    assert _cmd_setup(args) == 0
    assert sy_settings.load(sy_settings.plugin_settings(_config())).local.managed


def test_pin_is_saved_for_the_next_sidecar_start(client):
    assert _external_setup(client).status_code == 200
    assert client.post("/pin", json={"pin": "cloud"}).json() == {"ok": True, "pin": "cloud", "live": False}
    assert client.get("/state").json()["config"]["pin"] == "cloud"
    assert client.post("/pin", json={"pin": "off"}).json()["pin"] == ""
    assert client.post("/pin", json={"pin": "sideways"}).status_code == 400
