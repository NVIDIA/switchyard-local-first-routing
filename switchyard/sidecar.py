# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Switchyard sidecar: one process hosting the tap and Switchyard's in-process server.

``hermes switchyard serve`` runs it in the foreground; :func:`ensure_running` spawns it
detached by re-invoking the running Hermes entry point, which also works when
Hermes is a single frozen executable.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import re
import secrets
import tempfile
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Tuple
from urllib.parse import urlsplit

import httpx

from . import settings as sy_settings
from . import stages
from .decisions import LogFollower
from .monitor import Monitor
from .runtime import RuntimeMissing
from .tap import TapServer

logger = logging.getLogger(__name__)

_STARTUP_TIMEOUT_S = 20.0
# How long a restart waits for the old sidecar to stop answering before it kills it.
_RESTART_WAIT_S = 10.0
# Hermes's agent.local_stream_stale_timeout default.
_HERMES_LOCAL_STREAM_S = 900
# Past a VGR deadline, time for the cloud model to start answering.
_CLOUD_START_S = 300
# Hermes's auxiliary.title_generation.timeout default, and its HERMES_API_TIMEOUT.
_HERMES_TITLE_TIMEOUT_S = 30
_HERMES_API_TIMEOUT_S = 1800
# Past a VGR deadline, time for a queued title to come back from the local model.
_TITLE_SLACK_S = 60
_STREAM_TIMEOUT_KEYS = ("stale_timeout_seconds", "request_timeout_seconds")


def state_dir() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "switchyard"


# In state_dir(): the provider profile's restore record, kept apart from the plugin's settings.
RESTORE_FILE = "restore.json"

# Desktop's half touches this while it is loaded. Only it draws the route notes written into saved
# replies; with it off, Desktop shows them as raw ``::switchyard-route{...}`` text.
_RENDERER_FILE = "desktop-renderer"
RENDERER_FRESH_S = 300.0


def note_renderer(loaded: bool) -> None:
    path = state_dir() / _RENDERER_FILE
    if loaded:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    else:
        path.unlink(missing_ok=True)


def renderer_live() -> bool:
    """Whether Desktop's half reported being loaded within ``RENDERER_FRESH_S``."""
    try:
        return time.time() - (state_dir() / _RENDERER_FILE).stat().st_mtime < RENDERER_FRESH_S
    except OSError:
        return False


def _load_config() -> Dict[str, Any]:
    from hermes_cli.config import load_config_readonly

    return load_config_readonly() or {}


def load_settings() -> sy_settings.SwitchyardSettings:
    return sy_settings.load(sy_settings.plugin_settings(_load_config()))


def _merge(base: Mapping[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _leaves(updates: Mapping[str, Any], prefix: Tuple[str, ...] = ()):
    for key, value in updates.items():
        if isinstance(value, Mapping):
            yield from _leaves(value, (*prefix, key))
        else:
            yield (*prefix, key), value


_SWITCHYARD_ERROR = re.compile(
    r"(missing field `[^`]+`|unknown field `[^`]+`(?:, expected [^\n|]+)?|unknown variant `[^`]+`, expected [^\n|]+|invalid (?:type|value): [^\n|]+"
    r"|\w+ route \S+ (?:requires|uses|must) [^:\n]+)"
)


_UNKNOWN_FIELD = re.compile(r"unknown field `(?P<key>[^`]+)`")
_NO_VGR = re.compile(r"unknown variant `vgr`")


def _switchyard_error(error: Exception) -> str:
    text = str(error)
    if _NO_VGR.search(text):
        return _no_vgr_message()
    match = _SWITCHYARD_ERROR.search(text)
    return match.group(1).strip() if match else " ".join(text.split())[:300]


def _no_vgr_message() -> str:
    from .runtime import describe

    runtime = describe() or {}
    build = " ".join(str(value) for value in (runtime.get("project") or "nemo-switchyard", runtime.get("version")) if value)
    where = f" (from {runtime['location']})" if runtime.get("location") else ""
    return (f"{sy_settings.PRODUCT_NAME} needs VGR routing, which the installed Switchyard engine, {build}{where}, "
            "does not have. Hermes most likely rebuilt its "
            "plugin environment from a Switchyard build without it; see Troubleshooting in the plugin's README to "
            "reinstall the plugin's own build")


def check_routes(cfg: sy_settings.SwitchyardSettings) -> None:
    """Have Switchyard itself parse the deployment, so a bad preset fails before it is saved.

    Skipped when no Switchyard runtime is available to this interpreter.
    """
    from .runtime import server_class

    try:
        Server = server_class()
    except RuntimeMissing:
        return
    try:
        _parse_routes(Server, cfg, engine_lacks(Server, cfg))
    except Exception as error:
        raise sy_settings.SettingsError(f"Switchyard rejected the routes: {_switchyard_error(error)}") from None


def _parse_routes(Server: Any, cfg: sy_settings.SwitchyardSettings, omit: frozenset = frozenset()) -> None:
    fd, path = tempfile.mkstemp(suffix=".toml", prefix="switchyard-check-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(sy_settings.render_switchyard_toml(cfg, with_credentials=False, omit=omit))
        with _quiet_stderr():
            Server(path, port=0).close()
    finally:
        os.unlink(path)


@contextlib.contextmanager
def _quiet_stderr() -> Iterator[None]:
    """Hides what the engine logs straight to stderr, the user's terminal for a CLI command, as it starts and stops."""
    sys.stderr.flush()
    try:
        saved = os.dup(2)
    except OSError:  # no stderr, as in a windowed Hermes
        yield
        return
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def running_toml(cfg: sy_settings.SwitchyardSettings) -> str:
    """The deployment as the sidecar writes it for the installed Switchyard."""
    from .runtime import server_class

    try:
        omit = engine_lacks(server_class(), cfg)
    except RuntimeMissing:
        omit = frozenset()
    return sy_settings.render_switchyard_toml(cfg, omit=omit)


def consent_problem(cfg: sy_settings.SwitchyardSettings, *, saved: bool = True) -> Optional[str]:
    """Why ``cfg`` cannot route until the user agrees to the use of a subscription sign-in; None when it can.

    ``saved`` is False for a setup that was refused, so saved nothing for the flag alone to agree to.
    """
    from . import providers

    missing = providers.unconsented(cfg)
    if not missing:
        return None
    agree = ("run `hermes switchyard setup --allow-subscription-use`" if saved
             else "run the same `hermes switchyard setup` command again with `--allow-subscription-use` added")
    return (f"{providers.subscription_notice(missing)} To agree, {agree}, or turn on the agreement in the "
            "Switchyard panel's setup and save. Otherwise pick another model for that tier.")


def consent_refusal(cfg: sy_settings.SwitchyardSettings) -> Optional[str]:
    """The short form of :func:`consent_problem` a refusing sidecar answers each request with."""
    from . import providers

    missing = providers.unconsented(cfg)
    return providers.subscription_refusal(missing) if missing else None


def _record_consent(cfg: sy_settings.SwitchyardSettings, updates: Dict[str, Any], allow: Optional[bool],
                    confirm: Optional[Callable[[str], bool]]) -> sy_settings.SwitchyardSettings:
    """``cfg`` with the user's answer on its subscription sign-ins; refuses one they have not agreed to,
    except that a withdrawal is always saved (the sidecar then refuses every request)."""
    from . import providers

    consent = () if allow is False else cfg.subscription_consent
    missing = [p for p in providers.unconsented(cfg) if p not in consent]
    # Declining at the prompt refuses the setup; only an explicit withdrawal is saved without consent.
    if missing and allow is None and confirm is not None and confirm(providers.subscription_notice(missing)):
        allow = True
    if missing and allow:
        consent = (*consent, *missing)
    if consent != cfg.subscription_consent:
        updates[sy_settings.SUBSCRIPTION_CONSENT_KEY] = list(consent)
        cfg = dataclasses.replace(cfg, subscription_consent=consent)
    problem = consent_problem(cfg, saved=False)
    if problem and allow is not False:
        raise sy_settings.ConsentRequired(problem)
    return cfg


def startup_problem(cfg: sy_settings.SwitchyardSettings) -> Optional[str]:
    """Why the sidecar cannot start with ``cfg`` here, as the user should read it; None when nothing is known."""
    consent = consent_problem(cfg)
    if consent:
        return consent
    missing = [tier.api_key_env for tier in (cfg.local, cfg.cloud) if tier.source == "url" and tier.api_key_env
               and not _secret(tier.api_key_env)]
    if missing:
        return f"{' and '.join(missing)} not set; add {'it' if len(missing) == 1 else 'them'} with `hermes config env`"
    try:
        from .runtime import server_class

        server_class()
        check_routes(cfg)
    except (RuntimeMissing, sy_settings.SettingsError) as error:
        return str(error)
    return None


def _secret(name: str) -> str:
    from agent.secret_scope import get_secret

    return (get_secret(name) or "").strip()


_lacks: Dict[int, frozenset] = {}


def engine_lacks(Server: Any, cfg: sy_settings.SwitchyardSettings) -> frozenset:
    """The :data:`~settings.OPTIONAL_ROUTE_DEFAULTS` this Switchyard build refuses as unknown route keys."""
    known = _lacks.get(id(Server))
    if known is not None:
        return known
    omit: set = set()
    for _ in range(len(sy_settings.OPTIONAL_ROUTE_DEFAULTS) + 1):
        try:
            _parse_routes(Server, cfg, frozenset(omit))
        except Exception as error:
            match = _UNKNOWN_FIELD.search(str(error))
            key = match.group("key") if match else None
            if key in sy_settings.OPTIONAL_ROUTE_DEFAULTS and key not in omit:
                omit.add(key)
                continue
            # Another problem with the routes; the caller's own parse reports it.
            return frozenset(omit)
        _lacks[id(Server)] = frozenset(omit)
        break
    return frozenset(omit)


def _resolve_provider_tiers(
    cfg: sy_settings.SwitchyardSettings, updates: Dict[str, Any], merged: Dict[str, Any]
) -> Tuple[sy_settings.SwitchyardSettings, Dict[str, str]]:
    """Check each provider tier this update sets, and store the wire format its provider uses.

    A provider Hermes resolves from config.yaml's ``model`` block becomes the URL
    tier it resolves to now, before Switchyard takes that block over; returns the
    API keys those URL tiers need, by env var.
    """
    from . import providers

    keys: Dict[str, str] = {}
    changed = False
    for name, tier in (("local", cfg.local), ("cloud", cfg.cloud)):
        if name not in updates or tier.source != "provider":
            continue
        if tier.provider.lower() in providers.MODEL_BLOCK_PROVIDERS:
            fields, key = providers.pinned_url(tier)
            fields["api_key_env"] = sy_settings.tier_key_env(name) if key else ""
            if key:
                keys[fields["api_key_env"]] = key
        else:
            fields = {"format": providers.check(tier)}
        updates[name] = {**dict(updates[name] or {}), **fields}
        merged[name] = {**dict(merged.get(name) or {}), **fields}
        changed = changed or fields != {"format": tier.format}
    return (sy_settings.load(merged) if changed else cfg), keys


def configure(
    updates: Mapping[str, Any],
    *,
    make_default: bool = True,
    remove_presets: Tuple[str, ...] = (),
    progress: Callable[[str], None] = lambda step: None,
    allow_subscription_use: Optional[bool] = None,
    confirm_subscription_use: Optional[Callable[[str], bool]] = None,
) -> sy_settings.SwitchyardSettings:
    """Apply ``updates`` to the plugin settings and, optionally, make Switchyard Hermes's model.

    ``updates["presets"]`` maps preset names to whole presets (each replaces any
    preset of that name). The merged settings are validated, by this plugin and
    by Switchyard, before anything is written, so a rejected update leaves
    config.yaml untouched. ``progress`` is told "enable" before the plugin is
    first enabled and "save" before the settings are written.

    A tier on a subscription sign-in needs the user's agreement: ``allow_subscription_use``
    True gives it and False withdraws it; left None, ``confirm_subscription_use`` is shown
    the notice and asked, and without it the saved answer stands.
    """
    from hermes_cli.config import save_env_value_secure, set_config_value
    from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled
    current = sy_settings.plugin_settings(_load_config())
    updates = dict(updates)
    preset_updates = dict(updates.pop("presets", None) or {})
    for name, body in preset_updates.items():
        algorithm = (body or {}).get("algorithm") if isinstance(body, Mapping) else None
        sy_settings.require_available(str(name), str(algorithm or sy_settings.RECOMMENDED).strip().lower())
    presets = None
    if preset_updates or remove_presets:
        presets = sy_settings.preset_map(current.get("presets"))
        presets.update(preset_updates)
        for name in remove_presets:
            if name not in presets:
                raise sy_settings.SettingsError(f"no Switchyard preset named {name!r}")
            del presets[name]
    merged = _merge(current, updates)
    if presets is not None:
        merged["presets"] = presets
    if remove_presets and merged.get("default_preset") in remove_presets and "default_preset" not in updates:
        merged.pop("default_preset", None)
        updates["default_preset"] = None
    cfg = sy_settings.load(merged)
    cfg, keys = _resolve_provider_tiers(cfg, updates, merged)
    cfg = _record_consent(cfg, updates, allow_subscription_use, confirm_subscription_use)
    check_routes(cfg)
    # Plugins are opt-in: without this, `hermes switchyard serve` (the sidecar) does not exist.
    if sy_settings.PLUGIN_ID not in ((_load_config().get("plugins") or {}).get("enabled") or ()):
        progress("enable")
    enabled = dashboard_set_agent_plugin_enabled(sy_settings.PLUGIN_ID, enabled=True)
    if not enabled.get("ok"):
        raise sy_settings.SettingsError(f"could not enable the switchyard plugin: {enabled.get('error')}")
    progress("save")
    for key_env, key in keys.items():
        save_env_value_secure(key_env, key)
    for segments, value in _leaves(updates):
        sy_settings.save_setting(segments, value)
    if presets is not None:
        sy_settings.save_setting(("presets",), sy_settings.preset_records(presets))
    _fit_stream_timeouts(cfg)
    ensure_token()
    if make_default:
        _remember_previous_model()
        set_config_value("model.provider", sy_settings.PROVIDER)
        set_config_value("model.default", cfg.default.model)
        set_config_value("model.base_url", cfg.front_base_url)
    _fit_side_tasks(cfg, pin=make_default)
    _fit_model_guidance(cfg)
    sy_settings.save_setting((sy_settings.HERMES_SETTINGS_KEY,), sy_settings.HERMES_SETTINGS_REVISION)
    return cfg


def update_hermes_settings() -> bool:
    """Writes the Hermes settings this plugin's setup writes over an older plugin's, once.

    An upgrade otherwise keeps the older settings until setup runs again. Only a Hermes where
    Switchyard is set up is touched, and the same choices setup leaves alone stay as they are.
    """
    config = _load_config()
    model = config.get("model")
    default = isinstance(model, Mapping) and model.get("provider") == sy_settings.PROVIDER
    settings = sy_settings.plugin_settings(config)
    if not (default or settings.get("local")):
        return False
    applied = settings.get(sy_settings.HERMES_SETTINGS_KEY)
    if isinstance(applied, int) and applied >= sy_settings.HERMES_SETTINGS_REVISION:
        return False
    cfg = load_settings()
    _fit_stream_timeouts(cfg)
    _fit_side_tasks(cfg, pin=default)
    _fit_model_guidance(cfg)
    sy_settings.save_setting((sy_settings.HERMES_SETTINGS_KEY,), sy_settings.HERMES_SETTINGS_REVISION)
    return True


def _fit_side_tasks(cfg: sy_settings.SwitchyardSettings, *, pin: bool) -> None:
    """Pins the side tasks while Switchyard is Hermes's model. Otherwise a chat that picks a
    Switchyard model still sends its title to the sidecar, so only the title timeout is fitted."""
    if pin:
        _pin_side_tasks(cfg)
        return
    node = (_load_config().get("auxiliary") or {}).get("title_generation")
    node = node if isinstance(node, Mapping) else {}
    if _follows_the_chat_model(node):
        _fit_title_timeout(cfg, node, _local_tier_is_local(cfg))


def _follows_the_chat_model(node: Mapping[str, Any]) -> bool:
    """Whether a side task goes to the chat's model or Switchyard, not one the user chose for it."""
    provider = str(node.get("provider") or "").strip().lower()
    if provider == sy_settings.PROVIDER:
        return True
    return provider in ("", "auto") and not str(node.get("base_url") or "").strip()


def _fit_stream_timeouts(cfg: sy_settings.SwitchyardSettings) -> None:
    """Lets Hermes wait out the slowest VGR decision instead of killing the stream.

    VGR sends nothing while the local model drafts and is checked, for up to the preset's
    ``deadline_seconds``. Hermes kills and retries a stream from a local endpoint, which the
    sidecar is, after 900 s without content (keep-alive comments do not count), and gives up
    on a socket read after HERMES_API_TIMEOUT (1800 s). Setup writes Switchyard provider
    timeouts that cover the deadline, and removes them once no preset needs them. Timeouts
    the user set are left alone.
    """
    from hermes_cli.config import set_config_value, unset_config_value
    wait = int(_vgr_deadline(cfg)) + _CLOUD_START_S
    needed = wait if wait > _HERMES_LOCAL_STREAM_S else None
    config = _load_config()
    written = sy_settings.plugin_settings(config).get(sy_settings.STREAM_TIMEOUT_KEY)
    provider = (config.get("providers") or {}).get(sy_settings.PROVIDER)
    provider = provider if isinstance(provider, Mapping) else {}
    for key in _STREAM_TIMEOUT_KEYS:
        current = provider.get(key)
        if current is not None and current != written:
            continue
        if needed is not None and current != needed:
            set_config_value(f"providers.{sy_settings.PROVIDER}.{key}", str(needed))
        elif needed is None and current is not None:
            unset_config_value(f"providers.{sy_settings.PROVIDER}.{key}")
    if written != needed:
        sy_settings.save_setting((sy_settings.STREAM_TIMEOUT_KEY,), needed)


def _hermes_guidance_models() -> Dict[str, Tuple[str, ...]]:
    """Hermes's own model-name lists behind each guidance gate's "auto"; empty on a Hermes without them."""
    try:
        from agent import prompt_builder
    except ImportError:
        return {}
    lists = {"tool_use_enforcement": getattr(prompt_builder, "TOOL_USE_ENFORCEMENT_MODELS", None),
             "execution_guidance": getattr(prompt_builder, "EXECUTION_GUIDANCE_MODELS", None)}
    return {gate: tuple(models) for gate, models in lists.items() if isinstance(models, (tuple, list))}


def _fit_model_guidance(cfg: sy_settings.SwitchyardSettings) -> None:
    """Extends Hermes's model-name guidance gates to switchyard/ models when the efficient model matches them.
    Gates the user set are left alone."""
    from hermes_cli.config import load_config, save_config
    hermes_lists = _hermes_guidance_models()
    config = load_config()
    agent = config.setdefault("agent", {})
    written = sy_settings.plugin_settings(config).get(sy_settings.MODEL_GUIDANCE_KEY)
    written = dict(written) if isinstance(written, Mapping) else {}
    efficient = str(cfg.local.model or "").lower()
    record, changed = dict(written), False
    for gate in sy_settings.MODEL_GUIDANCE_GATES:
        current = agent.get(gate)
        ours = written.get(gate)
        if current not in (None, "auto") and current != ours:
            record.pop(gate, None)
            continue
        models = list(hermes_lists.get(gate) or ())
        needed = ([*models, sy_settings.MODEL_PREFIX]
                  if any(m in efficient for m in models) and sy_settings.MODEL_PREFIX not in models else None)
        if needed is not None:
            # A list; set_config_value would keep it as text, since the default is the string "auto".
            if current != needed:
                agent[gate], changed = needed, True
            record[gate] = needed
        else:
            if current is not None and current == ours:
                del agent[gate]
                changed = True
            record.pop(gate, None)
    if changed:
        save_config(config)
    if record != written:
        _save_restore(sy_settings.MODEL_GUIDANCE_KEY, record or None)


def _vgr_deadline(cfg: sy_settings.SwitchyardSettings) -> float:
    return max((float(p.route.get("deadline_seconds") or 0) for p in cfg.presets if p.algorithm == "vgr"),
               default=0.0)


def _remember_previous_model() -> None:
    """Saves the model and title settings Switchyard replaces, for the provider profile to put
    back once the plugin is disabled or removed. A rerun of setup keeps the first record."""
    config = _load_config()
    model = config.get("model")
    model = model if isinstance(model, Mapping) else {"default": model or ""}
    if model.get("provider") == sy_settings.PROVIDER:
        return
    previous: Dict[str, Any] = {
        "model": {key: str(model.get(key) or "") for key in ("provider", "default", "base_url")},
    }
    for task in sy_settings.PINNED_TASKS:
        node = (config.get("auxiliary") or {}).get(task)
        node = node if isinstance(node, Mapping) else {}
        previous[task] = {key: str(node.get(key) or "") for key in ("provider", "model", "base_url")}
    _save_restore(sy_settings.PREVIOUS_MODEL_KEY, previous)


def _save_restore(key: str, value: Any) -> None:
    """Saves what setup replaced in the plugin's settings and in its state folder, where the
    provider profile still finds it after Hermes's ``plugins remove`` deletes the settings."""
    sy_settings.save_setting((key,), value)
    _write_private(state_dir() / RESTORE_FILE, json.dumps({**_restore_record(), key: value}))


def _restore_record() -> Dict[str, Any]:
    try:
        record = json.loads((state_dir() / RESTORE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def keep_restore_record() -> bool:
    """Copies what setup replaced into the state folder on a Hermes whose setup ran before it was kept there."""
    settings = sy_settings.plugin_settings(_load_config())
    record = _restore_record()
    kept = {key: settings[key] for key in (sy_settings.PREVIOUS_MODEL_KEY, sy_settings.TITLE_TIMEOUT_KEY,
                                           sy_settings.MODEL_GUIDANCE_KEY)
            if key in settings}
    if all(record.get(key) == value for key, value in kept.items()):
        return False
    _write_private(state_dir() / RESTORE_FILE, json.dumps({**record, **kept}))
    return True


def _pin_side_tasks(cfg: sy_settings.SwitchyardSettings) -> None:
    """Sends Hermes's session titles and smart-approval checks straight to the efficient tier.

    Both run under Hermes's 30 s auxiliary timeout, which a VGR turn can outlast. A local
    efficient model is asked not to think for either, as Hermes's own request does not reach a
    llama.cpp template: an approval verdict is one word within 16 tokens, which a thinking model
    spends thinking, and a title asked alongside the first turn's draft thinks for a minute on the
    same GPU, holding the draft back. A cloud efficient model could reject the llama.cpp field. A
    provider or endpoint the user chose for a task, and an ``extra_body`` value they set, are
    left alone.
    """
    from hermes_cli.config import set_config_value

    local = _local_tier_is_local(cfg)
    for task in sy_settings.PINNED_TASKS:
        node = (_load_config().get("auxiliary") or {}).get(task)
        node = node if isinstance(node, Mapping) else {}
        if not _follows_the_chat_model(node):
            continue
        set_config_value(f"auxiliary.{task}.provider", sy_settings.PROVIDER)
        set_config_value(f"auxiliary.{task}.model", sy_settings.LOCAL_PIN_MODEL)
        set_config_value(f"auxiliary.{task}.base_url", cfg.front_base_url)
        if task == "title_generation":
            _fit_title_timeout(cfg, node, local)
        if not local:
            continue
        extra_body = json.loads(json.dumps(node.get("extra_body") if isinstance(node.get("extra_body"), Mapping) else {}))
        changed = False
        for leaf, value in sy_settings.THINKING_OFF.items():
            *parents, name = leaf.split(".")
            parent = extra_body
            for key in parents:
                parent = parent.setdefault(key, {}) if isinstance(parent, dict) else None
            if isinstance(parent, dict) and name not in parent:
                parent[name] = value
                changed = True
        if changed:
            # As JSON: Hermes stores a bare "none" as null.
            set_config_value(f"auxiliary.{task}.extra_body", json.dumps(extra_body))


def _fit_title_timeout(cfg: sy_settings.SwitchyardSettings, node: Mapping[str, Any], local: bool) -> None:
    """Lets a session title wait for a local model busy with the turn it was asked alongside.

    Hermes asks for the title as the first turn starts. A local server that answers one request
    at a time holds it until that turn's draft is done, and past the title's 30 s timeout Hermes
    asks the main model instead, which VGR can send to the cloud. A timeout the user set is left
    alone; one equal to what setup writes is taken as setup's, as a ``plugins remove`` leaves it.
    """
    from hermes_cli.config import set_config_value, unset_config_value
    written = sy_settings.plugin_settings(_load_config()).get(sy_settings.TITLE_TIMEOUT_KEY)
    current = node.get("timeout")
    deadline = _vgr_deadline(cfg)
    needed = int(min(deadline, _HERMES_API_TIMEOUT_S)) + _TITLE_SLACK_S if local and deadline else None
    if current not in (None, _HERMES_TITLE_TIMEOUT_S, written, needed):
        return
    if needed is not None and current != needed:
        set_config_value("auxiliary.title_generation.timeout", str(needed))
    elif needed is None and written is not None and current == written:
        unset_config_value("auxiliary.title_generation.timeout")
    if written != needed:
        _save_restore(sy_settings.TITLE_TIMEOUT_KEY, needed)


def _local_tier_is_local(cfg: sy_settings.SwitchyardSettings) -> bool:
    """Whether the efficient tier is served on this machine or its network."""
    if cfg.local.managed:
        return True
    try:
        endpoint = _local_endpoint(cfg)
    except Exception as error:
        logger.info("Switchyard could not resolve the local tier's address: %s", error)
        return False
    host = urlsplit(endpoint[0]).hostname if endpoint else None
    return bool(host) and sy_settings._private_host(host)


def managed_local_endpoint() -> Optional[Tuple[str, str]]:
    """``(base_url, api_key)`` of Hermes's managed llama.cpp server, or None while it is down."""
    from hermes_cli.local_runtime.endpoint import resolve_llamacpp_endpoint

    endpoint = resolve_llamacpp_endpoint()
    if not endpoint:
        return None
    return str(endpoint["base_url"]), str(endpoint.get("api_key") or "")


def get_token() -> str:
    from agent.secret_scope import (build_profile_secret_scope, current_secret_scope, current_secret_scope_home,
                                    get_secret, reset_secret_scope, set_secret_scope)
    from hermes_constants import get_hermes_home

    if current_secret_scope() is not None:
        token = (get_secret(sy_settings.TOKEN_ENV) or "").strip()
        if token:
            return token
    # No scope is bound, or the bound one was built before setup saved the token.
    home = current_secret_scope_home() or str(get_hermes_home())
    scope = set_secret_scope(build_profile_secret_scope(Path(home)), profile_home=home)
    try:
        return (get_secret(sy_settings.TOKEN_ENV) or "").strip()
    finally:
        reset_secret_scope(scope)


def ensure_token() -> str:
    """The front token Hermes presents to the sidecar, created on first use."""
    token = get_token()
    if token:
        return token
    from hermes_cli.config import save_env_value_secure

    token = secrets.token_urlsafe(32)
    save_env_value_secure(sy_settings.TOKEN_ENV, token)
    return token


def _write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)


def tier_credentials(cfg: sy_settings.SwitchyardSettings) -> Dict[str, Dict[str, str]]:
    """The credential headers the tap sends to each URL tier that has a key, in that tier's format.

    Switchyard never holds these; it calls the tap's tier routes with the tier-route key.
    """
    from agent.secret_scope import get_secret

    credentials: Dict[str, Dict[str, str]] = {}
    for name, tier in (("local", cfg.local), ("cloud", cfg.cloud)):
        if tier.source != "url" or not tier.api_key_env:
            continue
        value = (get_secret(tier.api_key_env) or "").strip()
        if not value:
            raise sy_settings.SettingsError(f"{tier.api_key_env} is not set; add it with `hermes config env`")
        credentials[name] = (
            {"x-api-key": value} if tier.format == "anthropic_messages" else {"Authorization": f"Bearer {value}"}
        )
    return credentials


def _managed_resolver():
    endpoint = managed_local_endpoint()
    if endpoint is None:
        return None
    base_url, api_key = endpoint
    return base_url, ({"Authorization": f"Bearer {api_key}"} if api_key else {})


_VERIFIER_PROBE_TIMEOUT_S = 30.0


def _local_endpoint(cfg: sy_settings.SwitchyardSettings) -> Optional[Tuple[str, Dict[str, str]]]:
    from agent.secret_scope import get_secret

    from . import providers

    tier = cfg.local
    if tier.managed:
        return _managed_resolver()
    if tier.source == "provider":
        return providers.endpoint(tier, providers._runtime(tier))
    key = (get_secret(tier.api_key_env) or "").strip() if tier.api_key_env else ""
    return tier.upstream_url, ({"Authorization": f"Bearer {key}"} if key else {})


def verifier_warning(cfg: sy_settings.SwitchyardSettings) -> Optional[str]:
    """Asks the local model one yes/no question the way VGR's scored verifier calls do.

    A warning when the reply carries no yes/no score, as a thinking model's does when its server
    ignores the request to skip thinking; VGR would then escalate every turn. None when no preset
    is judged locally from token probabilities, or the local server cannot be reached.
    """
    judged_locally = any(
        preset.algorithm in sy_settings.LOGPROB_ALGORITHMS and str(preset.route.get("judge_target") or "local") == "local"
        for preset in cfg.presets
    )
    if not judged_locally or cfg.local.format != "openai_chat":
        return None
    try:
        endpoint = _local_endpoint(cfg)
        if endpoint is None:
            return None
        base_url, headers = endpoint
        response = httpx.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, timeout=_VERIFIER_PROBE_TIMEOUT_S,
                              json={
                                  "model": cfg.local.model,
                                  "messages": [
                                      {"role": "system", "content": "Reply with exactly one word: yes or no."},
                                      {"role": "user", "content": "Is 2 + 2 equal to 4?"},
                                  ],
                                  "max_tokens": 4,
                                  "temperature": 0,
                                  "logprobs": True,
                                  "top_logprobs": 8,
                                  "reasoning_effort": "none",
                                  "chat_template_kwargs": {"enable_thinking": False},
                              })
        response.raise_for_status()
        choice = (response.json().get("choices") or [{}])[0]
    except Exception as error:
        logger.info("Switchyard could not check the local model's verifier score: %s", error)
        return None
    first = ((choice.get("logprobs") or {}).get("content") or [{}])[0]
    if stages.score(first.get("top_logprobs")) is not None:
        return None
    return (f"The local model {cfg.local.model} answered a VGR verifier question without a yes/no score, so VGR "
            "would escalate every turn. If it is a thinking model, have its server honour "
            "`chat_template_kwargs: {\"enable_thinking\": false}` or `reasoning_effort: \"none\"`, or start it with "
            "thinking off; the server must also return logprobs.")


# OpenAI's reasoning models refuse ``max_tokens``, so only a local server's probe caps its reply that way.
_ACCOUNT_PROBES = {
    "openai_chat": ("/chat/completions", lambda model, local: {
        "model": model, "messages": [{"role": "user", "content": "Reply with: ok"}], **({"max_tokens": 1} if local else {})}),
    "openai_responses": ("/responses", lambda model, local: {
        "model": model, "input": "Reply with: ok", "max_output_tokens": 16}),
    "anthropic_messages": ("/v1/messages", lambda model, local: {
        "model": model, "messages": [{"role": "user", "content": "Reply with: ok"}], "max_tokens": 1}),
}
_DETAIL_CHARS = 200


def _provider_detail(content: bytes) -> str:
    """The provider's own error message from a refused request, shortened for one line."""
    try:
        body = json.loads(content)
        error = body.get("error") if isinstance(body, dict) else None
        text = (error.get("message") if isinstance(error, dict) else error) or body.get("message") or body.get("detail")
    except (ValueError, AttributeError):
        text = content.decode("utf-8", errors="replace")
    text = " ".join(str(text or "").split())
    return text[:_DETAIL_CHARS] + ("..." if len(text) > _DETAIL_CHARS else "")


def _account_endpoint(tier: sy_settings.Tier) -> Tuple[str, Dict[str, str]]:
    from agent.secret_scope import get_secret

    from . import providers

    if tier.source == "provider":
        return providers.endpoint(tier, providers._runtime(tier))
    key = (get_secret(tier.api_key_env) or "").strip() if tier.api_key_env else ""
    if tier.format == "anthropic_messages":
        return tier.upstream_url, {"x-api-key": key, "anthropic-version": "2023-06-01"} if key else {}
    return tier.upstream_url, {"Authorization": f"Bearer {key}"} if key else {}


def _account_warning(name: str, tier: sy_settings.Tier) -> Optional[str]:
    from . import failure

    path, body = _ACCOUNT_PROBES[tier.format]
    role = sy_settings.ROLES[name]
    try:
        base_url, headers = _account_endpoint(tier)
        response = httpx.post(f"{base_url.rstrip('/')}{path}", headers=headers, json=body(tier.model, name == "local"),
                              timeout=_VERIFIER_PROBE_TIMEOUT_S)
    except httpx.ConnectError as error:
        later = " If you start it later, ignore this." if name == "local" else ""
        return (f"The {role} model's server ({tier.endpoint_label}) could not be reached ({type(error).__name__}), "
                f"so every call Switchyard makes to it fails until it is up.{later}")
    except Exception as error:
        logger.info("Switchyard could not check the %s model's account: %s", role, error)
        return None
    other = next(r for t, r in sy_settings.ROLES.items() if t != name)
    no_credit = failure.out_of_credits(response.status_code, response.content)
    if no_credit:
        return (f"The {role} model {tier.model} is out of credits, so Switchyard answers with the {other} model "
                f"whenever it would use it. Add credits or pick another {role} model. Its provider said: {no_credit}")
    if response.status_code == 401:
        return (f"The {role} model {tier.model} refused its API key (HTTP 401), so every call Switchyard makes to it "
                "fails. Check the key with `hermes config env`.")
    # A stream-only provider refuses this unstreamed probe whatever the account's state.
    stream_only = tier.source == "provider" and tier.provider.lower() in sy_settings.STREAM_ONLY_PROVIDERS
    # A timeout, conflict, rate limit or server error passes; it says nothing about the URL, model or key.
    if 400 <= response.status_code < 500 and response.status_code not in (408, 409, 429) and not stream_only:
        detail = _provider_detail(response.content)
        return (f"The {role} model {tier.model} refused a test request (HTTP {response.status_code}"
                f"{': ' + detail if detail else ''}), so calls to it would likely fail the same way. Check its "
                "URL, model name and API key.")
    return None


def account_warnings(cfg: sy_settings.SwitchyardSettings) -> list[str]:
    """Sends each tier one tiny request; a warning per tier that cannot be reached, refuses its key, has
    no credit, or refuses the request. A tier that answers, or is still working when the probe times out,
    gets none."""
    warnings = []
    for name, tier in (("local", cfg.local), ("cloud", cfg.cloud)):
        if not tier.managed and tier.format in _ACCOUNT_PROBES:
            warning = _account_warning(name, tier)
            if warning:
                warnings.append(warning)
    return warnings


def tier_endpoints(cfg: sy_settings.SwitchyardSettings) -> Dict[str, Any]:
    """The tap's resolver for each tier whose endpoint is not fixed in the config."""
    from . import providers

    endpoints: Dict[str, Any] = {}
    for name, tier in (("local", cfg.local), ("cloud", cfg.cloud)):
        if tier.managed:
            endpoints[name] = _managed_resolver
        elif tier.source == "provider":
            endpoints[name] = providers.resolver(tier)
    return endpoints


def stream_only_tiers(cfg: sy_settings.SwitchyardSettings) -> Tuple[str, ...]:
    """The tiers on a provider that only answers streaming requests."""
    return tuple(
        name for name, tier in (("local", cfg.local), ("cloud", cfg.cloud))
        if tier.source == "provider" and tier.provider.lower() in sy_settings.STREAM_ONLY_PROVIDERS
    )


def _preset_resolver(cfg: sy_settings.SwitchyardSettings):
    def resolve(model: Any) -> Optional[Tuple[str, str]]:
        preset = cfg.preset_for_model(model)
        return (preset.name, preset.model) if preset else None

    return resolve


def serve(cfg: Optional[sy_settings.SwitchyardSettings] = None) -> int:
    """Run the sidecar in the foreground until it is asked to stop."""
    from .runtime import describe, server_class

    cfg = cfg or load_settings()
    token = get_token()
    if not token:
        raise sy_settings.SettingsError(f"{sy_settings.TOKEN_ENV} is not set; run `hermes switchyard setup`")
    refusal = consent_refusal(cfg)
    if refusal:
        return _serve_refusal(cfg, token, refusal)
    credentials = tier_credentials(cfg)
    tier_route_key = secrets.token_urlsafe(32)
    os.environ[sy_settings.TIER_ROUTE_KEY_ENV] = tier_route_key
    directory = state_dir()
    config_path = directory / "switchyard.toml"
    Server = server_class()
    runtime = describe() or {}
    print(f"Switchyard runtime: {runtime.get('project')} {runtime.get('version')} from {runtime.get('location')}",
          file=sys.stderr, flush=True)
    _write_private(config_path, sy_settings.render_switchyard_toml(cfg, omit=engine_lacks(Server, cfg)))
    try:
        server = Server(str(config_path), port=0)
    except Exception as error:
        if _NO_VGR.search(str(error)):
            raise sy_settings.SettingsError(f"Switchyard cannot start: {_no_vgr_message()}") from None
        raise
    stop = threading.Event()
    pins = tuple(sy_settings.pin_preset(tier) for tier in sy_settings.PIN_TARGETS)
    monitor = Monitor(
        local_model=cfg.local.model,
        cloud_model=cfg.cloud.model,
        presets={
            p.name: {"algorithm": p.algorithm, "mode": p.vgr_mode, "route": p.route, "pinned": p in pins}
            for p in (*cfg.presets, *pins)
        },
        default_preset=cfg.default_preset,
        events_path=directory / "events.jsonl",
        pin=cfg.pin,
        runtime=runtime,
    )
    # A detached sidecar's stderr, where Switchyard logs each VGR decision, is sidecar.log.
    decisions = LogFollower(directory / "sidecar.log", monitor.record_decision)
    monitor.attach_decisions(decisions.poll)
    try:
        tap = TapServer(
            cfg.port,
            token=token,
            tier_route_key=tier_route_key,
            tier_credentials=credentials,
            local_base_url=cfg.local.base_url,
            cloud_base_url=cfg.cloud.upstream_url,
            monitor=monitor,
            presets=_preset_resolver(cfg),
            on_shutdown=stop.set,
            endpoints=tier_endpoints(cfg),
            stream_only=stream_only_tiers(cfg),
        )
    except Exception:
        server.close()
        raise
    tap.switchyard_url = server.base_url
    decisions.start()
    thread = threading.Thread(target=tap.serve_forever, name="switchyard-tap", daemon=True)
    thread.start()
    _write_private(directory / "sidecar.json", json.dumps({
        "pid": os.getpid(), "port": cfg.port, "switchyard": server.base_url, "started": time.time(),
    }))
    names = ", ".join(p.name for p in cfg.presets)
    logger.info("Switchyard sidecar on %s (switchyard %s, presets %s)", cfg.tap_url, server.base_url, names)
    print(f"Switchyard sidecar listening on {cfg.front_base_url} (presets {names}; default {cfg.default_preset})", flush=True)

    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(signum, lambda *_: stop.set())
        except (ValueError, OSError):
            pass
    try:
        while not stop.wait(0.5):
            pass
    finally:
        decisions.stop()
        tap.shutdown()
        tap.server_close()
        server.close()
        (directory / "sidecar.json").unlink(missing_ok=True)
    return 0


def _serve_refusal(cfg: sy_settings.SwitchyardSettings, token: str, refusal: str) -> int:
    """Answer every model request with ``refusal``, calling no tier, until the user agrees or a stop.

    Hermes then ends the turn with the reason at once, instead of retrying a port nothing answers.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    stop = threading.Event()
    expected = f"Bearer {token}".encode()
    problem = consent_problem(cfg) or refusal

    class Refusal(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _answer(self, status: int, body: Dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _handle(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                self.rfile.read(min(length, 64 * 1024 * 1024))
            if not secrets.compare_digest(self.headers.get("Authorization", "").encode(), expected):
                self._answer(401, {"error": {"message": "unauthorized"}})
            elif self.path == "/monitor/health":
                self._answer(503, {"refused": problem, "pid": os.getpid()})
            elif self.path == "/monitor/shutdown" and self.command == "POST":
                self._answer(200, {"ok": True})
                stop.set()
            elif self.path.startswith("/front/"):
                self._answer(403, {"error": {"message": refusal, "type": "permission_error",
                                             "code": sy_settings.CONSENT_ERROR_CODE}})
            else:
                self._answer(404, {"error": {"message": "not found"}})

        do_GET = do_POST = _handle

    server = ThreadingHTTPServer(("127.0.0.1", cfg.port), Refusal)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, name="switchyard-refusal", daemon=True)
    thread.start()
    directory = state_dir()
    _write_private(directory / "sidecar.json", json.dumps({"pid": os.getpid(), "port": cfg.port, "started": time.time()}))
    print(f"Switchyard sidecar refusing every request on {cfg.front_base_url}: {problem}", flush=True)
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(signum, lambda *_: stop.set())
        except (ValueError, OSError):
            pass
    try:
        # Exits once the user agrees, so the next request starts a sidecar that routes.
        while not stop.wait(1.0):
            try:
                if not consent_refusal(load_settings()):
                    break
            except sy_settings.SettingsError:
                pass
    finally:
        # Before the port frees, so it cannot remove the record of the sidecar that replaces it.
        (directory / "sidecar.json").unlink(missing_ok=True)
        server.shutdown()
        server.server_close()
    return 0


def _health_response(cfg: sy_settings.SwitchyardSettings, timeout: float) -> Optional[httpx.Response]:
    try:
        return httpx.get(
            f"{cfg.tap_url}/monitor/health",
            headers={"Authorization": f"Bearer {get_token()}"},
            timeout=timeout,
        )
    except httpx.HTTPError:
        return None


def health(cfg: sy_settings.SwitchyardSettings, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
    response = _health_response(cfg, timeout)
    return response.json() if response is not None and response.status_code == 200 else None


def refusing(cfg: sy_settings.SwitchyardSettings, timeout: float = 1.0) -> Optional[str]:
    """The reason a sidecar answering on ``cfg``'s port refuses every request; None when none does."""
    response = _health_response(cfg, timeout)
    if response is None or response.status_code != 503:
        return None
    try:
        return str(response.json().get("refused") or "") or None
    except (ValueError, AttributeError):
        return None


def fetch_state(cfg: sy_settings.SwitchyardSettings, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
    try:
        response = httpx.get(
            f"{cfg.tap_url}/monitor/state",
            headers={"Authorization": f"Bearer {get_token()}"},
            timeout=timeout,
        )
    except httpx.HTTPError:
        return None
    return response.json() if response.status_code == 200 else None


def _hermes_argv(args: list[str]) -> list[str]:
    """``hermes <args>`` with the running Hermes entry point, frozen or not."""
    if getattr(sys, "frozen", False):
        return [sys.executable, *args]
    from hermes_cli.relaunch import build_relaunch_argv

    return build_relaunch_argv(args, preserve_inherited=False)


def _serve_argv() -> list[str]:
    """``hermes switchyard serve`` on this Hermes, which loads the profile's ``.env`` for the sidecar.

    This interpreter runs it when it is a real Python, so a different ``hermes`` on PATH cannot.
    """
    if getattr(sys, "frozen", False) or not Path(sys.executable).stem.lower().startswith("python"):
        return _hermes_argv(["switchyard", "serve"])
    return [sys.executable, "-X", "utf8", "-m", "hermes_cli.main", "switchyard", "serve"]


def _serve_env() -> Dict[str, str]:
    return {**os.environ, "PYTHONPATH": os.pathsep.join(path for path in sys.path if path)}


def spawn_detached(argv: list[str], log_path: Path, env: Optional[Dict[str, str]] = None) -> subprocess.Popen:
    """Start ``argv`` in the background with no console window, its output in ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab")
    kwargs: Dict[str, Any] = {"stdin": subprocess.DEVNULL, "stdout": log, "stderr": log, "close_fds": True,
                              "env": env}
    try:
        if sys.platform == "win32":
            # Hermes's detach flags. Never add DETACHED_PROCESS: it voids CREATE_NO_WINDOW, and the
            # child's console-program descendants then each open a visible terminal. Breaking away
            # from Desktop's job object keeps the sidecar alive past it, where the job allows that.
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            try:
                return subprocess.Popen(argv, creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB, **kwargs)
            except OSError:
                return subprocess.Popen(argv, creationflags=flags, **kwargs)
        return subprocess.Popen(argv, start_new_session=True, **kwargs)
    finally:
        log.close()


def _recorded_port() -> Optional[int]:
    """The port of the sidecar ``sidecar.json`` records, which may predate the settings' port."""
    try:
        return int(json.loads((state_dir() / "sidecar.json").read_text(encoding="utf-8"))["port"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def ensure_running(cfg: Optional[sy_settings.SwitchyardSettings] = None) -> bool:
    """Start the sidecar detached unless one is already healthy; True once it answers.

    Without the user's agreement to a tier's subscription sign-in, the sidecar that starts refuses
    every request with the reason, and this returns False once it does.
    """
    cfg = cfg or load_settings()
    if health(cfg):
        return True
    refused = consent_refusal(cfg)
    if refusing(cfg):
        if refused:
            return False
        # The user agreed since it started.
        stop(cfg)
    recorded = _recorded_port()
    if recorded is not None and recorded != cfg.port:
        # Its record is about to be overwritten, after which nothing could find it to stop it.
        stop(dataclasses.replace(cfg, port=recorded))
    ensure_token()
    spawn_detached(_serve_argv(), state_dir() / "sidecar.log", env=_serve_env())
    deadline = time.monotonic() + _STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        status = health(cfg)
        if status and status.get("switchyard"):
            return True
        if refused and refusing(cfg, timeout=0.5):
            return False
        time.sleep(0.25)
    return False


def restart(cfg: sy_settings.SwitchyardSettings) -> bool:
    """Bring up a sidecar running ``cfg``, replacing one started with older settings."""
    old = health(cfg)
    if old:
        stop(cfg)
        # A sidecar still finishing a request can miss one probe and then answer again, which
        # would pass for the replacement and keep serving the old settings.
        deadline = time.monotonic() + _RESTART_WAIT_S
        while _still_running(cfg, old) and time.monotonic() < deadline:
            time.sleep(0.2)
        if _still_running(cfg, old):
            _kill_recorded(old.get("pid"))
            time.sleep(1.0)
    return ensure_running(cfg) and not (old and _still_running(cfg, old))


def _still_running(cfg: sy_settings.SwitchyardSettings, old: Mapping[str, Any]) -> bool:
    """Whether the sidecar that answered ``old`` still answers; one too old to report its pid counts."""
    now = health(cfg)
    return now is not None and now.get("pid") == old.get("pid")


def dashboard_url() -> Optional[str]:
    """The Switchyard tab of this host's running web dashboard; ``None`` when none answers.

    A headless ``hermes serve`` owner has no dashboard UI behind it, so it does not count.
    """
    try:
        from gateway import host_rendezvous as hr
        from hermes_constants import get_hermes_home, profile_name_for_home

        record = hr.read_record(hr.ROLE_SERVE)
        if record is None or not record.port or not hr.record_token_is_consistent(record):
            return None
        identity = hr.probe_owner(record)
        if not identity or not identity.get("servesSpa"):
            return None
        return f"http://{hr.dial_host(record)}:{record.port}/switchyard?profile={profile_name_for_home(get_hermes_home()) or 'custom'}"
    except Exception:
        return None


# What the desktop app gives the backend it owns. A dashboard started from that
# backend must not inherit them: the session token makes it a second
# desktop-owned backend, and the parent PID ties its lifetime to the app.
_DESKTOP_BACKEND_ENV = (
    "HERMES_DESKTOP",
    "HERMES_DASHBOARD_SESSION_TOKEN",
    "HERMES_DESKTOP_READY_FILE",
    "HERMES_SERVE_HEADLESS",
    "HERMES_PARENT_PID",
    "HERMES_PARENT_START_MARKER",
    "HERMES_PARENT_NONCE",
    "HERMES_SPAWN",
)


def _dashboard_env() -> Dict[str, str]:
    """This process's environment, fit for a standalone ``hermes dashboard``."""
    env = {key: value for key, value in os.environ.items() if key not in _DESKTOP_BACKEND_ENV}
    inherited = env.get("HERMES_WEB_DIST")
    # The packaged desktop names a web dist under its non-bundled install root
    # even when the bundled app carries its own, and the dashboard refuses a
    # HERMES_WEB_DIST without an index.html.
    if inherited and not (Path(inherited).expanduser() / "index.html").is_file():
        import hermes_cli

        own = Path(hermes_cli.__file__).resolve().parent / "web_dist"
        if (own / "index.html").is_file():
            env["HERMES_WEB_DIST"] = str(own)
        else:
            del env["HERMES_WEB_DIST"]
    return env


def ensure_dashboard() -> Optional[str]:
    """Start Hermes's web dashboard in the background unless one is up; its Switchyard URL."""
    url = dashboard_url()
    if url:
        return url
    process = spawn_detached(
        _hermes_argv(["dashboard", "--no-open"]), state_dir() / "dashboard.log", env=_dashboard_env()
    )
    deadline = time.monotonic() + 3 * _STARTUP_TIMEOUT_S
    while time.monotonic() < deadline:
        time.sleep(1.0)
        url = dashboard_url()
        if url:
            return url
        # A dashboard that attaches to a running one exits 0 once it is reachable.
        if process.poll() not in (None, 0):
            return None
    return None


def set_pin(pin: Any) -> Dict[str, Any]:
    """Pin every request to one tier, or ``""`` to route normally; a running sidecar applies it at once."""
    value = sy_settings.parse_pin(pin)
    cfg = load_settings()
    sy_settings.save_setting(("pin",), value)
    live = False
    if health(cfg):
        try:
            response = httpx.post(
                f"{cfg.tap_url}/monitor/pin",
                headers={"Authorization": f"Bearer {get_token()}"},
                json={"pin": value},
                timeout=2.0,
            )
            live = response.status_code == 200
        except httpx.HTTPError:
            live = False
    return {"pin": value, "live": live}


def stop(cfg: Optional[sy_settings.SwitchyardSettings] = None) -> bool:
    cfg = cfg or load_settings()
    try:
        httpx.post(
            f"{cfg.tap_url}/monitor/shutdown",
            headers={"Authorization": f"Bearer {get_token()}"},
            timeout=2.0,
        )
    except httpx.HTTPError:
        pass
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not health(cfg, timeout=0.5) and not refusing(cfg, timeout=0.5):
            return True
        time.sleep(0.2)
    return _kill_recorded()


def _kill_recorded(pid: Optional[int] = None) -> bool:
    record = state_dir() / "sidecar.json"
    try:
        pid = pid or int(json.loads(record.read_text(encoding="utf-8"))["pid"])
        os.kill(pid, signal.SIGTERM)
    except (OSError, ValueError, KeyError):
        return False
    return True
