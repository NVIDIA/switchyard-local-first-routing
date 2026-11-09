# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Switchyard plugin's settings (``plugins.entries.nvidia-switchyard.settings`` in config.yaml), validated.

Two tiers are shared by every preset: ``local``, the efficient model tried
first, and ``cloud``, the capable model it escalates to. The recommended pair
is an on-device model and a cloud model, but either tier can be any endpoint:
a URL, a provider the user is signed in to in Hermes, or Hermes's own
llama.cpp runtime. A preset is one Switchyard route over those tiers, picked
by algorithm, and Hermes selects it as the model ``switchyard/<preset>``.
"""

from __future__ import annotations

import ipaddress
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple
from urllib.parse import urlsplit

PLUGIN_ID = "nvidia-switchyard"
# The PluginContext Hermes registered the plugin with; None in a process that has not loaded it.
_context: Any = None
# The plugin's id before it took the vendor prefix; settings saved under it move to PLUGIN_ID on load.
LEGACY_PLUGIN_ID = "switchyard"
PRODUCT_NAME = "Switchyard Local-first Routing"
# How users see the plugin while it is in beta.
PRODUCT_LABEL = f"{PRODUCT_NAME} (Beta)"
PROVIDER = "switchyard"
MODEL_PREFIX = "switchyard/"
# Resolved by the sidecar to whichever preset is the default when the request arrives.
DEFAULT_MODEL = f"{MODEL_PREFIX}default"
TOKEN_ENV = "SWITCHYARD_API_KEY"
# Set only inside the sidecar process: the key Switchyard sends on its calls to the tap's tier routes.
TIER_ROUTE_KEY_ENV = "SWITCHYARD_TIER_ROUTE_KEY"
DEFAULT_PORT = 4110
TIERS = ("local", "cloud")
# What each tier does in a route, for display; the config keys stay ``local`` and ``cloud``.
ROLES = {"local": "efficient", "cloud": "capable"}
FORMATS = ("openai_chat", "anthropic_messages", "openai_responses")
# url: a fixed endpoint and key; provider: a Hermes provider resolved per request;
# managed: Hermes's own llama.cpp runtime.
SOURCES = ("url", "provider", "managed")
# The wire format of each Hermes provider ``api_mode`` the tap can carry.
API_MODE_FORMATS = {
    "chat_completions": "openai_chat",
    "codex_responses": "openai_responses",
    "anthropic_messages": "anthropic_messages",
}
# Providers whose wire format is fixed, used until setup records the resolved one.
PROVIDER_FORMATS = {"openai-codex": "openai_responses", "anthropic": "anthropic_messages"}
PROVIDER_EXTRA_BODY: Dict[str, Dict[str, Any]] = {"openai-codex": {"store": False}}
# Providers that refuse a request without ``stream: true`` (Codex answers 400); the tap streams to
# them and gives a non-streaming caller the completed response.
STREAM_ONLY_PROVIDERS = ("openai-codex",)
# Routes whose efficient tier is scored from first-token logprobs, which only Chat Completions returns.
LOGPROB_ALGORITHMS = ("vgr",)
PIN_TARGETS = ("local", "cloud")
PIN_PRESET_PREFIX = "pin-"
# The model picker's single-tier choices, named by role; they serve through the pin routes.
PIN_MODELS = {f"{MODEL_PREFIX}only-{ROLES[tier]}": tier for tier in PIN_TARGETS}
LOCAL_PIN_MODEL = f"{MODEL_PREFIX}only-{ROLES['local']}"
# Hermes auxiliary tasks setup sends straight to LOCAL_PIN_MODEL: session titles and smart approval.
PINNED_TASKS = ("title_generation", "approval")
# ``extra_body`` leaves that ask a local efficient model not to think, as the verifier probe does.
THINKING_OFF = {"reasoning_effort": "none", "chat_template_kwargs.enable_thinking": False}
# Plugin setting holding the model and title settings setup replaced; read by the provider profile.
PREVIOUS_MODEL_KEY = "previous"
# Plugin setting holding the stream timeout setup wrote into Hermes's Switchyard provider config.
STREAM_TIMEOUT_KEY = "stream_timeout"
# Plugin setting holding the session-title timeout setup wrote into Hermes's config.
TITLE_TIMEOUT_KEY = "title_timeout"
# Plugin setting holding the revision of the Hermes settings setup last wrote. A plugin whose
# setup writes more (stream and title timeouts, the approval pin, titles without thinking) bumps it
# and writes them at startup.
HERMES_SETTINGS_KEY = "hermes_settings"
HERMES_SETTINGS_REVISION = 4
# Plugin setting listing the Hermes providers whose subscription sign-in the user agreed
# Switchyard may call on its own as it routes.
SUBSCRIPTION_CONSENT_KEY = "subscription_consent"
# The error code a sidecar without that agreement answers each request with.
CONSENT_ERROR_CODE = "switchyard_subscription_consent_required"
MAX_RETRIES = 2
DEFAULT_CLOUD_URL = "https://api.anthropic.com"
DEFAULT_CLOUD_MODEL = "claude-opus-5-5"
# Sent with every request to a Claude model over Anthropic Messages, as in the TB 2.1 runs,
# unless the tier sets its own ``extra_body``.
CLAUDE_EXTRA_BODY: Dict[str, Any] = {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}
VGR_MODES = ("off", "shadow", "evaluate", "active")
# "managed": Hermes's own llama.cpp runtime serves the local tier; its address and key
# are resolved per request instead of configured.
RUNTIMES = ("", "managed")
RECOMMENDED = "vgr"
# The algorithms the plugin offers; the rest are not QA-aligned yet, so nothing shows or routes
# them. Presets saved with another algorithm stay in config.yaml but do not load.
AVAILABLE_ALGORITHMS = (RECOMMENDED,)
# VGR's per-request decision deadline when local processing is unbounded, so a runaway local
# model cannot hold one step for hours; the per-message budget stays unset.
UNBOUNDED_DEADLINE_SECONDS = 1800
UNBOUNDED_VGR_LIMITS: Dict[str, Any] = {"deadline_seconds": UNBOUNDED_DEADLINE_SECONDS, "local_turn_budget_seconds": None}
# The VGR route keys under the names upstream Switchyard gives them.
UPSTREAM_VGR_KEYS = {
    "local_target": "efficient_target",
    "cloud_target": "capable_target",
    "cloud_judge_target": "capable_judge_target",
    "local_turn_budget_seconds": "efficient_turn_budget_seconds",
}
# Route keys the plugin fills in only where the installed Switchyard reads them; one it does not
# know would stop every route from loading. A value the user sets is passed on either way.
OPTIONAL_ROUTE_DEFAULTS = ("compact_handoff",)
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PROVIDER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,95}$")
_PRIVATE_SUFFIXES = (".local", ".lan", ".home.arpa", ".internal")
_PRESET_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
_OPTION_KEY = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_MAX_OPTION_DEPTH = 3


@dataclass(frozen=True)
class Algorithm:
    label: str
    summary: str
    # Route keys Hermes fills in; ``local`` and ``cloud`` are the target names.
    defaults: Mapping[str, Any]
    # Guided algorithms work from these defaults alone; the rest need options the user supplies.
    guided: bool


ALGORITHMS: Dict[str, Algorithm] = {
    "vgr": Algorithm(
        "Verification-Gated Routing",
        "Attempt each turn locally, verify it, and escalate to the cloud only when the local answer is not licensed.",
        {
            "local_target": "local",
            "cloud_target": "cloud",
            "judge_target": "local",
            "cloud_judge_target": "cloud",
            # The judge-off configuration of the improved TB 2.1 runs (2026-09-29).
            "mode": "active",
            "active_approval": "prospective-validation-and-canary-approved",
            "deadline_seconds": 300,
            "task_typing": True,
            "agentic_handoff": True,
            # The cloud model gets a condensed history of the local model's tool work instead of all of it.
            "compact_handoff": True,
            "local_turn_budget_seconds": 600,
            "confirmed_recovery_min_clean_tail": 1,
        },
        True,
    ),
    "llm_classifier": Algorithm(
        "LLM classifier",
        "The local model judges whether it can handle each request; harder ones go to the cloud.",
        {
            "mode": "capability",
            "classifier_target": "local",
            "strong_target": "cloud",
            "weak_target": "local",
            "base_threshold": 0.5,
            "threshold_step": 0.1,
        },
        True,
    ),
    "stage_router": Algorithm(
        "Stage router",
        "Route by the stage of an agentic task from tool-call signals, without an extra model call.",
        {"capable_target": "cloud", "efficient_target": "local", "picker": "efficient_first", "confidence_threshold": 0.5},
        True,
    ),
    "random": Algorithm(
        "Random split",
        "Send each request to a tier at random by weight; useful for A/B comparisons.",
        {"targets": ["local", "cloud"], "weights": [1, 1]},
        True,
    ),
    "passthrough": Algorithm(
        "Single tier",
        "Send every request to one tier (local by default).",
        {"target": "local"},
        True,
    ),
    "plan_execute": Algorithm(
        "Plan and execute",
        "The cloud tier plans and the local tier executes.",
        {"capable_target": "cloud", "efficient_target": "local"},
        False,
    ),
    "auto": Algorithm("Auto", "Switchyard's automatic capable/efficient routing.", {"capable_target": "cloud", "efficient_target": "local"}, False),
    "advisor": Algorithm(
        "Advisor",
        "The local tier executes with advice from the cloud tier.",
        {"executor_target": "local", "advisor_target": "cloud"},
        False,
    ),
    "composite": Algorithm("Composite", "Combine a classifier with other routing signals (needs a classifier block).", {}, False),
    "prefill_router": Algorithm(
        "Prefill router",
        "Route on a trained prefill probe (needs a checkpoint).",
        {"targets": ["local", "cloud"]},
        False,
    ),
}

DEFAULT_PRESETS: Dict[str, Dict[str, Any]] = {RECOMMENDED: {"algorithm": RECOMMENDED}}


def tier_key_env(name: str) -> str:
    """The env var setup stores a URL tier's API key in."""
    return f"SWITCHYARD_{name.upper()}_API_KEY"


DEFAULTS: dict = {
    "port": DEFAULT_PORT,
    "default_preset": RECOMMENDED,
    "local": {"base_url": "http://127.0.0.1:8080/v1", "model": "qwen3.6-35b-a3b", "api_key_env": "", "runtime": ""},
    "cloud": {
        "base_url": DEFAULT_CLOUD_URL,
        "model": DEFAULT_CLOUD_MODEL,
        "api_key_env": tier_key_env("cloud"),
    },
}


class SettingsError(ValueError):
    """The ``switchyard`` settings cannot produce a valid deployment."""


class ConsentRequired(SettingsError):
    """A tier uses a subscription sign-in the user has not agreed Switchyard may call."""


def upstream_root(fmt: str, base_url: str) -> str:
    """Where the tap forwards a tier's calls for ``base_url``.

    Switchyard's Anthropic client appends ``/v1/messages`` to the tap's bare
    tier prefix, so an Anthropic endpoint's ``/v1`` root is dropped here.
    """
    base_url = base_url.rstrip("/")
    if fmt == "anthropic_messages" and base_url.endswith("/v1"):
        return base_url[: -len("/v1")]
    return base_url


@dataclass(frozen=True)
class Tier:
    base_url: str
    model: str
    api_key_env: str
    format: str = "openai_chat"
    source: str = "url"
    # The Hermes provider id of a ``provider`` tier.
    provider: str = ""
    extra_body: Mapping[str, Any] = field(default_factory=dict)

    @property
    def managed(self) -> bool:
        return self.source == "managed"

    @property
    def runtime(self) -> str:
        """The pre-``source`` spelling of a managed tier, still accepted in config.yaml."""
        return "managed" if self.managed else ""

    @property
    def upstream_url(self) -> str:
        return upstream_root(self.format, self.base_url)

    @property
    def identity(self) -> Tuple[str, str, str]:
        """What makes two tiers the same model at the same place."""
        where = self.provider if self.source == "provider" else "" if self.managed else self.base_url
        return self.source, where, self.model

    @property
    def endpoint_label(self) -> str:
        if self.source == "provider":
            return f"Hermes provider {self.provider}"
        if self.managed:
            return "Hermes llama.cpp"
        return self.base_url


@dataclass(frozen=True)
class Preset:
    name: str
    algorithm: str
    # The user's route keys; ``route`` merges them over the algorithm's defaults.
    options: Mapping[str, Any]
    # Keeps VGR's default decision deadline and per-turn local budget. Unbounded (in effect:
    # see UNBOUNDED_DEADLINE_SECONDS) otherwise; a limit set in the options applies either way.
    bounded_local: bool = False

    @property
    def model(self) -> str:
        return f"{MODEL_PREFIX}{self.name}"

    @property
    def route(self) -> Dict[str, Any]:
        unbounded = UNBOUNDED_VGR_LIMITS if self.algorithm == "vgr" and not self.bounded_local else {}
        merged = {**ALGORITHMS[self.algorithm].defaults, **unbounded, **self.options}
        # Switchyard refuses compact_handoff without agentic_handoff, so turning handoff off drops the default.
        if not merged.get("agentic_handoff") and "compact_handoff" not in self.options:
            merged.pop("compact_handoff", None)
        return {key: value for key, value in merged.items() if value is not None}

    @property
    def vgr_mode(self) -> Optional[str]:
        return str(self.route.get("mode", "")) if self.algorithm == "vgr" else None


@dataclass(frozen=True)
class SwitchyardSettings:
    port: int
    default_preset: str
    local: Tier
    cloud: Tier
    presets: Tuple[Preset, ...]
    # "" routes with the selected preset; "local" or "cloud" sends every request to that tier.
    pin: str = ""
    bounded_local: bool = False
    subscription_consent: Tuple[str, ...] = ()

    @property
    def front_base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/front/v1"

    @property
    def tap_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def default(self) -> Preset:
        return self.preset(self.default_preset)

    def preset(self, name: str) -> Preset:
        for preset in self.presets:
            if preset.name == name:
                return preset
        raise SettingsError(f"no Switchyard preset named {name!r}")

    def preset_for_model(self, model: Any) -> Optional[Preset]:
        """The preset a request's ``model`` selects; ``switchyard/default`` and an empty model pick the default."""
        name = str(model or "")
        if not name or name == DEFAULT_MODEL:
            return self.default
        if name in PIN_MODELS:
            return pin_preset(PIN_MODELS[name])
        for preset in self.presets:
            if name in (preset.model, preset.name):
                return preset
        return None


def _private_host(host: str) -> bool:
    """Loopback, a private-network address, or a name only a local network resolves."""
    host = host.lower().strip("[]")
    if host == "localhost" or host.endswith(_PRIVATE_SUFFIXES) or "." not in host and ":" not in host:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_private or address.is_link_local


def _base_url(name: str, base_url: str) -> str:
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise SettingsError(f"switchyard.{name}.base_url must be an http(s) URL, got {base_url!r}")
    if parts.scheme == "http" and not _private_host(parts.hostname):
        raise SettingsError(f"switchyard.{name}.base_url must use https unless it is on this machine or a private network")
    return base_url.rstrip("/")


def _source(name: str, merged: Mapping[str, Any]) -> str:
    runtime = str(merged.get("runtime") or "").strip().lower()
    if runtime not in RUNTIMES:
        raise SettingsError(f"switchyard.{name}.runtime must be empty or 'managed'")
    source = str(merged.get("source") or "").strip().lower() or ("managed" if runtime == "managed" else "url")
    if source not in SOURCES:
        raise SettingsError(f"switchyard.{name}.source must be one of {', '.join(SOURCES)}")
    return source


def _tier(name: str, raw: Any) -> Tier:
    merged = {**DEFAULTS[name], **(raw if isinstance(raw, Mapping) else {})}
    source = _source(name, merged)
    model = str(merged.get("model") or "").strip()
    if not model:
        raise SettingsError(f"switchyard.{name}.model is required")
    provider = ""
    key_env = ""
    base_url = str(merged.get("base_url") or "").strip().rstrip("/")
    if source == "provider":
        provider = str(merged.get("provider") or "").strip()
        if not _PROVIDER_ID.match(provider):
            raise SettingsError(f"switchyard.{name}.provider must be a Hermes provider id, got {provider!r}")
        # The endpoint is resolved per request; this only labels it.
        base_url = ""
        default_format = PROVIDER_FORMATS.get(provider, "openai_chat")
    else:
        if source == "url":
            base_url = _base_url(name, base_url)
            key_env = str(merged.get("api_key_env") or "").strip()
            if key_env and not _ENV_NAME.match(key_env):
                raise SettingsError(f"switchyard.{name}.api_key_env must be an environment variable name")
        # Claude is reached over Anthropic Messages, so the tier's thinking settings reach it.
        default_format = "anthropic_messages" if source == "url" and "claude" in model.lower() else "openai_chat"
    fmt = str(merged.get("format") or default_format)
    if fmt not in FORMATS or (source == "managed" and fmt != "openai_chat"):
        allowed = "openai_chat" if source == "managed" else ", ".join(FORMATS)
        raise SettingsError(f"switchyard.{name}.format must be one of {allowed}")
    extra_body = merged.get("extra_body")
    if extra_body is None:
        claude = fmt == "anthropic_messages" and "claude" in model.lower()
        extra_body = CLAUDE_EXTRA_BODY if claude else PROVIDER_EXTRA_BODY.get(provider, {})
    if not isinstance(extra_body, Mapping):
        raise SettingsError(f"switchyard.{name}.extra_body must be a mapping of request fields")
    return Tier(
        base_url=base_url,
        model=model,
        api_key_env=key_env,
        format=fmt,
        source=source,
        provider=provider,
        extra_body=_options(f"switchyard.{name}.extra_body", extra_body, 1),
    )


def _option_value(path: str, value: Any, depth: int) -> Any:
    if isinstance(value, bool) or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SettingsError(f"{path} must be a finite number")
        return value
    if isinstance(value, (list, tuple)):
        items = [_option_value(f"{path}[]", item, depth) for item in value]
        if any(isinstance(item, (dict, list)) for item in items):
            raise SettingsError(f"{path} may only list plain values")
        return items
    if isinstance(value, Mapping):
        if depth >= _MAX_OPTION_DEPTH:
            raise SettingsError(f"{path} is nested too deeply")
        return _options(path, value, depth + 1)
    raise SettingsError(f"{path} has an unsupported value type {type(value).__name__}")


def _options(path: str, raw: Mapping[str, Any], depth: int = 0) -> Dict[str, Any]:
    options: Dict[str, Any] = {}
    for key, value in raw.items():
        key = str(key)
        if not _OPTION_KEY.match(key):
            raise SettingsError(f"{path}.{key} is not a valid Switchyard route key")
        if depth == 0 and key in ("id", "type"):
            raise SettingsError(f"{path}.{key} is set by the plugin")
        # ``None`` drops a key the algorithm would otherwise fill in.
        options[key] = None if value is None and depth == 0 else _option_value(f"{path}.{key}", value, depth)
    return options


def pin_preset(tier: str) -> Preset:
    """The built-in route that sends every request to ``tier``."""
    return Preset(name=f"{PIN_PRESET_PREFIX}{tier}", algorithm="passthrough", options={"target": tier})


def parse_pin(value: Any) -> str:
    pin = str(value or "").strip().lower()
    if pin in ("", "off", "none", "smart"):
        return ""
    # The built-in name, never the caller's string: the tap's pin endpoint echoes this back.
    target = next((target for target in PIN_TARGETS if target == pin), None)
    if target is None:
        raise SettingsError(f"switchyard.pin must be one of {', '.join(PIN_TARGETS)}, or empty to route normally")
    return target


def require_available(name: str, algorithm: str) -> None:
    """Refuse to create or change a preset with an algorithm that is not offered."""
    if algorithm not in AVAILABLE_ALGORITHMS:
        offered = ", ".join(ALGORITHMS[a].label for a in AVAILABLE_ALGORITHMS)
        raise SettingsError(f"switchyard preset {name!r}: {PRODUCT_NAME} offers {offered} only")


def _flag(name: str, value: Any) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise SettingsError(f"switchyard.{name} must be true or false")
    return value


def _consent(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or not all(isinstance(p, str) and _PROVIDER_ID.match(p) for p in value):
        raise SettingsError(f"switchyard.{SUBSCRIPTION_CONSENT_KEY} must be a list of Hermes provider ids")
    return tuple(dict.fromkeys(value))


def subscription_notice(names: List[str]) -> str:
    """What the user agrees to before Switchyard may call the subscription sign-ins ``names``."""
    one = len(names) == 1
    return (
        f"{' and '.join(names)} {'is a subscription sign-in' if one else 'are subscription sign-ins'}. "
        f"{PRODUCT_NAME} calls {'it' if one else 'them'} on its own as it routes, including the checks of the "
        "efficient model's answers, without asking you each time. Those calls count against your subscription's "
        "usage. The provider can restrict your account, or Hermes's sign-in with it, for use its terms do not "
        "allow. `hermes switchyard pin local` keeps every request on the local model."
    )


def subscription_refusal(names: List[str]) -> str:
    """The answer a sidecar without the user's agreement gives each request, short enough that Hermes
    shows all of it (it keeps 300 characters of an error)."""
    return (
        f"You have not agreed that Switchyard may call {' and '.join(names)} on its own as it routes, so it "
        "routed nothing. Run `hermes switchyard setup --allow-subscription-use` to agree, or pick another "
        "model for that tier."
    )


def _preset(name: str, raw: Any, bounded_local: bool = False) -> Preset:
    reserved = name == "default" or name.startswith(PIN_PRESET_PREFIX) or f"{MODEL_PREFIX}{name}" in PIN_MODELS
    if not _PRESET_NAME.match(name) or reserved:
        raise SettingsError(
            f"switchyard preset name {name!r} must be lowercase letters, digits, '-' or '_' "
            f"(and not 'default', {', '.join(repr(m[len(MODEL_PREFIX):]) for m in PIN_MODELS)}, "
            f"or starting with '{PIN_PRESET_PREFIX}')"
        )
    raw = raw if isinstance(raw, Mapping) else {}
    algorithm = str(raw.get("algorithm") or RECOMMENDED).strip().lower()
    if algorithm not in ALGORITHMS:
        raise SettingsError(f"switchyard.presets.{name}.algorithm must be one of {', '.join(ALGORITHMS)}")
    options = raw.get("options") or {}
    if not isinstance(options, Mapping):
        raise SettingsError(f"switchyard.presets.{name}.options must be a mapping of route keys")
    preset = Preset(
        name=name,
        algorithm=algorithm,
        options=_options(f"switchyard.presets.{name}.options", options),
        bounded_local=bounded_local,
    )
    route = preset.route
    if algorithm == "vgr":
        mode = preset.vgr_mode
        if mode not in VGR_MODES:
            raise SettingsError(f"switchyard.presets.{name}: VGR mode must be one of {', '.join(VGR_MODES)}")
        if mode == "active" and not route.get("active_approval"):
            raise SettingsError(f"switchyard.presets.{name}: VGR mode active requires active_approval (Switchyard's operator attestation)")
    return preset


def preset_map(raw: Any) -> Dict[str, Any]:
    """``{name: {"algorithm", "options"}}`` from stored presets.

    config.yaml stores presets as a list of ``{name, algorithm, options}``
    because Hermes's plugin-setting writes deep-merge mappings and so can never
    remove a preset or an option; a list is replaced whole. A hand-written
    ``{name: preset}`` mapping is accepted too.
    """
    if raw is None:
        return {name: dict(body) for name, body in DEFAULT_PRESETS.items()}
    if isinstance(raw, Mapping):
        return {str(name): body for name, body in raw.items()}
    if not isinstance(raw, list):
        raise SettingsError("switchyard.presets must be a list of presets")
    presets: Dict[str, Any] = {}
    for item in raw:
        if not isinstance(item, Mapping) or not str(item.get("name") or "").strip():
            raise SettingsError("every switchyard preset needs a name")
        name = str(item["name"]).strip()
        if name in presets:
            raise SettingsError(f"switchyard preset {name!r} is defined twice")
        presets[name] = {key: value for key, value in item.items() if key != "name"}
    return presets


def preset_records(presets: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The stored (list) form of a ``{name: preset}`` mapping."""
    return [{"name": name, **dict(body or {})} for name, body in presets.items()]


def bind_context(ctx: Any) -> None:
    """Keeps the context Hermes registered the plugin with, for :func:`save_setting`."""
    global _context
    _context = ctx


def save_setting(segments: Tuple[str, ...], value: Any) -> None:
    """Writes ``plugins.entries.<PLUGIN_ID>.settings.<segments>``.

    Through the plugin's context once Hermes has registered it; in a process that has not
    loaded the plugin, as the same one-key merge into config.yaml through Hermes's config API.
    """
    if _context is not None:
        _context.set_config(".".join(segments), value)
        return
    from hermes_cli.config import is_managed, read_user_config_raw, save_config

    if is_managed():
        raise PermissionError("Plugin settings cannot be changed in a managed install")
    path = ("plugins", "entries", PLUGIN_ID, "settings", *segments)
    partial: Any = value
    for key in reversed(path):
        partial = {key: partial}
    # Fails on a malformed config.yaml, which save_config would otherwise overwrite.
    read_user_config_raw()
    save_config(partial, preserve_keys={path}, merge_existing=True)


def plugin_settings(config: Mapping[str, Any], plugin_id: str = PLUGIN_ID) -> Mapping[str, Any]:
    """``plugins.entries.<plugin_id>.settings`` from a loaded config.yaml."""
    node: Any = config
    for key in ("plugins", "entries", plugin_id, "settings"):
        node = node.get(key) if isinstance(node, Mapping) else None
    return node if isinstance(node, Mapping) else {}


def load(raw: Mapping[str, Any]) -> SwitchyardSettings:
    """Validated settings from the plugin's ``settings`` mapping."""
    try:
        port = int(raw.get("port", DEFAULT_PORT))
    except (TypeError, ValueError):
        raise SettingsError("switchyard.port must be a TCP port") from None
    if not 1 <= port <= 65535:
        raise SettingsError("switchyard.port must be a TCP port")
    local = _tier("local", raw.get("local"))
    cloud = _tier("cloud", raw.get("cloud"))
    if local.identity == cloud.identity:
        raise SettingsError("switchyard.local and switchyard.cloud are the same model at the same endpoint")
    raw_presets = preset_map(raw.get("presets"))
    if not raw_presets:
        raise SettingsError("switchyard.presets must define at least one preset")
    # The older ``unbounded_local`` key is ignored: setup wrote it on every save, so a saved
    # ``false`` does not mean the user chose the limits.
    bounded_local = _flag("bounded_local", raw.get("bounded_local"))
    loaded = tuple(_preset(str(name), body, bounded_local) for name, body in raw_presets.items())
    presets = tuple(p for p in loaded if p.algorithm in AVAILABLE_ALGORITHMS) or tuple(
        _preset(name, body, bounded_local) for name, body in DEFAULT_PRESETS.items())
    default_preset = str(raw.get("default_preset") or "").strip() or presets[0].name
    if default_preset not in {p.name for p in presets}:
        if "default_preset" in raw and default_preset not in {p.name for p in loaded}:
            raise SettingsError(f"switchyard.default_preset {default_preset!r} is not a configured preset")
        default_preset = presets[0].name
    return SwitchyardSettings(
        port=port,
        default_preset=default_preset,
        local=local,
        cloud=cloud,
        presets=presets,
        pin=parse_pin(raw.get("pin")),
        bounded_local=bounded_local,
        subscription_consent=_consent(raw.get(SUBSCRIPTION_CONSENT_KEY)),
    )


def compatibility_warnings(settings: SwitchyardSettings) -> List[str]:
    """Presets that would run but cannot work as intended with these tiers."""
    tiers = {"local": settings.local, "cloud": settings.cloud}
    notes = []
    for preset in settings.presets:
        if preset.algorithm not in LOGPROB_ALGORITHMS:
            continue
        judge = str(preset.route.get("judge_target") or "local")
        tier = tiers.get(judge)
        if tier is not None and tier.format != "openai_chat":
            notes.append(
                f"Preset {preset.name}: {ALGORITHMS[preset.algorithm].label} scores the {ROLES[judge]} model's answers "
                f"from token probabilities, which {tier.model} ({tier.format}) does not return, so every turn would "
                "escalate. Use an OpenAI-compatible Chat Completions endpoint for that model."
            )
    return notes


def setup_note(algorithm: str, route: Mapping[str, Any]) -> Optional[str]:
    """What a new user should expect from ``algorithm`` before relying on it."""
    if algorithm != "vgr":
        return None
    seconds = int(route.get("deadline_seconds") or ALGORITHMS["vgr"].defaults["deadline_seconds"])
    wait = f"up to {seconds // 60} minute{'s' if seconds != 60 else ''}" if seconds % 60 == 0 else f"up to {seconds} seconds"
    return (f"VGR checks each local answer before showing any of it, so a reply can take {wait} to appear, "
            "with no text streaming in the meantime. Requests with images always go to the cloud model.")


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        # A JSON string literal is a valid TOML basic string.
        return json.dumps(value)
    return "[" + ", ".join(_toml_value(item) for item in value) + "]"


def _toml_table(header: str, values: Mapping[str, Any]) -> list[str]:
    lines = [f"[{header}]"]
    tables = []
    for key, value in values.items():
        if isinstance(value, Mapping):
            tables.append((key, value))
        else:
            lines.append(f"{key} = {_toml_value(value)}")
    lines.append("")
    for key, value in tables:
        lines += _toml_table(f"{header}.{key}", value)
    return lines


def render_switchyard_toml(settings: SwitchyardSettings, *, with_credentials: bool = True,
                           upstream_names: Optional[bool] = None, omit: frozenset = frozenset()) -> str:
    """The Switchyard deployment: both tiers through the tap, one route per preset.

    Both tiers authenticate to the tap with the sidecar's tier-route key; the tap, not
    Switchyard, holds each tier's own credential.

    ``upstream_names`` spells VGR routes the way upstream Switchyard reads them; by default,
    whichever way the installed build does. ``omit`` names :data:`OPTIONAL_ROUTE_DEFAULTS`
    the installed Switchyard does not read.
    """
    if upstream_names is None:
        from .runtime import uses_upstream_names

        upstream_names = uses_upstream_names()
    lines = ["schema_version = 1", ""]
    for name, tier in (("local", settings.local), ("cloud", settings.cloud)):
        lines += [
            f"[llm_clients.{name}]",
            f"format = {_toml_value(tier.format)}",
            f"base_url = {_toml_value(f'{settings.tap_url}/{name}')}",
            f"max_retries = {MAX_RETRIES}",
        ]
        if with_credentials:
            lines.append(f"api_key_env = {_toml_value(TIER_ROUTE_KEY_ENV)}")
        lines.append("")
    for name, tier in (("local", settings.local), ("cloud", settings.cloud)):
        target: Dict[str, Any] = {"id": tier.model, "llm_client": name}
        if tier.extra_body:
            target["extra_body"] = tier.extra_body
        lines += _toml_table(f"targets.{name}", target)
    for preset in (*settings.presets, *(pin_preset(tier) for tier in PIN_TARGETS)):
        route = {"id": preset.model, "type": preset.algorithm,
                 **{k: v for k, v in preset.route.items() if k not in omit or k in preset.options}}
        if upstream_names and preset.algorithm == "vgr":
            route = {UPSTREAM_VGR_KEYS.get(key, key): value for key, value in route.items()}
        lines += _toml_table(f"routes.{preset.name}", route)
    return "\n".join(lines).rstrip("\n") + "\n"
