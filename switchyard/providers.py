# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tiers served by a provider the user is signed in to in Hermes.

The tap resolves such a tier's endpoint and credential on each call through
Hermes's own provider runtime, so subscription logins (OpenAI Codex, Nous
Portal) and their short-lived tokens refresh exactly as they do for Hermes.
Nothing the resolver returns is written to Switchyard's config or the
environment, except for the providers in ``MODEL_BLOCK_PROVIDERS``, which
setup saves as the URL tier they resolve to.

``hermes_models`` lists the models Hermes already has configured, for setup to
offer instead of asking for URLs and keys.
"""

from __future__ import annotations

import ipaddress
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
from urllib.parse import urlsplit

from . import settings as sy_settings

logger = logging.getLogger(__name__)

# Hermes refreshes OAuth tokens two minutes before they expire, so a shorter reuse is always valid.
_REUSE_S = 30.0
# Inventory rows that are not a model to route to: Switchyard itself and Mixture of Agents.
_NOT_TIERS = frozenset({sy_settings.PROVIDER, "nemo-switchyard", "moa"})
# Hermes reads these providers' endpoints from config.yaml's ``model`` block, which making
# Switchyard the model rewrites to point at Switchyard.
MODEL_BLOCK_PROVIDERS = frozenset({"custom", "azure-foundry"})
# What Hermes resolves as the key of an endpoint that takes none.
_NO_KEY = "no-key-required"
# The inventory's row for Hermes's llama.cpp runtime, served by a ``managed`` tier.
_MANAGED_ROW_SOURCE = "local-runtime"
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0")
_LOCAL_SUFFIXES = (".docker.internal", ".containers.internal", ".lima.internal", ".local")
_TAILSCALE_CGNAT = ipaddress.IPv4Network("100.64.0.0/10")

Endpoint = Tuple[str, Dict[str, str]]


class ProviderUnavailable(sy_settings.SettingsError):
    """A provider tier cannot be served: not signed in, or a wire Switchyard cannot carry."""


def _runtime(tier: sy_settings.Tier) -> Mapping[str, Any]:
    from hermes_cli.runtime_provider import resolve_runtime_provider

    return resolve_runtime_provider(requested=tier.provider, target_model=tier.model)


def _not_switchyard(tier: sy_settings.Tier, runtime: Mapping[str, Any]) -> None:
    parts = urlsplit(str(runtime.get("base_url") or ""))
    front = parts.hostname in ("127.0.0.1", "localhost") and parts.path.rstrip("/").endswith("/front/v1")
    if front or str(runtime.get("provider") or "") in _NOT_TIERS:
        raise ProviderUnavailable(
            f"provider {tier.provider!r} now resolves to Switchyard itself; run setup again and pick the model's own endpoint"
        )


def runtime_format(runtime: Mapping[str, Any]) -> str:
    api_mode = str(runtime.get("api_mode") or "chat_completions")
    fmt = sy_settings.API_MODE_FORMATS.get(api_mode)
    if fmt is None:
        raise ProviderUnavailable(f"provider {runtime.get('provider')!r} uses the {api_mode} API, which Switchyard cannot route")
    return fmt


def _is_anthropic_oauth(key: str) -> bool:
    """Hermes's rule for a Claude subscription token: every ``sk-ant-`` key outside the Console
    API-key families, plus setup JWTs and ``cc-`` tokens."""
    if key.startswith(("sk-ant-api", "sk-ant-usr", "sk-ant-svc", "sk-ant-admin")):
        return False
    return key.startswith(("sk-ant-", "eyJ", "cc-"))


def is_subscription(provider: str) -> bool:
    """Whether Hermes reaches ``provider`` through a subscription sign-in (OAuth), not an API key."""
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.models import normalize_provider

    config = PROVIDER_REGISTRY.get(normalize_provider(provider))
    return config is not None and str(config.auth_type).startswith("oauth")


def _label(provider: str) -> str:
    from hermes_cli.models import provider_label

    return provider_label(provider)


def unconsented(cfg: sy_settings.SwitchyardSettings) -> List[str]:
    """The subscription sign-ins ``cfg``'s tiers use that the user has not agreed Switchyard may call."""
    from hermes_cli.models import normalize_provider

    found = [normalize_provider(tier.provider) for tier in (cfg.local, cfg.cloud)
             if tier.source == "provider" and is_subscription(tier.provider)]
    agreed = {normalize_provider(p) for p in cfg.subscription_consent}
    return [p for p in dict.fromkeys(found) if p not in agreed]


def subscription_notice(provider_ids: List[str]) -> str:
    return sy_settings.subscription_notice([_label(p) for p in provider_ids])


def subscription_refusal(provider_ids: List[str]) -> str:
    return sy_settings.subscription_refusal([_label(p) for p in provider_ids])


def credential_headers(runtime: Mapping[str, Any]) -> Dict[str, str]:
    """The headers that authenticate one call to ``runtime``'s endpoint."""
    key = str(runtime.get("api_key") or "").strip()
    provider = str(runtime.get("provider") or "")
    if not key:
        detail = runtime.get("auth_error") or "no credential"
        raise ProviderUnavailable(f"provider {provider!r} is not signed in ({detail}); sign in with `hermes auth add {provider}`")
    if runtime_format(runtime) == "anthropic_messages" and provider == "anthropic":
        if _is_anthropic_oauth(key):
            raise ProviderUnavailable(
                "a Claude subscription login cannot be routed by Switchyard; use an Anthropic API key, "
                "or reach Claude through OpenRouter or Nous Portal"
            )
        return {"x-api-key": key}
    headers = {"Authorization": f"Bearer {key}"}
    if runtime_format(runtime) == "openai_responses" and provider == "openai-codex":
        from agent.codex_headers import codex_cloudflare_headers

        headers.update(codex_cloudflare_headers(key, base_url=str(runtime.get("base_url") or "")))
    return headers


def endpoint(tier: sy_settings.Tier, runtime: Mapping[str, Any]) -> Endpoint:
    """Where and how the tap reaches ``tier`` according to a resolved ``runtime``."""
    _not_switchyard(tier, runtime)
    fmt = runtime_format(runtime)
    if fmt != tier.format:
        raise ProviderUnavailable(
            f"provider {tier.provider!r} now uses {fmt}, but the tier was set up for {tier.format}; run setup again"
        )
    base_url = str(runtime.get("base_url") or "").strip()
    if not base_url:
        raise ProviderUnavailable(f"provider {tier.provider!r} has no endpoint")
    return sy_settings.upstream_root(fmt, base_url), credential_headers(runtime)


def _resolved(tier: sy_settings.Tier) -> Mapping[str, Any]:
    try:
        runtime = _runtime(tier)
    except ProviderUnavailable:
        raise
    except Exception as error:
        raise ProviderUnavailable(f"provider {tier.provider!r} did not resolve: {error}") from None
    _not_switchyard(tier, runtime)
    return runtime


def check(tier: sy_settings.Tier) -> str:
    """Confirm ``tier`` can be served now; returns the wire format its provider uses."""
    runtime = _resolved(tier)
    fmt = runtime_format(runtime)
    credential_headers(runtime)
    return fmt


def pinned_url(tier: sy_settings.Tier) -> Tuple[Dict[str, Any], str]:
    """The URL tier settings serving what ``tier``'s provider resolves to now, and its API key ("" for none)."""
    runtime = _resolved(tier)
    fmt = runtime_format(runtime)
    base_url = str(runtime.get("base_url") or "").strip()
    if not base_url:
        raise ProviderUnavailable(f"provider {tier.provider!r} has no endpoint")
    key = str(runtime.get("api_key") or "").strip()
    fields = {"source": "url", "runtime": "", "provider": None, "base_url": base_url, "format": fmt}
    return fields, "" if key == _NO_KEY else key


def _row_url(row: Mapping[str, Any]) -> str:
    if row.get("api_url"):
        return str(row["api_url"])
    from hermes_cli.auth import PROVIDER_REGISTRY

    return str(getattr(PROVIDER_REGISTRY.get(str(row.get("slug") or "")), "inference_base_url", "") or "")


def _is_local_host(host: str) -> bool:
    """Hermes's rule for a local endpoint: loopback, container and mDNS names, unqualified
    hosts, private and link-local addresses, and Tailscale's CGNAT range."""
    if host in _LOCAL_HOSTS or host.endswith(_LOCAL_SUFFIXES) or ("." not in host and ":" not in host):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # A dotted quad ``ipaddress`` rejects (leading zeros) is judged by its first two numbers.
        parts = host.split(".")
        if len(parts) != 4 or not (parts[0].isdigit() and parts[1].isdigit()):
            return False
        first, second = int(parts[0]), int(parts[1])
        return first == 10 or (first == 172 and 16 <= second <= 31) or (first == 192 and second == 168) or (
            first == 100 and 64 <= second <= 127)
    return address.is_private or address.is_loopback or address.is_link_local or (
        isinstance(address, ipaddress.IPv4Address) and address in _TAILSCALE_CGNAT)


def _is_local(row: Mapping[str, Any]) -> bool:
    if row.get("source") == _MANAGED_ROW_SOURCE:
        return True
    parts = urlsplit(_row_url(row).strip().rstrip("/"))
    return parts.scheme in ("http", "https") and bool(parts.hostname) and _is_local_host(parts.hostname)


def _choice(option: Mapping[str, Any], model: str) -> Dict[str, str]:
    return {"source": option["source"], "provider": option["provider"], "model": model}


def _defaults(options: List[Dict[str, Any]], current: Optional[Dict[str, Any]]) -> Dict[str, Optional[Dict[str, str]]]:
    """Each tier's starting choice: efficient, a local model; capable, a remote one.

    Hermes's current model fills the tier it belongs to. The efficient tier
    otherwise takes the llama.cpp model one-click setup downloaded, then any
    other local endpoint; the capable tier takes the first signed-in remote
    provider's default model.
    """
    signed_in = [o for o in options if o["authenticated"]]
    by_id = {o["provider"]: o for o in signed_in}
    held = by_id.get(current["provider"]) if current else None
    local = _choice(held, current["model"]) if held and held["local"] else None
    cloud = _choice(held, current["model"]) if held and not held["local"] else None
    if local is None:
        found = [o for o in signed_in if o["source"] == "managed"] or [o for o in signed_in if o["local"]]
        local = _choice(found[0], found[0]["default_model"]) if found else None
    if cloud is None:
        found = [o for o in signed_in if not o["local"]]
        cloud = _choice(found[0], found[0]["default_model"]) if found else None
    return {"local": local, "cloud": cloud}


def hermes_models() -> Dict[str, Any]:
    """The models Hermes has configured, by provider, with each tier's default choice.

    Each option's ``source`` is the tier source that serves it: ``managed`` for
    the models Hermes's llama.cpp runtime has downloaded, ``provider`` otherwise.
    Rows are the desktop model picker's (explicitly configured providers only),
    and ``default_model`` is the model Hermes picks for a provider without asking.
    """
    from hermes_cli.inventory import build_model_options_payload, load_picker_context
    from hermes_cli.models import pick_silent_default_model

    context = load_picker_context()
    options: List[Dict[str, Any]] = []
    current = None
    for row in build_model_options_payload(context, explicit_only=True).get("providers") or []:
        slug = str(row.get("slug") or "")
        models = [str(m) for m in row.get("models") or [] if m]
        if not slug or slug.lower() in _NOT_TIERS or not models:
            continue
        option = {
            "provider": slug,
            "name": str(row.get("name") or slug),
            "source": "managed" if row.get("source") == _MANAGED_ROW_SOURCE else "provider",
            "local": _is_local(row),
            "authenticated": bool(row.get("authenticated")),
            "models": models,
            "default_model": pick_silent_default_model(models, provider=slug) or models[0],
        }
        if option["source"] == "provider" and is_subscription(slug):
            option["subscription_notice"] = subscription_notice([slug])
        options.append(option)
        if row.get("is_current") and context.current_model in models:
            current = {"provider": slug, "model": context.current_model}
    return {"providers": options, "current": current, "defaults": _defaults(options, current)}


def resolver(tier: sy_settings.Tier) -> Callable[[], Optional[Endpoint]]:
    """A tap endpoint resolver for ``tier``, re-resolved at most every ``_REUSE_S`` seconds."""
    lock = threading.Lock()
    cached: Dict[str, Any] = {"at": 0.0, "endpoint": None}

    def resolve() -> Optional[Endpoint]:
        with lock:
            if cached["endpoint"] is not None and time.monotonic() - cached["at"] < _REUSE_S:
                return cached["endpoint"]
            try:
                found = endpoint(tier, _runtime(tier))
            except Exception as error:
                logger.warning("Switchyard tier on provider %s is unavailable: %s", tier.provider, error)
                cached["endpoint"] = None
                return None
            cached.update(at=time.monotonic(), endpoint=found)
            return found

    return resolve
