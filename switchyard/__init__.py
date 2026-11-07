# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Switchyard Local-first Routing for Hermes (beta).

Hermes talks to the ``switchyard`` provider; the provider is a loopback sidecar
hosting Switchyard in-process over two tiers, a local model and a cloud model.
Each preset is one Verification-Gated Routing (VGR) route over those tiers and
shows up as the model ``switchyard/<preset>``: each turn is attempted locally,
verified, and escalated to the cloud only when the evidence does not license a
local answer. ``hermes switchyard`` configures and runs the sidecar, and the
dashboard's Switchyard tab shows each decision live.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from . import escalation
from . import settings as sy_settings
from .runtime import RuntimeMissing

logger = logging.getLogger(__name__)

_HEALTH_CACHE_S = 10.0
_health_checked_at = 0.0
_health_lock = threading.Lock()

# ``hermes switchyard setup`` flag -> plugin-relative settings key.
_SETUP_FLAGS = {
    "local_url": "local.base_url",
    "local_model": "local.model",
    "local_key_env": "local.api_key_env",
    "cloud_url": "cloud.base_url",
    "cloud_model": "cloud.model",
    "cloud_key_env": "cloud.api_key_env",
    "local_provider": "local.provider",
    "cloud_provider": "cloud.provider",
    "local_format": "local.format",
    "cloud_format": "cloud.format",
    "port": "port",
    "bounded_local": "bounded_local",
}


_TIER_FLAGS = ("url", "model", "key_env", "provider", "format", "runtime")


def _tier_flags_given(args: argparse.Namespace) -> bool:
    return any(getattr(args, f"{name}_{flag}", None) for name in sy_settings.TIERS for flag in _TIER_FLAGS)


def _tier_from_choice(choice: Dict[str, str]) -> Dict[str, Any]:
    """The settings of a tier serving one of Hermes's models; the old source's fields are dropped."""
    managed = choice["source"] == "managed"
    return {
        "source": choice["source"],
        "runtime": "managed" if managed else "",
        "provider": None if managed else choice["provider"],
        "model": choice["model"],
        "format": None,
        "base_url": None,
    }


def _option_label(option: Dict[str, Any]) -> str:
    where = " (one-click local)" if option["source"] == "managed" else " (local)" if option["local"] else ""
    return f"{option['name']}{where}" + ("" if option["authenticated"] else " (not signed in)")


def _pick_hermes_model(name: str, catalog: Dict[str, Any], saved: bool) -> Optional[Dict[str, str]]:
    """The Hermes model the user picks for tier ``name``, or None to keep it."""
    from hermes_cli.setup import prompt_choice

    role = sy_settings.ROLES[name]
    # Local models first for the efficient tier, remote ones first for the capable tier.
    options = sorted(catalog["providers"], key=lambda o: o["local"] != (name == "local"))
    default = None if saved else catalog["defaults"].get(name)
    keep = "Keep the saved setting" if saved else f"Keep the built-in default (or rerun with --{name}-url)"
    start = next((i for i, o in enumerate(options) if default and o["provider"] == default["provider"]), len(options))
    picked = prompt_choice(f"The {role} ({name}) model's provider:", [_option_label(o) for o in options] + [keep], start)
    if picked == len(options):
        return None
    option = options[picked]
    models = option["models"]
    preferred = default["model"] if default and default["provider"] == option["provider"] else option["default_model"]
    model = prompt_choice(f"The {role} ({name}) model:", models, models.index(preferred) if preferred in models else 0)
    return {"source": option["source"], "provider": option["provider"], "model": models[model]}


def _hermes_tier_updates(current: Dict[str, Any], interactive: bool) -> Dict[str, Any]:
    """Tier settings from the models Hermes has.

    Interactively the user picks each tier's model; otherwise a tier with no saved
    settings takes its default (the efficient tier, a local model).
    """
    from . import providers

    catalog = providers.hermes_models()
    if interactive and not catalog["providers"]:
        if not all(current.get(name) for name in sy_settings.TIERS):
            print("Hermes has no models configured yet; pass --local-url / --cloud-url, or sign in with `hermes auth add`.")
        interactive = False
    updates: Dict[str, Any] = {}
    for name in sy_settings.TIERS:
        saved = bool(current.get(name))
        choice = _pick_hermes_model(name, catalog, saved) if interactive else (
            None if saved else catalog["defaults"].get(name))
        if choice:
            updates[name] = _tier_from_choice(choice)
    return updates


def _tier_source(args: argparse.Namespace, name: str) -> Optional[str]:
    """The source a setup run picks for tier ``name``; ``None`` keeps the saved one."""
    if getattr(args, f"{name}_provider", None):
        return "provider"
    if getattr(args, f"{name}_url", None) or getattr(args, f"{name}_runtime", None) == "external":
        return "url"
    if getattr(args, f"{name}_runtime", None) == "managed":
        return "managed"
    return None


def _uses_switchyard(provider: Any, base_url: Any) -> bool:
    if str(provider or "") == sy_settings.PROVIDER:
        return True
    return "/front/v1" in str(base_url or "") and str(base_url).startswith("http://127.0.0.1:")


def _on_pre_api_request(provider: Any = None, base_url: Any = None, **_: Any) -> None:
    """Bring the sidecar up before a Switchyard request needs it."""
    global _health_checked_at
    if not _uses_switchyard(provider, base_url):
        return
    with _health_lock:
        if time.monotonic() - _health_checked_at < _HEALTH_CACHE_S:
            return
        from . import sidecar

        try:
            if not sidecar.ensure_running():
                logger.warning("Switchyard sidecar did not come up; see %s", sidecar.state_dir() / "sidecar.log")
                return
        except sy_settings.SettingsError as error:
            logger.warning("Switchyard is not configured: %s (run `hermes switchyard setup`)", error)
            return
        _health_checked_at = time.monotonic()


def _fetch_state() -> Optional[Dict[str, Any]]:
    from . import sidecar

    try:
        return sidecar.fetch_state(sidecar.load_settings(), timeout=1.0)
    except (sy_settings.SettingsError, ValueError):
        return None


def _show_in_cli(text: str) -> None:
    try:
        from prompt_toolkit.application import get_app_or_none

        from hermes_cli.cli_render import _cprint
    except ImportError:
        return
    app = get_app_or_none()
    # No running REPL means a one-shot `hermes chat -q`, whose stdout must hold only the answer.
    if app is not None and getattr(app, "_is_running", False):
        _cprint(f"\033[2m  {text}\033[0m")


_cli_notice = escalation.CliNotice(_fetch_state, _show_in_cli)
_turn_calls: "OrderedDict[tuple[str, str], list]" = OrderedDict()  # (session, turn) -> [first call's start, calls]
_turn_calls_lock = threading.Lock()
_MAX_TRACKED_TURNS = 512


def _on_post_api_request(provider: Any = None, base_url: Any = None, platform: Any = None, session_id: Any = None,
                         turn_id: Any = None, started_at: Any = None, **_: Any) -> None:
    """Count the turn's Switchyard calls, and tell a classic CLI user when one changed tier."""
    if not session_id or not isinstance(started_at, (int, float)) or not _uses_switchyard(provider, base_url):
        return
    key = (str(session_id), str(turn_id))
    with _turn_calls_lock:
        calls = _turn_calls.setdefault(key, [float(started_at), 0])
        calls[1] += 1
        _turn_calls.move_to_end(key)
        while len(_turn_calls) > _MAX_TRACKED_TURNS:
            _turn_calls.popitem(last=False)
        turn_started = calls[0]
    if platform == "cli":
        _cli_notice.watch(*key, turn_started)


def _on_transform_llm_output(response_text: Any = None, session_id: Any = None, platform: Any = None,
                             turn_id: Any = None, **_: Any) -> Optional[str]:
    """Open a Desktop reply with where Switchyard changed tier during the turn, for the chat to show in place.

    Desktop's chat runs agents as ``desktop``; the TUI, as ``tui``, would print the directive as text,
    and so would Desktop with the plugin's Desktop half switched off."""
    with _turn_calls_lock:
        calls = _turn_calls.pop((str(session_id), str(turn_id)), None)
    if calls is None or platform != "desktop" or not isinstance(response_text, str):
        return None
    from . import sidecar

    if not sidecar.renderer_live():
        return None
    return escalation.mark_reply(response_text, escalation.wait_since(_fetch_state, str(session_id), *calls))


def _on_llm_request(request: Any = None, provider: Any = None, base_url: Any = None, turn_id: Any = None,
                    api_request_id: Any = None, session_id: Any = None, **_: Any) -> Optional[Dict[str, Any]]:
    """Keep the chat's route markers away from every model, and mark a Switchyard request
    ``x-initiator: user|agent``, so VGR knows when a person asked something new."""
    unmarked = escalation.unmark_request(request)
    if _uses_switchyard(provider, base_url):
        from . import initiator

        marked = initiator.mark_request(unmarked or request, turn_id=turn_id, api_request_id=api_request_id,
                                        session_id=session_id)
        if marked is not None:
            return {"request": marked, "source": "switchyard", "reason": initiator.HEADER}
    return None if unmarked is None else {"request": unmarked, "source": "switchyard", "reason": escalation.DIRECTIVE}


def _long_running_backend() -> bool:
    """Whether this process is a backend that serves chats, not a one-shot ``hermes`` command."""
    import sys

    from hermes_cli.process_identity import is_desktop_owned_backend

    return is_desktop_owned_backend() or sys.argv[1:2] in (["serve"], ["dashboard"], ["gateway"])


def _start_early() -> None:
    """Start the sidecar while the backend loads, so the first Switchyard request does not wait on it."""
    if not _long_running_backend():
        return

    def start() -> None:
        from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
        from hermes_constants import get_process_hermes_home

        from . import sidecar

        # Outside a turn no profile's secrets are bound, and a multiplexing backend refuses unscoped reads.
        home = get_process_hermes_home()
        scope = set_secret_scope(build_profile_secret_scope(home), profile_home=str(home))
        try:
            model = sidecar._load_config().get("model") or {}
            if isinstance(model, dict) and str(model.get("provider") or "") == sy_settings.PROVIDER:
                _on_pre_api_request(provider=sy_settings.PROVIDER)
        except Exception:
            logger.warning("Switchyard sidecar did not start early", exc_info=True)
        finally:
            reset_secret_scope(scope)

    threading.Thread(target=start, name="switchyard-start", daemon=True).start()


def _algorithm_help() -> str:
    names = ", ".join(sy_settings.AVAILABLE_ALGORITHMS)
    return f"{sy_settings.PRODUCT_NAME} algorithm: {names}"


def _setup_parser(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="switchyard_command")
    setup = sub.add_parser(
        "setup",
        help="Configure the two tiers and make Switchyard Hermes's model",
        description="The local tier is the efficient model and the cloud tier the capable one; either can be "
        "a server URL, a third-party provider you set up in Hermes (with an API key or a subscription "
        "sign-in; each provider's terms and conditions apply), or Hermes's own llama.cpp runtime. With no tier flags, each tier is picked from the models "
        "Hermes already has (without a terminal, an unconfigured efficient tier takes a local model).",
    )
    for name in ("local", "cloud"):
        role = sy_settings.ROLES[name]
        setup.add_argument(f"--{name}-url", help=f"Base URL of the {role} ({name}) model's server")
        setup.add_argument(f"--{name}-model", help=f"Model id of the {role} ({name}) tier")
        setup.add_argument(f"--{name}-key-env", help=f"Env var holding the {name} URL's API key, if it needs one")
        setup.add_argument(
            f"--{name}-provider",
            help=f"Use a Hermes provider for the {name} tier (a provider id that `hermes auth list` shows); "
            "its credentials come from `hermes auth`",
        )
        setup.add_argument(f"--{name}-format", choices=sy_settings.FORMATS, help=f"Wire format of the {name} tier")
        setup.add_argument(
            f"--{name}-runtime",
            choices=("managed", "external"),
            help=f"managed: serve the {name} tier from Hermes's own llama.cpp runtime; external: use --{name}-url",
        )
    setup.add_argument("--port", type=int, help=f"Loopback port of the sidecar (default: {sy_settings.DEFAULT_PORT})")
    setup.add_argument("--algorithm", choices=sy_settings.AVAILABLE_ALGORITHMS, help=_algorithm_help())
    setup.add_argument("--preset", help="Preset to create or select (default: the algorithm's name)")
    setup.add_argument("--no-default", action="store_true", help="Configure Switchyard without selecting it as the model")
    setup.add_argument(
        "--bounded-local", action=argparse.BooleanOptionalAction,
        help="Turn on (or, with --no-bounded-local, back off) VGR's time limits on local work: 300 s per "
        "request and 600 s per message (default: keep the saved setting)",
    )
    setup.add_argument(
        "--allow-subscription-use", action=argparse.BooleanOptionalAction,
        help="Agree that Switchyard may call a tier's subscription sign-in on its own as it routes, counting against that subscription's usage; --no-allow-subscription-use withdraws "
        "the agreement (default: ask in a terminal, else keep the saved answer)",
    )

    sub.add_parser("list", help="List presets and the routing algorithms Switchyard offers")

    configure = sub.add_parser("configure", help="Create or update a preset")
    configure.add_argument("name", nargs="?", help="Preset name (default: the default preset)")
    configure.add_argument("--algorithm", choices=sy_settings.AVAILABLE_ALGORITHMS, help=_algorithm_help())
    configure.add_argument(
        "--set", dest="options", action="append", default=[], metavar="KEY=VALUE",
        help="Set a Switchyard route key (VALUE is JSON, else a string; null drops a default). Repeatable",
    )
    configure.add_argument("--unset", action="append", default=[], metavar="KEY", help="Return a route key to its default")
    configure.add_argument("--default", action="store_true", help="Make this the default preset and Hermes's model")

    delete = sub.add_parser("delete", help="Delete a preset")
    delete.add_argument("name")

    pin = sub.add_parser("pin", help="Send every request to one tier, or route normally again")
    pin.add_argument("tier", choices=("local", "cloud", "off"), help="local or cloud pins to that tier; off routes with the preset")

    sub.add_parser("serve", help="Run the Switchyard sidecar in the foreground")
    sub.add_parser("start", help="Start the Switchyard sidecar in the background")
    sub.add_parser("stop", help="Stop the background Switchyard sidecar")
    status = sub.add_parser("status", help="Show the sidecar and the latest routing decisions")
    status.add_argument("--json", action="store_true", help="Print the raw monitor state")
    sub.add_parser("watch", help="Print each step of every turn as Switchyard works through it, until Ctrl+C")
    sub.add_parser("config", help="Print the Switchyard deployment the sidecar runs")


def _raw_presets() -> Dict[str, Any]:
    from . import sidecar

    return sy_settings.preset_map(sy_settings.plugin_settings(sidecar._load_config()).get("presets"))


def _missing_keys(cfg: sy_settings.SwitchyardSettings) -> list[str]:
    """The API key variables ``cfg``'s tiers name that hold no value; the sidecar refuses to start without them."""
    from hermes_cli.config import get_env_value

    return [tier.api_key_env for tier in (cfg.local, cfg.cloud)
            if tier.source == "url" and tier.api_key_env and not get_env_value(tier.api_key_env)]


def _apply_to_running(cfg: sy_settings.SwitchyardSettings) -> bool:
    """Restart a running sidecar on ``cfg``, which it reads only when it starts; False when ``cfg`` cannot run."""
    from . import sidecar

    running = sidecar.health(cfg)
    missing = _missing_keys(cfg)
    if missing:
        names = " and ".join(missing)
        consequence = ("the running sidecar keeps its previous settings" if running
                       else "the sidecar cannot start")
        print(f"{names} {'is' if len(missing) == 1 else 'are'} not set, so {consequence}. Set "
              f"{'it' if len(missing) == 1 else 'them'} in your Hermes .env (`hermes config env-path` shows where)"
              + (", then run `hermes switchyard stop` and `hermes switchyard start`." if running else "."))
        return False
    if not running:
        return True
    if sidecar.restart(cfg):
        print("Restarted the running sidecar with the new settings.")
        return True
    print(f"The running sidecar was stopped but did not restart; see {sidecar.state_dir() / 'sidecar.log'}. "
          "It starts again with the new settings on the next request.")
    return False


def _cmd_setup(args: argparse.Namespace) -> int:
    import sys

    from . import sidecar

    updates: Dict[str, Any] = {}
    for flag, key in _SETUP_FLAGS.items():
        value = getattr(args, flag, None)
        if value is not None:
            section, _, leaf = key.rpartition(".")
            (updates.setdefault(section, {}) if section else updates)[leaf] = value
    current = sy_settings.plugin_settings(sidecar._load_config())
    if not _tier_flags_given(args):
        # A run that only changes a setting keeps the tiers instead of asking for them again.
        interactive = (not updates and getattr(args, "allow_subscription_use", None) is None
                       and sys.stdin.isatty() and sys.stdout.isatty())
        updates.update(_hermes_tier_updates(dict(current), interactive=interactive))
    for name in ("local", "cloud"):
        source = _tier_source(args, name)
        if source is None:
            continue
        tier = updates.setdefault(name, {})
        tier["source"] = source
        tier["runtime"] = "managed" if source == "managed" else ""
        saved = current.get(name) if isinstance(current.get(name), dict) else {}
        if source != (saved.get("source") or ("managed" if saved.get("runtime") == "managed" else "url")):
            # The saved format (and provider or URL) belonged to the previous source.
            tier.setdefault("format", None)
            if source != "provider":
                tier.setdefault("provider", None)
            if source != "url":
                tier.setdefault("base_url", None)
    if args.algorithm or args.preset:
        name = args.preset or args.algorithm
        existing = _raw_presets().get(name) or {}
        algorithm = args.algorithm or existing.get("algorithm") or sy_settings.RECOMMENDED
        options = existing.get("options") if existing.get("algorithm") == algorithm else None
        updates["presets"] = {name: {"algorithm": algorithm, **({"options": options} if options else {})}}
        updates["default_preset"] = name
    confirm = None
    if sys.stdin.isatty() and sys.stdout.isatty():
        def confirm(notice: str) -> bool:
            from hermes_cli.setup import prompt_choice

            # Hermes's menu, as the tier pickers use: its line prompt can read no input on Windows.
            print(notice)
            return prompt_choice("Allow Switchyard to use it this way?", ["Yes, allow it", "No"], 1,
                                 description=notice) == 0
    try:
        cfg = sidecar.configure(
            updates, make_default=not args.no_default,
            allow_subscription_use=getattr(args, "allow_subscription_use", None), confirm_subscription_use=confirm,
        )
    except sy_settings.SettingsError as error:
        print(f"{sy_settings.PRODUCT_NAME} settings incomplete: {error}")
        return 1
    if sidecar.consent_problem(cfg):
        sidecar.stop(cfg)
        print(f"Withdrew the agreement. {sy_settings.PRODUCT_NAME} is stopped and won't start until you agree "
              "again or pick another model for the subscription tier.")
        return 0
    preset = cfg.default
    print(f"{sy_settings.PRODUCT_LABEL} configured: local {cfg.local.model}, cloud {cfg.cloud.model}; "
          f"default preset {preset.name} ({sy_settings.ALGORITHMS[preset.algorithm].label}).")
    if not args.no_default:
        print(f"Hermes now routes through {sy_settings.PRODUCT_NAME} ({preset.model} at {cfg.front_base_url}).")
        print(f"{sy_settings.PRODUCT_NAME} is in beta: its routing and settings may change between releases.")
    note = sy_settings.setup_note(preset.algorithm, preset.route)
    if note:
        print(f"Note: {note}")
    _report_warnings(cfg)
    score = sidecar.verifier_warning(cfg)
    for warning in ([score] if score else []) + sidecar.account_warnings(cfg):
        print(f"Warning: {warning}")
    return 0 if _apply_to_running(cfg) else 1


def _report_warnings(cfg: sy_settings.SwitchyardSettings) -> None:
    for note in sy_settings.compatibility_warnings(cfg):
        print(f"Warning: {note}")


def _cmd_list() -> int:
    from . import sidecar

    cfg = sidecar.load_settings()
    print(f"Tiers: local {cfg.local.model}{' (llama.cpp)' if cfg.local.managed else ''}, cloud {cfg.cloud.model}")
    print("Presets (select with /model <name> --provider switchyard):")
    for preset in cfg.presets:
        marker = "*" if preset.name == cfg.default_preset else " "
        overrides = ", ".join(f"{k}={json.dumps(v)}" for k, v in preset.options.items())
        print(f" {marker} {preset.name:<16} {preset.model:<28} {preset.algorithm}" + (f"  [{overrides}]" if overrides else ""))
    print("Algorithms:")
    for name in sy_settings.AVAILABLE_ALGORITHMS:
        algorithm = sy_settings.ALGORITHMS[name]
        tag = "recommended" if name == sy_settings.RECOMMENDED else "guided" if algorithm.guided else "advanced"
        print(f"   {name:<16} {tag:<12} {algorithm.summary}")
    return 0


def _parse_option(text: str) -> tuple[str, Any]:
    key, sep, value = text.partition("=")
    if not sep or not key.strip():
        raise sy_settings.SettingsError(f"--set expects KEY=VALUE, got {text!r}")
    try:
        parsed = json.loads(value)
    except ValueError:
        parsed = value
    return key.strip(), parsed


def _cmd_configure(args: argparse.Namespace) -> int:
    from . import sidecar

    cfg = sidecar.load_settings()
    name = args.name or cfg.default_preset
    existing = _raw_presets().get(name) or {}
    algorithm = args.algorithm or existing.get("algorithm") or sy_settings.RECOMMENDED
    options: Dict[str, Any] = dict(existing.get("options") or {}) if existing.get("algorithm", algorithm) == algorithm else {}
    for key in args.unset:
        options.pop(key, None)
    for text in args.options:
        key, value = _parse_option(text)
        options[key] = value
    updates: Dict[str, Any] = {"presets": {name: {"algorithm": algorithm, "options": options}}}
    if args.default:
        updates["default_preset"] = name
    cfg = sidecar.configure(updates, make_default=args.default)
    preset = cfg.preset(name)
    print(f"Preset {preset.name}: {sy_settings.ALGORITHMS[preset.algorithm].label} as {preset.model}"
          + (" (default)" if preset.name == cfg.default_preset else ""))
    _report_warnings(cfg)
    return 0 if _apply_to_running(cfg) else 1


def _cmd_delete(args: argparse.Namespace) -> int:
    from hermes_cli.config import load_config_readonly

    from . import sidecar

    model = (load_config_readonly() or {}).get("model") or {}
    in_use = (
        isinstance(model, dict)
        and model.get("provider") == sy_settings.PROVIDER
        and model.get("default") == f"{sy_settings.MODEL_PREFIX}{args.name}"
    )
    cfg = sidecar.configure({}, remove_presets=(args.name,), make_default=in_use)
    print(f"Deleted preset {args.name}; default preset is {cfg.default_preset}.")
    return 0 if _apply_to_running(cfg) else 1


def _cmd_status(args: argparse.Namespace) -> int:
    from . import sidecar

    cfg = sidecar.load_settings()
    state = sidecar.fetch_state(cfg)
    if args.json:
        print(json.dumps(state, indent=2))
        return 0 if state else 1
    if not state:
        print(f"{sy_settings.PRODUCT_LABEL} sidecar is not running on {cfg.tap_url}.")
        problem = sidecar.startup_problem(cfg)
        if problem:
            print(f"  It cannot start: {problem}")
        return 1
    totals = state["totals"]
    print(f"{sy_settings.PRODUCT_LABEL} sidecar: {cfg.front_base_url}  default preset {state['default_preset']}")
    runtime = state.get("runtime") or {}
    if runtime:
        print(f"  runtime: {runtime.get('project')} {runtime.get('version')} with VGR, from {runtime.get('location')}")
    if not sidecar.renderer_live():
        # Desktop turns a package's Desktop half on only from its own install dialog or Agent switch.
        print("  Desktop page: not loaded. If Hermes Desktop is open with no Switchyard in its sidebar, "
              "turn on Switchyard's Desktop switch under Capabilities > Plugins")
    if state.get("pin"):
        print(f"  pinned: every request goes to {state['pin']} (`hermes switchyard pin off` to route normally)")
    bound, flip = (("bounded", "--no-bounded-local") if cfg.bounded_local
                   else (f"unbounded per message, {sy_settings.UNBOUNDED_DEADLINE_SECONDS // 60} min per request", "--bounded-local"))
    print(f"  local thinking time: {bound} (`hermes switchyard setup {flip}` to change)")
    print(f"  local  {state['local_model']}: {totals['local']['calls']} calls, "
          f"{totals['local']['prompt']} in / {totals['local']['completion']} out tokens")
    print(f"  cloud  {state['cloud_model']}: {totals['cloud']['calls']} calls, "
          f"{totals['cloud']['prompt']} in / {totals['cloud']['completion']} out tokens")
    pins = {sy_settings.pin_preset(tier).name: tier for tier in sy_settings.PIN_TARGETS}
    for name, preset in totals["by_preset"].items():
        algorithm = (state["presets"].get(name) or {}).get("algorithm", "?")
        kind = f"{sy_settings.ROLES[pins[name]]} model only" if name in pins else algorithm
        print(f"  {name} ({kind}): served locally {preset['served']['local']} / cloud {preset['served']['cloud']}"
              f" of {preset['turns']} turns")
    current = state.get("current")
    if current:
        print(f"  now: {current['label']} ({current['tier'] or '-'}), {current['elapsed_ms']} ms,"
              f" {_turn_tokens(current.get('tokens'))}")
    for turn in state["turns"][:5]:
        path = " > ".join(stage["stage"] for stage in turn["stages"]) or "-"
        print(f"  #{turn['id']} {turn['preset']:<12} {_words(turn['outcome']):<30} {path}")
    return 0


def _cmd_watch() -> int:
    """The CLI's view of the step Desktop shows above the composer; Hermes's terminal has no place for it."""
    import sys

    from . import sidecar

    cfg = sidecar.load_settings()
    print(f"Watching {sy_settings.PRODUCT_LABEL} on {cfg.tap_url}; press Ctrl+C to stop.")
    shown: Any = None
    # Turns shown but not yet finished; turns running at once take turns as the current one.
    pending: List[int] = []
    # A terminal gets the running step's line rewritten each second; a log gets one line a step.
    redraw = sys.stdout.isatty()
    open_width = 0  # the rewritten line's width, 0 once a newline has ended it

    def end_line() -> None:
        nonlocal open_width
        if open_width:
            print()
            open_width = 0

    try:
        while True:
            state = sidecar.fetch_state(cfg)
            current = (state or {}).get("current")
            if state is None:
                if shown != "down":
                    end_line()
                    print("The sidecar is not running; waiting for it to start.")
                    shown = "down"
                    # A restarted sidecar numbers its turns from 1 again.
                    pending.clear()
            else:
                finished = {t["id"]: t for t in state["turns"]}
                for turn_id in [t for t in pending if t in finished]:
                    end_line()
                    _print_finished(finished[turn_id])
                    pending.remove(turn_id)
                    shown = None
                if current:
                    turn = current["turn"]
                    step = (turn["id"], current["label"], current["tier"])
                    line = (f"#{turn['id']} {current['elapsed_ms'] // 1000:>4} s  {current['label']} ({current['tier'] or '-'})"
                            f"  {_turn_tokens(current.get('tokens'))}")
                    if step != shown:
                        end_line()
                        shown = step
                        if turn["id"] not in pending:
                            pending.append(turn["id"])
                        if not redraw:
                            print(line)
                    if redraw:
                        print(f"\r{line.ljust(open_width)}", end="", flush=True)
                        open_width = len(line)
            time.sleep(1)
    except KeyboardInterrupt:
        end_line()
        return 0


def _print_finished(turn: Mapping[str, Any]) -> None:
    served = f", served by the {turn['served_tier']} model" if turn.get("served_tier") else ""
    tokens = {key: sum(t[key] for t in turn["tokens"].values()) for key in ("prompt", "completion")}
    print(f"#{turn['id']} finished: {_words(turn['outcome'])}{served}, {_turn_tokens(tokens)}")


def _words(outcome: Optional[str]) -> str:
    return str(outcome or "running").replace("_", " ")


def _turn_tokens(tokens: Optional[Mapping[str, Any]]) -> str:
    tokens = tokens or {}
    return f"{tokens.get('prompt', 0)} in / {tokens.get('completion', 0)} out tokens"


def _handle(args: argparse.Namespace) -> int:
    from . import sidecar

    command: Optional[str] = getattr(args, "switchyard_command", None)
    try:
        if command == "setup":
            return _cmd_setup(args)
        if command == "list":
            return _cmd_list()
        if command == "configure":
            return _cmd_configure(args)
        if command == "delete":
            return _cmd_delete(args)
        if command == "pin":
            result = sidecar.set_pin(args.tier)
            cfg = sidecar.load_settings()
            if result["pin"]:
                tier = getattr(cfg, result["pin"])
                print(f"{sy_settings.PRODUCT_NAME} pinned: every request goes to the "
                      f"{sy_settings.ROLES[result['pin']]} model {tier.model}.")
            else:
                print(f"{sy_settings.PRODUCT_NAME} uses the selected preset again (default {cfg.default_preset}).")
            if not result["live"]:
                print("The sidecar is not running; the pin applies when it starts.")
            return 0
        if command == "serve":
            return sidecar.serve()
        if command == "start":
            ok = sidecar.ensure_running()
            if ok:
                print(f"{sy_settings.PRODUCT_LABEL} sidecar running.")
                return 0
            problem = sidecar.startup_problem(sidecar.load_settings())
            print(f"{sy_settings.PRODUCT_LABEL} sidecar failed to start: {problem}" if problem
                  else f"{sy_settings.PRODUCT_LABEL} sidecar failed to start; see {sidecar.state_dir() / 'sidecar.log'}")
            return 1
        if command == "stop":
            return 0 if sidecar.stop() else 1
        if command == "config":
            print(sidecar.running_toml(sidecar.load_settings()), end="")
            return 0
        if command == "watch":
            return _cmd_watch()
        return _cmd_status(argparse.Namespace(json=getattr(args, "json", False)))
    except (sy_settings.SettingsError, RuntimeMissing, PermissionError) as error:
        print(f"{sy_settings.PRODUCT_NAME}: {error}")
        return 1


def _ensure_front_token() -> None:
    """Fresh installs seeded with ``provider: switchyard`` have no front token yet, and
    Hermes resolves provider credentials before any request hook runs."""
    try:
        from hermes_cli.config import load_config_readonly

        model = load_config_readonly().get("model")
        if isinstance(model, dict) and model.get("provider") == sy_settings.PROVIDER:
            from . import sidecar

            sidecar.ensure_token()
    except Exception as error:  # never block Hermes startup on this
        logger.warning("Switchyard could not create its front token: %s", error)


def _update_hermes_settings() -> None:
    import contextlib
    import io

    try:
        from . import sidecar

        # Hermes's config writer prints each key it sets, which would land in whatever command started Hermes.
        with contextlib.redirect_stdout(io.StringIO()):
            sidecar.keep_restore_record()
            updated = sidecar.update_hermes_settings()
        if updated:
            logger.info("Switchyard updated the Hermes settings an older version of the plugin wrote")
    except (Exception, SystemExit) as error:  # never block Hermes startup on this; a managed key exits
        logger.warning("Switchyard could not update its Hermes settings: %s", error)


# ``__init__.py`` last: Hermes imports the folder by it, so it never sees a half-copied profile.
_PROVIDER_FILES = ("plugin.yaml", "LICENSE", "__init__.py")


def _install_provider() -> None:
    """Copies the bundled provider profile to ``HERMES_HOME/plugins/model-providers/switchyard/``
    where it differs. Hermes loads provider profiles only from there, and the copy outlives the
    plugin long enough to undo setup once the plugin is removed."""
    try:
        from hermes_constants import get_hermes_home

        source = Path(__file__).resolve().with_name("provider")
        target = get_hermes_home() / "plugins" / "model-providers" / sy_settings.PROVIDER
        for name in _PROVIDER_FILES:
            data = (source / name).read_bytes()
            path = target / name
            if path.is_file() and path.read_bytes() == data:
                continue
            target.mkdir(parents=True, exist_ok=True)
            staging = target / f".{name}.{secrets.token_hex(4)}"
            try:
                staging.write_bytes(data)
                os.replace(staging, path)
            finally:
                staging.unlink(missing_ok=True)
    except Exception as error:  # never block Hermes startup on this
        logger.warning("Switchyard could not install its provider profile: %s", error)


def _adopt_legacy_settings() -> None:
    """Moves settings saved under the plugin's old id to its current one, unless it already has its own."""
    import contextlib
    import copy
    import io

    try:
        from hermes_cli.config import load_config_readonly, unset_config_value

        config = load_config_readonly() or {}
        entries = (config.get("plugins") or {}).get("entries")
        if not (isinstance(entries, dict) and isinstance(entries.get(sy_settings.LEGACY_PLUGIN_ID), dict)):
            return
        legacy = copy.deepcopy(dict(sy_settings.plugin_settings(config, sy_settings.LEGACY_PLUGIN_ID)))
        with contextlib.redirect_stdout(io.StringIO()):
            if legacy and not sy_settings.plugin_settings(config):
                for key, value in legacy.items():
                    sy_settings.save_setting((key,), value)
            unset_config_value(f"plugins.entries.{sy_settings.LEGACY_PLUGIN_ID}")
        logger.info("Switchyard moved its settings from plugin id %r to %r",
                    sy_settings.LEGACY_PLUGIN_ID, sy_settings.PLUGIN_ID)
    except (Exception, SystemExit) as error:  # never block Hermes startup on this; a managed key exits
        logger.warning("Switchyard could not move its settings from its old plugin id: %s", error)


def _forget_legacy_id() -> None:
    """Takes the plugin's old id out of the plugin lists; enabled, it would load an earlier build
    copied in as ``plugins/switchyard``."""
    import contextlib
    import io

    try:
        from hermes_cli.config import load_config, load_config_readonly, save_config

        plugins = (load_config_readonly() or {}).get("plugins")
        lists = [key for key in ("enabled", "disabled")
                 if isinstance(plugins, dict) and isinstance(plugins.get(key), list)
                 and sy_settings.LEGACY_PLUGIN_ID in plugins[key]]
        if not lists:
            return
        config = load_config()
        for key in lists:
            config["plugins"][key] = [name for name in config["plugins"][key] if name != sy_settings.LEGACY_PLUGIN_ID]
        with contextlib.redirect_stdout(io.StringIO()):
            save_config(config)
        logger.info("Switchyard took its old plugin id %r out of plugins.%s",
                    sy_settings.LEGACY_PLUGIN_ID, " and plugins.".join(lists))
    except (Exception, SystemExit) as error:  # never block Hermes startup on this; a managed key exits
        logger.warning("Switchyard could not take its old plugin id out of the plugin lists: %s", error)


def register(ctx) -> None:
    sy_settings.bind_context(ctx)
    _adopt_legacy_settings()
    _forget_legacy_id()
    _install_provider()
    _ensure_front_token()
    _update_hermes_settings()
    ctx.register_cli_command(
        "switchyard",
        help=f"Route Hermes between a local and a cloud model with {sy_settings.PRODUCT_LABEL}, using VGR",
        setup_fn=_setup_parser,
        handler_fn=_handle,
        description=__doc__,
    )
    ctx.register_hook("pre_api_request", _on_pre_api_request)
    ctx.register_hook("post_api_request", _on_post_api_request)
    # A Hermes without request middleware sends no x-initiator; Switchyard then reads the message shape.
    # Nor could it keep route markers away from the model, so its chats get none.
    if callable(getattr(ctx, "register_middleware", None)):
        from . import initiator

        ctx.register_hook("pre_llm_call", initiator.on_pre_llm_call)
        ctx.register_middleware("llm_request", _on_llm_request)
        ctx.register_hook("transform_llm_output", _on_transform_llm_output)
    _start_early()
