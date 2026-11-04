# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Switchyard dashboard backend, mounted at /api/plugins/nvidia-switchyard/.

Relays the sidecar's live monitor state. The sidecar holds the state; this
process only reads it, so the tab works for traffic from any Hermes surface.
``/setup`` and ``/presets`` back the desktop plugin's setup page; ``/local/*``
relays Hermes's llama.cpp quickstart so that page stays inside this plugin's
API namespace.
"""

from __future__ import annotations

import contextvars
import importlib
import importlib.util
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

_PACKAGE_ALIAS = "hermes_switchyard_plugin"


def _plugin_package() -> ModuleType:
    """This plugin's package, whether Hermes bundled it or installed it under HERMES_HOME."""
    root = Path(__file__).resolve().parent.parent
    init = root / "__init__.py"
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if file and getattr(module, "__path__", None) is not None and Path(file).resolve() == init:
            return module
    spec = importlib.util.spec_from_file_location(_PACKAGE_ALIAS, init, submodule_search_locations=[str(root)])
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the switchyard plugin from {root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PACKAGE_ALIAS] = module
    spec.loader.exec_module(module)
    return module


_package = _plugin_package().__name__
sy_settings = importlib.import_module(f"{_package}.settings")
sidecar = importlib.import_module(f"{_package}.sidecar")
tier_providers = importlib.import_module(f"{_package}.providers")
router = APIRouter()

_API_KEY = re.compile(r"^[\x21-\x7e]{8,4096}$")

def _preset_view(preset: Any) -> Dict[str, Any]:
    return {
        "name": preset.name,
        "model": preset.model,
        "algorithm": preset.algorithm,
        "label": sy_settings.ALGORITHMS[preset.algorithm].label,
        "options": dict(preset.options),
        "route": preset.route,
    }


def _tier_view(name: str, tier: Any) -> Dict[str, Any]:
    return {
        "role": sy_settings.ROLES[name],
        "model": tier.model,
        "source": tier.source,
        "provider": tier.provider,
        "base_url": tier.base_url,
        "format": tier.format,
        "runtime": tier.runtime,
        "endpoint": tier.endpoint_label,
    }


def _summary(cfg: Any) -> Dict[str, Any]:
    return {
        "endpoint": cfg.front_base_url,
        "default_preset": cfg.default_preset,
        "pin": cfg.pin,
        "bounded_local": cfg.bounded_local,
        "local": _tier_view("local", cfg.local),
        "cloud": _tier_view("cloud", cfg.cloud),
        "presets": [_preset_view(preset) for preset in cfg.presets],
        "warnings": sy_settings.compatibility_warnings(cfg),
    }


def _algorithms() -> list[Dict[str, Any]]:
    return [
        {
            "name": name,
            "label": algorithm.label,
            "summary": algorithm.summary,
            "guided": algorithm.guided,
            "recommended": name == sy_settings.RECOMMENDED,
            "available": name in sy_settings.AVAILABLE_ALGORITHMS,
            "defaults": dict(algorithm.defaults),
            "note": sy_settings.setup_note(name, sy_settings.Preset(name=name, algorithm=name, options={}).route),
            "bounded_note": sy_settings.setup_note(
                name, sy_settings.Preset(name=name, algorithm=name, options={}, bounded_local=True).route
            ),
        }
        for name, algorithm in sy_settings.ALGORITHMS.items()
        if name in sy_settings.AVAILABLE_ALGORITHMS
    ]


def _selected_model() -> Optional[str]:
    model = sidecar._load_config().get("model") or {}
    if isinstance(model, dict) and str(model.get("provider") or "").lower() == sy_settings.PROVIDER:
        return str(model.get("default") or "") or None
    return None


@router.get("/state")
def get_state() -> Dict[str, Any]:
    try:
        cfg = sidecar.load_settings()
    except sy_settings.SettingsError as error:
        return {"configured": False, "running": False, "error": str(error), "algorithms": _algorithms()}
    state = sidecar.fetch_state(cfg)
    selected = _selected_model()
    return {
        "configured": True,
        "active": selected is not None,
        "selected_model": selected,
        "running": state is not None,
        "config": _summary(cfg),
        "algorithms": _algorithms(),
        "state": state,
    }


class RendererBody(BaseModel):
    loaded: bool = True


@router.post("/renderer")
def renderer(body: RendererBody) -> Dict[str, Any]:
    """Desktop's half reports it is loaded, so replies carry route notes only while something draws them."""
    sidecar.note_renderer(body.loaded)
    return {"ok": True}


@router.post("/start")
def start() -> Dict[str, Any]:
    try:
        cfg = sidecar.load_settings()
    except sy_settings.SettingsError as error:
        return {"ok": False, "error": str(error)}
    ok = sidecar.ensure_running(cfg)
    return {"ok": ok, "error": None if ok else f"see {sidecar.state_dir() / 'sidecar.log'}"}


@router.post("/dashboard")
def open_dashboard() -> Dict[str, Any]:
    """The browser monitor's URL, starting Hermes's web dashboard first if it is not up."""
    url = sidecar.ensure_dashboard()
    return {"ok": url is not None, "url": url,
            "error": None if url else f"The dashboard did not start; see {sidecar.state_dir() / 'dashboard.log'}"}


@router.post("/local/quickstart")
def local_quickstart(profile: Optional[str] = None) -> Dict[str, Any]:
    """Hermes's llama.cpp quickstart, reachable from the plugin's own API namespace."""
    from hermes_cli.web_routers.local_models import QuickstartBody, local_models_quickstart

    return local_models_quickstart(QuickstartBody(), profile=profile)


@router.get("/local/jobs/{job_id}")
async def local_job(job_id: str) -> Dict[str, Any]:
    from hermes_cli.web_routers.local_models import local_models_job

    return await local_models_job(job_id)


class TierBody(BaseModel):
    # "managed": Hermes's llama.cpp runtime (the model the quickstart just made the default);
    # "url": an OpenAI- or Anthropic-compatible server at base_url;
    # "provider": a Hermes provider (API key or subscription such as Codex or Nous Portal).
    source: Optional[Literal["managed", "url", "provider"]] = None
    # Older desktop builds: "external" is source "url".
    runtime: Optional[Literal["managed", "external"]] = None
    base_url: Optional[str] = Field(default=None, max_length=2048)
    model: Optional[str] = Field(default=None, max_length=256)
    provider: Optional[str] = Field(default=None, max_length=64)
    api_key: Optional[str] = Field(default=None, max_length=4096)

    def chosen_source(self, default: str) -> str:
        if self.source:
            return self.source
        return {"external": "url", "managed": "managed"}.get(self.runtime or "", default)


class PresetBody(BaseModel):
    algorithm: str = Field(default=sy_settings.RECOMMENDED, max_length=64)
    options: Dict[str, Any] = Field(default_factory=dict)


class SetupBody(BaseModel):
    local: TierBody = TierBody()
    cloud: TierBody = TierBody()
    # The preset to create (or replace) and make the default; omitted keeps the configured ones.
    preset_name: Optional[str] = Field(default=None, max_length=32)
    preset: Optional[PresetBody] = None
    # Turns on VGR's decision deadline and per-turn local budget; omitted keeps the saved value.
    bounded_local: Optional[bool] = None


class SavePresetBody(PresetBody):
    make_default: bool = False


def _managed_model() -> Optional[str]:
    """The model the managed runtime serves as Hermes's main model, if that is llama.cpp."""
    from hermes_cli.local_runtime.endpoint import LLAMACPP_ALIASES

    model = (sidecar._load_config().get("model") or {})
    if isinstance(model, dict) and str(model.get("provider") or "").lower() in LLAMACPP_ALIASES:
        return str(model.get("default") or "").strip() or None
    return None


def _tier_updates(name: str, body: TierBody, current: Any) -> Dict[str, Any]:
    """The settings one tier of a setup request writes; a changed source drops the old one's fields.

    Omitted fields keep their saved (or default) values.
    """
    source = body.chosen_source(current.source if current is not None else ("managed" if name == "local" else "url"))
    model = (body.model or "").strip()
    updates: Dict[str, Any] = {"source": source}
    if current is not None and current.source != source:
        updates.update(format=None, provider=None, base_url=None)
    if source == "managed":
        model = model or _managed_model() or ""
        if not model and current is not None and current.managed:
            return {}
        if not model:
            raise HTTPException(status_code=409, detail="Set up the local model first")
        return {**updates, "runtime": "managed", "model": model}
    updates["runtime"] = ""
    fields = {"model": model, "provider": body.provider, "base_url": body.base_url}
    updates.update({k: v.strip() for k, v in fields.items() if v and v.strip()})
    # A request that names its source spells out that source's fields.
    if body.source or body.runtime:
        needed = ("model", "provider") if source == "provider" else ("model", "base_url")
        missing = [field.replace("_", " ") for field in needed if field not in updates]
        if missing:
            raise HTTPException(
                status_code=400, detail=f"Enter the {sy_settings.ROLES[name]} model's {' and '.join(missing)}"
            )
    return updates


def _key_env(name: str, current: Any) -> str:
    if current is not None and current.source == "url" and current.api_key_env:
        return current.api_key_env
    return sy_settings.tier_key_env(name)


def _tier_url(name: str, updates: Dict[str, Any], current: Any) -> str:
    if "base_url" in updates:
        return updates["base_url"] or ""
    if current is not None and current.base_url:
        return current.base_url
    return sy_settings.DEFAULTS[name]["base_url"]


def _needs_key(base_url: str) -> bool:
    """A remote URL tier needs an API key; a server on this machine or network may not."""
    from urllib.parse import urlsplit

    host = urlsplit(base_url).hostname or ""
    return not sy_settings._private_host(host)


def _configure(updates: Dict[str, Any], **kwargs: Any) -> Any:
    try:
        return sidecar.configure(updates, **kwargs)
    except sy_settings.SettingsError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from None


def _apply(cfg: Any, *, start: bool) -> Dict[str, Any]:
    """Run ``cfg``: restart a running sidecar, or start one when asked to."""
    running = sidecar.restart(cfg) if start or sidecar.health(cfg) else False
    return {
        "ok": True,
        "running": running,
        "config": _summary(cfg),
        "error": None if running or not start else (
            f"Switchyard is configured but did not start; see {sidecar.state_dir() / 'sidecar.log'}"
        ),
    }


_SETUP_LOCK = threading.Lock()

# What the desktop shows for each setup step, in order.
_SETUP_STEPS = {
    "check": "Checking the model settings",
    "enable": "Turning on the Switchyard plugin. This can take a few minutes",
    "save": "Saving the settings and API keys",
    "start": "Starting the Switchyard sidecar",
}
_setup_jobs: Dict[str, Dict[str, Any]] = {}


def _acquire_setup() -> None:
    # Two setups at once would race each other on config.yaml.
    if not _SETUP_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail=f"{sy_settings.PRODUCT_NAME} setup is still running; wait for it to finish")


@router.post("/setup")
def setup(body: SetupBody) -> Dict[str, Any]:
    """Point Switchyard at its two tiers, store any URL tier's key, and make it the default model."""
    _acquire_setup()
    try:
        return _setup(body)
    finally:
        _SETUP_LOCK.release()


@router.post("/setup/jobs")
def start_setup(body: SetupBody) -> Dict[str, Any]:
    """``/setup`` in the background, so the desktop can show each step of a slow first run."""
    _acquire_setup()
    job_id = uuid.uuid4().hex
    job: Dict[str, Any] = {"status": "running", "started": time.monotonic(), "result": None, "error": None}
    _setup_jobs.clear()
    _setup_jobs[job_id] = job

    def report(step: str) -> None:
        job.update(step=list(_SETUP_STEPS).index(step) + 1, detail=_SETUP_STEPS[step])

    def run() -> None:
        try:
            job["result"] = _setup(body, progress=report)
            job["status"] = "done"
        except HTTPException as error:
            job.update(status="error", error=str(error.detail))
        except Exception as error:
            job.update(status="error", error=f"{sy_settings.PRODUCT_NAME} setup failed: {error}")
        finally:
            job["elapsed_s"] = round(time.monotonic() - job["started"])
            _SETUP_LOCK.release()

    report("check")
    # Hermes binds this profile's secrets to the request's context; a bare thread would read them unscoped.
    context = contextvars.copy_context()
    threading.Thread(target=context.run, args=(run,), name="switchyard-setup", daemon=True).start()
    return {"job_id": job_id}


@router.get("/setup/jobs/{job_id}")
def setup_job(job_id: str) -> Dict[str, Any]:
    job = _setup_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such Switchyard setup job")
    view = {key: value for key, value in job.items() if key != "started"}
    view.setdefault("elapsed_s", round(time.monotonic() - job["started"]))
    return {**view, "steps": len(_SETUP_STEPS)}


def _setup(body: SetupBody, progress: Callable[[str], None] = lambda step: None) -> Dict[str, Any]:
    from agent.secret_scope import get_secret
    from hermes_cli.config import save_env_value_secure

    try:
        current = sidecar.load_settings()
    except sy_settings.SettingsError:
        current = None
    updates: Dict[str, Any] = {}
    keys: Dict[str, str] = {}
    for name in ("local", "cloud"):
        tier_body = getattr(body, name)
        tier = getattr(current, name) if current is not None else None
        tier_updates = _tier_updates(name, tier_body, tier)
        updates[name] = tier_updates
        if tier_updates.get("source") != "url":
            continue
        api_key = (tier_body.api_key or "").strip()
        if api_key and not _API_KEY.match(api_key):
            raise HTTPException(status_code=400, detail="That API key does not look valid")
        key_env = _key_env(name, tier)
        if api_key:
            keys[key_env] = api_key
        elif not get_secret(key_env):
            if _needs_key(_tier_url(name, tier_updates, tier)):
                raise HTTPException(status_code=400, detail=f"Enter the API key for the {sy_settings.ROLES[name]} model")
            # The sidecar refuses to start on a named key that is not set.
            key_env = ""
        tier_updates["api_key_env"] = key_env
    if body.preset is not None:
        name = (body.preset_name or body.preset.algorithm).strip()
        preset = body.preset.model_dump()
        try:
            existing = sidecar.load_settings().preset(name)
        except sy_settings.SettingsError:
            existing = None
        # Re-running setup must not wipe a preset's route settings.
        if existing is not None and existing.algorithm == preset["algorithm"] and not preset["options"]:
            preset["options"] = dict(existing.options)
        updates["presets"] = {name: preset}
        updates["default_preset"] = name
    if body.bounded_local is not None:
        updates["bounded_local"] = body.bounded_local
    cfg = _configure(updates, progress=progress)
    for key_env, api_key in keys.items():
        save_env_value_secure(key_env, api_key)
    progress("start")
    result = _apply(cfg, start=True)
    score = sidecar.verifier_warning(cfg)
    result["config"]["warnings"].extend([score] if score else [])
    result["config"]["warnings"].extend(sidecar.account_warnings(cfg))
    return result


@router.get("/providers")
def providers() -> Dict[str, Any]:
    """The models Hermes has configured, by provider, and the one each tier starts on."""
    return tier_providers.hermes_models()


class PinBody(BaseModel):
    # "local" or "cloud" sends every request to that tier; "" or "off" routes with the preset.
    pin: str = Field(default="", max_length=8)


@router.post("/pin")
def set_pin(body: PinBody) -> Dict[str, Any]:
    try:
        return {"ok": True, **sidecar.set_pin(body.pin)}
    except sy_settings.SettingsError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None


@router.put("/presets/{name}")
def save_preset(name: str, body: SavePresetBody) -> Dict[str, Any]:
    updates: Dict[str, Any] = {"presets": {name: {"algorithm": body.algorithm, "options": body.options}}}
    if body.make_default:
        updates["default_preset"] = name
    cfg = _configure(updates, make_default=body.make_default)
    return _apply(cfg, start=False)


@router.delete("/presets/{name}")
def delete_preset(name: str) -> Dict[str, Any]:
    in_use = _selected_model() == f"{sy_settings.MODEL_PREFIX}{name}"
    cfg = _configure({}, remove_presets=(name,), make_default=in_use)
    return _apply(cfg, start=False)
