# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Switchyard's native runtime for the sidecar.

The plugin declares ``nemo-switchyard`` under ``python_dependencies``, so
Hermes installs it from PyPI into its own environment when it installs the
plugin, and the sidecar imports it from there. The plugin is pinned to one
release, and the sidecar imports it only once every installed file matches that
release's published wheels.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Optional, Tuple

_PROJECT = "nemo-switchyard"
# The first nemo-switchyard release with VGR, as upstream names its tiers.
_FIRST_UPSTREAM = (0, 4)
_RELEASE = re.compile(r"^(\d+)\.(\d+)")
_MODULE = "switchyard_rust"
# The release plugin.yaml pins. PyPI never replaces a released file, so its files are fixed.
_PINNED = "0.4.0a1"
_PACKAGES = ("switchyard", "switchyard_rust")
# Each wheel holds one native module: the .pyd on Windows, the .so elsewhere.
_NATIVE = ("switchyard_rust/_switchyard_rust.abi3.so", "switchyard_rust/_switchyard_rust.pyd")
# SHA-256 of every file in the six published 0.4.0a1 wheels, by path under site-packages. The
# Windows wheels' Python files differ in line endings, and each platform's native module differs.
_RELEASE_FILES: Dict[str, FrozenSet[str]] = {
    "switchyard/__init__.py": frozenset({
        "803225e89d5948090669b891470a33319b2525376c32297dd4bd0b02540e01b9",
        "b192b7077c10b10efefc314f815cf10376c9ca67e901bf6d2903d47579f012f1",
    }),
    "switchyard/libsy/__init__.py": frozenset({
        "2d59832883fbbebe913b3f8f33adbb8e5725d1246e3df7bfb9c6385b7d580b85",
        "3d6603701beaecaba22b495dec71f53bb429124c5c0b8799bd9e0dd1f3949fb5",
    }),
    "switchyard/libsy/algorithms.py": frozenset({
        "3abd2152af929fc1c8467d6dfbfec820d70f7952a88089fbba184058f432ef8d",
        "4ac5347284e2c6c26bdd63846da3b9f3852fca2b2abb6f9456fcb4c5a6e44185",
    }),
    "switchyard_rust/__init__.py": frozenset({
        "bf318bbd0080f7f225a8a31a5a17a79234a8a4d1e96331c085bc640c2bd11263",
        "c7d706f2440ed6fe8e56790fb545ee81533fbf4ceeef45b860bcad82e97e66c4",
    }),
    "switchyard_rust/_libsy_async.py": frozenset({
        "066b9ead958bbeaff36e1a1e86d0856ac096e02b6613473dfb6fe9a538c4d69e",
        "498295319b647933c5cd0077abf53471af5f55780a9de816a2358e41adcbc521",
    }),
    "switchyard_rust/_native.py": frozenset({
        "4ea429285a34d5a97702ee448eef80ad5b4f15662a03d5c2206a83f2aa0e5482",
        "f6718707a7071cce525c57db6681b02b115af022781f847b46bde1f729e3ceb1",
    }),
    "switchyard_rust/_switchyard_rust.abi3.so": frozenset({
        "03b47de76c3c22eb0c3836a4104527c6eafc69b839617866a872bb43e3ebefa3",
        "6c20a0c63019cb702a24afe602ddcf40e3a2420a80c4435e9ddbc9132f2b3d1b",
        "7ea87db246dccb11e360e288cb455a8de0dfe196fb20b186b12c8858a611f587",
        "dfcae93d5e238985fa73597deb4bae1dd4d5f05212d9c16c73814f1b366f02cc",
    }),
    "switchyard_rust/_switchyard_rust.pyd": frozenset({
        "64d9d137ed6a73f31503f762ab8364f3608d251af9dbfc80c1671ea54ab18463",
        "b6fccc3df93374e14f8950ec16c6cfd3fb0dc6e460875449c870563ef0ae9943",
    }),
    "switchyard_rust/libsy.py": frozenset({
        "2db4ff49629bd8471a0b59c90d393f53895e2927598fad7cad7cd352b11ed356",
        "d40adc27fbd20aca7466b78981c2ec5155f908b67a3382820b935b0ccfb6f325",
    }),
    "switchyard_rust/py.typed": frozenset({
        "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
        "7eb70257593da06f682a3ddda54a9d260d4fc514f645237f5ca74b08f8da61a6",
    }),
    "switchyard_rust/server.py": frozenset({
        "444bd4bce8cef3ce9554dda133c267a5636730089d280081f6f3b46fff40f21b",
        "649a0d0b79bd250f87cd5846e15c18681aba09182450a90c80fbf2e84658df69",
    }),
}
_REINSTALL = "reinstall the plugin with `hermes plugins install nvidia-switchyard --force`, which installs it"


@dataclass(frozen=True)
class Runtime:
    server: Any
    project: str
    version: str
    location: str


_chosen: Optional[Runtime] = None


class RuntimeMissing(RuntimeError):
    """``nemo-switchyard`` is not installed in this interpreter's environment, is not the pinned
    release, or does not load."""


def installed() -> Optional[Tuple[str, str]]:
    """``(project, version)`` of the installed Switchyard build, if any."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return _PROJECT, version(_PROJECT)
    except PackageNotFoundError:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def unverified(site: Path) -> Optional[str]:
    """Why the Switchyard files under ``site`` are not the pinned release's, or None when they are.

    Bytecode caches are skipped: the installer writes them, and only the account that owns the
    environment can.
    """
    found = {
        path.relative_to(site).as_posix(): path
        for package in _PACKAGES for path in (site / package).rglob("*")
        if path.is_file() and "__pycache__" not in path.relative_to(site).parts
    }
    for name, path in sorted(found.items()):
        if name not in _RELEASE_FILES:
            return f"{name} is not in the release"
        if _sha256(path) not in _RELEASE_FILES[name]:
            return f"{name} differs from the release"
    missing = sorted(name for name in _RELEASE_FILES if name not in found and name not in _NATIVE)
    if missing:
        return f"{missing[0]} is missing"
    if sum(name in found for name in _NATIVE) != 1:
        return "it does not hold exactly one native module"
    return None


def _site() -> Path:
    """The folder the installed ``nemo-switchyard`` package's files are under."""
    from importlib.metadata import distribution

    return Path(str(distribution(_PROJECT).locate_file("")))


def uses_upstream_names() -> bool:
    """Whether the installed build reads VGR routes as upstream Switchyard names them.

    Upstream VGR routes call the tiers efficient and capable, from ``nemo-switchyard``
    0.4 on and in ``nemo-switchyard-vgr``; the 0.3 ``nemo-switchyard`` builds the plugin
    has vendored call them local and cloud. Once a build is loaded, its names apply: Hermes can
    rebuild or take away the environment it came from while the process keeps it.
    """
    found = (_chosen.project, _chosen.version) if _chosen is not None else installed()
    if found is None:
        return False
    project, version = found
    release = _RELEASE.match(version)
    return project != _PROJECT or (release is not None and tuple(map(int, release.groups())) >= _FIRST_UPSTREAM)


def select() -> Runtime:
    """The installed Switchyard build."""
    global _chosen
    if _chosen is not None:
        return _chosen
    found = installed()
    if found is None:
        raise RuntimeMissing(f"{_PROJECT} is not installed in Hermes's environment; {_REINSTALL}")
    project, version = found
    if version != _PINNED:
        raise RuntimeMissing(f"Hermes's environment has {project} {version}, but the plugin is pinned to {_PINNED}; {_REINSTALL}")
    site = _site()
    problem = unverified(site)
    if problem:
        raise RuntimeMissing(f"the installed {project} {version} is not the published release: {problem}; {_REINSTALL}")
    spec = importlib.util.find_spec(_MODULE)
    origin = Path(spec.origin) if spec is not None and spec.origin else None
    if origin is None or origin.resolve() != (site / _MODULE / "__init__.py").resolve():
        raise RuntimeMissing(f"{_MODULE} would load from {origin}, not from the checked {project} {version}; {_REINSTALL}")
    try:
        from switchyard_rust.server import Server
    except ImportError as error:
        raise RuntimeMissing(f"the installed {project} {version} does not load ({error}); {_REINSTALL}") from None
    _chosen = Runtime(Server, project, version, str(Path(sys.modules[_MODULE].__file__ or "").parent))
    return _chosen


def server_class() -> Any:
    """Switchyard's in-process ``Server`` class; see :func:`select`."""
    return select().server


def describe() -> Optional[dict]:
    """The selected build, for status output; ``None`` before one is selected."""
    if _chosen is None:
        return None
    return {"project": _chosen.project, "version": _chosen.version, "location": _chosen.location}
