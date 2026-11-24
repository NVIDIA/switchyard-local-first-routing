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
_PINNED = "0.4.0"
_PACKAGES = ("switchyard", "switchyard_rust")
# Each wheel holds one native module: the .pyd on Windows, the .so elsewhere.
_NATIVE = ("switchyard_rust/_switchyard_rust.abi3.so", "switchyard_rust/_switchyard_rust.pyd")
# SHA-256 of every file in the six published 0.4.0 wheels, by path under site-packages. The
# Windows wheels' Python files differ in line endings, and each platform's native module differs.
_RELEASE_FILES: Dict[str, FrozenSet[str]] = {
    "switchyard/__init__.py": frozenset({
        "803225e89d5948090669b891470a33319b2525376c32297dd4bd0b02540e01b9",
        "b192b7077c10b10efefc314f815cf10376c9ca67e901bf6d2903d47579f012f1",
    }),
    "switchyard/libsy/__init__.py": frozenset({
        "e607c609dd74dc8eef4bd5e3b971d4e1e699b52794cc430d80f4cc927652ab73",
        "fb85157198eb7adf910dbe3bafe5a3b78caebe09fc962439782f9f9eacca4e9b",
    }),
    "switchyard/libsy/algorithms.py": frozenset({
        "3abd2152af929fc1c8467d6dfbfec820d70f7952a88089fbba184058f432ef8d",
        "4ac5347284e2c6c26bdd63846da3b9f3852fca2b2abb6f9456fcb4c5a6e44185",
    }),
    "switchyard/translation.py": frozenset({
        "415ff086488628742b0c7fe64cbc99b7caa80c1cfa09e45cf3c4bdf697b2ecbe",
        "6b812738aa76134b3e484f15cabdb68e5d753689b68b475d617a303a55efefa3",
    }),
    "switchyard_rust/__init__.py": frozenset({
        "bf318bbd0080f7f225a8a31a5a17a79234a8a4d1e96331c085bc640c2bd11263",
        "c7d706f2440ed6fe8e56790fb545ee81533fbf4ceeef45b860bcad82e97e66c4",
    }),
    "switchyard_rust/_libsy_async.py": frozenset({
        "5035577e8bbe06f132656076de43a0da04a2aba1fa5368eec973c6e60be44bee",
        "c327421c07cb1cc08ad0f700113d43e138391ac6fffd84b8140eabe25b7eb574",
    }),
    "switchyard_rust/_native.py": frozenset({
        "4ea429285a34d5a97702ee448eef80ad5b4f15662a03d5c2206a83f2aa0e5482",
        "f6718707a7071cce525c57db6681b02b115af022781f847b46bde1f729e3ceb1",
    }),
    "switchyard_rust/_switchyard_rust.abi3.so": frozenset({
        "2c2ec71e9b92aa56cad699456b833cec6c1d474113f2e4fd55eeb12556bf8207",
        "d63c902d0a94dcf144511e77575ce5feed8268508f34202206dda472b3f7a987",
        "d7d60518a087e42efa4c279bdeb508fe2d17a3e5beb3a2aa297576cf045214d8",
        "e680d19b89a2b15aeae7ee690c589be14b1f219d07378f498f68f1ed31baeb04",
    }),
    "switchyard_rust/_switchyard_rust.pyd": frozenset({
        "d3cc515019cc3ed08f4a9d240e8087add9515488c600677c001da7e9348ecfe0",
        "ed1ba5310d4c39b41a8694edf2fcba5e9c4924f051509bc94f74711d2ff6036b",
    }),
    "switchyard_rust/_translation_stream.py": frozenset({
        "9f2fd7a3cc524c70681d39000688cb7903437d5bc52050a730ec77a08a424d1f",
        "e5f13d9db51ca760342b271acfec61bd5793e45f044433267c85d5ae4cb33052",
    }),
    "switchyard_rust/libsy.py": frozenset({
        "f0bbb48619d66a0b7a4e27cd16a31949a24da47fda4c4745b70d892e28750faf",
        "f1dc26d9f02841ca2aec7887162f03c0fd5d91bb3279b270846707e3e9c0d9a4",
    }),
    "switchyard_rust/py.typed": frozenset({
        "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
        "7eb70257593da06f682a3ddda54a9d260d4fc514f645237f5ca74b08f8da61a6",
    }),
    "switchyard_rust/server.py": frozenset({
        "444bd4bce8cef3ce9554dda133c267a5636730089d280081f6f3b46fff40f21b",
        "649a0d0b79bd250f87cd5846e15c18681aba09182450a90c80fbf2e84658df69",
    }),
    "switchyard_rust/translation.py": frozenset({
        "5111f2762246f97e20faa951f28cd75d70e5eb7ebb60b840090ce90e8814af63",
        "d3c4a9627dba1160b55065009dcf3b364b5aadb6c1813244ad63eea5c3286a21",
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
