# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Switchyard's native runtime for the sidecar.

The plugin declares ``nemo-switchyard`` under ``python_dependencies``, so
Hermes installs it from PyPI into its own environment when it installs the
plugin, and the sidecar imports it from there.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

_PROJECT = "nemo-switchyard"
# The first nemo-switchyard release with VGR, as upstream names its tiers.
_FIRST_UPSTREAM = (0, 4)
# Builds of the same modules under other names, used when the declared one is absent.
_ALSO = ("nemo-switchyard-vgr",)
_RELEASE = re.compile(r"^(\d+)\.(\d+)")
_MODULE = "switchyard_rust"
_REINSTALL = "reinstall the plugin with `hermes plugins install nvidia-switchyard --force`, which installs it"


@dataclass(frozen=True)
class Runtime:
    server: Any
    project: str
    version: str
    location: str


_chosen: Optional[Runtime] = None


class RuntimeMissing(RuntimeError):
    """``nemo-switchyard`` is not installed in this interpreter's environment, or does not load."""


def installed() -> Optional[Tuple[str, str]]:
    """``(project, version)`` of the installed Switchyard build, if any."""
    from importlib.metadata import PackageNotFoundError, version

    for project in (_PROJECT, *_ALSO):
        try:
            return project, version(project)
        except PackageNotFoundError:
            continue
    return None


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
