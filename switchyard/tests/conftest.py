# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""These tests live outside ``tests/``, so its HERMES_HOME sandbox does not apply."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    # A spawned ``hermes`` finds its bundled plugins from where ``hermes_cli`` resolves, which a
    # symlinked tree points at another checkout; the plugin it loads must be this one.
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(Path(__file__).parents[2]))
    return home


@pytest.fixture(autouse=True)
def _no_sidecar_outlives_its_test(monkeypatch):
    """A started sidecar is detached on purpose, so one a test leaves behind keeps serving after the run:
    a failed start whose process came up late, or a stop that missed it."""
    from plugins.switchyard import sidecar

    spawned: list[subprocess.Popen] = []
    spawn = sidecar.spawn_detached

    def tracked(*args, **kwargs):
        process = spawn(*args, **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr(sidecar, "spawn_detached", tracked)
    yield
    for process in spawned:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
