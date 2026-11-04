# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loading the Switchyard build Hermes installed with the plugin."""

from __future__ import annotations

import pytest

from plugins.switchyard import runtime


@pytest.fixture(autouse=True)
def _unselected(monkeypatch):
    monkeypatch.setattr(runtime, "_chosen", None)


def test_the_installed_build_is_loaded_once_and_described():
    chosen = runtime.select()
    assert chosen.server.__name__ == "Server"
    assert runtime.select() is chosen
    project, version = runtime.installed()
    assert runtime.describe() == {"project": project, "version": version, "location": chosen.location}


@pytest.mark.parametrize(("found", "upstream"), [
    (("nemo-switchyard", "0.4.0a1"), True),
    (("nemo-switchyard", "0.4.0"), True),
    (("nemo-switchyard", "0.3.0"), False),
    (("nemo-switchyard-vgr", "0.3.0"), True),
    (None, False),
])
def test_vgr_route_names_follow_the_installed_build(monkeypatch, found, upstream):
    monkeypatch.setattr(runtime, "installed", lambda: found)
    assert runtime.uses_upstream_names() is upstream


def test_vgr_route_names_follow_the_loaded_build_once_its_environment_is_gone(monkeypatch):
    runtime.select()
    upstream = runtime.uses_upstream_names()
    monkeypatch.setattr(runtime, "installed", lambda: None)
    assert runtime.uses_upstream_names() is upstream


def test_a_missing_build_names_the_reinstall(monkeypatch):
    monkeypatch.setattr(runtime, "installed", lambda: None)
    with pytest.raises(runtime.RuntimeMissing, match="not installed .*`hermes plugins install nvidia-switchyard --force`"):
        runtime.select()
    assert runtime.describe() is None
