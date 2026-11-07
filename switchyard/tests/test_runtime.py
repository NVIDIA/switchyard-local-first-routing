# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loading the Switchyard build Hermes installed with the plugin."""

from __future__ import annotations

import shutil

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


def test_only_the_published_release_files_pass(tmp_path):
    for package in runtime._PACKAGES:
        shutil.copytree(runtime._site() / package, tmp_path / package,
                        ignore=shutil.ignore_patterns("__pycache__"))
    assert runtime.unverified(tmp_path) is None
    server = tmp_path / "switchyard_rust" / "server.py"
    original = server.read_bytes()
    server.write_bytes(original + b"\n")
    assert runtime.unverified(tmp_path) == "switchyard_rust/server.py differs from the release"
    server.write_bytes(original)
    # A version-tagged extension module would load ahead of the release's abi3 one.
    shadow = tmp_path / "switchyard_rust" / "_switchyard_rust.cpython-314-x86_64-linux-gnu.so"
    shadow.write_bytes(b"")
    assert runtime.unverified(tmp_path) == f"{shadow.relative_to(tmp_path).as_posix()} is not in the release"
    shadow.unlink()
    server.unlink()
    assert runtime.unverified(tmp_path) == "switchyard_rust/server.py is missing"


def test_a_build_that_is_not_the_pinned_release_is_not_loaded(monkeypatch):
    monkeypatch.setattr(runtime, "installed", lambda: ("nemo-switchyard", "0.4.0"))
    with pytest.raises(runtime.RuntimeMissing, match="pinned to 0.4.0a1"):
        runtime.select()
    monkeypatch.setattr(runtime, "installed", lambda: ("nemo-switchyard", "0.4.0a1"))
    monkeypatch.setattr(runtime, "unverified", lambda site: "switchyard_rust/server.py differs from the release")
    with pytest.raises(runtime.RuntimeMissing, match="not the published release: switchyard_rust/server.py differs"):
        runtime.select()
    assert runtime.describe() is None


def test_a_missing_build_names_the_reinstall(monkeypatch):
    monkeypatch.setattr(runtime, "installed", lambda: None)
    with pytest.raises(runtime.RuntimeMissing, match="not installed .*`hermes plugins install nvidia-switchyard --force`"):
        runtime.select()
    assert runtime.describe() is None
